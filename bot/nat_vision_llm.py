"""
Custom LLM service that automatically fetches user images for vision queries.

This integrates with the NAT router to automatically capture images when
the LLM classifier determines a query requires visual understanding.
"""

import asyncio
import base64
import io
import aiohttp
from typing import Optional

from loguru import logger
from PIL import Image
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    UserImageRequestFrame,
    UserImageRawFrame,
    StartInterruptionFrame,
    InputAudioRawFrame,
    UserSpeakingFrame,
    BotSpeakingFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService


# Quick regex patterns for obvious vision questions (avoid LLM call)
import re
VISION_KEYWORDS_PATTERN = re.compile(
    r'(手[上裡中]?|拿|握|holding|hold|showing|show me|'  # hand/holding
    r'看[起来來]|look|appear|'  # appearance
    r'穿|wearing|wear|衣服|clothes|'  # clothing
    r'臉|face|表情|expression|'  # face
    r'背景|background|房間|room|環境|周[圍围]|surroundings|'  # environment
    r'這是|这是|what is this|what\'s this|'  # pointing at something
    r'我[的是]什[麼么]|what am i|how do i look)',  # about user
    re.IGNORECASE
)

# Few-shot prompt for ambiguous cases
VISION_CLASSIFIER_PROMPT = """Determine if this question requires seeing the user through their camera.

Examples:
- "我手上有什麼？" (What's in my hand?) -> YES
- "你看到什麼？" (What do you see?) -> YES
- "我穿得好看嗎？" (Do I look good?) -> YES
- "這是什麼東西？" (What is this thing?) -> YES
- "今天天氣如何？" (How's the weather today?) -> NO
- "法國的首都是哪裡？" (What's the capital of France?) -> NO
- "幫我寫一首詩" (Write me a poem) -> NO

Question: {question}
Answer (YES or NO):"""


class NATVisionLLMService(NvidiaLLMService):
    """
    Custom NVIDIA LLM service that uses LLM classification to decide when to fetch images.

    When it receives an LLMContextFrame, it:
    1. Uses a quick LLM call to classify if the question needs vision
    2. Only fetches camera image if classification returns YES
    3. Adds the image to context in OpenAI's multimodal format
    4. Sends to NAT for processing

    This saves ~16K tokens per turn for non-visual questions.
    """

    # vLLM model name for vision requests
    VISION_MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"

    def __init__(self, *args, user_id: Optional[str] = None, max_image_dimension: int = 256,
                 image_quality: int = 40, classifier_url: str = "http://localhost:8000/v1",
                 vision_url: str = "http://localhost:8000/v1", **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("NATVisionLLMService: Initialized with LLM-based vision classification")
        logger.info(f"  max_dimension={max_image_dimension}, quality={image_quality}")
        logger.info(f"  classifier_url={classifier_url}")
        logger.info(f"  vision_url={vision_url} (direct vLLM for vision requests)")
        self._user_id = user_id
        self._pending_image_future: Optional[asyncio.Future] = None
        self._last_image: Optional[UserImageRawFrame] = None
        self._current_turn_has_image = False
        self._max_image_dimension = max_image_dimension
        self._image_quality = image_quality
        self._classifier_url = classifier_url
        self._vision_url = vision_url  # Direct vLLM URL for vision requests
        self._pending_messages_frame: Optional[Frame] = None

    def set_user_id(self, user_id: str):
        """Set the user ID for image requests."""
        self._user_id = user_id
        logger.debug(f"NATVisionLLMService: User ID set to {user_id}")

    async def _needs_vision(self, user_message: str) -> bool:
        """
        Use LLM to classify if the user's question requires visual understanding.

        Args:
            user_message: The user's text question

        Returns:
            True if the question needs vision, False otherwise
        """
        if not user_message or len(user_message.strip()) < 2:
            return False

        try:
            prompt = VISION_CLASSIFIER_PROMPT.format(question=user_message)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._classifier_url}/chat/completions",
                    json={
                        "model": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 5,  # Only need YES or NO
                        "temperature": 0.0,  # Deterministic
                    },
                    timeout=aiohttp.ClientTimeout(total=3.0)  # Quick timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                        needs_vision = answer.startswith("YES")
                        logger.info(f"Vision classifier: '{user_message[:50]}...' -> {answer} -> needs_vision={needs_vision}")
                        return needs_vision
                    else:
                        logger.warning(f"Vision classifier failed with status {response.status}, defaulting to NO")
                        return False

        except asyncio.TimeoutError:
            logger.warning("Vision classifier timeout, defaulting to NO")
            return False
        except Exception as e:
            logger.error(f"Vision classifier error: {e}, defaulting to NO")
            return False

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to stay within max dimension while preserving aspect ratio."""
        width, height = image.size
        max_dim = self._max_image_dimension

        if width <= max_dim and height <= max_dim:
            return image

        if width > height:
            new_width = max_dim
            new_height = int((max_dim / width) * height)
        else:
            new_height = max_dim
            new_width = int((max_dim / height) * width)

        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return resized

    def _encode_image_to_base64(self, frame: UserImageRawFrame) -> Optional[str]:
        """
        Encode UserImageRawFrame to base64 JPEG data URL.

        Returns:
            Data URL string like "data:image/jpeg;base64,..." or None on error
        """
        try:
            image_format = getattr(frame, 'format', 'RGB')
            image_size = getattr(frame, 'size', (640, 360))

            image = Image.frombytes(image_format, image_size, frame.image)
            image = self._resize_image(image)

            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None
                )
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._image_quality)
            image_bytes = buffer.getvalue()

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            logger.info(f"Image encoded: {len(image_b64)} bytes")

            return f"data:image/jpeg;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _add_image_to_context(self, context, image_data_url: str, user_message: str):
        """
        Manually add image to the last user message in context using OpenAI's multimodal format.
        """
        if not context:
            logger.warning("No context provided, cannot add image")
            return

        messages = context.messages
        if not messages:
            logger.warning("No messages in context, cannot add image")
            return

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                current_content = messages[i].get("content", "")

                if isinstance(current_content, list):
                    messages[i]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "low"}  # Use low detail to save tokens
                    })
                    logger.info("Appended image to existing multimodal user message")
                else:
                    messages[i]["content"] = [
                        {"type": "text", "text": current_content or user_message},
                        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "low"}}
                    ]
                    logger.info("Converted user message to multimodal format with image")

                logger.info("Image successfully added to context!")
                return

        logger.warning("No user message found in context to add image to")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Intercept frames to handle smart image fetching based on LLM classification.
        """
        if not isinstance(frame, (InputAudioRawFrame, UserImageRawFrame, UserSpeakingFrame, BotSpeakingFrame)):
            logger.info(f"NATVisionLLMService.process_frame: {type(frame).__name__}, direction={direction}")

        # Reset image flag on new turn (interruption or user starts speaking)
        if isinstance(frame, (StartInterruptionFrame, UserStartedSpeakingFrame)):
            self._current_turn_has_image = False
            logger.debug(f"NATVisionLLMService: Reset _current_turn_has_image on {type(frame).__name__}")

        if isinstance(frame, UserImageRawFrame):
            self._last_image = frame

            if self._pending_image_future and not self._pending_image_future.done():
                logger.info("NATVisionLLMService: Image captured for pending request")
                self._pending_image_future.set_result(frame)

            return

        if isinstance(frame, LLMContextFrame):
            has_user_message = any(msg.get("role") == "user" for msg in frame.context.messages)

            if not has_user_message:
                logger.debug("NATVisionLLMService: No user message, skipping")
                await super().process_frame(frame, direction)
                return

            if not self._current_turn_has_image:
                # Extract user message for classification
                user_message = ""
                for msg in reversed(frame.context.messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            user_message = content
                        elif isinstance(content, list):
                            for item in content:
                                if item.get("type") == "text":
                                    user_message = item.get("text", "")
                                    break
                        break

                # Use LLM to classify if vision is needed
                needs_vision = await self._needs_vision(user_message)

                if needs_vision:
                    logger.info(f"NATVisionLLMService: Vision needed for '{user_message[:30]}...', fetching image")

                    await self._fetch_and_wait_for_image(frame)

                    if self._last_image:
                        image_data_url = self._encode_image_to_base64(self._last_image)

                        if image_data_url:
                            context = getattr(frame, 'context', None)
                            self._add_image_to_context(context, image_data_url, user_message)
                            self._current_turn_has_image = True
                            logger.info("NATVisionLLMService: Image added to context")
                    else:
                        logger.warning("NATVisionLLMService: No image received")
                else:
                    logger.info(f"NATVisionLLMService: No vision needed for '{user_message[:30]}...', skipping image")
                    self._current_turn_has_image = True  # Mark as processed (no image needed)

            # Check if context has image - if so, use vLLM instead of NAT
            if self._context_has_image(frame):
                logger.info(f"NATVisionLLMService: Context has image, routing to vLLM at {self._vision_url}")
                # Temporarily swap base_url and model to use vLLM for vision requests
                original_base_url = self._client.base_url
                original_model = self.model_name  # Use property, not _model_name
                self._client.base_url = self._vision_url
                self.set_model_name(self.VISION_MODEL)
                try:
                    await super().process_frame(frame, direction)
                finally:
                    # Restore original base_url and model
                    self._client.base_url = original_base_url
                    self.set_model_name(original_model)
                return

        await super().process_frame(frame, direction)

    def _context_has_image(self, frame: Frame) -> bool:
        """Check if the LLMContextFrame has any image content."""
        if not isinstance(frame, LLMContextFrame):
            return False
        for msg in frame.context.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return True
        return False

    async def _process_vision_request_direct(self, frame: LLMContextFrame, direction: FrameDirection):
        """Send vision request directly to vLLM, bypassing NAT."""
        from pipecat.frames.frames import LLMFullResponseStartFrame, LLMFullResponseEndFrame, TextFrame

        try:
            messages = frame.context.messages

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._vision_url}/chat/completions",
                    json={
                        "model": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD",
                        "messages": messages,
                        "max_tokens": 500,
                        "temperature": 0.7,
                        "stream": False,
                    },
                    timeout=aiohttp.ClientTimeout(total=60.0)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        logger.info(f"NATVisionLLMService: Vision response received: {answer[:100]}...")

                        # Push response frames
                        await self.push_frame(LLMFullResponseStartFrame(), direction)
                        await self.push_frame(TextFrame(text=answer), direction)
                        await self.push_frame(LLMFullResponseEndFrame(), direction)
                    else:
                        error_text = await response.text()
                        logger.error(f"NATVisionLLMService: Vision request failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"NATVisionLLMService: Vision request error: {e}")

    async def _fetch_and_wait_for_image(self, context_frame: LLMContextFrame):
        """
        Request a user image and wait for it to arrive.
        """
        if not self._user_id:
            logger.warning("NATVisionLLMService: No user_id set, cannot fetch image")
            return

        user_message = ""
        for msg in reversed(context_frame.context.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                    break

        self._pending_image_future = asyncio.Future()

        await self.push_frame(
            UserImageRequestFrame(
                user_id=self._user_id,
                text=user_message,
                append_to_context=False,
            ),
            FrameDirection.UPSTREAM,
        )

        try:
            await asyncio.wait_for(self._pending_image_future, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("NATVisionLLMService: Timeout waiting for user image")
        finally:
            self._pending_image_future = None
