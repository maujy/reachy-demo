#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
import aiohttp
import asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nat_vision_llm import NATVisionLLMService
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import OutputImageRawFrame, Frame


load_dotenv(override=True)


class BotVideoSource(FrameProcessor):
    """Streams video from Reachy robot camera, falls back to placeholder."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._running = False
        self._task = None
        self._frame_count = 0

    def _create_placeholder_frame(self) -> bytes:
        """Create a placeholder image when robot camera unavailable."""
        img = Image.new('RGB', (self._width, self._height), color=(30, 30, 50))
        draw = ImageDraw.Draw(img)

        # Animated circle (breathing effect)
        pulse = abs(np.sin(self._frame_count * 0.1)) * 20 + 80
        center_x, center_y = self._width // 2, self._height // 2
        radius = int(pulse)
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=(100, 150, 200),
            outline=(150, 200, 255)
        )

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = "小芝 AI"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (self._width - text_width) // 2
        draw.text((text_x, center_y + radius + 20), text, fill=(255, 255, 255), font=font)

        self._frame_count += 1
        return img.tobytes()

    def _get_robot_frame(self) -> bytes | None:
        """Try to get frame from Reachy robot camera."""
        try:
            reachy = ReachyService.get_instance()
            if reachy.connected and reachy.robot:
                frame = reachy.robot.media.get_frame()
                if frame is not None:
                    # Convert to PIL, resize, and return RGB bytes
                    img = Image.fromarray(frame)
                    img = img.resize((self._width, self._height))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    return img.tobytes()
        except Exception as e:
            logger.debug(f"Robot camera unavailable: {e}")
        return None

    async def _generate_frames(self):
        """Background task to generate video frames."""
        interval = 1.0 / self._fps
        while self._running:
            try:
                # Try robot camera first, fall back to placeholder
                frame_data = self._get_robot_frame()
                if frame_data is None:
                    frame_data = self._create_placeholder_frame()

                frame = OutputImageRawFrame(
                    image=frame_data,
                    size=(self._width, self._height),
                    format="RGB"
                )
                await self.push_frame(frame, FrameDirection.DOWNSTREAM)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"BotVideoSource error: {e}")
                await asyncio.sleep(interval)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Start generating frames on first frame
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._generate_frames())

        await self.push_frame(frame, direction)

    async def cleanup(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,  # Enable video output for bot avatar
        video_out_width=640,
        video_out_height=480,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Connect to Reachy robot early so camera is available
    reachy = ReachyService.get_instance()
    if not reachy.connected:
        reachy.connect()

    async with aiohttp.ClientSession() as session:

        stt = ElevenLabsSTTService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            aiohttp_session=session,
            params=ElevenLabsSTTService.InputParams(
                language=Language.ZH,  # Force Chinese transcription
            ),
        )

        tts = ElevenLabsHttpTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model="eleven_multilingual_v2",  # Support Chinese TTS
            aiohttp_session=session,
        )

        llm = NATVisionLLMService(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="http://localhost:8001/v1",
        )

        # Get current date/time for system prompt
        now = datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_date = f"{now.year}年{now.month}月{now.day}日 {weekdays[now.weekday()]}"
        current_time = f"{now.hour}時{now.minute}分"

        messages = [
            {
                "role": "system",
                "content": f"""你是小芝（Sage），一個友善的AI語音助理。

當前時間：{current_date} {current_time}

重要規則：
1. 必須使用繁體中文回答（台灣用語）
2. 回答要簡潔，適合語音輸出（2-3句話）
3. 避免使用特殊符號、emoji、項目符號
4. 保持親切友善的語氣
5. 你可以看到並描述使用者的攝影機畫面

你支援的語言：繁體中文、English、日本語
請根據使用者的語言回應。""",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()

        # Create video source for bot avatar
        video_source = BotVideoSource(width=640, height=480, fps=10)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI protocol processor
                stt,  # STT
                transcript.user(),  # Capture user transcripts
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                ReachyWobblerProcessor(),
                video_source,  # Generate placeholder video frames
                transport.output(),  # Transport bot output
                transcript.assistant(),  # Capture assistant transcripts
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_transcript_update(processor, frame):
            """Handle transcript updates and send them to the web UI"""
            for message in frame.messages:
                logger.info(f"Transcript [{message.role}]: {message.content}")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")

            await maybe_capture_participant_camera(transport, client)

            client_id = get_transport_client_id(transport, client)
            
            # Set the user_id for automatic image fetching
            llm.set_user_id(client_id)

            # Send static greeting directly to TTS (bypasses LLM for instant response)
            greeting = "你好！我是小芝，很高興認識你。有什麼我可以幫助你的嗎？"
            await task.queue_frames([TTSSpeakFrame(text=greeting)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys
    from pipecat.runner.run import main

    # Default to binding to all interfaces for network access
    if "--host" not in sys.argv:
        sys.argv.extend(["--host", "0.0.0.0"])

    main()