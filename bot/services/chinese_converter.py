"""
Simplified to Traditional Chinese converter processor.

Converts all text (user transcripts and LLM responses) to Traditional Chinese (Taiwan).
"""

import opencc
from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    LLMTextFrame,
)
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService


# Global converter instance for reuse
_converter = None

def get_converter():
    """Get or create the OpenCC converter singleton."""
    global _converter
    if _converter is None:
        _converter = opencc.OpenCC('s2twp')
    return _converter


def convert_to_traditional(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese (Taiwan style)."""
    if not text:
        return text
    converter = get_converter()
    converted = converter.convert(text)
    if converted != text:
        logger.debug(f"Converted: '{text}' → '{converted}'")
    return converted


class ConvertingElevenLabsSTTService(ElevenLabsSTTService):
    """
    ElevenLabs STT service that converts output to Traditional Chinese.

    This wraps the STT service so conversion happens BEFORE frames are pushed,
    ensuring RTVI observers see the converted text.
    """

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Override push_frame to convert text before pushing."""
        # Convert transcription frames before they're observed by RTVI
        if isinstance(frame, TranscriptionFrame):
            frame.text = convert_to_traditional(frame.text)
        elif isinstance(frame, InterimTranscriptionFrame):
            frame.text = convert_to_traditional(frame.text)

        await super().push_frame(frame, direction)


class SimplifiedToTraditionalProcessor(FrameProcessor):
    """Converts Simplified Chinese text to Traditional Chinese (Taiwan style)."""

    def __init__(self):
        super().__init__()
        logger.info("SimplifiedToTraditionalProcessor: Initialized with s2twp conversion")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Convert user transcriptions (from STT) - backup in case STT wrapper missed
        if isinstance(frame, TranscriptionFrame):
            frame.text = convert_to_traditional(frame.text)

        elif isinstance(frame, InterimTranscriptionFrame):
            frame.text = convert_to_traditional(frame.text)

        # Convert LLM text output
        elif isinstance(frame, TextFrame):
            frame.text = convert_to_traditional(frame.text)

        elif isinstance(frame, LLMTextFrame):
            frame.text = convert_to_traditional(frame.text)

        await self.push_frame(frame, direction)
