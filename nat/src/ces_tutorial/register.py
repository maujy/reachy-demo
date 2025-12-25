# Import functions module to trigger registration of custom functions
from ces_tutorial import functions

# Monkey-patch NAT's ChatRequest to use our OpenAI-compatible schema
import nat.data_models.api_server
from ces_tutorial.openai_chat_request import OpenAIChatRequest

# Replace NAT's ChatRequest with our custom one that supports multimodal content
nat.data_models.api_server.ChatRequest = OpenAIChatRequest

__all__ = ["functions"]
