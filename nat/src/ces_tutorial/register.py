# Import functions module to trigger registration of custom functions
from ces_tutorial import functions

# Monkey-patch NAT's ChatRequest to use our OpenAI-compatible schema
import nat.data_models.api_server
from ces_tutorial.openai_chat_request import OpenAIChatRequest, MessageDict

# Replace NAT's ChatRequest with our custom one that supports multimodal content
nat.data_models.api_server.ChatRequest = OpenAIChatRequest

# -------------------------------------------------------------------------
# NEW: Register converter for React Agent compatibility
# -------------------------------------------------------------------------
from nat.data_models.api_server import ChatRequestOrMessage
from nat.utils.type_converter import GlobalTypeConverter

def crom_to_openai_request(crom: ChatRequestOrMessage) -> OpenAIChatRequest:
    """
    Converts a standard NAT ChatRequestOrMessage into the custom OpenAIChatRequest.
    Manually wraps message dicts into MessageDict to satisfy react_agent's 
    expectation of objects with a .model_dump() method.
    """
    # Extract data from the standard object
    data = crom.model_dump(exclude_none=True)
    
    # Handle the input_message case (string input)
    if "input_message" in data:
        input_msg = data.pop("input_message")
        model = data.pop("model", "gpt-4o")
        
        # Create a MessageDict for the user message
        msgs = [MessageDict(role="user", content=input_msg)]
        
        return OpenAIChatRequest(
            messages=msgs,
            model=model,
            **data
        )

    # Handle the conversation case (messages list)
    # We MUST pop 'messages' and 'model' to avoid "multiple values" TypeError
    raw_messages = data.pop("messages", [])
    model = data.pop("model", "gpt-4o") 
    
    # Wrap standard dicts in MessageDict so they have .model_dump()
    wrapped_messages = []
    for msg in raw_messages:
        if isinstance(msg, dict) and not isinstance(msg, MessageDict):
            wrapped_messages.append(MessageDict(msg))
        else:
            wrapped_messages.append(msg)

    return OpenAIChatRequest(messages=wrapped_messages, model=model, **data)

GlobalTypeConverter.register_converter(crom_to_openai_request)
# -------------------------------------------------------------------------

__all__ = ["functions"]