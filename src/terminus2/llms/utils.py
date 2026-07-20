from model_library.base import LLM


async def get_model_context_limit(llm: LLM) -> int:
    """Return the context window loaded with the registry-configured model."""
    context_window = llm.input_context_window
    assert context_window is not None, "no context window found for " + llm.model_name
    return context_window
