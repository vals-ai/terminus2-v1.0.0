from types import SimpleNamespace

from terminus2.llms.chat import _without_unsendable_turns


def test_an_assistant_turn_with_neither_content_nor_tool_calls_is_dropped():
    """A reasoning model can return no content, and providers reject that turn."""
    history = [
        {"role": "user", "content": "solve it"},
        {"role": "assistant", "content": None, "tool_calls": None},
        {"role": "user", "content": "continue"},
    ]

    assert _without_unsendable_turns(history) == [history[0], history[2]]


def test_assistant_turns_carrying_content_or_tool_calls_are_kept():
    history = [
        {"role": "assistant", "content": "done"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    ]

    assert _without_unsendable_turns(history) == history


def test_non_assistant_turns_are_never_dropped():
    """Only the assistant turn is subject to the provider's rule."""
    history = [{"role": "user", "content": ""}, {"role": "system", "content": None}]

    assert _without_unsendable_turns(history) == history


def test_object_shaped_history_entries_are_handled():
    """History carries model_library input objects, not only dicts."""
    kept = SimpleNamespace(role="assistant", content="ok", tool_calls=None)
    dropped = SimpleNamespace(role="assistant", content=None, tool_calls=None)
    text = SimpleNamespace(role="user", content="hi")

    assert _without_unsendable_turns([text, kept, dropped]) == [text, kept]
