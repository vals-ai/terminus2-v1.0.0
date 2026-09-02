"""History entries a provider will not accept back are dropped before reuse."""

from terminus2.llms.chat import _without_unsendable_turns


class _Message:
    """Shaped like the provider message model-library wraps in a RawResponse."""

    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _RawResponse:
    """model-library stores an assistant turn as RawResponse(response=<message>)."""

    kind = "raw_response"

    def __init__(self, response):
        self.response = response


class _TextInput:
    """A prompt carries no `response` field at all."""

    kind = "text_input"

    def __init__(self, text):
        self.text = text


def test_an_assistant_turn_with_no_content_and_no_tool_calls_is_dropped():
    """This is the turn a reasoning model leaves when it emits nothing."""
    prompt = _TextInput("solve it")
    empty = _RawResponse(_Message(content=None, tool_calls=None))

    assert _without_unsendable_turns([prompt, empty]) == [prompt]


def test_a_turn_with_content_is_kept():
    kept = _RawResponse(_Message(content="running ls"))

    assert _without_unsendable_turns([kept]) == [kept]


def test_a_turn_with_only_tool_calls_is_kept():
    """An action with no prose is still an action."""
    kept = _RawResponse(_Message(content=None, tool_calls=[{"id": "1"}]))

    assert _without_unsendable_turns([kept]) == [kept]


def test_prompts_and_tool_results_are_never_dropped():
    """Only the assistant wrapper is inspected; other InputItem kinds pass through."""
    items = [_TextInput(""), _TextInput("next")]

    assert _without_unsendable_turns(items) == items


def test_a_dict_shaped_provider_message_is_handled():
    empty = _RawResponse({"role": "assistant", "content": None, "tool_calls": None})
    kept = _RawResponse({"role": "assistant", "content": "ok"})

    assert _without_unsendable_turns([empty, kept]) == [kept]


def test_a_non_assistant_role_inside_the_wrapper_is_kept():
    kept = _RawResponse(_Message(role="user", content=None))

    assert _without_unsendable_turns([kept]) == [kept]


def test_a_responses_api_output_list_is_left_alone():
    """That path stores a list of output items; judging it is not attempted."""
    kept = _RawResponse([{"type": "message"}])

    assert _without_unsendable_turns([kept]) == [kept]
