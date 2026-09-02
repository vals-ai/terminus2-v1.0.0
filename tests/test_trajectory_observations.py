import terminus2.terminus_2 as terminus_module


def test_single_command_is_one_tool_call() -> None:
    commands = [terminus_module.Command(keystrokes="ls\n", duration_sec=1.0)]

    tool_calls = terminus_module._terminal_tool_calls(commands, episode=3)

    assert [call.tool_call_id for call in tool_calls] == ["call_3_1"]
    assert tool_calls[0].function_name == "bash_command"


def test_batched_commands_are_one_tool_call() -> None:
    commands = [
        terminus_module.Command(keystrokes="ls\n", duration_sec=1.0),
        terminus_module.Command(keystrokes="pwd\n", duration_sec=1.0),
    ]

    tool_calls = terminus_module._terminal_tool_calls(commands, episode=3)

    assert [call.tool_call_id for call in tool_calls] == ["call_3_terminal_batch"]
    assert tool_calls[0].function_name == "terminal_batch"
    assert tool_calls[0].arguments == {
        "commands": [
            {"keystrokes": "ls\n", "duration": 1.0},
            {"keystrokes": "pwd\n", "duration": 1.0},
        ],
        "task_complete": False,
    }


def test_task_completion_is_one_tool_call() -> None:
    tool_calls = terminus_module._terminal_tool_calls([], episode=3, is_task_complete=True)

    assert [call.tool_call_id for call in tool_calls] == ["call_3_task_complete"]
    assert tool_calls[0].function_name == "mark_task_complete"


def test_command_plus_completion_is_one_batch_tool_call() -> None:
    commands = [terminus_module.Command(keystrokes="ls\n", duration_sec=1.0)]

    tool_calls = terminus_module._terminal_tool_calls(commands, episode=3, is_task_complete=True)

    assert [call.tool_call_id for call in tool_calls] == ["call_3_terminal_batch"]
    assert tool_calls[0].arguments["task_complete"] is True
