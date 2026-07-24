import terminus2.terminus_2 as terminus_module


def test_single_command_observation_references_command_tool_call() -> None:
    commands = [terminus_module.Command(keystrokes="ls\n", duration_sec=1.0)]

    assert (
        terminus_module._terminal_observation_source_call_id(commands, episode=3)
        == "call_3_1"
    )


def test_batched_command_observation_stays_step_level() -> None:
    commands = [
        terminus_module.Command(keystrokes="ls\n", duration_sec=1.0),
        terminus_module.Command(keystrokes="pwd\n", duration_sec=1.0),
    ]

    assert (
        terminus_module._terminal_observation_source_call_id(commands, episode=3)
        is None
    )


def test_task_completion_observation_references_completion_tool_call() -> None:
    assert (
        terminus_module._terminal_observation_source_call_id(
            [], episode=3, is_task_complete=True
        )
        == "call_3_task_complete"
    )


def test_command_plus_completion_observation_stays_step_level() -> None:
    commands = [terminus_module.Command(keystrokes="ls\n", duration_sec=1.0)]

    assert (
        terminus_module._terminal_observation_source_call_id(
            commands, episode=3, is_task_complete=True
        )
        is None
    )
