# Terminus2-v1.0.0

Terminus2 contract for Valkyrie. The wrapper installs Terminus2, runs it against the task prompt, and writes the trajectory under `/logs/terminus2-v1.0.0`.

For local CLI usage, install the package with `uv sync` and run `terminus2 --help`.

## Configuration

- Secrets: provide provider API keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) at run time with `-s <ENV_VAR> <CLOUD_KEYNAME>`
- Model: required via `--model`
- Final output: `/logs/terminus2-v1.0.0`
- Optional kwargs: `temperature`, `max_turns`, `parser`, `reasoning`, `reasoning_effort`, `api_base`

The model is passed through to Terminus2 as `--model`. When set, `api_base` is passed through as `--api-base`.

## Usage with Valkyrie

We assume if you are here that you have [Valkyrie](https://github.com/vals-ai/Valkyrie) installed. If not, navigate to the [main repository](https://github.com/vals-ai/Valkyrie) and get started.

Install the agent from GitHub using Valkyrie for future use.

```bash
valkyrie agent install https://github.com/vals-ai/terminus2-v1.0.0
```

Download the agent, visit the [documentation](https://github.com/vals-ai/Valkyrie/blob/dev/docs/CONTRACTS.md) on **contract.yaml** if you would like to make any modifications.

```bash
valkyrie agent download terminus2-v1.0.0
```

Upload the agent after changes are made.

```bash
valkyrie agent push terminus2-v1.0.0
```

Run Terminal-Bench using Terminus2 as the agent. Specify `--model <MODEL>` to set the model passed into Terminus2.

```bash
valkyrie run start --benchmark terminal-bench --agent terminus2-v1.0.0 --model openai/gpt-4o -s OPENAI_API_KEY <CLOUD_KEYNAME> --concurrency 10 --slice :10
```

## Output Files

The run writes the Terminus2 output directory to `/logs/terminus2-v1.0.0`, including:

- `trajectory.json`
- `trajectory.cont-*.json` when linear history splits are enabled
- `trajectory.summarization-*.json` when summarization subagents run

## Run options

| Flag | Description |
|---|---|
| `-m, --model` | Model name (required) |
| `--logs-dir` | Logs/trajectory output directory (default: `./logs`) |
| `--parser` | Response format: `json` or `xml` (default: `json`) |
| `--max-turns` | Cap on agent turns |
| `--temperature` | Sampling temperature |
| `--max-tokens` | Max output tokens |
| `--api-base` | Custom API endpoint URL |
| `--no-summarize` | Disable context summarization |
| `--raw-trajectory` | Store raw LLM responses in trajectory (for SFT export) |
| `--linear-history` | Split trajectory files on summarization boundaries |
