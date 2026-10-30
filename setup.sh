#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq && apt-get install -y -q curl

# Install uv
uv_installer="$(mktemp)"
curl -LsSf https://astral.sh/uv/0.13.5/install.sh -o "$uv_installer"
printf '%s  %s\n' '1c6e7f025499664633ac2dce43a210800510e9ef0625c52e034a8afdf8bf177a' "$uv_installer" | sha256sum -c -
sh "$uv_installer"
rm "$uv_installer"

# Install the package
/root/.local/bin/uv sync

# Wrapper script so `terminus2` is on PATH
PROJECT_DIR=$(pwd)
cat > /usr/local/bin/terminus2 << WRAPPER
#!/bin/bash
exec "$PROJECT_DIR/.venv/bin/python" -m terminus2.cli "\$@"
WRAPPER
chmod +x /usr/local/bin/terminus2
