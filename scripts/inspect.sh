#!/usr/bin/env bash
# Launch MCP Inspector pre-configured for El Paso.
#
# Usage:
#   ./scripts/inspect.sh          # Docker mode (uses docker compose)
#   ./scripts/inspect.sh --local  # Local mode (uses npx, connects to localhost:3000)

set -euo pipefail

ELPASO_URL="${ELPASO_URL:-http://localhost:3000/mcp}"
INSPECTOR_PORT="${CLIENT_PORT:-6274}"

if [[ "${1:-}" == "--local" ]]; then
    echo "Starting MCP Inspector (local/npx) pointed at $ELPASO_URL"
    echo "Inspector UI will open at http://localhost:$INSPECTOR_PORT"
    echo ""
    echo "In the Inspector UI:"
    echo "  1. Set transport type to 'Streamable HTTP'"
    echo "  2. Set URL to: $ELPASO_URL"
    echo "  3. Click 'Connect'"
    echo ""
    DANGEROUSLY_OMIT_AUTH=true CLIENT_PORT="$INSPECTOR_PORT" \
        npx @modelcontextprotocol/inspector
else
    echo "Starting El Paso + Inspector via Docker Compose..."
    docker compose --profile dev up -d

    echo ""
    echo "Services:"
    echo "  El Paso MCP:  http://localhost:3000/mcp"
    echo "  Inspector UI: http://localhost:6274"
    echo ""
    echo "In the Inspector UI:"
    echo "  1. Set transport type to 'Streamable HTTP'"
    echo "  2. Set URL to: http://elpaso:8080/mcp"
    echo "  3. Click 'Connect'"
    echo ""
    echo "Logs: docker compose logs -f elpaso inspector"

    # Auto-open browser if on macOS
    if command -v open &>/dev/null; then
        sleep 2
        open "http://localhost:6274"
    fi
fi
