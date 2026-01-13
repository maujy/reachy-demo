#!/bin/bash
# Stop all Reachy Demo components
#
# Usage: ./scripts/stop-all.sh [--models]
#   --models    Also stop model servers (vLLM + llama-server) (default: keep running)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_MODELS=false

for arg in "$@"; do
    case $arg in
        --models|--containers)
            STOP_MODELS=true
            ;;
    esac
done

echo "=== Stopping Reachy Demo ==="
echo ""

# Stop tmux session
if tmux has-session -t "reachy-demo" 2>/dev/null; then
    echo "Stopping tmux session 'reachy-demo'..."
    tmux kill-session -t "reachy-demo"
    echo "  Done"
else
    echo "Tmux session 'reachy-demo' not running"
fi
echo ""

# Optionally stop model servers
if [ "$STOP_MODELS" = true ]; then
    echo "Stopping model servers..."
    "$SCRIPT_DIR/stop-vllm.sh"
else
    echo "Model servers kept running (use --models to stop)"
fi

echo ""
echo "All services stopped."
