#!/bin/bash
# Run complete Reachy Demo system in tmux
#
# This script:
# 1. Checks if vLLM containers are ready
# 2. Creates a tmux session with 3 panes for all services
# 3. Attaches to the session for monitoring
#
# Usage: ./scripts/run-demo.sh [--sim]
#   --sim    Run Reachy daemon in simulation mode (default)
#   --real   Run Reachy daemon with real hardware

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION_NAME="reachy-demo"

# Parse arguments
SIM_FLAG="--sim"
for arg in "$@"; do
    case $arg in
        --real)
            SIM_FLAG=""
            ;;
        --sim)
            SIM_FLAG="--sim"
            ;;
    esac
done

echo "=== Reachy Demo Launcher ==="
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  sudo apt install tmux"
    exit 1
fi

# Check if vLLM containers are running
echo "Checking vLLM containers..."
VLLM_READY=true

if ! curl -s --connect-timeout 2 http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  Warning: VLM (port 8000) not ready"
    VLLM_READY=false
fi

if ! curl -s --connect-timeout 2 http://localhost:8081/v1/models > /dev/null 2>&1; then
    echo "  Warning: Agent (port 8081) not ready"
    VLLM_READY=false
fi

if [ "$VLLM_READY" = false ]; then
    echo ""
    echo "vLLM containers are not ready. Start them first:"
    echo "  ./scripts/start-vllm.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  VLM (port 8000): Ready"
    echo "  Agent (port 8081): Ready"
fi

echo ""

# Kill existing session if exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Creating tmux session '$SESSION_NAME'..."
echo ""

# Detect if running on macOS (needs mjpython for mujoco)
if [[ "$OSTYPE" == "darwin"* ]]; then
    DAEMON_CMD="cd $PROJECT_DIR/bot && uv run mjpython -m reachy_mini.daemon.app.main $SIM_FLAG --no-localhost-only"
else
    DAEMON_CMD="cd $PROJECT_DIR/bot && uv run python -m reachy_mini.daemon.app.main $SIM_FLAG --no-localhost-only"
fi

BOT_CMD="cd $PROJECT_DIR/bot && uv run --env-file $PROJECT_DIR/.env python main.py"
NAT_CMD="cd $PROJECT_DIR/nat && uv run --env-file $PROJECT_DIR/.env nat serve --config_file src/ces_tutorial/config.yml --port 8001"

# Create tmux session with first pane (Daemon)
tmux new-session -d -s "$SESSION_NAME" -n "services"

# Split into 3 panes
# Layout:
#  ┌─────────┬─────────┐
#  │ Daemon  │   Bot   │
#  ├─────────┴─────────┤
#  │    NAT Agent      │
#  └───────────────────┘

tmux split-window -h -t "$SESSION_NAME"
tmux split-window -v -t "$SESSION_NAME:0.0"

# Select panes and send commands
tmux select-pane -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.0" "echo '=== Reachy Daemon ===' && sleep 1 && $DAEMON_CMD" C-m

tmux select-pane -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.1" "echo '=== NAT Agent Service ===' && sleep 3 && $NAT_CMD" C-m

tmux select-pane -t "$SESSION_NAME:0.2"
tmux send-keys -t "$SESSION_NAME:0.2" "echo '=== Bot Service ===' && sleep 5 && $BOT_CMD" C-m

# Set pane titles
tmux select-pane -t "$SESSION_NAME:0.0" -T "Daemon"
tmux select-pane -t "$SESSION_NAME:0.1" -T "NAT"
tmux select-pane -t "$SESSION_NAME:0.2" -T "Bot"

echo "Started services in tmux session '$SESSION_NAME'"
echo ""
echo "Tmux controls:"
echo "  Ctrl+B, arrow keys  - Switch between panes"
echo "  Ctrl+B, d           - Detach (services keep running)"
echo "  Ctrl+B, z           - Zoom current pane"
echo "  Ctrl+C              - Stop service in current pane"
echo ""
echo "To reattach later: tmux attach -t $SESSION_NAME"
echo "To stop all: ./scripts/stop-all.sh"
echo ""

# Attach to session
tmux attach -t "$SESSION_NAME"
