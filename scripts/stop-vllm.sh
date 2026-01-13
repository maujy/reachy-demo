#!/bin/bash
# Stop model servers for Reachy Demo

echo "=== Stopping Model Servers ==="
echo ""

# Stop vLLM container
echo ">>> Stopping vLLM..."
if docker ps --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
    docker stop vllm-reachy
    echo "  vllm-reachy stopped"
else
    echo "  vllm-reachy not running"
fi

# Stop llama-server
echo ""
echo ">>> Stopping llama-server..."
if pgrep -f "llama-server" > /dev/null 2>&1; then
    pkill -f "llama-server"
    echo "  llama-server stopped"
else
    echo "  llama-server not running"
fi

echo ""
echo "Done. Use './scripts/start-vllm.sh' to restart."
