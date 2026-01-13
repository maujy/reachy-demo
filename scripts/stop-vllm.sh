#!/bin/bash
# Stop vLLM containers for Reachy Demo

echo "Stopping vLLM containers..."

if docker ps --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
    echo "Stopping vllm-reachy..."
    docker stop vllm-reachy
else
    echo "vllm-reachy is not running"
fi

if docker ps --format '{{.Names}}' | grep -q '^vllm-agent$'; then
    echo "Stopping vllm-agent..."
    docker stop vllm-agent
else
    echo "vllm-agent is not running"
fi

echo ""
echo "Done. Use './scripts/start-vllm.sh' to restart."
