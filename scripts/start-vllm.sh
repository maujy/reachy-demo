#!/bin/bash
# Start vLLM containers for Reachy Demo on DGX Spark (GB10)
#
# This script starts two vLLM containers:
# - Port 8000: Vision model (Nemotron-Nano-12B-v2-VL)
# - Port 8081: Agent/Router/Chitchat model (Nemotron-3-Nano-30B)

set -e

VLLM_IMAGE="avarok/vllm-dgx-spark:v11"

echo "Starting vLLM containers for Reachy Demo..."

# Check if containers already exist
if docker ps -a --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
    echo "Container vllm-reachy exists. Checking status..."
    if docker ps --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
        echo "  vllm-reachy is already running"
    else
        echo "  Starting existing vllm-reachy container..."
        docker start vllm-reachy
    fi
else
    echo "Creating vllm-reachy (Vision model on port 8000)..."
    docker run -d \
        --name vllm-reachy \
        --gpus all \
        -p 8000:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/vllm:/root/.cache/vllm \
        ${VLLM_IMAGE} \
        serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.5 \
        --max-model-len 16384 \
        --trust-remote-code
fi

if docker ps -a --format '{{.Names}}' | grep -q '^vllm-agent$'; then
    echo "Container vllm-agent exists. Checking status..."
    if docker ps --format '{{.Names}}' | grep -q '^vllm-agent$'; then
        echo "  vllm-agent is already running"
    else
        echo "  Starting existing vllm-agent container..."
        docker start vllm-agent
    fi
else
    echo "Creating vllm-agent (Agent model on port 8081)..."
    docker run -d \
        --name vllm-agent \
        --gpus all \
        --shm-size=16g \
        -p 8081:8081 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/vllm:/root/.cache/vllm \
        -v ~/.cache/flashinfer:/root/.cache/flashinfer \
        ${VLLM_IMAGE} \
        serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --host 0.0.0.0 \
        --port 8081 \
        --gpu-memory-utilization 0.4 \
        --max-model-len 4096 \
        --trust-remote-code \
        --enforce-eager
fi

echo ""
echo "Waiting for models to load (this may take 2-5 minutes)..."
echo ""

# Wait for vision model
echo -n "Waiting for vision model (port 8000)..."
for i in {1..120}; do
    if curl -s --connect-timeout 2 http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 5
done

# Wait for agent model
echo -n "Waiting for agent model (port 8081)..."
for i in {1..120}; do
    if curl -s --connect-timeout 2 http://localhost:8081/v1/models > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "vLLM containers status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(NAMES|vllm)"

echo ""
echo "Models available:"
echo "  - Port 8000: $(curl -s http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 || echo 'Not ready')"
echo "  - Port 8081: $(curl -s http://localhost:8081/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 || echo 'Not ready')"
