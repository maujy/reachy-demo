#!/bin/bash
# Start model servers for Reachy Demo on DGX Spark (GB10)
#
# This script starts:
# - Port 8000: Vision + Routing model via vLLM (Nemotron-12B-VL-NVFP4) ~18GB
# - Port 8081: Agent/Chitchat model via llama.cpp (Nemotron-3-Nano-30B) ~38GB
#
# Note: vLLM's FlashInfer MoE kernels don't compile on GB10 (sm_121),
# so we use llama.cpp for the Nemotron-3-Nano-30B MoE model.

set -e

VLLM_IMAGE="avarok/vllm-dgx-spark:v11"
LLAMA_CPP_DIR="$HOME/llama.cpp"
MODEL_PATH="$HOME/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf"

echo "=== Starting Model Servers for Reachy Demo ==="
echo ""

# ─────────────────────────────────────────────────────────
# Part 1: Start vLLM container for Vision model
# ─────────────────────────────────────────────────────────
echo ">>> Starting vLLM (Vision model on port 8000)..."

if docker ps -a --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
    if docker ps --format '{{.Names}}' | grep -q '^vllm-reachy$'; then
        echo "  vllm-reachy is already running"
    else
        echo "  Starting existing vllm-reachy container..."
        docker start vllm-reachy
    fi
else
    echo "  Creating vllm-reachy container..."
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

# ─────────────────────────────────────────────────────────
# Part 2: Start llama-server for Agent model
# ─────────────────────────────────────────────────────────
echo ""
echo ">>> Starting llama-server (Agent model on port 8081)..."

# Check if llama.cpp is built
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-server" ]; then
    echo "  Error: llama-server not found at $LLAMA_CPP_DIR/build/bin/llama-server"
    echo "  Please build llama.cpp first:"
    echo "    cd ~/llama.cpp && mkdir -p build && cd build"
    echo "    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=\"121\""
    echo "    cmake --build . -j\$(nproc)"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "  Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Check if already running
if pgrep -f "llama-server.*8081" > /dev/null 2>&1; then
    echo "  llama-server already running on port 8081"
else
    echo "  Starting llama-server..."
    # --reasoning-format none: Required for REACT agent compatibility
    # Without this, stop sequences trigger on reasoning_content, breaking agents
    nohup "$LLAMA_CPP_DIR/build/bin/llama-server" \
        -m "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8081 \
        -ngl 99 \
        --ctx-size 8192 \
        --alias nemotron \
        --reasoning-format none \
        > "$HOME/.llama-server.log" 2>&1 &
    echo "  Started with PID: $!"
    echo "  Log: ~/.llama-server.log"
fi

# ─────────────────────────────────────────────────────────
# Part 3: Wait for models to be ready
# ─────────────────────────────────────────────────────────
echo ""
echo ">>> Waiting for models to load..."
echo ""

# Wait for vision model
echo -n "Vision model (port 8000)..."
for i in {1..120}; do
    if curl -s --connect-timeout 2 http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 5
done

# Wait for agent model
echo -n "Agent model (port 8081)..."
for i in {1..120}; do
    if curl -s --connect-timeout 2 http://localhost:8081/v1/models > /dev/null 2>&1; then
        echo " Ready!"
        break
    fi
    echo -n "."
    sleep 5
done

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
echo ""
echo "=== Status ==="
echo ""
echo "Docker containers:"
docker ps --format "  {{.Names}}: {{.Status}}" | grep vllm || echo "  (none)"
echo ""
echo "llama-server:"
if pgrep -f "llama-server" > /dev/null 2>&1; then
    echo "  Running (PID: $(pgrep -f 'llama-server'))"
else
    echo "  Not running"
fi
echo ""
echo "Endpoints:"
echo "  Port 8000: $(curl -s http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 || echo 'Not ready')"
echo "  Port 8081: $(curl -s http://localhost:8081/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 || echo 'Not ready')"
