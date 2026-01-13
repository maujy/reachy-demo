#!/bin/bash
# Check status of all Reachy Demo components
#
# Usage: ./scripts/status.sh

echo "=== Reachy Demo Status ==="
echo ""

# Check vLLM container
echo "Docker Containers:"
echo "─────────────────────────────────────────────────────────"
if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "(NAMES|vllm)" | head -5; then
    :
else
    echo "  No vLLM containers found"
fi
echo ""

# Check llama-server process
echo "llama-server Process:"
echo "─────────────────────────────────────────────────────────"
if pgrep -f "llama-server" > /dev/null 2>&1; then
    PID=$(pgrep -f "llama-server")
    echo "  llama-server: RUNNING (PID: $PID)"
else
    echo "  llama-server: NOT RUNNING"
fi
echo ""

# Check model endpoints
echo "Model Endpoints:"
echo "─────────────────────────────────────────────────────────"

# Port 8000 - VLM (vLLM)
if curl -s --connect-timeout 2 http://localhost:8000/v1/models > /dev/null 2>&1; then
    MODEL=$(curl -s http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "  Port 8000 (VLM+Router): OK - $MODEL"
else
    echo "  Port 8000 (VLM+Router): NOT READY"
fi

# Port 8081 - Agent (llama.cpp)
if curl -s --connect-timeout 2 http://localhost:8081/v1/models > /dev/null 2>&1; then
    MODEL=$(curl -s http://localhost:8081/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "  Port 8081 (Agent):      OK - $MODEL"
else
    echo "  Port 8081 (Agent):      NOT READY"
fi
echo ""

# Check NAT service
echo "NAT Agent Service:"
echo "─────────────────────────────────────────────────────────"
if curl -s --connect-timeout 2 http://localhost:8001/health > /dev/null 2>&1; then
    echo "  Port 8001: OK"
elif curl -s --connect-timeout 2 http://localhost:8001/ > /dev/null 2>&1; then
    echo "  Port 8001: OK (no health endpoint)"
else
    echo "  Port 8001: NOT RUNNING"
fi
echo ""

# Check tmux session
echo "Tmux Session:"
echo "─────────────────────────────────────────────────────────"
if tmux has-session -t "reachy-demo" 2>/dev/null; then
    echo "  Session 'reachy-demo': ACTIVE"
    echo "  Attach with: tmux attach -t reachy-demo"
else
    echo "  Session 'reachy-demo': NOT RUNNING"
fi
echo ""

# GPU memory usage
echo "GPU Memory:"
echo "─────────────────────────────────────────────────────────"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while read line; do
        USED=$(echo $line | cut -d',' -f1 | tr -d ' ')
        TOTAL=$(echo $line | cut -d',' -f2 | tr -d ' ')
        # Skip if values are not numeric
        if [[ "$USED" =~ ^[0-9]+$ ]] && [[ "$TOTAL" =~ ^[0-9]+$ ]] && [ "$TOTAL" -gt 0 ]; then
            PCT=$((USED * 100 / TOTAL))
            echo "  ${USED}MB / ${TOTAL}MB (${PCT}%)"
        else
            echo "  $USED / $TOTAL"
        fi
    done
else
    echo "  nvidia-smi not available"
fi
echo ""
