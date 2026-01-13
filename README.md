# Reachy Mini Robot with NeMo Agent Toolkit Tutorial

This tutorial showcases a real-time AI agent built with the **NVIDIA NeMo Agent Toolkit**, powered by **NVIDIA Nemotron models**, controlling a **Reachy Mini Robot**. The agent uses an intelligent LLM router to dynamically route between:
- **Nemotron VLM** (Vision Language Model) for routing and visual understanding
- **Nemotron 30B NVFP4** for chitchat and complex reasoning
- **REACT agent** for tool-based actions

![Reachy Mini Robot Demo](ces_tutorial.png)

## Architecture

The system consists of three main components running in parallel:

1. **Reachy Mini Daemon** - Controls the robot hardware (or simulation)
2. **Bot Service** - Processes vision and speech, coordinates robot actions
3. **NeMo Agent Service** - Handles AI agent logic with intelligent routing between models

![System Architecture](ces_tutorial_arch.png)

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- ElevenLabs API Key (for text-to-speech)
- Local NIM containers (see below) OR NVIDIA API Key for cloud inference

### Local Model Setup (DGX Spark / GB10)

The default configuration runs models locally:
- **Port 8000**: Vision + Routing model via vLLM (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD`)
- **Port 8081**: Agent/Chitchat model via llama.cpp (`Nemotron-3-Nano-30B-A3B` GGUF)

> **Note**: We use NVFP4 quantization for GB10 compatibility. FP8 models don't work on sm_121 architecture. The MoE model (Nemotron-3-Nano-30B) runs via llama.cpp because vLLM's FlashInfer MoE kernels fail to compile on GB10.

**Prerequisites:**
1. Build llama.cpp with CUDA:
   ```bash
   cd ~/llama.cpp
   mkdir -p build && cd build
   cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121"
   cmake --build . -j$(nproc)
   ```
2. Download Nemotron-3-Nano GGUF model to `~/models/nemotron3-gguf/`

**Start model servers:**
```bash
./scripts/start-vllm.sh
```

**Stop model servers:**
```bash
./scripts/stop-vllm.sh
```

To use cloud NIM instead, update `nat/src/ces_tutorial/config.yml` to remove `base_url` entries and use the NVIDIA API.

## Setup Instructions

### 1. Clone and Navigate to Repository

```bash
git clone https://github.com/maujy/reachy-demo.git
cd reachy-demo
```

### 2. Create Environment File

Create a `.env` file in the main directory with your API keys:

```bash
NVIDIA_API_KEY=your_nvidia_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 3. Setup Bot Service

In a terminal window:

```bash
cd bot
uv venv
uv sync
```

### 4. Setup NeMo Agent Service

In a separate terminal window:

```bash
cd nat
uv venv
uv sync
```

## Running the System

### Quick Start (Recommended)

```bash
# 1. Start model servers (wait 2-5 minutes for models to load)
./scripts/start-vllm.sh

# 2. Run all services in tmux
./scripts/run-demo.sh
```

This opens a tmux session with all 3 services running in separate panes.

**Tmux controls:**
- `Ctrl+B, arrow keys` - Switch between panes
- `Ctrl+B, d` - Detach (services keep running)
- `Ctrl+B, z` - Zoom current pane
- `Ctrl+C` - Stop service in current pane

**Other commands:**
```bash
./scripts/status.sh        # Check status of all components
./scripts/stop-all.sh      # Stop all services
./scripts/stop-all.sh --models  # Stop services AND model servers
```

### Manual Setup (Alternative)

If you prefer running in separate terminals:

**Terminal 1: Reachy Mini Daemon**
```bash
cd bot
uv run python -m reachy_mini.daemon.app.main --sim --no-localhost-only
# macOS: use mjpython instead of python
```

**Terminal 2: Bot Service**
```bash
cd bot
uv run --env-file ../.env python main.py
```

**Terminal 3: NeMo Agent Service**
```bash
cd nat
uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

*Note: The `--sim` flag runs the robot in simulation mode. Remove it for real hardware.*

## How It Works

1. **Vision & Audio Input**: The bot captures visual information and listens for speech
2. **Agent Processing**: The NeMo Agent router (VLM) intelligently classifies user intent:
   - `chit_chat` → Nemotron 30B NVFP4 for casual conversation
   - `image_understanding` → Nemotron VLM for visual queries
   - `other` → REACT agent with tool calling for complex tasks
3. **Robot Actions**: Based on the agent's response, the bot executes movements, expressions, or speaks

## Demo

Check out `ces_tutorial.mp4` to see the system in action!

## Project Structure

```
reachy-demo/
├── bot/                    # Robot control and vision/speech processing
│   ├── main.py            # Main bot orchestration
│   ├── nat_vision_llm.py  # Vision and LLM integration with smart classification
│   └── services/          # Robot services (moves, speech, etc.)
├── nat/                    # NeMo Agent Toolkit configuration
│   └── src/ces_tutorial/
│       ├── config.yml     # Agent and local vLLM endpoint configuration
│       └── functions/     # Router and agent implementations
├── scripts/               # Utility scripts
│   ├── start-vllm.sh     # Start vLLM containers
│   ├── stop-vllm.sh      # Stop vLLM containers
│   ├── run-demo.sh       # Run all services in tmux
│   ├── status.sh         # Check status of all components
│   └── stop-all.sh       # Stop all services
├── CLAUDE.md              # Claude Code guidance
└── .env                   # API keys (create this file)
```

## Troubleshooting

- **Port conflicts**: Ensure ports 8000, 8001, 8081 are available
- **API key errors**: Verify your `.env` file is properly formatted and contains valid keys
- **Robot connection issues**: Check that the Reachy daemon started successfully before launching the bot service
- **vLLM connection errors**: Run `./scripts/start-vllm.sh` and wait for models to load. Check vLLM logs with `docker logs vllm-reachy`
- **llama-server issues**: Check logs at `~/.llama-server.log`. Ensure llama.cpp is built with CUDA for sm_121

## Resources

- [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
- [Reachy Mini Robot](https://www.pollen-robotics.com/)
- [NVIDIA Nemotron Models](https://build.nvidia.com/)

