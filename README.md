# Reachy Mini Robot with NeMo Agent Toolkit Tutorial

This tutorial showcases a real-time AI agent built with the **NVIDIA NeMo Agent Toolkit**, powered by **NVIDIA Nemotron models**, controlling a **Reachy Mini Robot**. The agent uses an intelligent LLM router to dynamically route between:
- **Nemotron nano text** for text-based interactions
- **Nemotron nano VLM** (Vision Language Model) for visual understanding
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

### Local NIM Setup (Optional)

The default configuration expects local NIM containers on:
- **Port 8000**: Vision model (nemotron-nano-12b-v2-vl)
- **Port 8080**: Routing model (phi-3-mini)
- **Port 8081**: Agent/Chitchat model (nemotron-3-nano)

To use cloud NIM instead, update `nat/src/ces_tutorial/config.yml` to remove `base_url` entries and use full model names (e.g., `nvidia/nemotron-3-nano-30b-a3b`).

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

You'll need **three terminal windows** running simultaneously.

### Terminal 1: Start Reachy Mini Daemon

Navigate to the `bot` directory and start the robot daemon:

**For macOS:**
```bash
cd bot
uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only
```

**For Linux:**
```bash
cd bot
uv run -m reachy_mini.daemon.app.main --sim --no-localhost-only
```

*Note: The `--sim` flag runs the robot in simulation mode. Remove it if using actual hardware.*

### Terminal 2: Start Bot Service

In the `bot` directory:

```bash
cd bot
uv run --env-file ../.env python main.py
```

This service handles:
- Vision processing through the robot's camera
- Speech recognition and text-to-speech
- Robot movement coordination
- Emotional expression through dance moves

### Terminal 3: Start NeMo Agent Service

In the `nat` directory:

```bash
cd nat
uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

This launches the NeMo Agent Toolkit server with intelligent model routing capabilities.

## How It Works

1. **Vision & Audio Input**: The bot captures visual information and listens for speech
2. **Agent Processing**: The NeMo Agent router intelligently selects the appropriate model:
   - Text queries → Nemotron nano text model
   - Visual queries → Nemotron nano VLM
   - Action requests → REACT agent with tool calling
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
│       ├── config.yml     # Agent and local NIM endpoint configuration
│       └── functions/     # Router and agent implementations
├── CLAUDE.md              # Claude Code guidance
└── .env                   # API keys (create this file)
```

## Troubleshooting

- **Port conflicts**: Ensure ports 8000, 8001, 8080, 8081 are available
- **API key errors**: Verify your `.env` file is properly formatted and contains valid keys
- **Robot connection issues**: Check that the Reachy daemon started successfully before launching the bot service
- **NIM connection errors**: Ensure local NIM containers are running on the expected ports, or switch to cloud NIM in config.yml

## Resources

- [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit)
- [Reachy Mini Robot](https://www.pollen-robotics.com/)
- [NVIDIA Nemotron Models](https://build.nvidia.com/)

