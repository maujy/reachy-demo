# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A real-time AI agent system controlling a Reachy Mini Robot using NVIDIA NeMo Agent Toolkit (NAT). The system uses intelligent LLM routing between Nemotron nano text, Vision Language Model, and REACT agent to dynamically select the best processing path for user queries.

## Architecture

Three independent services running in parallel:

```
┌─────────────────────────────────────────────────────────────┐
│                  Reachy Mini Daemon                         │
│         mjpython -m reachy_mini.daemon.app.main             │
└────────────────────────┬────────────────────────────────────┘
                         │ gRPC
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼─────────────┐         ┌────────▼─────────────┐
│    Bot Service      │◄────────│   NAT Agent Service  │
│  bot/main.py        │  HTTP   │  nat/config.yml      │
│                     │ :8001   │                      │
│ - Pipecat pipeline  │         │ - Router classifier  │
│ - Audio I/O (STT)   │         │ - Chitchat LLM       │
│ - Vision I/O        │         │ - Image LLM (VLM)    │
│ - TTS (ElevenLabs)  │         │ - REACT Agent        │
│ - Robot movements   │         │                      │
└─────────────────────┘         └──────────────────────┘
```

**Data Flow**: User speech/camera → Bot (STT) → NAT Router (classifies intent: chit_chat/image_understanding/other) → Appropriate LLM → Response → Bot (TTS + robot movements)

## Commands

### Setup
```bash
cd bot && uv venv && uv sync
cd ../nat && uv venv && uv sync
```

### Running (three terminals)
```bash
# Terminal 1: Reachy daemon (simulator)
cd bot && uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only

# Terminal 2: Bot service
cd bot && uv run --env-file ../.env python main.py

# Terminal 3: NAT Agent service
cd nat && uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

## Key Components

### Bot Service (`/bot`)
- `main.py`: Entry point - Pipecat pipeline with Daily transport, Silero VAD, ElevenLabs STT/TTS
- `nat_vision_llm.py`: Custom NVIDIA LLM service with smart vision classification (determines if query needs camera input before fetching images to save tokens)
- `services/reachy_service.py`: Singleton robot connection manager
- `services/moves.py`: MovementManager - 100Hz control loop with command queue
- `services/wobbler.py`: Audio-driven head animation

### NAT Service (`/nat/src/ces_tutorial`)
- `config.yml`: Model endpoints and workflow configuration
- `functions/router.py`: Intent classifier (chit_chat/image_understanding/other)
- `functions/router_agent.py`: Orchestrates routing to appropriate LLM
- `openai_chat_request.py`: Bridges OpenAI API schema with NAT internals
- `register.py`: Component registration and monkey-patching for REACT agent compatibility

## Key Patterns

**Singleton**: `ReachyService.get_instance()` ensures single robot connection

**Threading**: MovementManager and HeadWobbler use dedicated threads with command queues for real-time control

**Multimodal Content**: OpenAI format with `type: image_url` and base64 data URLs

**Token Optimization**: Vision classification before image fetch; image compression (256px max, JPEG quality 40, low detail mode)

**Graceful Degradation**: Pipeline continues if Reachy daemon unavailable

## Environment Variables

Required in `.env`:
- `ELEVENLABS_API_KEY`
- `NVIDIA_API_KEY` (if using external endpoints)

## Language

System prompts and user-facing text use Traditional Chinese (台灣用語).
