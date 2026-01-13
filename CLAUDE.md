# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A real-time AI agent system controlling a Reachy Mini Robot using NVIDIA NeMo Agent Toolkit (NAT). The system uses intelligent LLM routing to dynamically select the best processing path for user queries.

**GB10-optimized architecture** (2 models):
- **Nemotron VLM** (12B NVFP4 via vLLM) - Handles routing + vision understanding
- **Nemotron 30B** (GGUF Q8 via llama.cpp) - Handles chitchat + REACT agent

> FP8 models don't work on GB10 (sm_121). vLLM's MoE kernels fail to compile on GB10, so we use llama.cpp for the Nemotron-3-Nano-30B model.

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
│ - Pipecat pipeline  │         │ - VLM (router+vision)│
│ - Audio I/O (STT)   │         │ - Chitchat LLM       │
│ - Vision I/O        │         │ - REACT Agent        │
│ - TTS (ElevenLabs)  │         │                      │
│ - Robot movements   │         │  Model servers:      │
└─────────────────────┘         │  :8000 vLLM (VLM)    │
                                │  :8081 llama.cpp     │
                                └──────────────────────┘
```

**Data Flow**: User speech/camera → Bot (STT) → NAT Router (classifies intent: chit_chat/image_understanding/other) → Appropriate LLM → Response → Bot (TTS + robot movements)

## Commands

### Setup
```bash
cd bot && uv venv && uv sync
cd ../nat && uv venv && uv sync
```

### Start Model Servers (DGX Spark / GB10)
```bash
./scripts/start-vllm.sh   # Start vLLM + llama-server, wait for models to load
./scripts/stop-vllm.sh    # Stop all model servers
```

Models served:
- Port 8000: `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD` via vLLM (routing + vision)
- Port 8081: `Nemotron-3-Nano-30B-A3B` GGUF via llama.cpp (agent + chitchat)

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
- `services/chinese_converter.py`: Simplified → Traditional Chinese converter using OpenCC (s2twp)

### NAT Service (`/nat/src/ces_tutorial`)
- `config.yml`: Model endpoints, tools, and workflow configuration
- `functions/router.py`: Reasoning-based intent classifier with tool capability awareness
- `functions/router_agent.py`: Orchestrates routing to appropriate LLM, strips `<think>` tags
- `openai_chat_request.py`: Bridges OpenAI API schema with NAT internals
- `register.py`: Component registration and monkey-patching for REACT agent compatibility

## Key Patterns

**Singleton**: `ReachyService.get_instance()` ensures single robot connection

**Threading**: MovementManager and HeadWobbler use dedicated threads with command queues for real-time control

**Multimodal Content**: OpenAI format with `type: image_url` and base64 data URLs

**Token Optimization**: Vision classification before image fetch; image compression (256px max, JPEG quality 40, low detail mode)

**Graceful Degradation**: Pipeline continues if Reachy daemon unavailable

## Routing and Web Search

The router uses **reasoning-based classification** to determine which route can handle a query:

**Routes:**
- `other` → REACT agent with tools (web_search, wikipedia_search)
- `chit_chat` → Direct LLM response (no tools)
- `image_understanding` → VLM for visual queries

**Reasoning-based routing** (`router.py`):
- Router prompt explicitly lists tool capabilities per route
- Weather, news, factual questions → `other` (has web_search)
- Simple greetings → `chit_chat` (no tools needed)

**Web Search** (Tavily):
- REACT agent calls `web_search` tool for real-time data
- Returns results from weather sites, news, etc.
- Agent synthesizes response from search results

**Think Tag Filtering** (`router_agent.py`):
- System prompt instructs model to use `<think></think>` tags for reasoning
- `_strip_think_tags()` removes thinking content before TTS
- Also filters English reasoning patterns from REACT agent output

## Environment Variables

Required in `.env`:
- `ELEVENLABS_API_KEY` - Speech-to-text and text-to-speech
- `TAVILY_API_KEY` - Web search for real-time data (weather, news, etc.)
- `NVIDIA_API_KEY` (optional, if using external endpoints)

## Language

System prompts and user-facing text use Traditional Chinese (台灣用語).

**Chinese Text Conversion**:
- ElevenLabs STT outputs Simplified Chinese (API limitation)
- `ConvertingElevenLabsSTTService` converts STT output to Traditional before RTVI displays it
- `SimplifiedToTraditionalProcessor` converts LLM output to Traditional
- Uses OpenCC `s2twp` (Simplified to Traditional with Taiwan phrases)
