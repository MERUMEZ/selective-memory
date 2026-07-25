# ARCHITECTURE MANIFESTO: DYNAMIC AI BRAIN

## Project Concept
We are building a dynamic, bio-inspired language model prototype ("digital organism").
It is NOT a standard RAG/LLM wrapper. It operates on biological brain principles:
1. Spike Memory: Information is saved ONLY when emotional density or perplexity exceeds a plasticity threshold.
2. Decay & Time (t_0): Unused memory nodes fade over time (forgetting curve).
3. Stress Protection: High overload triggers self-preservation, dampening input plasticity.
4. Instincts Layer: A lightweight, low-cost baseline logic (self-preservation, mimicry/echolalia, consistency).

## Tech Stack
- Python 3.11+
- Local DB: SQLite / FAISS / NetworkX (for dynamic graph weights)
- Execution: Local CLI first (`main.py`), Telegram Bot interface (`bot.py`) later.

## Coding Principles for Agent
- Write modular, lightweight Python code.
- Always include clear logs for internal state changes (e.g., `[SPIKE DETECTED]`, `[DECAY APPLIED]`).
- Do not add complex frameworks unless explicitly requested.