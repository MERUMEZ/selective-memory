"""
decaymem — динамическая память: пишет избирательно, забывает со временем,
укрепляет то, что пригодилось.

    from decaymem import Memory

    memory = Memory("brain.db")
    memory.observe("меня зовут Паша", "приятно познакомиться", emotion=0.4)
    memory.feedback(+1.0)
    memory.recall("как меня зовут")

Всё остальное — MemoryGraph, PlasticityGate, ReinforcementLoop — доступно
по своим модулям: фасад покрывает обычный случай, но ничего не запирает.
"""

from decaymem.memory import Memory, MemoryStats, Observation
from decaymem.settings import MemorySettings

__all__ = ["Memory", "MemoryStats", "Observation", "MemorySettings"]
__version__ = "0.1.0"
