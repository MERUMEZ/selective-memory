# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# Эта программа — свободное ПО: вы можете распространять и изменять её
# на условиях GNU Affero General Public License версии 3, изданной
# Free Software Foundation. Полный текст — в файле LICENSE.
#
# Программа распространяется В НАДЕЖДЕ, ЧТО БУДЕТ ПОЛЕЗНОЙ, но БЕЗ
# ВСЯКИХ ГАРАНТИЙ, включая подразумеваемые гарантии товарного
# состояния и пригодности для определённой цели.
#
# Для использования в закрытых продуктах существует коммерческая
# лицензия — см. COMMERCIAL.md.
"""
selectivemem — динамическая память: пишет избирательно, забывает со временем,
укрепляет то, что пригодилось.

    from selectivemem import Memory

    memory = Memory("brain.db")
    memory.observe("меня зовут Паша", "приятно познакомиться", emotion=0.4)
    memory.feedback(+1.0)
    memory.recall("как меня зовут")

Всё остальное — MemoryGraph, PlasticityGate, ReinforcementLoop — доступно
по своим модулям: фасад покрывает обычный случай, но ничего не запирает.
"""

from selectivemem.memory import Memory, MemoryStats, Observation
from selectivemem.settings import MemorySettings

__all__ = ["Memory", "MemoryStats", "Observation", "MemorySettings"]
__version__ = "0.1.0"
