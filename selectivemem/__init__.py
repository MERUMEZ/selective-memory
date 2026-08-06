# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# A commercial licence is available for use in closed products — see
# COMMERCIAL.md.
"""
selectivemem — dynamic memory: stores selectively, fades with time,
strengthens what turned out to be useful.

    from selectivemem import Memory

    memory = Memory("brain.db")
    memory.observe("my name is Pasha", "nice to meet you", emotion=0.4)
    memory.feedback(+1.0)
    memory.recall("what is my name")

Everything else — MemoryGraph, PlasticityGate, ReinforcementLoop — stays
available through its own module: the facade covers the ordinary case
without locking anything away.
"""

from selectivemem.memory import Memory, MemoryStats, Observation
from selectivemem.interoception import InternalState
from selectivemem.settings import MemorySettings

__all__ = ["Memory", "MemoryStats", "Observation", "MemorySettings",
           "InternalState"]
__version__ = "0.1.1.dev1"
