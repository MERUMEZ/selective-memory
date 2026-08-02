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
================================================================================
 MEMORY.PY — The public facade of selective memory
================================================================================
Five calls are enough to work with it:

    memory = Memory("brain.db")

    obs = memory.observe("my name is Pasha", "nice to meet you")
    memory.feedback(+1.0)                      # approval — reinforce
    memory.recall("what is my name")           # -> [MemoryMatch]
    memory.context_for("what is my name")      # -> text for a prompt
    memory.stats()                             # -> what is going on inside

Underneath sit MemoryGraph (the graph and forgetting), PlasticityGate
(the write threshold) and ReinforcementLoop (dopamine and retrospective
correction). All three stay reachable as memory.graph / memory.gate /
memory.loop: the facade covers the ordinary case without locking the
rest away.

HOW THIS DIFFERS FROM A VECTOR STORE. A vector store answers "how do I
find the right thing". This answers a different question: "what should be
forgotten". Writing does NOT happen on every message but when emotion or
surprise clears a threshold; what is stored fades with time and is
strengthened by use.

Measured over five seeds: after two weeks of silence the organism recalls
100% of what the user marked as important against 60% of the ordinary —
a gap of +40 pp. On UNIFORM questions, though, it is merely level with a
random sample (92.4% against 90.8%), and that is measured too. This is a
tool for keeping what matters, not for losing nothing.

TIME COMES FROM OUTSIDE. Every method takes a timestamp, and the default
clock is time.time. An application may substitute its own: an organism's
subjective time, an accelerated demo clock, a frozen test clock.
Forgetting is computed on that scale, so the scale must be single and
monotonic.

THE ENCODER SETS THE LANGUAGE, not the library. The bundled one is navec,
Russian static vectors: light, no torch. For English or any other
language, pass your own function:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    memory = Memory("brain.db", encoder=lambda text: model.encode(text))

The encoder can be anything returning a sequence of numbers or None: a
local model, an API call, fastText. The library ships no model and does
not choose one for you.

The core knows no words of evaluation either: `valence` in feedback() is
a number the application obtains however it likes — a button, an emoji, a
classifier, parsing the reply.
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from selectivemem.database import Database
from selectivemem.graph_memory import MemoryGraph, MemoryMatch
from selectivemem.plasticity import PlasticityDecision, PlasticityGate
from selectivemem.reinforcement import ReinforcementLoop, ReinforcementOutcome
from selectivemem.settings import MemorySettings

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """What happened to a single observation."""

    text: str
    surprise: float
    decision: PlasticityDecision
    node_id: Optional[int] = None
    superseded_ids: List[int] = field(default_factory=list)
    learned_words: int = 0  # how many WORDS were seen for the first time

    @property
    def written(self) -> bool:
        return self.node_id is not None

    @property
    def reason(self) -> str:
        """Why it was stored, or why it was not — in plain words."""
        if self.superseded_ids and self.node_id is not None:
            return f"contradiction with nodes {self.superseded_ids}"
        if self.node_id is not None:
            return f"spike, density {self.decision.density:.3f}"
        return f"routine, {abs(self.decision.headroom):.3f} short of the threshold"


@dataclass
class MemoryStats:
    """A snapshot of the memory's state."""

    nodes: int
    episodes: int
    vocabulary: int
    threshold: float
    # Whether meaning-based search works. False means only entries that
    # share words with the query will be found — by far the most common
    # reason behind "why does the memory find nothing".
    semantic: bool = False


class Memory:
    """
    Selective memory: stores what is worth storing, fades with time,
    strengthens what turned out to be useful.

        memory = Memory("brain.db")
        memory.observe("I have a cat", "tell me about him")
        memory.recall("cat")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        settings: Optional[MemorySettings] = None,
        clock: Optional[Callable[[], float]] = None,
        encoder: Optional[Callable[[str], Any]] = None,
    ):
        self.settings = settings or MemorySettings()
        self.clock = clock or time.time
        self.graph = MemoryGraph(
            db=Database(db_path=db_path, settings=self.settings),
            settings=self.settings,
            encoder=encoder,
        )
        self.gate = PlasticityGate(settings=self.settings)
        # amygdala=None: the reinforcement loop only consults it for
        # marker trust, and markers belong to the application, not memory.
        self.loop = ReinforcementLoop(memory=self.graph, amygdala=None, settings=self.settings)
        # Что было вспомнено с прошлой записи. Нужен ОТДЕЛЬНЫЙ список, а не
        # trace.node_ids: тот принадлежит уже созданному действию, а связывать
        # новый узел надо с тем, что было активно ДО его появления.
        self._recently_recalled: List[int] = []

    # ----------------------------------------------------------------------
    # 1. Observing
    # ----------------------------------------------------------------------

    def observe(
        self,
        text: str,
        response: str = "",
        emotion: float = 0.0,
        load: float = 0.0,
        timestamp: Optional[float] = None,
        action_type: str = "observation",
    ) -> Observation:
        """
        Show the memory an event and let it decide whether to keep it.

        text     — what the user said (or what happened);
        response — what the system replied, if it did;
        emotion  — how charged the event is, [0, 1]. Where that number
                   comes from is the application's call: the core does not
                   infer it;
        load     — current overload, [0, 1]: raises the write threshold.

        THE ORDER MATTERS and mirrors the living one: the organism is
        surprised by the input first and learns it only afterwards. The
        other way round would mean being surprised by what it memorised a
        moment ago — that is, never being surprised at all.
        """
        ts = timestamp if timestamp is not None else self.clock()

        surprise = self.graph.compute_surprise(text).total
        decision = self.gate.evaluate(emotion=emotion, surprise=surprise, load=load)

        # A contradiction is a reason to store in its own right. A
        # correction is unsurprising by nature ("no, her name is Mia") and
        # never reaches the density threshold; without this branch,
        # corrections never made it into memory at all — that was a live
        # bug, not a hypothesis.
        superseded = self.graph.find_superseded(text, exclude_id=None)

        node_id = None
        if decision.is_spike or superseded:
            weight = decision.density
            if superseded:
                # The new version inherits the weight of what it replaces:
                # it is the same fact, updated, and has every right to take
                # its predecessor's place. Otherwise the write would happen
                # and the stale version would still win the search.
                inherited = max(
                    (self.graph.db.get_node(n.id)["weight"] for n in superseded),
                    default=0.0,
                )
                weight = max(weight, inherited)
            node_id = self.graph.save_connection(
                context=text, response=response, weight=weight, timestamp=ts,
            )
            self._associate_with_recalled(node_id, ts)

        learned = self.graph.process_language_input(text, timestamp=ts)

        self.loop.record_action(
            user_input=text, bot_output=response, node_id=node_id, action_type=action_type,
        )

        return Observation(
            text=text,
            surprise=surprise,
            decision=decision,
            node_id=node_id,
            superseded_ids=[n.id for n in superseded],
            learned_words=learned.new_words,
        )

    # ----------------------------------------------------------------------
    # 2. Feedback
    # ----------------------------------------------------------------------

    def feedback(self, valence: float, timestamp: Optional[float] = None) -> ReinforcementOutcome:
        """
        Rate the last action: valence in [-1, 1].

        It works on prediction error rather than on the rating itself: a
        node that is praised every time barely moves on another round of
        praise, while unexpected praise moves it a lot. That is the
        Rescorla-Wagner rule, and it is also what keeps weights from
        drifting off to infinity.
        """
        ts = timestamp if timestamp is not None else self.clock()
        return self.loop.apply(valence=valence, timestamp=ts)

    # ----------------------------------------------------------------------
    # 3. Recall
    # ----------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 3,
        timestamp: Optional[float] = None,
        with_associations: bool = True,
    ) -> List[MemoryMatch]:
        """
        Find what fits. Touching a node is not a free read: it raises the
        node's stability, so recall itself resists forgetting — the
        spacing effect.
        """
        ts = timestamp if timestamp is not None else self.clock()
        matches = self.graph.search(
            query, top_k=top_k, timestamp=ts, with_associations=with_associations,
        )
        self._remember_used([m.id for m in matches])
        # То, что было активно сейчас, свяжется со следующей записью — см.
        # associate_recalled_limit в observe.
        for match in matches:
            if match.id not in self._recently_recalled:
                self._recently_recalled.append(match.id)
        return matches

    def _associate_with_recalled(self, node_id: Optional[int], timestamp: float) -> None:
        """
        Links a fresh memory to whatever was ACTIVE when it appeared.

        Measured gap this closes: after 200 observe() calls the store held
        201 episodic nodes and ZERO edges between them. Not few — none. The
        library never linked memories to each other at all; the edges seen
        in the demo are created by the showcase (core/brain_session.py),
        which orchestrates recall and write together and therefore has both
        ends in hand.

        For a library user that meant two things at once: connectivity
        could never serve as an importance signal, and spreading activation
        — advertised as multi-hop retrieval and occupying a fair share of
        the search code — had nothing to travel along.

        The rule is Hebbian: what fires together wires together. What the
        application had just pulled out of memory is exactly the context in
        which the new memory formed.

        associate_recalled_limit = 0 disables this and restores the
        previous behaviour.
        """
        limit = self.settings.associate_recalled_limit
        if node_id is None or limit <= 0 or not self._recently_recalled:
            self._recently_recalled = []
            return

        # Свежайшие вспоминания первыми: связь с тем, что доставали только
        # что, осмысленнее связи с началом длинного разговора.
        for source_id in reversed(self._recently_recalled[-limit:]):
            if source_id != node_id:
                self.graph.connect_nodes(
                    source_id, node_id,
                    weight_boost=self.settings.associate_edge_weight,
                    timestamp=timestamp,
                )
        self._recently_recalled = []

    def _remember_used(self, node_ids: List[int]) -> None:
        """
        Marks nodes as INVOLVED in the current action so the next
        feedback() reaches them too.

        Without this, praise only reached the node that had just been
        WRITTEN — while with an assistant praise usually follows a good
        answer built from what was RECALLED. That is, the main case ("you
        remembered my allergy correctly — well done") reinforced nothing
        at all, and reward expectation stayed stuck at the value of the
        very first praise. Measured: eight consecutive praises produced
        the same expectation of 0.300 as one.

        Nodes ACCUMULATE within a turn: an application may call recall
        several times before answering, and the rating applies to the
        whole answer.
        """
        trace = self.loop.last_action_trace
        if trace is None or not node_ids:
            return
        existing = list(trace.node_ids or [])
        for node_id in node_ids:
            if node_id not in existing and node_id != trace.node_id:
                existing.append(node_id)
        trace.node_ids = existing

    def context_for(self, query: str, top_k: int = 3, timestamp: Optional[float] = None) -> str:
        """
        The same recall, but as a ready block of text for a prompt.

        An empty string is a legitimate answer meaning "I have nothing to
        add". The caller must handle it: a memory that always injects
        something is injecting noise.
        """
        matches = self.recall(query, top_k=top_k, timestamp=timestamp)
        if not matches:
            return ""
        return "\n".join(
            f"- {m.context}" + (f" -> {m.response}" if m.response else "")
            for m in matches
        )

    # ----------------------------------------------------------------------
    # 4. The passage of time
    # ----------------------------------------------------------------------

    def forget(self, now: Optional[float] = None) -> int:
        """
        Apply forgetting as of now. Returns the number of nodes affected.

        Call it EXPLICITLY and regularly — forgetting does not happen by
        itself between calls. Once per message is enough: decay is
        computed from elapsed time, not from the number of calls.
        """
        return self.graph.apply_decay(now=now if now is not None else self.clock())

    # ----------------------------------------------------------------------
    # 5. State
    # ----------------------------------------------------------------------

    def stats(self) -> MemoryStats:
        """A snapshot for a dashboard, a status command or debugging."""
        return MemoryStats(
            nodes=self.graph.count_nodes(),
            episodes=self.graph.db.count_nodes_by_type("episodic"),
            vocabulary=self.graph.get_vocabulary_size(),
            threshold=self.gate.base_threshold,
            semantic=self.graph._encode("probe") is not None,
        )

    def close(self) -> None:
        self.graph.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
