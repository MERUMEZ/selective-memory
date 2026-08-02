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
 REINFORCEMENT.PY — The reinforcement loop: importance from the user
================================================================================
CORE, not persona. This is the mechanism the whole project rests on:
the importance of a memory is NOT declared at write time and is not
decided by an LLM judge — it is earned by how the person reacted.

Measured (tools/compare_retention.py, 5 seeds, 14 days of silence):
    praised  100%   against  ordinary 60%
    naive stores show a gap near zero — they know nothing about reactions

Extracted from core/cortex.py, where 300 lines of the reinforcement loop
were tangled with speech generation, speech stages and prompt assembly.
For reuse as a memory library that split is mandatory: reinforcement
works with nodes and must know nothing about babbling, mood or LLMs.

WHAT LIVES HERE:
    - the dopamine signal: reward prediction error (Rescorla-Wagner),
      which is why habitual praise stops having an effect;
    - applying the effect to nodes: reinforcement or penalty, scaled by
      how unexpected the outcome was;
    - retrospective correction: if the user contradicts themselves
      (sarcasm, "no, wait, that is wrong"), the previous effect is rolled
      back and trust in the markers that fired is lowered.

WHAT MUST NEVER LIVE HERE: mood, echolalia, speech stages, prompts. The
caller receives a ReinforcementOutcome and decides for itself what that
means for its persona (see Cortex.apply_feedback).
================================================================================
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from selectivemem.settings import MemorySettings
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionTrace:
    """
    A trace of input -> action, awaiting a rating on the NEXT turn.

    node_id   — a single node (whether one particular memory was used).
    node_ids  — several nodes (babbling draws on a set of syllables, a
                one-word reply on a set of words). None when the action
                did not lean on memory at all.
    """
    user_input: str
    bot_output: str
    node_id: Optional[int]
    action_type: str
    node_ids: Optional[List[int]] = None


@dataclass
class FeedbackHistoryEntry:
    """One already-applied effect — the window for retrospective correction."""
    timestamp: Optional[float]
    valence: float
    matched_markers: List[str]
    node_id: Optional[int]
    user_input: str
    bot_output: str
    action_type: str
    applied_delta: float = 0.0
    reversed: bool = False
    node_ids: Optional[List[int]] = None


@dataclass
class RetrospectiveCorrectionResult:
    """Outcome of checking whether an earlier rating was later reversed."""
    triggered: bool
    reversed_entry: Optional[FeedbackHistoryEntry] = None
    reversal_delta: float = 0.0
    penalized_markers: List[str] = field(default_factory=list)


@dataclass
class ReinforcementOutcome:
    """
    What reinforcement did to MEMORY. Anything persona-related (mood, a
    tendency towards echolalia) is for the caller to derive from this
    data — the core knows no such notions.

    congruence — the reward prediction error. That, rather than raw
    valence, is the right input for an emotion: habitual praise should not
    feel good.
    """
    effect: str  # "rewarded" | "penalized" | "neutral" | "no_trace"
    valence: float
    congruence: float
    applied_delta: float
    trace: Optional[ActionTrace]
    retrospective: Optional[RetrospectiveCorrectionResult] = None


class ReinforcementLoop:
    """
    Turns a human reaction into a change in memory.

    Usage:
        loop = ReinforcementLoop(memory=graph, amygdala=amygdala)
        loop.record_action(user_input, bot_output, node_id=..., action_type=...)
        outcome = loop.apply(valence=0.8, timestamp=brain_time)
    """

    def __init__(self, memory, amygdala, settings: Optional[MemorySettings] = None):
        self.memory = memory
        # Default to the memory's own settings: the reinforcement loop and
        # the graph must live by the same constants.
        self.settings = settings or getattr(memory, "settings", None) or MemorySettings()
        # It must be the SAME amygdala instance that detects markers:
        # penalties and trust recovery have to affect later parsing.
        self.amygdala = amygdala
        self.last_action_trace: Optional[ActionTrace] = None
        self.feedback_history: "deque[FeedbackHistoryEntry]" = deque(
            maxlen=self.settings.retrospective_window_size
        )

    # ----------------------------------------------------------------------
    # Recording an action
    # ----------------------------------------------------------------------

    def record_action(
        self,
        user_input: str,
        bot_output: str,
        node_id: Optional[int],
        action_type: str,
        node_ids: Optional[List[int]] = None,
    ) -> ActionTrace:
        """Remembers the pairing so the next reaction can rate it."""
        self.last_action_trace = ActionTrace(
            user_input=user_input,
            bot_output=bot_output,
            node_id=node_id,
            action_type=action_type,
            node_ids=list(node_ids) if node_ids else None,
        )
        return self.last_action_trace

    # ----------------------------------------------------------------------
    # Applying a rating
    # ----------------------------------------------------------------------

    def apply(
        self,
        valence: float,
        timestamp: Optional[float] = None,
        matched_markers: Optional[List[str]] = None,
    ) -> ReinforcementOutcome:
        """
        Applies the user's reaction to the previous action.

        The order matters: retrospective correction first (does this
        reaction reverse a rating already applied?), then the dopamine
        signal (which sets the RATE of consolidation), and only then the
        effect on the nodes themselves.
        """
        trace = self.last_action_trace
        matched_markers = list(matched_markers or [])

        if valence == 0.0:
            return ReinforcementOutcome("neutral", valence, 0.0, 0.0, None)

        if trace is None:
            logger.debug("[REWARD] valence=%.2f, but there is no action to evaluate", valence)
            return ReinforcementOutcome("no_trace", valence, valence, 0.0, None)

        retrospective = self._check_retrospective_correction(valence, timestamp)

        # --- DOPAMINE: reward prediction error ---
        # Computed BEFORE any effects, because it — not raw valence — sets
        # the rate of consolidation: unexpected praise consolidates
        # strongly, fully predicted praise almost not at all.
        # ALL nodes involved in the action are rewarded: both the one
        # just written and the ones the action leaned on.
        #
        # This used to be either/or, which silently lost the main case:
        # if a turn wrote something, what was recalled got nothing. With
        # an assistant, praise usually lands on an answer built from
        # memory — so it is the recalled part that must be reinforced.
        # Measured through the facade: eight consecutive praises produced
        # the same expectation of 0.300 as one, because only the first
        # ever arrived.
        reward_nodes = list(dict.fromkeys(
            ([trace.node_id] if trace.node_id is not None else [])
            + list(trace.node_ids or [])
        ))
        signals = [
            s for s in (
                self.memory.apply_reward(node_id, valence, timestamp=timestamp)
                for node_id in reward_nodes
            ) if s is not None
        ]
        # One multiplier for the whole action: the strength of the most
        # unexpected node involved. The action is rated as a whole.
        learning_scale = (
            max(self.memory.learning_scale(s.prediction_error) for s in signals)
            if signals else 1.0
        )
        # If the action leaned on no node at all there was nothing to
        # expect, so the rating itself counts as the surprise.
        congruence = (
            max((s.prediction_error for s in signals), key=abs) if signals else valence
        )

        effect, applied_delta = self._apply_effect(
            trace, valence, learning_scale, timestamp
        )

        self._record_history(
            FeedbackHistoryEntry(
                timestamp=timestamp,
                valence=valence,
                matched_markers=matched_markers,
                node_id=trace.node_id,
                user_input=trace.user_input,
                bot_output=trace.bot_output,
                action_type=trace.action_type,
                applied_delta=applied_delta,
                reversed=False,
                node_ids=trace.node_ids,
            )
        )

        return ReinforcementOutcome(
            effect=effect,
            valence=valence,
            congruence=congruence,
            applied_delta=applied_delta,
            trace=trace,
            retrospective=retrospective,
        )

    def _involved_nodes(self, trace) -> List[int]:
        """
        Every node involved in the action: the one written plus the ones
        the action leaned on.

        This used to be either/or — if a turn wrote something, the rating
        never reached what was recalled. That silently lost the main
        assistant case: praise applies to a good answer BUILT FROM MEMORY,
        so it is the recalled part that must be reinforced.
        """
        return list(dict.fromkeys(
            ([trace.node_id] if trace.node_id is not None else [])
            + list(trace.node_ids or [])
        ))

    def _apply_effect(self, trace, valence, learning_scale, timestamp):
        """Reinforces or penalises the nodes involved."""
        nodes = self._involved_nodes(trace)
        if not nodes:
            return "neutral", 0.0

        if valence > 0:
            boost = valence * self.settings.reward_positive_boost * learning_scale
            for node_id in nodes:
                self.memory.reinforce_node(node_id, boost=boost, timestamp=timestamp)
                # The access mark is pushed forward: the node looks
                # fresher than it is and decays more slowly.
                if timestamp is not None:
                    self.memory.touch_node(
                        node_id,
                        timestamp=timestamp + self.settings.reward_positive_freshness_bonus,
                    )
            if len(nodes) > 1:
                # A successful combination is reinforced as a whole and
                # the links BETWEEN its parts are strengthened: that is how
                # a repeatedly successful set crystallises into a stable
                # association.
                self.memory.reinforce_coactivation(
                    nodes, weight_boost=boost, timestamp=timestamp
                )
            logger.info("[REWARD] +%.2f -> nodes %s", valence, nodes)
            return "rewarded", boost

        penalty = abs(valence) * self.settings.reward_negative_penalty * learning_scale
        for node_id in nodes:
            self.memory.penalize_node(node_id, penalty=penalty, timestamp=timestamp)
        logger.info("[REWARD] %.2f -> penalty to nodes %s", valence, nodes)
        return "penalized", -penalty

    # ----------------------------------------------------------------------
    # Retrospective correction
    # ----------------------------------------------------------------------

    def _check_retrospective_correction(
        self, valence: float, timestamp: Optional[float]
    ) -> RetrospectiveCorrectionResult:
        """
        Looks through the history window for a not-yet-reverted rating of
        the OPPOSITE sign. If one is found, the current reaction is read as
        a delayed reversal (sarcasm, "no, wait, that is wrong"): the
        earlier effect is rolled back with an amplified correction, and
        trust in the markers that caused it is lowered.
        """
        if not self.settings.retrospective_correction_enabled:
            return RetrospectiveCorrectionResult(triggered=False)
        if timestamp is None or valence == 0.0:
            return RetrospectiveCorrectionResult(triggered=False)

        for entry in reversed(self.feedback_history):
            if entry.reversed or entry.node_id is None or entry.valence == 0.0:
                continue
            if (entry.valence > 0) == (valence > 0) or entry.timestamp is None:
                continue

            elapsed = timestamp - entry.timestamp
            if elapsed < 0 or elapsed > self.settings.retrospective_time_window_seconds:
                continue

            # A confirmed false rating is a far stronger learning signal
            # than ordinary one-off feedback, hence the multiplier.
            reversal = -entry.applied_delta * self.settings.retrospective_reversal_strength
            if reversal > 0:
                self.memory.reinforce_node(entry.node_id, boost=reversal, timestamp=timestamp)
            elif reversal < 0:
                self.memory.penalize_node(entry.node_id, penalty=abs(reversal), timestamp=timestamp)

            entry.reversed = True
            if entry.matched_markers:
                self.amygdala.penalize_markers(entry.matched_markers)

            logger.info(
                "[RETROSPECTIVE] Reversal: was %.2f (t=%.1f), now %.2f (t=%.1f) "
                "-> node %s, reverted %.3f, markers %s",
                entry.valence, entry.timestamp, valence, timestamp,
                entry.node_id, reversal, entry.matched_markers,
            )

            return RetrospectiveCorrectionResult(
                triggered=True,
                reversed_entry=entry,
                reversal_delta=reversal,
                penalized_markers=list(entry.matched_markers),
            )

        return RetrospectiveCorrectionResult(triggered=False)

    def _record_history(self, entry: FeedbackHistoryEntry) -> None:
        """
        Puts an entry into the window. If the oldest one is evicted and
        was NEVER reversed, its markers survived the whole window and have
        earned some trust back.
        """
        if len(self.feedback_history) == self.feedback_history.maxlen:
            oldest = self.feedback_history[0]
            if not oldest.reversed and oldest.matched_markers:
                self.amygdala.recover_markers(oldest.matched_markers)

        self.feedback_history.append(entry)
