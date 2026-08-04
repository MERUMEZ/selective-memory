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
 SYNAPSES.PY — Уровень связи: вес, затухание, предел ёмкости
================================================================================
Затухание узлов и связей, пол угасания, вытеснение по ёмкости.

НАЗВАНО ПО НОСИТЕЛЮ, А НЕ ПО ПРОЦЕССУ. Прежде файл звался forgetting.py,
и это была ошибка того же рода, что разнесение словаря и понятий:
забывание — не участок мозга и не орган, а СЛЕДСТВИЕ того, что
происходит с отдельной связью. Здесь описано именно это: как слабеет
синапс, до какого предела и что случается, когда их становится слишком
много.

Участку принадлежит решение, ЧТО хранить; синапсу — насколько прочно.

ЗДЕСЬ ЖИВУТ ДВЕ ТЕОРИИ СРАЗУ, и это надо знать, читая код.

Ранжирование выдачи уже переведено на ИНТЕРФЕРЕНЦИЮ: забывание есть
проигрыш в конкуренции, важное определяется долей накопленной силы, и
часы её не трогают. Это лучше подтверждённое объяснение человеческого
забывания.

А само затухание здесь по-прежнему РАСПАД: вес убывает экспонентой от
прошедшего времени.

    вес *= exp(-decay_rate * прошло / (базовый_срок(тип) * стабильность))

Переход остановлен на середине сознательно: замеры удержания стоят на
нынешней формуле, и менять её надо отдельной работой с полным
перемером, а не заодно.

ПО ВОЗРАСТУ НИЧЕГО НЕ УДАЛЯЕТСЯ — delete_on_decay выключен, и это подняло
полноту на внешнем наборе на 18.6 пункта. Затухание теперь только
понижает вес; предел памяти держится на отборе при записи и на связях,
которые зарастают, если по ним не ходят.

Класс — миксин: состояние принадлежит MemoryGraph.
================================================================================
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SynapsesMixin:
    """Затухание, пол угасания, вытеснение по ёмкости."""

    def apply_decay(self, now: Optional[float] = None) -> int:
        """
        Applies exponential weight decay to every long-term node and to
        the edges of the associative graph. Rare or unused edges (weight
        below edge_forget_threshold) are deleted outright.
        """
        current_time = now if now is not None else time.time()

        decayed_count = self._decay_nodes(current_time)
        self._decay_edges(current_time)
        self._evict_over_capacity()

        return decayed_count

    def _keep_score(self, row: Any) -> float:
        """
        How much a memory DESERVES to stay when the store is full.

        Deliberately does NOT use weight. Weight is dominated by age —
        decay is built into it — so evicting by weight would evict the
        oldest, which is exactly the failure already measured: for
        knowledge-update questions the evidence was on average 16 days
        older than the question and age alone erased all of it, 12 nodes
        out of 12.

        What is used instead survives the passage of time:
            reward_expectation — approval the user actually gave;
            stability          — grows 1.5x on every recall, so it counts
                                 how often the memory PROVED USEFUL;
            spike_strength     — how hard the event hit when it happened.
        """
        if self.settings.use_relative_strength:
            # В модели интерференции заслуга УЖЕ НАКОПЛЕНА в strength:
            # туда сложились и одобрение, и польза от каждого извлечения, и
            # сила рождения. Складывать те же слагаемые второй раз незачем,
            # а главное — сюда не просачивается возраст, потому что часы
            # силу не трогают.
            return float(row["strength"] if row["strength"] is not None else row["weight"])

        expectation = max(0.0, row["reward_expectation"] or 0.0)
        stability = (row["stability"] or self.settings.stability_initial)
        stability_norm = min(1.0, stability / max(1e-9, self.settings.stability_max))
        spike = row["spike_strength"] or 0.0
        return expectation + stability_norm + spike

    def _evict_over_capacity(self) -> int:
        """
        Keeps the store within memory_capacity by dropping the LEAST
        DESERVING memories rather than the oldest.

        Why capacity rather than age. Age-based deletion cannot be told
        apart from importance-based deletion by anyone looking at the
        result, and measurement showed it was deciding on its own: a tenth
        of memory was removed on every pass, and the answers to later
        questions were always inside that tenth.

        Capacity is also what an application actually wants to control.
        A game designer sets "this character remembers two hundred things";
        nobody wants to set "this character forgets after eleven days".

        memory_capacity = 0 means unlimited and nothing is evicted.
        """
        capacity = self.settings.memory_capacity
        if capacity <= 0:
            return 0

        rows = [
            row for row in self.db.fetch_all_nodes()
            if not row["is_meta"]
            and row["node_type"] not in self.LEXICAL_NODE_TYPES
        ]
        if len(rows) <= capacity:
            return 0

        rows.sort(key=self._keep_score)
        doomed = rows[: len(rows) - capacity]
        for row in doomed:
            self.db.delete_node(row["id"])

        logger.info(
            "[CAPACITY] %d memories evicted, %d kept (limit %d)",
            len(doomed), capacity, capacity,
        )
        return len(doomed)

    def _age_t0_for(self, node_type: Optional[str]) -> float:
        """
        The characteristic lifetime of a node for the decay formula.
        Vocabulary fades on lexical_age_t0 (~30 days), everything else on
        age_t0 (~7 subjective hours).

        Without that split, a mastered word lost its status overnight and
        was deleted from the database within a day — the vocabulary could
        never accumulate at all.
        """
        if node_type in self.LEXICAL_NODE_TYPES:
            return self.settings.lexical_age_t0
        return self.settings.age_t0

    def _decay_nodes(self, current_time: float) -> int:
        """Exponential weight decay for nodes."""
        rows = self.db.fetch_all_nodes()

        updates = []
        to_forget = []
        decayed_count = 0
        skipped_meta_count = 0

        for row in rows:
            if row["is_meta"]:
                skipped_meta_count += 1
                continue

            # NULL guard: if last_decayed_at was never set — some path
            # of creation or update missed it — last_accessed serves as
            # the origin instead of crashing with a TypeError. The same
            # pass fills last_decayed_at in through updates[] below.
            last_decayed = row["last_decayed_at"]
            if last_decayed is None:
                last_decayed = row["last_accessed"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = row["weight"]
            # Effective lifetime = the base for this node type times
            # stability. Stability grows on every recall (see
            # Database.update_last_accessed), so memory that is in demand
            # resists time while memory that is not disappears quickly.
            stability = row["stability"] if row["stability"] else self.settings.stability_initial
            effective_t0 = self._age_t0_for(row["node_type"]) * max(1e-9, stability)
            decay_factor = math.exp(-self.settings.decay_rate * dt / effective_t0)

            # THE DECAY FLOOR. A node the user praised dims but does not
            # vanish: it decays towards a non-zero level rather than to
            # nothing.
            #
            # Why. Stability grows through RECALL, so something marked as
            # important but never needed again still sank to zero and was
            # deleted. Measured: after two weeks of silence, NOTHING
            # survived a fifteen-message conversation — including what the
            # user had explicitly called important. For an assistant that
            # was told "I am allergic to penicillin" that is unacceptable:
            # such a thing must never be forgotten, even if it never came
            # up again.
            #
            # Where the floor's height comes from: reward_expectation, the
            # very quantity the Rescorla-Wagner rule maintains (see
            # apply_reward). It already means "how much approval this node
            # has earned" — a ready-made answer to "how important is this
            # TO THE USER". No extra column is needed and no threshold is
            # assigned by hand.
            #
            # A node that was never reinforced has an expectation of 0 and
            # a floor of 0 — nothing changes for it, and it fades as
            # before. So this does not make memory immortal: only a
            # handful of nodes ever get a floor.
            expectation = row["reward_expectation"] or 0.0
            floor = max(
                # Earned by approval: the more a node was reinforced, the
                # higher it settles.
                max(0.0, expectation) * self.settings.memory_floor_max,
                # Earned by being written at all. Measured: without this,
                # decay DELETED every evidence node for knowledge-update
                # questions — 12 of 12, five instances out of five — while
                # only a tenth of memory was removed overall. The evidence
                # is simply the oldest thing in the store, and age alone
                # decided its fate.
                self.settings.memory_floor_base,
                # Earned by the force of the event. A flat floor saves
                # everything and so forgets nothing, which costs the whole
                # praised-over-routine gap; this term keeps the gap, because
                # routine that barely cleared the gate gets a floor below
                # forget_threshold and still dies of old age.
                (row["spike_strength"] or 0.0) * self.settings.memory_floor_spike_factor,
            )
            floor = min(floor, old_weight)          # the floor never raises a weight
            new_weight = floor + (old_weight - floor) * decay_factor

            if new_weight < self.settings.forget_threshold and self.settings.delete_on_decay:
                to_forget.append(row["id"])
            else:
                updates.append({
                    "id": row["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_weights(updates)
            logger.info(
                "[DECAY APPLIED] Weights updated: %d nodes (meta skipped: %d)",
                len(updates), skipped_meta_count,
            )

        for node_id in to_forget:
            self.db.delete_node(node_id)

        if to_forget:
            logger.info("[MEMORY FORGOTTEN] Nodes deleted (weight < FORGET_THRESHOLD): %d", len(to_forget))

        return decayed_count

    def _decay_edges(self, current_time: float) -> int:
        """
        Exponential weight decay for edges. Edges fade faster than nodes
        (edge_decay_rate is normally above decay_rate), modelling the fact
        that associations between memories are more fragile than the
        memories themselves. Edges below edge_forget_threshold are deleted
        outright.
        """
        edges = self.db.fetch_all_edges()

        updates = []
        to_forget = []
        decayed_count = 0

        for edge in edges:
            # NULL guard — see the comment in _decay_nodes.
            last_decayed = edge["last_decayed_at"]
            if last_decayed is None:
                last_decayed = edge["last_activated"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = edge["weight"]
            # Ассоциации между эпизодами тают МЕДЛЕННЕЕ словарных связей, и
            # это не вкусовщина. Словарное ребро подкрепляется каждым
            # повтором фразы и обязано выветриваться быстро, если слово
            # перестали произносить. Пара реплик не повторяется никогда, и
            # на общей скорости связи исчезали между четвёртыми и
            # одиннадцатыми сутками — вместе с многошаговым извлечением,
            # выигрыш от которого падал с +6.7 пункта до нуля за неделю.
            rate = (
                self.settings.associate_edge_decay_rate
                if edge["edge_type"] == "association"
                else self.settings.edge_decay_rate
            )
            decay_factor = math.exp(-rate * dt / self.settings.age_t0)
            new_weight = old_weight * decay_factor

            if new_weight < self.settings.edge_forget_threshold:
                to_forget.append(edge["id"])
            else:
                updates.append({
                    "id": edge["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_edge_weights(updates)
            logger.info(
                "[EDGE DECAY APPLIED] Weights updated: %d edges (t=%.2f)",
                len(updates), current_time,
            )

        for edge_id in to_forget:
            self.db.delete_edge(edge_id)

        if to_forget:
            logger.info(
                "[EDGE FORGOTTEN] Edges deleted (weight < EDGE_FORGET_THRESHOLD): %d",
                len(to_forget),
            )

        return decayed_count
