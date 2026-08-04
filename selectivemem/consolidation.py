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
 CONSOLIDATION.PY — Что память делает, когда её не спрашивают
================================================================================
Здесь собрано всё, что превращает груду эпизодов в структуру: реактивация,
гомеостатическое понижение, подрезка, поиск плотных кластеров, свёртка их
в абстракцию и свёртка кратковременного буфера в эпизод.

ПОРЯДОК СТАДИЙ ПОВТОРЯЕТ НОЧНОЙ и задан в Memory.sleep():

    1. replay            — сильнейшие следы переигрываются вместе, связи
                           между ними крепнут. Это то, что переносит след
                           из гиппокампа в кору;
    2. downscale_edges   — все связи слабеют пропорционально, порядок
                           сохраняется, держащееся на волоске уходит само;
    3. run_synaptic_pruning — под нож идёт не выдержавшее сжатия;
    4. find_hub_clusters + create_abstract_node — плотный кластер
                           сворачивается в одну схему.

Стадий 1 и 2 у нас не было вовсе: сон умел чистить, но не консолидировать.
Обе написаны и ВЫКЛЮЧЕНЫ по умолчанию — срабатывание проверено, пользы
измерить не удалось, а включать умолчание без замера здесь не принято.

ИЗВЕСТНОЕ ПРОТИВОРЕЧИЕ, записанное честно. Свёртка АРХИВИРУЕТ источники,
понижая им вес и силу, но не удаляет: удаление по возрасту отключено, и
это подняло полноту на 18.6 пункта. Значит сон складывает в архив,
который никто не разбирает, и сжатия в штуках узлов ждать неоткуда. Два
решения, принятые в разное время, друг другу противоречат — см. раздел
2.17 аудита.

Класс — миксин: состояние принадлежит MemoryGraph, который его
подмешивает.
================================================================================
"""

import logging
import time
from typing import Any, Dict, List, Optional

from selectivemem.records import (
    ConsolidationResult,
    HubCluster,
    PruningReport,
)

logger = logging.getLogger(__name__)


class ConsolidationMixin:
    """Реактивация, понижение, подрезка, абстракция, свёртка эпизода."""

    def prune_weak_edges(self, min_weight: Optional[float] = None) -> int:
        """
        Explicit, non-decay pruning of edges: deletes EVERY edge with
        weight < min_weight. Used by the sleep phase, unlike _decay_edges,
        which first lowers weights and deletes only what fell below the
        threshold DURING that call.

        Returns the number of edges deleted.
        """
        threshold = min_weight if min_weight is not None else self.settings.edge_forget_threshold
        deleted = self.db.delete_edges_below_weight(threshold)
        return deleted

    def prune_orphan_nodes(
        self,
        min_edge_weight: Optional[float] = None,
        max_node_weight: Optional[float] = None,
    ) -> int:
        """
        Deletes orphaned nodes: those with no strong edge (weight >=
        min_edge_weight) AND weak in themselves (node weight below
        max_node_weight). Strong isolated memories are left alone — they
        stand on their own even when linked to nothing.

        Returns the number of nodes deleted.
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.edge_activation_threshold
        )
        effective_node_weight = (
            max_node_weight if max_node_weight is not None else self.settings.sleep_orphan_weight_threshold
        )

        orphans = self.db.get_orphan_nodes(
            min_edge_weight=effective_edge_weight,
            max_node_weight=effective_node_weight,
        )

        for orphan in orphans:
            self.db.delete_node(orphan["id"])

        if orphans:
            logger.info(
                "[SLEEP PRUNING] Orphan nodes deleted (weight < %.2f, no edges >= %.2f): %d",
                effective_node_weight, effective_edge_weight, len(orphans),
            )

        return len(orphans)

    def replay(self, limit: Optional[int] = None,
               timestamp: Optional[float] = None) -> int:
        """
        Реактивация во сне: сильнейшие следы переигрываются вместе, и
        связи между ними крепнут.

        ЭТО ПЕРВАЯ СТАДИЯ КОНСОЛИДАЦИИ, И ЕЁ У НАС НЕ БЫЛО. Сон делал
        только последние две — подрезку и свёртку кластера, — тогда как
        именно реактивация переносит след из гиппокампа в кору. В
        медленном сне дневные последовательности переигрываются пачками
        разрядов, причём избирательно: то, что привело к награде, чаще.

        Отбор здесь по накопленной силе, то есть по тому, что уже
        доказало пользу или получило одобрение. Это и есть избирательность
        реплея, выраженная тем, что у нас для неё есть.

        УКРЕПЛЯЮТСЯ СВЯЗИ, А НЕ СИЛА УЗЛОВ, и это не мелочь. Поднимать
        силу вслепую мы уже пробовали: замер показал, что так закрепляется
        и ошибочно извлечённое (раздел 2.16 аудита). Связь же говорит не
        "это важно", а "эти двое про одно и то же" — из неё на следующей
        стадии и вырастает схема, которую сворачивает create_abstract_node.

        Возвращает число переигранных узлов.
        """
        count = limit if limit is not None else self.settings.sleep_replay_nodes
        if count <= 0:
            return 0

        rows = [
            row for row in self.db.fetch_searchable_nodes()
            if row["node_type"] == "episodic"
        ]
        if len(rows) < 2:
            logger.info("[SLEEP REPLAY] Too few nodes to replay: %d", len(rows))
            return 0

        rows.sort(
            key=lambda r: (r["strength"] if r["strength"] is not None else r["weight"]),
            reverse=True,
        )
        selected = [row["id"] for row in rows[:count]]
        self.reinforce_coactivation(
            selected,
            weight_boost=self.settings.sleep_replay_edge_boost,
            timestamp=timestamp,
        )
        logger.info(
            "[SLEEP REPLAY] Replayed %d strongest traces, %d pairs linked",
            len(selected), len(selected) * (len(selected) - 1) // 2,
        )
        return len(selected)

    def downscale_edges(self, factor: Optional[float] = None) -> int:
        """
        Гомеостатическое понижение: ВСЕ связи слабеют пропорционально.

        Гипотеза синаптического гомеостаза (Тонони и Чирелли): за день
        синапсы в среднем усиливаются, и сон нужен, чтобы вернуть их к
        рабочему уровню, понизив все разом. Относительный порядок при этом
        сохраняется — теряется только то, что и так держалось на волоске.

        Тот же принцип, на который мы уже перевели ранжирование: значение
        имеет доля, а не абсолютная величина. После понижения подрезка
        перестаёт быть отдельной политикой с собственным порогом и
        становится его следствием: под нож идёт ровно то, что не выдержало
        общего сжатия.

        Возвращает число изменённых связей.
        """
        scale = factor if factor is not None else self.settings.sleep_downscale_factor
        if scale >= 1.0:
            return 0

        edges = self.db.fetch_all_edges()
        if not edges:
            logger.info("[SLEEP DOWNSCALE] No edges to scale")
            return 0

        # last_decayed_at переносится КАК ЕСТЬ. Понижение — не затухание:
        # оно не отсчитывает время заново, иначе сон незаметно продлевал
        # бы жизнь всем связям разом, сдвигая им точку отсчёта.
        updates = [
            {
                "id": row["id"],
                "weight": row["weight"] * scale,
                "last_decayed_at": row["last_decayed_at"],
            }
            for row in edges
        ]
        self.db.bulk_update_edge_weights(updates)
        logger.info(
            "[SLEEP DOWNSCALE] %d edges scaled by %.2f",
            len(updates), scale,
        )
        return len(updates)

    def run_synaptic_pruning(self) -> PruningReport:
        """
        The full synaptic pruning cycle: weak edges are cut first, orphan
        nodes are sought AFTERWARDS. The order matters — removing weak
        edges can orphan nodes that were held up by nothing else.
        """
        edges_pruned = self.prune_weak_edges()
        orphan_nodes_pruned = self.prune_orphan_nodes()

        # Печатается ВСЕГДА, даже когда резать нечего. Иначе "механизм не
        # вызывали" и "механизм отработал вхолостую" выглядят одинаково —
        # tools/check_liveness.py считает срабатывания по логам и объявил
        # подрезку мёртвой, хотя она честно отработала на пустом графе.
        logger.info(
            "[SLEEP PRUNING] Pass complete: %d edges, %d orphan nodes removed",
            edges_pruned, orphan_nodes_pruned,
        )
        return PruningReport(
            edges_pruned=edges_pruned,
            orphan_nodes_pruned=orphan_nodes_pruned,
        )

    def find_hub_clusters(
        self,
        min_edge_weight: Optional[float] = None,
        min_spokes: Optional[float] = None,
        max_spokes: Optional[float] = None,
        limit: int = 1,
        timestamp: Optional[float] = None,
    ) -> List[HubCluster]:
        """
        Looks for hub-and-spoke clusters: a dominant node with the
        largest sum of strong edge weights (its hub score) and its top-N
        strongest neighbours, as candidates for semantic consolidation
        during the sleep phase.

        Returns up to `limit` clusters sorted by hub score. A cluster is
        included only when the hub has at least min_spokes strong
        neighbours — otherwise it is not a cluster but merely a pair.
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.sleep_hub_min_edge_weight
        )
        effective_min_spokes = (
            min_spokes if min_spokes is not None else self.settings.sleep_min_cluster_spokes
        )
        effective_max_spokes = (
            max_spokes if max_spokes is not None else self.settings.sleep_max_cluster_spokes
        )

        hub_rows = self.db.get_hub_candidates(min_edge_weight=effective_edge_weight)

        clusters: List[HubCluster] = []
        used_node_ids: set = set()

        for hub_row in hub_rows:
            if len(clusters) >= limit:
                break

            hub_id = hub_row["id"]
            if hub_id in used_node_ids:
                continue

            associated = self.get_associated_nodes(
                hub_id,
                min_weight=effective_edge_weight,
                limit=effective_max_spokes,
                timestamp=timestamp,
            )

            # Drop spokes already claimed by another cluster in this same
            # pass, so no node is consolidated twice
            available_spokes = [a for a in associated if a.id not in used_node_ids]

            if len(available_spokes) < effective_min_spokes:
                continue

            cluster = HubCluster(
                hub_id=hub_id,
                hub_context=hub_row["context"],
                hub_response=hub_row["response"],
                hub_weight=hub_row["weight"],
                spoke_ids=[a.id for a in available_spokes],
                spoke_contexts=[a.context for a in available_spokes],
                spoke_responses=[a.response for a in available_spokes],
                spoke_weights=[a.weight for a in available_spokes],
                edge_weights=[a.edge_weight for a in available_spokes],
            )

            clusters.append(cluster)
            used_node_ids.add(hub_id)
            used_node_ids.update(cluster.spoke_ids)

            logger.info(
                "[SLEEP CLUSTER] Cluster found: hub=%s (weight=%.2f) + spokes=%s",
                hub_id, hub_row["weight"], cluster.spoke_ids,
            )

        return clusters

    def create_abstract_node(
        self,
        summary_context: str,
        summary_response: str,
        source_node_ids: List[int],
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Stores a new abstract long-term node — the result of semantically
        consolidating a cluster — and ARCHIVES the cluster's source nodes
        by multiplying their weight by sleep_archive_weight_multiplier.
        They are not deleted immediately: they become candidates for
        deletion at the next ordinary decay or pruning pass if they remain
        unused.

        The new abstract node is also linked by edges to every source node
        of the cluster, preserving a trace of where it came from.

        Returns the id of the new abstract node.
        """
        effective_weight = weight if weight is not None else self.settings.sleep_abstract_node_weight
        ts = timestamp if timestamp is not None else time.time()

        # ТОТ ЖЕ ТИП, ЧТО У СВЁРТОК ЭПИЗОДА, и по той же измеренной
        # причине: абстракция, пущенная в общий поиск, содержит слишком
        # много слов и совпадает почти с любым запросом. У консолидации
        # это роняло R@1 с 76% до 42%, а понижение силы спасало лишь
        # частично (56% и 60%), потому что выигрывала она НА
        # РЕЛЕВАНТНОСТИ, а не на важности.
        #
        # Схема достаётся отдельным ходом — Memory.summaries().
        abstract_node_id = self.db.insert_node(
            context=summary_context,
            response=summary_response,
            weight=effective_weight,
            timestamp=ts,
            node_type="episode_summary",
        )

        for source_id in source_node_ids:
            source_row = self.db.get_node(source_id)
            if source_row is None:
                continue

            archived_weight = source_row["weight"] * self.settings.sleep_archive_weight_multiplier
            self.db.update_weight(source_id, archived_weight)

            # АБСТРАКЦИЯ ЗАБИРАЕТ ДОЛЮ, а не прибавляется сверху. Прежде
            # сон добавлял узел и не убирал ни одного, то есть память
            # росла монотонно с каждым циклом. В модели интерференции
            # ценность — доля от суммы, поэтому источники обязаны отдать
            # ровно столько, сколько получила свёртка.
            if self.settings.use_relative_strength:
                base = source_row["strength"]
                if base is None:
                    base = source_row["weight"]
                self.db.add_strength(
                    source_id,
                    base * (self.settings.sleep_archive_weight_multiplier - 1.0),
                    self.settings.strength_max,
                )

            self.connect_nodes(abstract_node_id, source_id, weight_boost=self.settings.edge_initial_weight, timestamp=ts)

        logger.info(
            "[SLEEP CONSOLIDATION] Abstract node id=%s (weight=%.2f) created from cluster %s",
            abstract_node_id, effective_weight, source_node_ids,
        )

        return abstract_node_id

    def consolidate_from_stm(
        self,
        entries: List["STMEntry"],
        timestamp: Optional[float] = None,
        already_captured_by_spike: bool = False,
    ) -> ConsolidationResult:
        """
        Judges an accumulated short-term episode and decides according to
        the physics of selective consolidation:

            a) EMOTIONAL NODE — if the highest emotion_score among the STM
               entries reaches stm_emotional_threshold, the episode is
               packed and stored with a high weight (that same maximum).

            b) STRUCTURAL NODE — if the episode's average surprise reaches
               stm_structural_threshold (a lot of new information), it is
               packed and stored with a moderate weight
               (stm_structural_weight).

            c) ROUTINE NOISE — neither of the above: the episode is simply
               discarded without touching the database, which is how
               memory is spared everyday chatter.

        `entries` is a list of STMEntry, normally the result of
        working_memory.consume_all(). An empty list counts as routine
        noise straight away — there is no data.

        Returns a ConsolidationResult with the decision and the details
        for logs and debugging.
        """
        if not entries:
            return ConsolidationResult(decision="routine_noise", reason="STM is empty, no data")

        # Duplicate guard: if this flush was triggered by a spike that
        # already created an exact long-term node for the current
        # exchange, and STM at flush time holds only that same exchange
        # (at most two entries — one user turn and one bot turn), then
        # consolidation would create a near-identical duplicate. The write
        # is skipped: the content is already stored.
        if already_captured_by_spike and len(entries) <= 2:
            logger.info(
                "[CONSOLIDATION] Skipped: the exchange is already stored as a spike node "
                "(entries=%d)",
                len(entries),
            )
            return ConsolidationResult(
                decision="routine_noise",
                reason="Already stored as a spike node, duplicate skipped",
            )

        max_emotion = max(e.emotion_score for e in entries)

        # Среднее удивление берётся ТОЛЬКО по репликам собеседника.
        # Ответы бота попадают в буфер с нулевым удивлением — оно и
        # считается лишь для входящего текста, — поэтому усреднение по
        # всему буферу делило результат пополам: 0.523 превращалось в
        # 0.26, и порог 0.55 не брался НИКОГДА.
        #
        # Замер на настоящем разговоре: медиана среднего удивления эпизода
        # 0.523, порог 0.55 сворачивает треть эпизодов. При усреднении по
        # всему буферу сворачивалось ноль.
        speaker_entries = [e for e in entries if e.role != "bot"] or entries
        avg_perplexity = sum(e.perplexity for e in speaker_entries) / len(speaker_entries)

        packed_context, packed_response = self._pack_episode(entries)

        # --- a) Emotional node (takes priority over the structural one) ---
        if max_emotion >= self.settings.stm_emotional_threshold:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=max_emotion,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Emotional node id=%s weight=%.3f (max_emotion=%.3f)",
                node_id, max_emotion, max_emotion,
            )
            return ConsolidationResult(
                decision="emotional_node",
                node_id=node_id,
                weight=max_emotion,
                reason=f"max_emotion={max_emotion:.3f} >= {self.settings.stm_emotional_threshold}",
            )

        # --- b) Structural node ---
        if avg_perplexity >= self.settings.stm_structural_threshold:
            # СВЁРТКА — ОТДЕЛЬНЫЙ ТИП, и она НЕ УЧАСТВУЕТ в обычном поиске.
            #
            # Замер: с консолидацией на равных R@1 падает с 76% до 52%.
            # Понижение силы помогает монотонно, но не спасает — 56% при
            # x0.5 и 60% при x0.2. Причина видна из чисел: сила входит в
            # оценку долей 0.15, а свёрнутый узел выигрывает НА
            # РЕЛЕВАНТНОСТИ. Восемь обменов содержат столько слов, что
            # совпадают почти с любым запросом; штрафом по важности
            # преимущество по смыслу не отменить.
            #
            # R@10 при этом стабильно 92% во всех вариантах: улика в
            # памяти есть всегда, ломается только порядок.
            #
            # Правильное разделение: схема достаётся ДРУГИМ ЗАПРОСОМ, а не
            # соревнуется с эпизодом за одну полку. Человека спрашивают
            # "какой антибиотик выписали" — он не перебирает "мы обсуждали
            # лекарства" как кандидата; общее знание достаётся иначе.
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=self.settings.stm_structural_weight,
                timestamp=timestamp,
                node_type="episode_summary",
            )
            # СВЁРТКА УСТУПАЕТ ПОДРОБНОСТИ. Свёрнутый эпизод — это восемь
            # обменов в одном узле: он широко совпадает почти с любым
            # запросом и занимает первое место, вытесняя настоящую улику.
            # Замер: R@1 падает с 76% до 42% при буфере 16 и до 52% при
            # буфере 4, а R@5 и R@10 почти не страдают — улика остаётся в
            # выдаче, её просто оттесняют сверху.
            #
            # Понижение силы делает абстракцию тем, чем она является у
            # людей: схемой, которая всплывает, КОГДА ПОДРОБНОСТЬ УЖЕ
            # НЕДОСТУПНА. "Мы обсуждали лекарства" вспоминается тогда,
            # когда "азитромицин" уже нет.
            #
            # Сила при этом может отрасти от пользы: схема, к которой
            # постоянно обращаются, законно становится сильной.
            if node_id is not None and self.settings.consolidated_strength_factor < 1.0:
                row = self.db.get_node(node_id)
                if row is not None:
                    base = row["strength"] if row["strength"] is not None else row["weight"]
                    self.db.add_strength(
                        node_id,
                        base * (self.settings.consolidated_strength_factor - 1.0),
                        self.settings.strength_max,
                    )
            logger.info(
                "[CONSOLIDATION] Structural node id=%s weight=%.3f (avg_perplexity=%.3f)",
                node_id, self.settings.stm_structural_weight, avg_perplexity,
            )
            return ConsolidationResult(
                decision="structural_node",
                node_id=node_id,
                weight=self.settings.stm_structural_weight,
                reason=f"avg_perplexity={avg_perplexity:.3f} >= {self.settings.stm_structural_threshold}",
            )

        # --- c) Routine noise — discarded without touching the database ---
        logger.info(
            "[STM FLUSH] Routine noise discarded (max_emotion=%.3f, avg_perplexity=%.3f, %d entries)",
            max_emotion, avg_perplexity, len(entries),
        )
        return ConsolidationResult(
            decision="routine_noise",
            reason=f"max_emotion={max_emotion:.3f}, avg_surprise={avg_perplexity:.3f} — below both thresholds",
        )

    @staticmethod
    def _pack_episode(entries: List["STMEntry"]) -> "tuple[str, str]":
        """
        Packs a list of STMEntry into a (context, response) pair for
        storage as one long-term node. User turns are joined into context
        and bot turns into response, preserving their order.
        """
        user_lines = [e.text.strip() for e in entries if e.role == "user"]
        bot_lines = [e.text.strip() for e in entries if e.role != "user"]

        context = " | ".join(user_lines) if user_lines else "(no user turns)"
        response = " | ".join(bot_lines) if bot_lines else "(no bot replies)"

        return context, response
