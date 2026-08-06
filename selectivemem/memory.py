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
from selectivemem.prefrontal import WorkingMemory
from selectivemem.context import TemporalContext
from selectivemem.interoception import Interoception, InternalState


@dataclass
class SleepReport:
    """
    Что сделал сон, по стадиям — в том же порядке, в каком они идут.

    replayed_nodes    — сколько сильнейших следов переиграно (реактивация);
    edges_downscaled  — сколько связей понижено гомеостатически;
    edges_pruned      — сколько связей срезано;
    orphan_nodes_pruned — сколько узлов осталось без связей и убрано;
    clusters_consolidated — сколько кластеров свёрнуто в абстракцию.
    """
    edges_pruned: int
    orphan_nodes_pruned: int
    clusters_consolidated: int
    replayed_nodes: int = 0
    edges_downscaled: int = 0

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
        self._recently_stored: List[int] = []
        # ВРЕМЕННОЙ КОНТЕКСТ. Не сохраняется намеренно: это состояние, а
        # не знание. Проснувшись, организм не помнит, о чём шла речь
        # неделю назад, — как человек, вернувшийся к прерванному разговору.
        self.context = TemporalContext(
            half_life=self.settings.context_half_life)
        # Недавние записи вместе с фоном, на котором они сделаны. Связь
        # ставится по СХОЖЕСТИ ФОНОВ, а не по номеру в очереди.
        self._recent_contexts: List[tuple] = []
        # Кратковременная память фасада. Была написана в пакете и не
        # использовалась им: консолидацию звала только витрина
        # (core/brain_session.py), поэтому библиотечный пользователь её не
        # получал вовсе — то же расхождение витрины с пакетом, что было с
        # ассоциациями. Найдено tools/check_liveness.py.
        self.stm = WorkingMemory(settings=self.settings)
        # ВНУТРЕННЯЯ СРЕДА. Не хранится в базе намеренно: это состояние,
        # а не знание. Проснувшись, организм не вспоминает вчерашнюю
        # тесноту — он заново её чувствует, если она никуда не делась.
        self.interoception = Interoception(settings=self.settings)
        # Последняя выдача: (запрос, id узлов, время). Нужна, чтобы
        # СЛЕДУЮЩИЙ вопрос мог её оценить — см. _judge_previous_recall.
        self._last_recall: Optional[tuple] = None
        # Помеченные слабые следы: (текст, ответ, плотность, время).
        # Ждут сильного события рядом по времени — см. _tag/_capture_tags.
        self._tags: List[tuple] = []

    # ----------------------------------------------------------------------
    # 1. Observing
    # ----------------------------------------------------------------------

    def observe(
        self,
        text: str,
        response: str = "",
        emotion: Optional[float] = None,
        load: Optional[float] = None,
        fills_gap: bool = False,
        timestamp: Optional[float] = None,
        action_type: str = "observation",
    ) -> Observation:
        """
        Show the memory an event and let it decide whether to keep it.

        text     — what the user said (or what happened);
        response — what the system replied, if it did;
        emotion  — how charged the event is, [0, 1]. Pass it when the
                   application knows more about the event than the core
                   can: a model read the text, a human rated it. LEFT OUT
                   (or None), the organism answers for itself out of its
                   own internal state — see interoception.py. Explicitly
                   passed 0.0 still means 0.0: "not charged" and "did not
                   say" are different things, and the signature tells them
                   apart;
        fills_gap — this event answers something the application ASKED
                   FOR. Not a hint but a fact only the caller has: the
                   assistant asked "what is your dog called" because it
                   did not know, and "Levi" is the answer.

                   WHY THE LIBRARY CANNOT WORK THIS OUT ITSELF, measured
                   rather than assumed. A gap was built inside: recall
                   returns little or nothing -> the next event gets a
                   boost. It never fired once in thirty-six turns. The
                   search for "what is the dog called" returned 0.776 for
                   "I have a dog" — memory FOUND something, confidently
                   and wrong. It measures similarity, not whether a need
                   was met, and no confidence threshold separates the two.

                   Novelty cannot stand in for it either: "Levi" is one
                   familiar word, surprise 0.06, and nothing in the
                   conversation matters more.

        load     — current overload, [0, 1]: raises the write threshold.
                   LEFT OUT, the organism reports its own strain — how
                   crowded its store is and how little of the world it
                   currently understands;

        THE ORDER MATTERS and mirrors the living one: the organism is
        surprised by the input first and learns it only afterwards. The
        other way round would mean being surprised by what it memorised a
        moment ago — that is, never being surprised at all.
        """
        ts = timestamp if timestamp is not None else self.clock()

        # ВРЕМЯ ИДЁТ — даже когда никто ничего не говорил. Пауза ослабляет
        # фон: через минуту разговор тот же, через неделю почти новый.
        # Часов у библиотеки нет, дрейф считается от разницы переданных
        # отметок.
        self.context.advance(ts)

        surprise = self.graph.compute_surprise(text).total

        # ЗНАЧИМОСТЬ СОБЫТИЯ. Явно переданная всегда главнее: приложение
        # знает про событие то, чего ядро знать не может.
        #
        # Состояние берётся ТО, В КОТОРОМ ОРГАНИЗМ ВСТРЕТИЛ СОБЫТИЕ, — до
        # того, как оно переварено. Иначе он реагировал бы на последствия
        # того, что ещё не случилось.
        # Два канала расходятся по двум входам гейта: срочность запись
        # облегчает, напряжение — затрудняет. Свести их в один было бы
        # неверно: организм, которому тесно, должен писать МЕНЬШЕ, а
        # организм, чья модель мира разъехалась, — БОЛЬШЕ.
        # ЗАПОЛНЕНИЕ ЗАЯВЛЕННОЙ НУЖДЫ ИДЁТ ОТДЕЛЬНЫМ ВХОДОМ, а не
        # множителем. Плотность равна новизна * (1 + значимость) / 2, то
        # есть при новизне 0.33 не превысит 0.33 ни при какой значимости
        # ниже единицы, — а порог под напряжением поднимается к 0.40.
        # Проверено: со значимостью 0.8 кличка по-прежнему терялась.
        if fills_gap and emotion is None:
            emotion = self.settings.gap_fill_significance

        inner = self.interoception.state
        if self.settings.intrinsic_emotion:
            if emotion is None:
                emotion = inner.urgency
            if load is None:
                load = inner.strain
        if emotion is None:
            emotion = 0.0
        if load is None:
            load = 0.0

        decision = self.gate.evaluate(emotion=emotion, surprise=surprise, load=load)

        # A contradiction is a reason to store in its own right. A
        # correction is unsurprising by nature ("no, her name is Mia") and
        # never reaches the density threshold; without this branch,
        # corrections never made it into memory at all — that was a live
        # bug, not a hypothesis.
        # СОБЫТИЕ ВПИТЫВАЕТСЯ В ФОН ДО ЗАПИСИ. След привязывается к
        # состоянию контекста, ВКЛЮЧАЮЩЕМУ его самого, — иначе у первой
        # записи фона нет вовсе, и связаться с ней невозможно ничем.
        # Проверено: первый узел уходил в базу с пустым контекстом, и
        # временных связей не возникало ни одной.
        self.context.absorb(self.graph._encode(text))

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
                    (self.graph.gate.node.get(n.id)["weight"] for n in superseded),
                    default=0.0,
                )
                weight = max(weight, inherited)
            node_id = self.graph.save_connection(
                context=text, response=response, weight=weight, timestamp=ts,
            )
            self._associate_with_recalled(node_id, ts)
            self._associate_with_recent(node_id, ts)
            # ЗАХВАТ: сильное событие производит "белки" на всю клетку, и
            # помеченные слабые следы рядом по времени их перехватывают.
            self._capture_tags(decision.density, ts)
        else:
            # МЕТКА: гейт не пропустил, но след не исчезает совсем — он
            # лабилен и ждёт своего часа в пределах окна.
            self._tag(text, response, decision.density, ts)

        self._consolidation = None
        if self.settings.consolidate_from_stm:
            # Удивление передаётся ОБЯЗАТЕЛЬНО. У консолидации два входа —
            # эмоция и удивление, — и без второго она глохнет у любого, кто
            # не передаёт эмоцию, то есть у обычного библиотечного
            # пользователя: emotion по умолчанию 0.0. Замер до правки:
            # решение "routine_noise" при любой эмоции, причина
            # "avg_surprise=0.000 — below both thresholds".
            self.stm.add_message(
                "user", text, emotion_score=emotion, perplexity=surprise, timestamp=ts,
            )
            if response:
                self.stm.add_message("bot", response, timestamp=ts)
            if self.stm.is_full():
                # Консолидация ТОЛЬКО по заполнению буфера. В витрине её
                # когда-то дёргал ещё и спайк, и это вытирало
                # кратковременную память ровно в тот момент, когда разговор
                # становился интересным: замер показывал STM пустой после
                # 39 сообщений из 40.
                self._consolidation = self.graph.consolidate_from_stm(
                    self.stm.consume_all(),
                    timestamp=ts,
                    already_captured_by_spike=node_id is not None,
                )

        # ПЕРЕВАРИВАНИЕ СОБЫТИЯ ВНУТРЕННЕЙ СРЕДОЙ. После решения о записи:
        # теснота изменилась именно этой записью, а противоречие стало
        # известно именно сейчас.
        self.interoception.sense(
            surprise=surprise,
            contradiction=bool(superseded),
            stored=self.graph.gate.episodic.count(),
            capacity=self.settings.memory_capacity,
        )

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

    def describe_setup(self) -> str:
        """
        В какой конфигурации работает ЭТОТ экземпляр памяти.

        Семантика необязательна и отваливается тихо, а разница между
        режимами — троекратная. Строка отвечает на вопрос «почему поиск
        стал хуже» до того, как его зададут.

            >>> print(memory.describe_setup())
            смысл: potion-base-8M | значимость: своя | пишет: при плотности >= 0.25
        """
        from selectivemem import embeddings

        if self.graph.encoder is not None:
            meaning = "кодировщик приложения"
        elif self.graph.perception is not None:
            grown = self.graph.perception
            meaning = (f"выращенное восприятие "
                       f"(показов {grown.exposures}, словарь {grown.vocabulary})")
        else:
            meaning = embeddings.describe()

        significance = ("своя, из внутренней среды" if self.settings.intrinsic_emotion
                        else "только переданная приложением")
        capacity = (f"{self.settings.memory_capacity}" if self.settings.memory_capacity
                    else "без предела")
        return (f"смысл: {meaning} | значимость: {significance} | "
                f"порог записи: {self.settings.base_plasticity_threshold} | "
                f"ёмкость: {capacity}")

    def feel(self) -> InternalState:
        """
        Самочувствие организма прямо сейчас.

        Возвращает отклонения по трём нуждам плюс две оси аффекта:
        возбуждение (насколько нехорошо) и валентность (становится лучше
        или хуже). Приложение может показывать это в /status, писать в
        логи или подавать в подкрепление как собственную оценку организма:

            state = memory.feel()
            if abs(state.valence) > 0.3:
                memory.feedback(state.valence)
        """
        return self.interoception.state

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
        # СНАЧАЛА СУДИМ ПРЕДЫДУЩУЮ ВЫДАЧУ, потом ищем. Если пришёл тот же
        # вопрос, значит прошлый ответ не подошёл — и узнать это можно
        # только сейчас, до того как новый поиск затрёт след.
        self._judge_previous_recall(query, ts)
        matches = self.graph.search(
            query, top_k=top_k, timestamp=ts, with_associations=with_associations,
        )
        self._last_recall = (query, [m.id for m in matches], ts)
        self._remember_used([m.id for m in matches])
        # То, что было активно сейчас, свяжется со следующей записью — см.
        # associate_recalled_limit в observe.
        for match in matches:
            if match.id not in self._recently_recalled:
                self._recently_recalled.append(match.id)
        return matches

    def summaries(self, limit: int = 5) -> List[str]:
        """
        Свёрнутые эпизоды — «о чём вообще был разговор».

        ОТДЕЛЬНЫЙ ВЫЗОВ, а не часть recall, и это измерено. Со свёртками в
        общей выдаче R@1 падает с 76% до 52%: восемь обменов в одном узле
        содержат столько слов, что совпадают почти с любым запросом и
        занимают первое место. Понижение их силы помогает монотонно, но не
        спасает (56% при x0.5, 60% при x0.2) — сила входит в оценку долей
        0.15, а выигрывают свёртки НА РЕЛЕВАНТНОСТИ, и штрафом по важности
        преимущество по смыслу не отменить.

        R@10 при этом остаётся 92% в любом варианте: подробность из памяти
        никуда не девается, ломается только порядок.

        Так и у людей: схема и эпизод достаются РАЗНЫМИ ходами. На вопрос
        "какой антибиотик выписали" человек не перебирает "мы обсуждали
        лекарства" как кандидата.
        """
        rows = self.graph.gate.semantic.schemas()
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return [row["context"] for row in rows[:limit]]

    def sleep(self, timestamp: Optional[float] = None, summarise=None) -> "SleepReport":
        """
        Housekeeping the memory cannot do while it is being used: pruning
        weak links and folding dense clusters into one abstract memory.

        CALL IT EXPLICITLY, like forget(). The library has no scheduler and
        does not start threads; the application decides when it is idle —
        after a session, on a timer, overnight.

        Why this method exists at all. Pruning, hub clusters and abstract
        nodes are all written in this package, and NOTHING in the package
        called them: the only caller was the showcase's sleep cycle
        (core/sleep_cycle.py). So a library user got none of it, though it
        is described as part of how the memory works. That is the same gap
        that association and consolidation turned out to have, and it was
        found the same way — tools/check_liveness.py counts how often each
        mechanism fires, and these three showed zero.

        summarise(cluster) -> (context, response) turns a cluster into the
        text of the abstract memory. Without it the hub's own text is used:
        the library has no language model and will not invent one, but an
        application that has one can pass it in.
        """
        ts = timestamp if timestamp is not None else self.clock()

        # ПОРЯДОК СТАДИЙ ПОВТОРЯЕТ НОЧНОЙ, и он не переставляем.
        #
        # 1. Реактивация: сильнейшие следы переигрываются вместе, связи
        #    между ними крепнут. Раньше этой стадии не было вовсе, а без
        #    неё сон только чистит — тогда как переносит память в кору
        #    именно она.
        # 2. Гомеостатическое понижение: все связи слабеют пропорционально.
        #    ПОСЛЕ реплея, иначе оно погасило бы то, что он только что
        #    построил.
        # 3. Подрезка: под нож идёт не выдержавшее сжатия.
        # 4. Свёртка кластера в абстракцию — на структуре, которую
        #    подготовил реплей.
        # 5. Пересмотр выведенного корой: что оказалось не темой, а
        #    оборотом речи. ПОСЛЕДНЕЙ, потому что судит по накопленному
        #    языку, а он к этому моменту полнее всего.
        replayed = self.graph.replay(timestamp=ts)
        scaled = self.graph.downscale_edges()
        report = self.graph.run_synaptic_pruning()
        clusters = self.graph.find_hub_clusters(limit=1, timestamp=ts)

        abstracts = 0
        for cluster in clusters:
            if summarise is not None:
                context, response = summarise(cluster)
            else:
                # Без языковой модели свернуть смысл нечем. Берём текст
                # хаба: он и так самый связный узел кластера, а спицы
                # уходят в архив, что и есть суть консолидации.
                context, response = cluster.hub_context, cluster.hub_response
            self.graph.create_abstract_node(
                summary_context=context,
                summary_response=response,
                source_node_ids=[cluster.hub_id] + list(cluster.spoke_ids),
                timestamp=ts,
            )
            abstracts += 1

        reviewed = self.graph.review_cortex_facts()
        if reviewed:
            logger.info("[SLEEP] Снято тем, оказавшихся оборотами: %d", reviewed)

        return SleepReport(
            edges_pruned=report.edges_pruned,
            orphan_nodes_pruned=report.orphan_nodes_pruned,
            clusters_consolidated=abstracts,
            replayed_nodes=replayed,
            edges_downscaled=scaled,
        )

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
                    edge_type="association",
                )
        self._recently_recalled = []

    def _associate_with_recent(self, node_id: Optional[int], timestamp: float) -> None:
        """
        Связать новое с тем, что записано на ПОХОЖЕМ ФОНЕ.

        ЗАЧЕМ ОТДЕЛЬНО ОТ СВЯЗЫВАНИЯ ПО ПРИПОМИНАНИЮ. То опирается на
        поиск и потому глохнет ровно там, где нужнее всего — при
        перегрузке ключа: когда одну подсказку делят четыре записи,
        связь завязывается в 3 случаях из 60 вместо 23. Механизм,
        опирающийся на поиск, не может починить провал поиска.

        ЧТО ИЗМЕНИЛОСЬ. Раньше связывалось окно из двух последних записей
        — независимо от того, прошла между ними минута или месяц. Теперь
        решает схожесть ВРЕМЕННОГО КОНТЕКСТА: фон дрейфует по паузе, и
        сказанное через неделю просто не окажется на том же фоне.

        Так это и работает в живом: вспомнив эпизод, человек чаще всего
        следующим называет соседа по времени, потому что оба привязаны к
        одному состоянию медленно меняющегося контекста.

        Без кодировщика фона нет, и тогда работает прежнее окно — иначе
        механизм молчал бы у всех, кто обходится без семантики.
        """
        window = self.settings.temporal_link_window
        if node_id is None or window <= 0:
            return

        here = self.context.snapshot()
        threshold = self.settings.context_link_threshold

        if here is None or threshold <= 0.0:
            # Фона нет — связываем по очереди, как прежде.
            partners = [(other, 1.0) for other in self._recently_stored[-window:]]
        else:
            partners = []
            for other_id, other_context in self._recent_contexts[-window * 4:]:
                closeness = self.context.similarity(other_context)
                if closeness >= threshold:
                    partners.append((other_id, closeness))
            partners.sort(key=lambda pair: pair[1], reverse=True)
            partners = partners[:window]

        for source_id, closeness in reversed(partners):
            if source_id != node_id:
                self.graph.connect_nodes(
                    source_id, node_id,
                    # Чем ближе фоны, тем крепче связь: соседство во
                    # времени — величина степенная, а не да/нет.
                    weight_boost=self.settings.temporal_link_weight * closeness,
                    timestamp=timestamp,
                    edge_type="temporal",
                )

        self._recently_stored.append(node_id)
        self._recent_contexts.append((node_id, here))
        keep = max(window * 4, 8)
        if len(self._recently_stored) > keep * 2:
            del self._recently_stored[:-keep]
        if len(self._recent_contexts) > keep * 2:
            del self._recent_contexts[:-keep]

    def _tag(self, text: str, response: str, density: float, ts: float) -> None:
        """
        МЕТКА НА СЛАБОМ СЛЕДЕ (Фрей и Моррис, 1997).

        Гейт решает в момент события и необратимо: не прошло — следа нет
        никогда. В мозге не так. Слабое воздействие ставит на синапсе
        метку, белков для закрепления не хватает, и след угаснет — но
        если в ближайший час-два рядом случится сильное событие, клетка
        произведёт эти белки на всю себя, и помеченный синапс их
        перехватит.

        Так спокойно сказанное закрепляется задним числом. Наши
        предпочтения страдают от отсутствия этого больше всего: «я больше
        люблю кофе, чем чай» не несёт удивления и порога не берёт, а это
        худший тип на внешнем наборе — 20% R@1.

        Метка живёт в памяти процесса, а не в базе: след ЛАБИЛЕН по
        определению, и переживать перезапуск ему незачем.
        """
        if self.settings.tagging_window <= 0.0:
            return
        self._tags.append((text, response, density, ts))
        limit = max(1, self.settings.tagging_buffer)
        if len(self._tags) > limit:
            del self._tags[:-limit]

    def _capture_tags(self, spike_density: float, ts: float) -> int:
        """
        ЗАХВАТ: сильное событие закрепляет помеченные следы рядом по времени.

        Ждёт метка не одобрения, а ЛЮБОГО сильного события — знак не
        важен, важна интенсивность. Так и в мозге: новизна, возбуждение и
        награда открывают пластичность одинаково.

        Чем дальше метка по времени, тем меньше достаётся: доля падает
        линейно до края окна. Захваченный след получает лишь часть
        плотности спайка (tagging_capture_factor) — закрепляется слабое
        воспоминание, а не копия сильного.

        Возвращает число закреплённых следов.
        """
        window = self.settings.tagging_window
        if window <= 0.0 or not self._tags:
            return 0

        captured = 0
        remaining = []
        for text, response, density, tag_ts in self._tags:
            age = ts - tag_ts
            if age < 0 or age > window:
                continue  # вне окна — след угас, и это нормально
            proximity = 1.0 - (age / window)
            weight = max(
                density,
                spike_density * self.settings.tagging_capture_factor * proximity,
            )
            node_id = self.graph.save_connection(
                context=text, response=response, weight=weight, timestamp=tag_ts,
            )
            if node_id is not None:
                captured += 1
        self._tags = remaining

        if captured:
            logger.info(
                "[TAG CAPTURE] Spike (density %.3f) consolidated %d tagged traces",
                spike_density, captured,
            )
        return captured

    def _judge_previous_recall(self, query: str, ts: float) -> int:
        """
        ПОДКРЕПЛЕНИЕ ПО ПОСЛЕДСТВИЮ, отрицательная ветвь.

        Тот же вопрос, заданный снова, — свидетельство, что прошлый ответ
        не подошёл. Это единственный сигнал о качестве выдачи, который
        библиотека получает БЕЗ участия приложения: `feedback()` почти
        никто не зовёт, а переспрашивают все.

        ЗАЧЕМ ИМЕННО ОТРИЦАТЕЛЬНАЯ. Сила уже растёт от извлечения, но
        растёт у всего, что попало в выдачу, — у верного и неверного
        одинаково. Замер показал, к чему это ведёт: ускорение подкрепления
        ускоряло и закрепление ошибок (раздел 2.16 аудита). Недоставало не
        скорости, а провала — того самого, который у дофамина случается на
        обещанную и не полученную награду.

        Положительной ветви здесь намеренно нет. «Разговор пошёл дальше» —
        слишком слабое свидетельство: оно наступает по умолчанию, и
        награждать по нему значило бы снова растить всё подряд.

        Возвращает число наказанных узлов.
        """
        penalty = self.settings.consequence_penalty
        if penalty <= 0.0 or not self._last_recall:
            return 0
        prev_query, prev_ids, prev_ts = self._last_recall
        if not prev_ids or ts - prev_ts > self.settings.consequence_window:
            return 0
        if self._query_similarity(query, prev_query) < self.settings.consequence_repeat_threshold:
            return 0

        for node_id in prev_ids:
            self.graph.gate.node.add_strength(node_id, -penalty, self.settings.strength_max)
        logger.info(
            "[CONSEQUENCE] Question repeated (%r ~ %r) — %d nodes weakened by %.3f",
            query[:30], prev_query[:30], len(prev_ids), penalty,
        )
        self._last_recall = None
        return len(prev_ids)

    def _query_similarity(self, a: str, b: str) -> float:
        """
        Насколько два запроса — один и тот же вопрос.

        По смыслу, если кодировщик есть: «когда у меня отпуск» и «а отпуск
        когда» — один вопрос разными словами, и строковое сравнение это
        упустит. Без кодировщика остаётся строковое, и тогда механизм ловит
        только буквальные повторы. Деградация честная и объявлена.
        """
        from selectivemem import embeddings

        va, vb = self.graph._encode(a), self.graph._encode(b)
        if va is not None and vb is not None:
            return max(0.0, embeddings.cosine(va, vb))
        return self.graph._compute_fuzzy_similarity(a.strip().lower(), b.strip().lower())

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

    def remember_about_user(self, fact: str, timestamp: Optional[float] = None) -> None:
        """
        Записать ВЫВЕДЕННЫЙ факт о человеке: предпочтение, привычку,
        ограничение.

        ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ observe. `observe` показывает памяти
        СОБЫТИЕ, и она сама решает, стоит ли его хранить. Здесь приходит
        не событие, а вывод из него — «предпочитает материалы на
        французском», — и решать тут нечего: если приложение сумело это
        вывести, значит оно того стоит.

        ЛОЖИТСЯ В КОРУ, А НЕ В ЭПИЗОДЫ, и это не мелочь. Корковые факты не
        вытесняются по ёмкости: аллергия переживёт тесноту, а разговор о
        погоде — нет. Для персонализации это и нужно.

        ЗАЧЕМ ВООБЩЕ. Предпочтения — худший тип вопросов у памяти (R@1
        17%), и разбор показал, что дело не в весах: улика выглядит как
        ЗАПРОС («ищу подкасты на французском»), а спрашивают потом иначе
        («посоветуй что послушать»). Общих слов нет, связывает знание о
        мире. Превратить запрос в утверждение способна только модель — и
        она стоит в цикле приложения, а не здесь.

        Проверено и отвергнуто по дороге: надбавка за речь о себе (в
        разговоре с ассистентом о себе КАЖДАЯ реплика), кодировщик
        вчетверо крупнее (разделяет хуже), отбор профиля по тем же
        признакам (втрое хуже обычного поиска).
        """
        text = (fact or "").strip()
        if not text:
            return
        ts = timestamp if timestamp is not None else self.clock()
        self.graph.gate.semantic.record_fact(
            theme=text, text=text, meaning=text,
            strength_step=self.settings.cortex_fact_strength,
            cap=self.settings.strength_max,
            timestamp=ts,
        )
        logger.info("[О ПОЛЬЗОВАТЕЛЕ] %r", text[:60])

    def profile(self, limit: int = 10) -> List[MemoryMatch]:
        """
        Что память знает О ЧЕЛОВЕКЕ — независимо от того, о чём спросили.

        ЗАЧЕМ ОТДЕЛЬНО ОТ recall. `recall` ищет ПО ЗАПРОСУ, и это верно для
        вопроса «когда у меня рейс». Но ассистенту нужно и другое: держать
        при себе то, что о человеке известно ВСЕГДА — предпочтения,
        ограничения, обстоятельства, — и подмешивать это в каждый ответ, а
        не только когда спросили впрямую.

        Живой разговор показал, чем оборачивается отсутствие такого вызова.
        На «что ты обо мне знаешь» ассистент назвал предпочтения (запрос
        случайно совпал с записями), а на «может ещё что-то?» ответил, что
        больше ничего нет — при двадцати записях в памяти. Запрос не
        совпал ни с чем, и память промолчала, хотя знала.

        Отбор идёт по РЕЧИ О СЕБЕ («я», «мне», «мой», «i», «my»). Это
        грубый признак, и он не отличает «я люблю кофе» от «я вчера
        заходил» — зато не требует ни модели, ни разметки. Эффект
        самореференции в психологии памяти именно про это: сказанное о
        себе запоминается лучше прочего.

        Порядок — по накопленной силе: наверху то, что подтверждалось
        повторениями и припоминанием, а не то, что сказано последним.
        """
        # ВЫВЕДЕННЫЕ ФАКТЫ ИДУТ ПЕРВЫМИ, и лишь потом догадки по речи.
        #
        # То, что приложение вывело явно («предпочитает французское»),
        # стоит дороже того, что память угадала по слову «мой». Замер
        # показал, чего стоит одна догадка: отбор профиля по речи о себе
        # находит предпочтение в 3 случаях из 30 против 15 у обычного
        # поиска — потому что в разговоре с ассистентом о себе говорит
        # КАЖДАЯ реплика, и признак не различает ничего.
        derived = [
            MemoryMatch(
                id=row["id"], context=row["text"], response="",
                weight=row["weight"], similarity=1.0,
                created_at=row["created_at"], last_accessed=row["last_accessed"],
            )
            for row in self.graph.gate.semantic.facts(limit=limit)
        ]

        rows = self.graph.gate.episodic.searchable()
        pattern = self.graph._SELF_REFERENCE
        about_user = [
            row for row in rows
            if not row["is_meta"]
            and pattern.search(row["context"] or "")
            # ВОПРОСЫ — НЕ ФАКТЫ О ЧЕЛОВЕКЕ. «Что мне заказать на ужин»
            # содержит «мне» и проходило отбор наравне с «я не ем мясо»,
            # то есть профиль засорялся тем, что человек СПРАШИВАЛ, а не
            # тем, что он о себе сообщил. Признак грубый, но верный:
            # утверждение о себе вопросительным знаком не кончается.
            and not (row["context"] or "").rstrip().endswith("?")
        ]
        about_user.sort(
            key=lambda row: (row["strength"] if row["strength"] is not None
                             else row["weight"]),
            reverse=True,
        )
        guessed = [
            MemoryMatch(
                id=row["id"], context=row["context"], response=row["response"],
                weight=row["weight"], similarity=1.0,
                created_at=row["created_at"], last_accessed=row["last_accessed"],
            )
            for row in about_user
        ]
        return (derived + guessed)[:limit]

    def profile_text(self, limit: int = 10) -> str:
        """Профиль готовой строкой для промпта. Пусто — законный ответ."""
        return "\n".join(f"- {m.context}" for m in self.profile(limit))

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
            episodes=self.graph.gate.episodic.count(),
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
