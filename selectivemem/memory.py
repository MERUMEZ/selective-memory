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
================================================================================
 MEMORY.PY — Публичный фасад динамической памяти
================================================================================
Пять действий, которых достаточно для работы:

    memory = Memory("brain.db")

    obs = memory.observe("меня зовут Паша", "приятно познакомиться")
    memory.feedback(+1.0)                      # "молодец" — закрепить
    memory.recall("как меня зовут")            # -> [MemoryMatch]
    memory.context_for("как меня зовут")       # -> текст для промпта
    memory.stats()                             # -> что происходит внутри

Под фасадом — MemoryGraph (граф и забывание), PlasticityGate (порог
записи) и ReinforcementLoop (дофамин и ретроспективная коррекция). Все
три остаются доступны как memory.graph / memory.gate / memory.loop: фасад
покрывает обычный случай, но не запирает остальное.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ВЕКТОРНОЙ БАЗЫ. Векторная база отвечает на вопрос
"как найти нужное". Здесь решается другой: "что забыть". Запись
происходит НЕ на каждое сообщение, а когда эмоция или удивление
переваливают порог; сохранённое угасает со временем и укрепляется
употреблением. Замер на пяти сидах: после двух недель молчания
организм помнит 100% отмеченного пользователем как важное против 60%
обычного — разрыв +40 п.п. При этом на РАВНОМЕРНЫХ вопросах он лишь
вровень со случайной выборкой (92.4% против 90.8%), и это тоже
измерено. Инструмент
для "удержать важное", а не для "не потерять ничего".

ВРЕМЯ ПРИХОДИТ ИЗВНЕ. Все методы принимают timestamp, а часы по
умолчанию — time.time. Приложение вправе подставить свои: субъективное
время организма, ускоренное время демонстрации, замороженное время
теста. Забывание считается по этой шкале, поэтому она обязана быть одна
и монотонная.

ЯЗЫК ЗАДАЁТСЯ КОДИРОВЩИКОМ, а не библиотекой. Встроенный — navec,
русские статические векторы: лёгкий, без torch. Для английского или
любого другого языка передаётся своя функция:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    memory = Memory("brain.db", encoder=lambda text: model.encode(text))

Кодировщиком может быть что угодно, возвращающее последовательность
чисел или None: локальная модель, вызов API, fastText. Библиотека не
возит модель с собой и не выбирает её за вас.

Ядро не знает и ни одного слова оценки: valence в feedback() — число,
которое приложение получает откуда угодно (кнопка, эмодзи,
классификатор, разбор реплики).
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
    """Что случилось с одним наблюдением."""

    text: str
    surprise: float
    decision: PlasticityDecision
    node_id: Optional[int] = None
    superseded_ids: List[int] = field(default_factory=list)
    learned_words: int = 0  # сколько СЛОВ увидено впервые

    @property
    def written(self) -> bool:
        return self.node_id is not None

    @property
    def reason(self) -> str:
        """Почему записали или почему нет — человеческим языком."""
        if self.superseded_ids and self.node_id is not None:
            return f"противоречие с узлами {self.superseded_ids}"
        if self.node_id is not None:
            return f"спайк, плотность {self.decision.density:.3f}"
        return f"рутина, не хватило {abs(self.decision.headroom):.3f} до порога"


@dataclass
class MemoryStats:
    """Снимок состояния памяти."""

    nodes: int
    episodes: int
    vocabulary: int
    threshold: float


class Memory:
    """
    Динамическая память: пишет избирательно, забывает со временем,
    укрепляет то, что пригодилось.

        memory = Memory("brain.db")
        memory.observe("у меня есть кот", "расскажи про него")
        memory.recall("кот")
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
        # amygdala=None: контур подкрепления обращается к ней только ради
        # доверия к маркерам, а маркеры — дело приложения, не памяти.
        self.loop = ReinforcementLoop(memory=self.graph, amygdala=None, settings=self.settings)

    # ----------------------------------------------------------------------
    # 1. Наблюдение
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
        Показать памяти событие и дать ей решить, стоит ли его хранить.

        text     — что сказал пользователь (или что произошло);
        response — что ответила система, если ответ был;
        emotion  — насколько событие заряжено, [0, 1]. Откуда взять оценку,
                   решает приложение: ядро её не выводит;
        load     — текущая перегрузка, [0, 1]: поднимает порог записи.

        ПОРЯДОК ВАЖЕН и повторяет живой: сначала организм удивляется входу,
        и только потом учит его. Наоборот — значит удивляться тому, что
        уже успел выучить секунду назад, то есть не удивляться никогда.
        """
        ts = timestamp if timestamp is not None else self.clock()

        surprise = self.graph.compute_surprise(text).total
        decision = self.gate.evaluate(emotion=emotion, surprise=surprise, load=load)

        # Противоречие — самостоятельный повод записать. Поправка по своей
        # природе неудивительна ("нет, её зовут Мия"), плотности ей не
        # хватает, и без этой ветки исправления в память не попадали
        # вовсе — это был живой баг, а не гипотеза.
        superseded = self.graph.find_superseded(text, exclude_id=None)

        node_id = None
        if decision.is_spike or superseded:
            weight = decision.density
            if superseded:
                # Новая версия наследует вес того, что заменяет: это тот же
                # факт, обновлённый, и он вправе занять место предшественника.
                # Иначе запись состоялась бы, а устаревшее всё равно
                # выигрывало бы поиск.
                inherited = max(
                    (self.graph.db.get_node(n.id)["weight"] for n in superseded),
                    default=0.0,
                )
                weight = max(weight, inherited)
            node_id = self.graph.save_connection(
                context=text, response=response, weight=weight, timestamp=ts,
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

    # ----------------------------------------------------------------------
    # 2. Обратная связь
    # ----------------------------------------------------------------------

    def feedback(self, valence: float, timestamp: Optional[float] = None) -> ReinforcementOutcome:
        """
        Оценить последнее действие: valence в [-1, 1].

        Работает по ошибке предсказания, а не по самой оценке: узел,
        которого и так хвалят каждый раз, от очередной похвалы почти не
        укрепляется, а неожиданная — двигает сильно. Это правило
        Рескорлы-Вагнера, и оно же не даёт весам разъехаться в бесконечность.
        """
        ts = timestamp if timestamp is not None else self.clock()
        return self.loop.apply(valence=valence, timestamp=ts)

    # ----------------------------------------------------------------------
    # 3. Вспоминание
    # ----------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 3,
        timestamp: Optional[float] = None,
        with_associations: bool = True,
    ) -> List[MemoryMatch]:
        """
        Найти подходящее. Обращение к узлу — не бесплатное чтение: оно
        обновляет его стабильность, то есть вспоминание само по себе
        сопротивляется забыванию (эффект интервального повторения).
        """
        ts = timestamp if timestamp is not None else self.clock()
        matches = self.graph.search(
            query, top_k=top_k, timestamp=ts, with_associations=with_associations,
        )
        self._remember_used([m.id for m in matches])
        return matches

    def _remember_used(self, node_ids: List[int]) -> None:
        """
        Отмечает узлы как ЗАДЕЙСТВОВАННЫЕ в текущем действии, чтобы
        следующий feedback() достался и им.

        Без этого похвала доставалась только что ЗАПИСАННОМУ узлу — а у
        ассистента похвала обычно следует за хорошим ответом, построенным
        на ВСПОМНЕННОМ. То есть главный случай ("ты правильно вспомнил
        мою аллергию — молодец") не подкреплял вообще ничего, и
        ожидание награды навсегда застревало на величине одной первой
        похвалы. Замер: восемь похвал подряд давали то же ожидание 0.300,
        что и одна.

        Узлы НАКАПЛИВАЮТСЯ в пределах хода: приложение вправе позвать
        recall несколько раз перед ответом, и оценка относится ко всему
        ответу целиком.
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
        То же вспоминание, но готовым куском текста для промпта.

        Пустая строка — законный ответ и означает "мне нечего добавить".
        Вызывающий обязан это уметь: память, которая всегда что-то
        подмешивает, подмешивает шум.
        """
        matches = self.recall(query, top_k=top_k, timestamp=timestamp)
        if not matches:
            return ""
        return "\n".join(
            f"- {m.context}" + (f" -> {m.response}" if m.response else "")
            for m in matches
        )

    # ----------------------------------------------------------------------
    # 4. Течение времени
    # ----------------------------------------------------------------------

    def forget(self, now: Optional[float] = None) -> int:
        """
        Провести забывание на текущий момент. Возвращает число забытых узлов.

        Вызывать ЯВНО и регулярно — забывание не происходит само по себе
        между обращениями. Раз в сообщение достаточно: угасание считается
        от времени, а не от числа вызовов.
        """
        return self.graph.apply_decay(now=now if now is not None else self.clock())

    # ----------------------------------------------------------------------
    # 5. Состояние
    # ----------------------------------------------------------------------

    def stats(self) -> MemoryStats:
        """Снимок для дашборда, /status и отладки."""
        return MemoryStats(
            nodes=self.graph.count_nodes(),
            episodes=self.graph.db.count_nodes_by_type("episodic"),
            vocabulary=self.graph.get_vocabulary_size(),
            threshold=self.gate.base_threshold,
        )

    def close(self) -> None:
        self.graph.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
