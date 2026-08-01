"""
================================================================================
 ENGRAM.PY — Публичный фасад динамической памяти
================================================================================
Пять действий, которых достаточно для работы:

    memory = Engram("brain.db")

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
организм помнит 97% отмеченного пользователем как важное против 43%
обычного — разрыв +53 п.п. При этом на РАВНОМЕРНЫХ вопросах он не лучше
случайной выборки (86.4% против 86.4%), и это тоже измерено. Инструмент
для "удержать важное", а не для "не потерять ничего".

ВРЕМЯ ПРИХОДИТ ИЗВНЕ. Все методы принимают timestamp, а часы по
умолчанию — time.time. Приложение вправе подставить свои: субъективное
время организма, ускоренное время демонстрации, замороженное время
теста. Забывание считается по этой шкале, поэтому она обязана быть одна
и монотонная.

ЯЗЫК. Ядро не знает ни одного слова оценки: valence в feedback() —
число, которое приложение получает откуда угодно (кнопка, эмодзи,
классификатор, разбор реплики). Семантический поиск опирается на
навешиваемую модель эмбеддингов и деградирует до строкового сходства,
если её нет.
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from engram.database import Database
from engram.graph_memory import MemoryGraph, MemoryMatch
from engram.plasticity import PlasticityDecision, PlasticityGate
from engram.reinforcement import ReinforcementLoop, ReinforcementOutcome
from engram.settings import MemorySettings

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


class Engram:
    """
    Динамическая память: пишет избирательно, забывает со временем,
    укрепляет то, что пригодилось.

        memory = Engram("brain.db")
        memory.observe("у меня есть кот", "расскажи про него")
        memory.recall("кот")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        settings: Optional[MemorySettings] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.settings = settings or MemorySettings()
        self.clock = clock or time.time
        self.graph = MemoryGraph(
            db=Database(db_path=db_path, settings=self.settings),
            settings=self.settings,
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
        return self.graph.search(
            query, top_k=top_k, timestamp=ts, with_associations=with_associations,
        )

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

    def __enter__(self) -> "Engram":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
