"""
================================================================================
 REINFORCEMENT.PY — Контур подкрепления: важность из реакции пользователя
================================================================================
ЯДРО, а не персона. Это тот самый механизм, ради которого проект вообще
имеет ценность: важность воспоминания НЕ ОБЪЯВЛЯЕТСЯ при записи и не
определяется LLM-судьёй — она зарабатывается тем, как человек
отреагировал.

Измерено (tools/compare_retention.py, 5 сидов, 14 суток молчания):
    похвалённое  97%   против  обычного 53%
    у наивных хранилищ разрыв около нуля — они про реакцию не знают

Вынесено из core/cortex.py, где 300 строк контура подкрепления были
перемешаны с генерацией речи, речевыми стадиями и сборкой промптов. Для
переиспользования как библиотеки памяти это разделение обязательно:
подкрепление работает с узлами и не должно ничего знать ни про лепет, ни
про настроение, ни про LLM.

ЧТО ЗДЕСЬ ЕСТЬ:
    - дофаминовый сигнал: ошибка предсказания награды (Рескорла-Вагнер),
      из-за которой привычная похвала перестаёт действовать;
    - применение эффекта к узлам: усиление/штраф, масштабированные
      неожиданностью;
    - ретроспективная коррекция: если пользователь сам себя опроверг
      (сарказм, "нет, стой, неправильно"), прежний эффект откатывается, а
      доверие к сработавшим маркерам понижается.

ЧЕГО ЗДЕСЬ НЕТ И БЫТЬ НЕ ДОЛЖНО: настроения, эхолалии, речевых стадий,
промптов. Вызывающий код получает ReinforcementOutcome и сам решает, какие
последствия это имеет для его персонажа (см. Cortex.apply_feedback).
================================================================================
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from memory.settings import MemorySettings
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActionTrace:
    """
    След связки Вход -> Действие, ожидающий оценки на СЛЕДУЮЩЕМ шаге.

    node_id   — единичный узел (использовался ли конкретный фрагмент памяти).
    node_ids  — несколько узлов (лепет задействует набор слогов, ответ на
                однословной стадии — набор слов). None, если действие не
                опиралось на память вообще.
    """
    user_input: str
    bot_output: str
    node_id: Optional[int]
    action_type: str
    node_ids: Optional[List[int]] = None


@dataclass
class FeedbackHistoryEntry:
    """Один уже применённый эффект — окно для ретроспективной коррекции."""
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
    """Результат проверки на отложенное опровержение прошлой оценки."""
    triggered: bool
    reversed_entry: Optional[FeedbackHistoryEntry] = None
    reversal_delta: float = 0.0
    penalized_markers: List[str] = field(default_factory=list)


@dataclass
class ReinforcementOutcome:
    """
    Что подкрепление сделало с ПАМЯТЬЮ. Всё, что касается персонажа
    (настроение, склонность к эхолалии), вызывающий код решает сам по
    этим данным — ядру такие понятия неизвестны.

    congruence — ошибка предсказания награды. Именно она, а не сырая
    валентность, годится как вход для эмоции: привычная похвала не должна
    радовать.
    """
    effect: str  # "rewarded" | "penalized" | "neutral" | "no_trace"
    valence: float
    congruence: float
    applied_delta: float
    trace: Optional[ActionTrace]
    retrospective: Optional[RetrospectiveCorrectionResult] = None


class ReinforcementLoop:
    """
    Превращает реакцию человека в изменение памяти.

    Использование:
        loop = ReinforcementLoop(memory=graph, amygdala=amygdala)
        loop.record_action(user_input, bot_output, node_id=..., action_type=...)
        outcome = loop.apply(valence=0.8, timestamp=brain_time)
    """

    def __init__(self, memory, amygdala, settings: Optional[MemorySettings] = None):
        self.memory = memory
        # По умолчанию берём настройки самой памяти: контур подкрепления и
        # граф обязаны жить по одним и тем же константам.
        self.settings = settings or getattr(memory, "settings", None) or MemorySettings()
        # Нужна ТА ЖЕ инстанция амигдалы, что детектирует маркеры: штрафы и
        # реабилитация доверия должны влиять на последующие разборы реплик.
        self.amygdala = amygdala
        self.last_action_trace: Optional[ActionTrace] = None
        self.feedback_history: "deque[FeedbackHistoryEntry]" = deque(
            maxlen=self.settings.retrospective_window_size
        )

    # ----------------------------------------------------------------------
    # Фиксация действия
    # ----------------------------------------------------------------------

    def record_action(
        self,
        user_input: str,
        bot_output: str,
        node_id: Optional[int],
        action_type: str,
        node_ids: Optional[List[int]] = None,
    ) -> ActionTrace:
        """Запоминает связку, чтобы оценить её реакцией на следующем шаге."""
        self.last_action_trace = ActionTrace(
            user_input=user_input,
            bot_output=bot_output,
            node_id=node_id,
            action_type=action_type,
            node_ids=list(node_ids) if node_ids else None,
        )
        return self.last_action_trace

    # ----------------------------------------------------------------------
    # Применение оценки
    # ----------------------------------------------------------------------

    def apply(
        self,
        valence: float,
        timestamp: Optional[float] = None,
        matched_markers: Optional[List[str]] = None,
    ) -> ReinforcementOutcome:
        """
        Применяет реакцию пользователя к предыдущему действию.

        Порядок важен: сначала ретроспективная коррекция (не опровергает ли
        эта реакция уже применённую оценку), затем дофаминовый сигнал (он
        задаёт ТЕМП закрепления), и только потом сам эффект на узлах.
        """
        trace = self.last_action_trace
        matched_markers = list(matched_markers or [])

        if valence == 0.0:
            return ReinforcementOutcome("neutral", valence, 0.0, 0.0, None)

        if trace is None:
            logger.debug("[REWARD] valence=%.2f, но действия для оценки нет", valence)
            return ReinforcementOutcome("no_trace", valence, valence, 0.0, None)

        retrospective = self._check_retrospective_correction(valence, timestamp)

        # --- ДОФАМИН: ошибка предсказания награды ---
        # Считается ДО применения эффектов, потому что именно она, а не сама
        # валентность, задаёт темп закрепления: неожиданная похвала
        # закрепляет сильно, полностью предсказанная — почти никак.
        reward_nodes = (
            [trace.node_id] if trace.node_id is not None else list(trace.node_ids or [])
        )
        signals = [
            s for s in (
                self.memory.apply_reward(node_id, valence, timestamp=timestamp)
                for node_id in reward_nodes
            ) if s is not None
        ]
        # Один множитель на всё действие: сила самого неожиданного из
        # задействованных узлов. Действие оценивается целиком.
        learning_scale = (
            max(self.memory.learning_scale(s.prediction_error) for s in signals)
            if signals else 1.0
        )
        # Если действие не опиралось ни на один узел, ожидать было нечему —
        # тогда неожиданностью считается сама оценка.
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

    def _apply_effect(self, trace, valence, learning_scale, timestamp):
        """Усиливает или штрафует задействованные узлы."""
        if valence > 0:
            boost = valence * self.settings.reward_positive_boost * learning_scale

            if trace.node_id is not None:
                self.memory.reinforce_node(trace.node_id, boost=boost, timestamp=timestamp)
                # Продвигаем метку доступа вперёд: узел выглядит свежее, чем
                # есть, и будет угасать медленнее.
                if timestamp is not None:
                    self.memory.touch_node(
                        trace.node_id,
                        timestamp=timestamp + self.settings.reward_positive_freshness_bonus,
                    )
                logger.info("[REWARD] +%.2f -> узел %s", valence, trace.node_id)
                return "rewarded", boost

            if trace.node_ids:
                # Удачная комбинация усиливается целиком, а связи МЕЖДУ её
                # частями укрепляются: так из повторяющегося удачного набора
                # выкристаллизовывается устойчивая связка.
                for node_id in trace.node_ids:
                    self.memory.reinforce_node(node_id, boost=boost, timestamp=timestamp)
                self.memory.reinforce_coactivation(
                    trace.node_ids, weight_boost=boost, timestamp=timestamp
                )
                logger.info("[REWARD] +%.2f -> узлы %s", valence, trace.node_ids)
                return "rewarded", boost

            return "neutral", 0.0

        penalty = abs(valence) * self.settings.reward_negative_penalty * learning_scale

        if trace.node_id is not None:
            self.memory.penalize_node(trace.node_id, penalty=penalty, timestamp=timestamp)
            logger.info("[REWARD] %.2f -> штраф узлу %s", valence, trace.node_id)
            return "penalized", -penalty

        if trace.node_ids:
            for node_id in trace.node_ids:
                self.memory.penalize_node(node_id, penalty=penalty, timestamp=timestamp)
            logger.info("[REWARD] %.2f -> штраф узлам %s", valence, trace.node_ids)
            return "penalized", -penalty

        return "neutral", 0.0

    # ----------------------------------------------------------------------
    # Ретроспективная коррекция
    # ----------------------------------------------------------------------

    def _check_retrospective_correction(
        self, valence: float, timestamp: Optional[float]
    ) -> RetrospectiveCorrectionResult:
        """
        Ищет в окне истории ещё не откатанную оценку ПРОТИВОПОЛОЖНОГО знака.
        Если находит — трактует текущую реакцию как отложенное опровержение
        (сарказм, "нет, стой, неправильно"): откатывает прежний эффект
        усиленной коррекцией и понижает доверие к маркерам, которые к нему
        привели.
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

            # Подтверждённая ложная оценка — куда более сильный обучающий
            # сигнал, чем обычный однократный фидбэк, отсюда множитель.
            reversal = -entry.applied_delta * self.settings.retrospective_reversal_strength
            if reversal > 0:
                self.memory.reinforce_node(entry.node_id, boost=reversal, timestamp=timestamp)
            elif reversal < 0:
                self.memory.penalize_node(entry.node_id, penalty=abs(reversal), timestamp=timestamp)

            entry.reversed = True
            if entry.matched_markers:
                self.amygdala.penalize_markers(entry.matched_markers)

            logger.info(
                "[RETROSPECTIVE] Опровержение: было %.2f (t=%.1f), стало %.2f (t=%.1f) "
                "-> узел %s, откат %.3f, маркеры %s",
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
        Кладёт запись в окно. Если вытесняется самая старая и она НИКОГДА не
        была опровергнута — её маркеры выжили всё окно и заслуживают
        восстановления доверия.
        """
        if len(self.feedback_history) == self.feedback_history.maxlen:
            oldest = self.feedback_history[0]
            if not oldest.reversed and oldest.matched_markers:
                self.amygdala.recover_markers(oldest.matched_markers)

        self.feedback_history.append(entry)
