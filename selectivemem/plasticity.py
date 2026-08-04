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
 PLASTICITY.PY — The write threshold: what is worth remembering at all
================================================================================
The principle the whole project is built around: information is NOT always
stored. It is stored when emotional density or prediction error exceeds
the plasticity threshold.

    density = (emotion + surprise) / 2
    store if density >= threshold

Half and half is not a tuned coefficient but a claim: the organism has two
independent reasons to remember, and they are equal. One is "this affected
me", the other is "I did not see this coming". Neither outranks the other.

THE THRESHOLD MOVES. Under load it rises: an overloaded organism absorbs
new things worse. That is not decoration but self-preservation — otherwise
a stream of unfamiliar input under stress would flood memory with noise.
The size of the rise is set by plasticity_stress_modifier.

THIS IS THE DECISION ONLY, NOT THE WRITE. The gate answers "is it worth
it"; whether to store is up to the caller, who may have reasons of their
own to keep routine input. See MemoryGraph.find_superseded, where the very
fact of a correction justifies storing despite low density.

WHY THIS LIVES IN THE CORE. The decision used to be made by Amygdala in
core/, together with recognising Russian words of praise. Two very
different things: a write threshold belongs to memory, while "well done"
and "wrong" belong to one particular language of one particular product.
They were split so the memory package can be used with any language and
any source of evaluation.
================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional

from selectivemem.settings import MemorySettings

logger = logging.getLogger(__name__)


@dataclass
class PlasticityDecision:
    """The gate's decision and everything it was made of."""

    emotion: float
    surprise: float
    density: float
    threshold: float
    is_spike: bool

    @property
    def headroom(self) -> float:
        """
        By how much density cleared the threshold (negative if it fell short).

        Useful when tuning: when someone complains that the bot failed to
        remember something, "0.185 short of the threshold" is a better
        answer than "it did not fire".
        """
        return self.density - self.threshold


class PlasticityGate:
    """
    Decides whether an incoming event deserves long-term storage.

        gate = PlasticityGate(settings)
        decision = gate.evaluate(emotion=0.2, surprise=0.9)
        if decision.is_spike:
            memory.save_connection(...)
    """

    def __init__(
        self,
        settings: Optional[MemorySettings] = None,
        base_threshold: Optional[float] = None,
    ):
        self.settings = settings or MemorySettings()
        self.base_threshold = (
            base_threshold
            if base_threshold is not None
            else self.settings.base_plasticity_threshold
        )

    def effective_threshold(self, load: float = 0.0) -> float:
        """
        The threshold adjusted for load. `load` is in [0, 1]: zero means
        calm, one means overload.
        """
        load = max(0.0, min(1.0, load))
        return min(1.0, self.base_threshold + load * self.settings.plasticity_stress_modifier)

    def evaluate(
        self,
        emotion: float,
        surprise: float = 0.0,
        load: float = 0.0,
    ) -> PlasticityDecision:
        """Density against the threshold. Logs both outcomes, not just spikes."""
        if self.settings.gate_emotion_gain > 0.0:
            # МОДУЛЯЦИЯ, А НЕ СРЕДНЕЕ. Норадреналин не складывается с
            # новизной — он УМНОЖАЕТ пластичность, которую новизна уже
            # открыла. Событие страшное, но привычное, и событие
            # безразличное, но небывалое, получали одинаковую оценку;
            # в мозге это разные вещи.
            #
            # Новизна остаётся основанием, эмоция только усиливает. Иначе
            # произведение обнулило бы гейт у любого, кто не передаёт
            # эмоцию, — а по умолчанию она 0.0, то есть у обычного
            # библиотечного пользователя.
            # НОРМИРОВАННОЕ ПРОИЗВЕДЕНИЕ, и нормировка здесь не украшение.
            #
            # Чистое произведение выходит за единицу: новизна 1.0 при
            # эмоции 1.0 и усилении 1.0 даёт 2.0. А плотность становится
            # НАЧАЛЬНЫМ ВЕСОМ узла, который обязан жить в [0,1]. Первая
            # попытка ограничивала сверху — и узел рождался прямо на
            # потолке, так что похвале было некуда расти: тест поймал
            # «вес не вырос: 1.000 -> 0.950».
            #
            # Деление на (1 + усиление) держит границы по построению и,
            # что важнее, СОВПАДАЕТ СО СРЕДНИМ ПРИ НУЛЕВОЙ ЭМОЦИИ: обе
            # формы дают 0.5·новизна. Значит для тех, кто эмоцию не
            # передаёт, ничего не меняется и порог трогать не надо.
            gain = self.settings.gate_emotion_gain
            density = surprise * (1.0 + emotion * gain) / (1.0 + gain)
        else:
            density = (emotion * 0.5) + (surprise * 0.5)
        threshold = self.effective_threshold(load)
        is_spike = density >= threshold

        # ВТОРОЙ ВХОД: сильное возбуждение пишет само по себе.
        #
        # Умножение — правильная форма, но у неё есть следствие: привычное
        # не спасти никакой эмоцией, потому что произведение с околонулевой
        # новизной остаётся околонулевым. Биологически это верно, а обещание
        # продукта — другое: emotion=1.0 означает «пользователь сказал
        # запомни», и это обязано работать.
        #
        # В мозге такой путь тоже отдельный. Вспышечные воспоминания —
        # когда помнишь, где стоял, услышав новость, — возникают не потому,
        # что обстановка была новой, а потому что миндалина при сильном
        # возбуждении модулирует гиппокамп напрямую, минуя обычный порог.
        #
        # Устроено как вытеснение устаревшего: самостоятельный повод для
        # записи, а не слагаемое в формуле.
        override = self.settings.gate_emotion_override
        if not is_spike and override > 0.0 and emotion >= override:
            is_spike = True
            logger.info(
                "[EMOTION OVERRIDE] density=%.3f below threshold=%.3f, "
                "but emotion=%.3f >= %.3f — stored anyway",
                density, threshold, emotion, override,
            )

        logger.info(
            "[%s] density=%.3f (emotion=%.3f, surprise=%.3f) %s threshold=%.3f",
            "SPIKE DETECTED" if is_spike else "ROUTINE",
            density, emotion, surprise,
            ">=" if is_spike else "<",
            threshold,
        )

        return PlasticityDecision(
            emotion=emotion,
            surprise=surprise,
            density=density,
            threshold=threshold,
            is_spike=is_spike,
        )
