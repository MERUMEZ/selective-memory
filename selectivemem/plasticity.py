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
 PLASTICITY.PY — Порог записи: что вообще достойно памяти
================================================================================
Принцип, ради которого построен весь проект: информация сохраняется НЕ
всегда, а только когда эмоциональная плотность или ошибка предсказания
превышают порог пластичности.

    плотность = (эмоция + удивление) / 2
    писать, если плотность >= порог

Половина на половину — не подобранный коэффициент, а утверждение: у
организма два независимых повода запомнить, и они равноправны. Первый —
"это меня задело", второй — "я этого не ожидал". Ни один не главнее.

ПОРОГ ПОДВИЖЕН. При перегрузке он поднимается: организм под нагрузкой
хуже усваивает новое. Это не украшение, а самосохранение — иначе поток
незнакомого входа в стрессе забивал бы память шумом. Величина подъёма
задаётся plasticity_stress_modifier.

ЗДЕСЬ ТОЛЬКО РЕШЕНИЕ, НЕ ЗАПИСЬ. Гейт отвечает на вопрос "стоит ли", а
писать или нет — дело вызывающего: у него могут быть свои причины
сохранить рутину (например, противоречие с уже известным — см.
MemoryGraph.find_superseded, где сам факт исправления служит поводом
записать вопреки низкой плотности).

ПОЧЕМУ ЭТО В ЯДРЕ. Раньше решение принимала Amygdala из core/ — вместе с
распознаванием русских маркеров похвалы. Две очень разные вещи: порог
записи это память, а "молодец" и "неверно" — конкретный язык конкретного
продукта. Разделены, чтобы пакет памяти можно было использовать с любым
языком и любым источником оценки.
================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional

from selectivemem.settings import MemorySettings

logger = logging.getLogger(__name__)


@dataclass
class PlasticityDecision:
    """Решение гейта и всё, из чего оно сложилось."""

    emotion: float
    surprise: float
    density: float
    threshold: float
    is_spike: bool

    @property
    def headroom(self) -> float:
        """
        Насколько плотность превысила порог (или не дотянула, если минус).

        Нужна для настройки: когда пользователь жалуется, что бот чего-то
        не запомнил, ответ "не хватило 0.185 до порога" полезнее, чем
        "не сработало".
        """
        return self.density - self.threshold


class PlasticityGate:
    """
    Решает, достойно ли входящее событие записи в долговременную память.

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
        Порог с поправкой на нагрузку. load в [0, 1]: ноль — покой,
        единица — перегрузка.
        """
        load = max(0.0, min(1.0, load))
        return min(1.0, self.base_threshold + load * self.settings.plasticity_stress_modifier)

    def evaluate(
        self,
        emotion: float,
        surprise: float = 0.0,
        load: float = 0.0,
    ) -> PlasticityDecision:
        """Плотность против порога. Логирует оба исхода, не только спайк."""
        density = (emotion * 0.5) + (surprise * 0.5)
        threshold = self.effective_threshold(load)
        is_spike = density >= threshold

        logger.info(
            "[%s] плотность=%.3f (эмоция=%.3f, удивление=%.3f) %s порог=%.3f",
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
