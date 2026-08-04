# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Внутренняя среда: организм чувствует собственное состояние.

Проверяется НЕ красота чисел, а три обязательных свойства:
  * состояние РАЗЛИЧАЕТ обстоятельства (иначе это украшение);
  * два канала расходятся по смыслу (срочность облегчает запись,
    напряжение затрудняет);
  * явно переданная приложением оценка ВСЕГДА главнее внутренней.
"""

import random

import pytest

from selectivemem import Memory
from selectivemem.settings import MemorySettings

CONS = "бвгджзклмнпрстфхцчшщ"
VOW = "аеиоуыэюя"


def nonsense(rng):
    """Каждый раз новые слова: удивление остаётся высоким."""
    def word():
        return "".join(rng.choice(CONS) + rng.choice(VOW)
                       for _ in range(rng.randint(2, 4)))
    return " ".join(word() for _ in range(5))


def familiar(rng):
    """Один и тот же уклад: удивление падает."""
    return f"{rng.choice(['утром', 'днём', 'вечером'])} я пил чай на кухне"


@pytest.fixture
def settings():
    s = MemorySettings()
    s.intrinsic_emotion = True
    return s


def test_newborn_feels_nothing(settings):
    """У новорождённого нет прошлого, относительно которого что-то изменилось."""
    memory = Memory(":memory:", settings=settings)
    state = memory.feel()
    assert state.valence == 0.0
    assert state.arousal == 0.0
    memory.close()


def test_predictable_and_chaotic_worlds_feel_different(settings):
    """
    Главная проверка модуля.

    Если предсказуемая и бессвязная среда дают одно самочувствие, никакого
    чувства состояния нет. Первая версия стенда именно это и показывала:
    восемь строк тарабарщины, повторённые восемь раз, становятся
    привычными — поэтому здесь бессвязность ПОРОЖДАЕТСЯ заново каждый раз.
    """
    rng = random.Random(1)
    calm = Memory(":memory:", settings=settings)
    for _ in range(40):
        calm.observe(familiar(rng))
    calm_state = calm.feel()
    calm.close()

    wild = Memory(":memory:", settings=settings)
    for _ in range(40):
        wild.observe(nonsense(rng))
    wild_state = wild.feel()
    wild.close()

    assert wild_state.mean_surprise > calm_state.mean_surprise + 0.5, (
        f"среды неразличимы: {calm_state.mean_surprise:.2f} против "
        f"{wild_state.mean_surprise:.2f}"
    )


def test_crowding_rises_with_a_full_store(settings):
    """Теснота — единственная нужда, при которой организм теряет часть себя."""
    settings.memory_capacity = 4
    rng = random.Random(2)
    memory = Memory(":memory:", settings=settings)
    for _ in range(30):
        memory.observe(nonsense(rng))
    assert memory.feel().crowding == pytest.approx(1.0)
    assert memory.feel().strain > 0.5, "полное хранилище обязано давать напряжение"
    memory.close()


def test_unlimited_memory_feels_no_crowding(settings):
    """
    Без предела ёмкости терять нечего — и нужды нет.

    Выдумать организму нехватку места, которой у него не бывает, было бы
    враньём: нужда должна быть настоящей, иначе всё устройство держится на
    подделке.
    """
    settings.memory_capacity = 0
    rng = random.Random(3)
    memory = Memory(":memory:", settings=settings)
    for _ in range(20):
        memory.observe(nonsense(rng))
    assert memory.feel().crowding == 0.0
    memory.close()


def test_explicit_emotion_always_wins(settings):
    """
    Приложение знает про событие то, чего ядро знать не может.

    И отдельно: явный ноль — это НЕ «не сказали». Подпись различает их
    через None, и проверка сторожит именно это.
    """
    rng = random.Random(4)
    memory = Memory(":memory:", settings=settings)
    for _ in range(20):
        memory.observe(nonsense(rng))

    # emotion=1.0 означает "запомни" и обязано пройти при любом самочувствии.
    result = memory.observe("запомни: ключи лежат под ковриком", emotion=1.0)
    assert result.node_id is not None
    memory.close()


def test_valence_tracks_change_not_level(settings):
    """
    Хорошо не тогда, когда всё хорошо, а когда становится лучше.

    То же устройство, что в подкреплении: дофамин кодирует ошибку
    предсказания, а не награду. Постоянная теснота перестаёт быть плохой
    новостью — плохая новость это теснота, которая РАСТЁТ.
    """
    settings.memory_capacity = 3
    rng = random.Random(5)
    memory = Memory(":memory:", settings=settings)
    # Долгий ровный поток: отклонение перестаёт меняться.
    for _ in range(60):
        memory.observe(nonsense(rng))
    assert abs(memory.feel().valence) < 0.2, (
        "при установившемся состоянии валентность обязана осесть у нуля"
    )
    memory.close()


def test_interoception_is_off_by_default():
    """
    Умолчание не меняется молча.

    Внутренняя среда трогает решение о записи у КАЖДОГО пользователя
    библиотеки, поэтому включаться она может только осознанно — пока
    замер на полном стенде не покажет, что от неё лучше.
    """
    assert MemorySettings().intrinsic_emotion is False
