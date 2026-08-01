"""
================================================================================
 TEST_DECAY_FLOOR.PY — Отмеченное важным тускнеет, но не исчезает
================================================================================
Стабильность растёт от ВСПОМИНАНИЯ. Поэтому факт, который пользователь
прямо назвал важным, но о котором разговор больше не заходил, уходил в
ноль вместе с рутиной и удалялся. Замер: через полгода от разговора не
оставалось НИЧЕГО, включая аллергию на пенициллин.

Для ассистента это неприемлемо. Теперь подкреплённый узел угасает не в
ноль, а к полу, высота которого берётся из reward_expectation — той
величины, которую и так ведёт правило Рескорлы-Вагнера. Пол ЗАСЛУЖИВАЕТСЯ:
одна похвала даёт низкий, повторные поднимают.

Идея взята у memory-decay-core (soft-floor decay). Реализация своя:
у них высота пола задаётся полем impact, которое передаёт вызывающий, а
здесь она выводится из накопленного одобрения, то есть из поведения
пользователя, а не из числа, которое кто-то должен придумать.
================================================================================
"""

import pytest

from decaymem import Memory, MemorySettings

YEAR = 365 * 86400


def _run(praises: int, floor: float = 0.25):
    """Запоминает факт, хвалит его N раз за ВСПОМИНАНИЕ, ждёт год."""
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0],
                    settings=MemorySettings(memory_floor_max=floor))
    obs = memory.observe("у меня аллергия на пенициллин", timestamp=now[0])
    now[0] += 300

    for _ in range(praises):
        memory.observe("что мне нельзя принимать", timestamp=now[0])
        memory.recall("какие у меня аллергии", timestamp=now[0])
        memory.feedback(+1.0, timestamp=now[0])
        now[0] += 300

    memory.forget(now=now[0] + YEAR)
    row = memory.graph.db.get_node(obs.node_id)
    memory.close()
    return row


def test_unpraised_memory_still_disappears():
    """
    Пол не делает память бессмертной: без подкрепления узел уходит, как
    и раньше. Иначе библиотека превратилась бы в свалку.
    """
    assert _run(praises=0) is None


def test_praised_memory_survives_a_year():
    """Одной похвалы достаточно, чтобы факт пережил год молчания."""
    row = _run(praises=1)
    assert row is not None
    assert row["weight"] > 0.0


def test_floor_is_earned_not_granted():
    """
    Высота пола растёт с одобрением. Это и отличает наш пол от заданного
    числом: важность зарабатывается поведением пользователя.
    """
    weights = [_run(praises=n)["weight"] for n in (1, 2, 5)]
    assert weights[0] < weights[1] < weights[2], weights


def test_floor_never_raises_weight():
    """
    Угасание обязано оставаться угасанием: пол не может ПОДНЯТЬ вес выше
    того, что был. Иначе молчание усиливало бы память, что абсурдно.
    """
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])
    obs = memory.observe("важный факт про пенициллин", timestamp=now[0])
    memory.recall("пенициллин", timestamp=now[0])
    memory.feedback(+1.0, timestamp=now[0])

    before = memory.graph.db.get_node(obs.node_id)["weight"]
    for _ in range(6):
        now[0] += 30 * 86400
        memory.forget(now=now[0])
        current = memory.graph.db.get_node(obs.node_id)["weight"]
        assert current <= before + 1e-9, "вес вырос при угасании"
        before = current
    memory.close()


def test_floor_can_be_switched_off():
    """
    Нулевой пол возвращает прежнее поведение. Компромисс должен быть в
    руках вызывающего: кому нужна память, забывающая всё, тот её получит.
    """
    assert _run(praises=3, floor=0.0) is None


# ---------------------------------------------------------------------------
# Подкрепление должно доставать до ВСПОМНЕННОГО, а не только до записанного
# ---------------------------------------------------------------------------

def test_praise_reaches_recalled_nodes():
    """
    У ассистента похвала следует за хорошим ответом, построенным на
    памяти. Значит подкрепляться должно вспомненное.

    Раньше feedback доставался только что ЗАПИСАННОМУ узлу, а вспомненное
    игнорировалось. Замер через фасад: восемь похвал подряд давали то же
    ожидание 0.300, что и одна.
    """
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])

    obs = memory.observe("мой телефон восемь девятьсот двенадцать", timestamp=now[0])
    now[0] += 300
    before = memory.graph.db.get_node(obs.node_id)["reward_expectation"] or 0.0

    memory.observe("какой у меня телефон", timestamp=now[0])
    memory.recall("телефон", timestamp=now[0])
    memory.feedback(+1.0, timestamp=now[0])

    after = memory.graph.db.get_node(obs.node_id)["reward_expectation"] or 0.0
    assert after > before, "вспомненный узел обязан получить подкрепление"
    memory.close()


def test_praise_reaches_both_written_and_recalled():
    """
    Если ход и записал новое, и опирался на старое — награду получают
    оба. Раньше стояло "либо-либо", и записанное вытесняло вспомненное.
    """
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])

    old = memory.observe("моя дочь Лиза, ей шесть лет", timestamp=now[0])
    now[0] += 300

    new = memory.observe("Лиза пошла в первый класс этой осенью", timestamp=now[0])
    memory.recall("дочь Лиза", timestamp=now[0])
    memory.feedback(+1.0, timestamp=now[0])

    assert new.written, "второй факт должен был записаться"
    assert (memory.graph.db.get_node(old.node_id)["reward_expectation"] or 0.0) > 0.0
    assert (memory.graph.db.get_node(new.node_id)["reward_expectation"] or 0.0) > 0.0
    memory.close()
