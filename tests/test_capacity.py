"""
================================================================================
 TEST_CAPACITY.PY — Память ограничена ОБЪЁМОМ, а не сроком годности
================================================================================
Прежняя политика удаляла по возрасту, и замер показал, что она решала
судьбу памяти сама: на каждом проходе стиралась десятая часть узлов, и
ответы на последующие вопросы всегда оказывались внутри этой десятой.
Улики к вопросам LongMemEval категории knowledge-update старше вопроса в
среднем на 16 дней — стёрлись все до единой, 12 из 12 в каждом из пяти
разобранных случаев.

Замена: приложение задаёт ЁМКОСТЬ, при переполнении уходят наименее
заслужившие. Это ещё и то, чем приложение действительно хочет управлять —
дизайнер игры задаёт "персонаж помнит двести вещей", а не "персонаж
забывает через одиннадцать дней".

Ключевое требование, которое здесь и проверяется: вытеснение НЕ ДОЛЖНО
БЫТЬ СОРТИРОВКОЙ ПО ВОЗРАСТУ. Иначе мы переименовали болезнь, а не
вылечили её.
================================================================================
"""

import pytest

from selectivemem import Memory, MemorySettings

DAY = 86400.0



# РАЗНООБРАЗНЫЙ ПОТОК, а не двадцать раз одно и то же.
#
# Раньше здесь стояло f"событие номер {index} про разное" — и с точки
# зрения организма это ДВАДЦАТЬ ОДИНАКОВЫХ ТЕКСТОВ: цифры в словарь не
# попадают, поэтому новизна падает до 0.000 на третьем сообщении.
#
# Прежний гейт писал их все, потому что эмоция 0.9 сама перетаскивала
# через порог: плотность = 0.5·0.9 + 0.5·0 = 0.45. То есть память плодила
# дубликаты бесконечно, пока приложение передаёт заряд. Нормированное
# произведение это прекратило (0.9·0 -> 0), и тест справедливо покраснел.
#
# Здесь проверяется ЁМКОСТЬ, а не гейт, поэтому поток сделан таким, каким
# и должен быть: двадцать разных событий.
_SUBJECTS = ["кот", "сосед", "врач", "поезд", "чайник", "магазин", "парк",
             "телефон", "зонт", "лестница", "окно", "письмо", "ключи",
             "лампа", "кресло", "забор", "мост", "ручей", "пирог", "шарф"]
_ACTIONS = ["пропал", "сломался", "нашёлся", "подорожал", "закрылся",
            "переехал", "загорелся", "остановился", "промок", "потерялся"]


def _varied(index: int) -> str:
    return f"{_SUBJECTS[index % len(_SUBJECTS)]} {_ACTIONS[index % len(_ACTIONS)]} этим утром"


def _make(capacity: int, **extra):
    now = [1_700_000_000.0]
    settings = MemorySettings(memory_capacity=capacity, delete_on_decay=False, **extra)
    memory = Memory(":memory:", settings=settings, clock=lambda: now[0])
    return memory, now


def _episodic(memory):
    return [r for r in memory.graph.db.fetch_all_nodes() if r["node_type"] == "episodic"]


def test_capacity_is_respected():
    memory, now = _make(capacity=5)
    for index in range(20):
        memory.observe(_varied(index), emotion=0.9)
        now[0] += 60.0
    memory.forget(now=now[0])

    assert len(_episodic(memory)) <= 5
    memory.close()


def test_zero_capacity_means_unlimited():
    memory, now = _make(capacity=0)
    for index in range(20):
        memory.observe(_varied(index), emotion=0.9)
        now[0] += 60.0
    memory.forget(now=now[0])

    assert len(_episodic(memory)) > 5
    memory.close()


def test_praised_survives_a_flood_of_newer_memories():
    """
    Главная проверка. Отмеченное важным записано ПЕРВЫМ и дальше только
    стареет, а сверху валится два десятка свежих. При вытеснении по
    возрасту оно ушло бы первым же; при вытеснении по заслугам обязано
    остаться.
    """
    memory, now = _make(capacity=5)
    memory.observe("у меня аллергия на пенициллин", emotion=0.9)
    memory.feedback(+1.0)

    for index in range(20):
        now[0] += 60.0
        memory.observe(_varied(index), emotion=0.9)

    now[0] += 3 * DAY
    memory.forget(now=now[0])

    texts = " | ".join(r["context"] for r in _episodic(memory))
    assert "пенициллин" in texts, "похвалённое вытеснено более свежим мусором"
    memory.close()


def test_recalled_survives_a_flood_of_newer_memories():
    """
    То же самое, но заслуга другая: факт не хвалили, к нему ВОЗВРАЩАЛИСЬ.
    Стабильность растёт при каждом обращении, и это единственная величина
    в системе, которая движется против возраста.
    """
    memory, now = _make(capacity=5)
    memory.observe("мой самолёт вылетает в четверг утром", emotion=0.9)
    for _ in range(4):
        now[0] += 60.0
        memory.recall("самолёт")

    for index in range(20):
        now[0] += 60.0
        memory.observe(_varied(index), emotion=0.9)

    now[0] += 3 * DAY
    memory.forget(now=now[0])

    texts = " | ".join(r["context"] for r in _episodic(memory))
    assert "самолёт" in texts, "вспоминавшееся вытеснено более свежим мусором"
    memory.close()


def test_eviction_is_not_sorting_by_age():
    """
    Прямая проверка того, что мы не переименовали болезнь: среди
    выживших должны быть и старые узлы, а не только последние N.
    """
    memory, now = _make(capacity=6)
    memory.observe("первое важное событие про аллергию", emotion=0.9)
    memory.feedback(+1.0)
    memory.observe("второе важное событие про самолёт", emotion=0.9)
    memory.feedback(+1.0)

    for index in range(15):
        now[0] += 60.0
        memory.observe(_varied(index), emotion=0.9)

    now[0] += DAY
    memory.forget(now=now[0])

    survivors = _episodic(memory)
    ids = sorted(r["id"] for r in survivors)
    newest = sorted(r["id"] for r in survivors)[-1]
    assert ids[0] < newest - len(survivors), (
        "выжили только последние по счёту — это сортировка по возрасту"
    )
    memory.close()


def test_meta_and_vocabulary_are_not_evicted():
    """
    Ёмкость про ЭПИЗОДЫ. Словарь — инфраструктура языка, мета-узлы —
    служебные; выбрасывать их вместе с проходными репликами нельзя.
    """
    memory, now = _make(capacity=3)
    for index in range(20):
        memory.observe(f"событие номер {index} про кота и собаку", emotion=0.9)
        now[0] += 60.0
    memory.forget(now=now[0])

    assert memory.graph.get_vocabulary_size() > 0, "словарь вытеснен вместе с эпизодами"
    memory.close()
