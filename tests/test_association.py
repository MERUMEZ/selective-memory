"""
================================================================================
 TEST_ASSOCIATION.PY — Библиотека обязана связывать воспоминания между собой
================================================================================
Замер, из которого родился этот файл: после 200 вызовов observe() в базе
оказался 201 эпизодический узел и НОЛЬ рёбер между ними. Не мало — ни
одного. Библиотека не связывала воспоминания вообще; рёбра, которые видно
в демонстрации, создаёт витрина (core/brain_session.py), у которой под
рукой сразу оба конца — и вспомненный узел, и записываемый.

Для человека, поставившего пакет, это значило две вещи сразу:

  1. связность не могла работать сигналом важности — нечего измерять;
  2. растекающейся активации, заявленной в README как multi-hop и
     занимающей заметную часть кода поиска, было НЕ ПО ЧЕМУ растекаться.

Второе — расхождение витрины с пакетом, то есть ровно тот разряд ошибок,
который вскрывается на демонстрации у покупателя.

Правило связывания хеббовское: что было возбуждено вместе, то и
связывается. Приложение только что достало из памяти какие-то узлы — это
и есть контекст, в котором появилось новое воспоминание.
================================================================================
"""

import pytest

from selectivemem import Memory, MemorySettings


@pytest.fixture
def memory():
    m = Memory(":memory:", settings=MemorySettings(associate_recalled_limit=3))
    yield m
    m.close()


def _episodic_ids(m):
    return [r["id"] for r in m.graph.db.fetch_all_nodes() if r["node_type"] == "episodic"]


def test_recalled_node_links_to_the_next_memory(memory):
    """Вспомнили — записали. Между ними должно появиться ребро."""
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    assert first is not None

    memory.recall("аллергия")
    second = memory.observe("врач выписал другой антибиотик", emotion=0.8).node_id
    assert second is not None

    edges = memory.graph.db.get_edges_between([first, second])
    assert edges, "новая запись не связалась с тем, что было вспомнено"


def test_without_recall_nothing_is_linked(memory):
    """
    Связь возникает от СОВМЕСТНОЙ АКТИВНОСТИ, а не от соседства во времени.
    Две записи подряд без единого вспоминания связывать не за что.
    """
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    second = memory.observe("во дворе растёт высокое дерево", emotion=0.9).node_id

    assert memory.graph.db.get_edges_between([first, second]) == []


def test_zero_limit_restores_previous_behaviour():
    """Ноль обязан полностью выключать механизм: это путь отката."""
    m = Memory(":memory:", settings=MemorySettings(associate_recalled_limit=0))
    first = m.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    m.recall("аллергия")
    second = m.observe("врач выписал другой антибиотик", emotion=0.8).node_id

    assert m.graph.db.get_edges_between([first, second]) == []
    m.close()


def test_connectivity_stops_being_empty(memory):
    """
    Главное, ради чего всё затевалось: степень узлов перестаёт быть нулевой.
    Без этого связность как сигнал важности измеряет ровно ничто — что и
    показал замер на LongMemEval: 201 узел, у всех 201 степень ноль.
    """
    # Запрос совпадает со СЛОВАМИ записей дословно: тест не должен
    # зависеть от наличия семантической модели. Прежняя версия спрашивала
    # "кот" про записи со словом "кота" и проходила только там, где
    # доступен navec, — то есть на машине автора.
    for index in range(6):
        memory.observe(f"событие номер {index} про кота и собаку", emotion=0.9)
        memory.recall("событие про кота и собаку")

    ids = _episodic_ids(memory)
    degrees = memory.graph.db.get_degrees(ids)
    connected = [node_id for node_id in ids if degrees.get(node_id, 0) > 0]

    assert connected, "граф эпизодов остался пустым"


def test_a_memory_is_never_linked_to_itself(memory):
    """
    Узел, только что найденный поиском, может оказаться и тем, который
    сейчас пишется, — петля в графе никому не нужна.
    """
    first = memory.observe("у меня аллергия на пенициллин", emotion=0.9).node_id
    memory.recall("аллергия на пенициллин")
    second = memory.observe("у меня аллергия на пенициллин совсем", emotion=0.9).node_id

    for node_id in {first, second} - {None}:
        edges = memory.graph.db.get_edges_between([node_id])
        assert all(e["node_from"] != e["node_to"] for e in edges)
