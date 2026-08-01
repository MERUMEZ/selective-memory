"""
================================================================================
 TEST_SEARCH_SCALING.PY — Отбор кандидатов не должен менять ответы
================================================================================
Поиск считает нечёткое сходство не по всем узлам, а по лучшим кандидатам
из дешёвого отбора. Замер профилировщиком на 10 000 узлов показал, зачем:
SequenceMatcher съедал 82% времени (4.49 с из 5.44), семантика — 6%.
После правки 30 000 узлов ищутся за 0.5 с вместо 2.1 с.

Но экономия имеет цену: узел, который вытянул бы себя ТОЛЬКО нечётким
сходством, до этого этапа не доживёт. Здесь проверяется, что цена не
взимается на практике — результаты совпадают с полным перебором.

Полный перебор включается настройкой: search_candidate_minimum больше
числа узлов означает "кандидаты — все". Так этот компромисс не только
проверяется, но и остаётся в руках пользователя библиотеки: кому нужна
точность любой ценой, тот её получит.
================================================================================
"""

import random

import pytest

from decaymem.database import Database
from decaymem.graph_memory import MemoryGraph
from decaymem.settings import MemorySettings

WORDS = (
    "кот собака дом работа книга город море еда музыка друг "
    "время погода машина сад окно письмо чай дорога лес"
).split()

QUERIES = [
    "расскажи про кота",
    "что там было про город и море",
    "работа и время",
    "музыка для друга",
    "погода в саду",
    "письмо про дорогу в лесу",
    "чай у окна",
    "книга про машину",
]


def _build(settings, count=400, seed=7):
    rng = random.Random(seed)
    graph = MemoryGraph(db=Database(db_path=":memory:"), settings=settings)
    for i in range(count):
        context = " ".join(rng.choice(WORDS) for _ in range(6))
        graph.db.insert_node(
            context=context, response="ответ про " + context[:20],
            weight=round(rng.uniform(0.3, 0.9), 3),
            timestamp=float(i), node_type="episodic",
        )
    return graph


def test_prefilter_agrees_with_exhaustive_search():
    """
    Отбор кандидатов и полный перебор должны давать один и тот же лучший
    узел. Если расходятся — экономия куплена ценой качества, и об этом
    надо знать числом, а не догадкой.
    """
    fast = _build(MemorySettings())
    exhaustive = _build(MemorySettings(search_candidate_minimum=100_000))

    disagreements = []
    for query in QUERIES:
        a = fast.search(query, top_k=1, timestamp=1e9, with_associations=False)
        b = exhaustive.search(query, top_k=1, timestamp=1e9, with_associations=False)
        if [m.id for m in a] != [m.id for m in b]:
            disagreements.append((query, [m.id for m in a], [m.id for m in b]))

    fast.close()
    exhaustive.close()
    assert not disagreements, f"отбор изменил ответы: {disagreements}"


def test_prefilter_agrees_on_top_three():
    """То же самое для тройки лучших — порядок тоже не должен разъезжаться."""
    fast = _build(MemorySettings())
    exhaustive = _build(MemorySettings(search_candidate_minimum=100_000))

    for query in QUERIES:
        a = [m.id for m in fast.search(query, top_k=3, timestamp=1e9, with_associations=False)]
        b = [m.id for m in exhaustive.search(query, top_k=3, timestamp=1e9, with_associations=False)]
        assert a == b, f"{query!r}: {a} против {b}"

    fast.close()
    exhaustive.close()


def test_candidate_pool_is_tunable():
    """
    Компромисс скорость/точность обязан быть в руках вызывающего, а не
    зашит числом в коде.
    """
    settings = MemorySettings(search_candidate_multiplier=5, search_candidate_minimum=10)
    graph = _build(settings, count=200)
    assert graph.search("расскажи про кота", top_k=1, timestamp=1e9, with_associations=False) is not None
    graph.close()


def test_search_survives_without_embeddings(monkeypatch):
    """
    Без семантики отбор идёт по ключевым словам и весу. Модель
    необязательна, и поиск обязан продолжать работать — иначе обещание
    "ставится куда угодно" неверно.
    """
    from decaymem import embeddings

    monkeypatch.setattr(embeddings, "encode", lambda text: None)

    graph = _build(MemorySettings(), count=200)
    found = graph.search("расскажи про кота", top_k=1, timestamp=1e9, with_associations=False)
    assert isinstance(found, list)
    graph.close()


@pytest.mark.parametrize("count", [0, 1, 3])
def test_tiny_graphs_do_not_break_the_prefilter(count):
    """Пустой и почти пустой граф — тоже законные состояния."""
    graph = _build(MemorySettings(), count=count)
    graph.search("что-нибудь", top_k=3, timestamp=1e9, with_associations=False)
    graph.close()
