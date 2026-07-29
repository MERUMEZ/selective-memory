"""
Regression-тесты на decay-формулу (узлы + рёбра) и на все пути создания
узлов, которые должны проставлять last_decayed_at.

Контекст: 2026-07-29 в проде дважды ловили баги в decay из-за поля
last_decayed_at:
  1. Decay считал dt от last_accessed/last_activated (которые двигаются
     только при реальном использовании), а apply_decay гоняется на
     КАЖДОЕ сообщение для ВСЕХ узлов -> угасание накапливалось
     квадратично вместо линейного по времени (узел с весом 1.0 удалялся
     за 2 виртуальных часа вместо ожидаемых ~0.905 веса).
  2. После фикса (1) выяснилось, что upsert_concept_node и
     upsert_lexical_node создавали/обновляли узлы БЕЗ last_decayed_at,
     что приводило к TypeError в проде при первом же apply_decay после
     создания такого узла.

Эти тесты фиксируют оба фикса и защищают от повторения в будущем.
"""
import math

import pytest

import config
from memory.database import Database
from memory.graph_memory import MemoryGraph


@pytest.fixture
def mg():
    """Свежий MemoryGraph на in-memory SQLite для каждого теста."""
    return MemoryGraph(db=Database(db_path=":memory:"))


# ---------------------------------------------------------------------------
# 1. Compounding-тест: N вызовов apply_decay на нетронутом узле должны
#    давать ТОТ ЖЕ результат, что и один расчёт по формуле на суммарный dt.
# ---------------------------------------------------------------------------
def test_node_decay_no_compounding(mg):
    node_id = mg.db.insert_node(
        context="x", response="y", weight=1.0, timestamp=0.0, node_type="episodic"
    )

    tick = 120.0
    n_ticks = 60  # 2 виртуальных часа

    for i in range(1, n_ticks + 1):
        mg.apply_decay(now=i * tick)

    row = mg.db.get_node(node_id)
    assert row is not None, "узел не должен быть забыт: корректный вес выше FORGET_THRESHOLD"

    expected = 1.0 * math.exp(-config.DECAY_RATE * (n_ticks * tick) / config.AGE_T0)
    assert row["weight"] == pytest.approx(expected, rel=1e-6)


def test_edge_decay_no_compounding(mg):
    a = mg.db.insert_node(context="a", response="a", weight=1.0, timestamp=0.0, node_type="episodic")
    b = mg.db.insert_node(context="b", response="b", weight=1.0, timestamp=0.0, node_type="episodic")
    mg.connect_nodes(a, b, weight_boost=0.5, timestamp=0.0)

    tick = 120.0
    n_ticks = 60

    for i in range(1, n_ticks + 1):
        mg.apply_decay(now=i * tick)

    edges = mg.db.fetch_all_edges()
    assert len(edges) == 1, "ребро не должно быть забыто раньше срока"

    expected = 0.5 * math.exp(-config.EDGE_DECAY_RATE * (n_ticks * tick) / config.AGE_T0)
    assert edges[0]["weight"] == pytest.approx(expected, rel=1e-6)

# ---------------------------------------------------------------------------
# 2. Touch-тест: touch_node должен сбрасывать last_decayed_at, чтобы decay
#    после реального использования не "доначислял" за прошедший период.
# ---------------------------------------------------------------------------
def test_touch_node_resets_decay_clock(mg):
    node_id = mg.db.insert_node(
        context="x", response="y", weight=1.0, timestamp=0.0, node_type="episodic"
    )

    mg.apply_decay(now=500.0)
    row = mg.db.get_node(node_id)
    weight_after_first_decay = row["weight"]
    assert row["last_decayed_at"] == 500.0

    mg.touch_node(node_id, timestamp=1000.0)
    row = mg.db.get_node(node_id)
    assert row["last_accessed"] == 1000.0
    assert row["last_decayed_at"] == 1000.0
    assert row["weight"] == weight_after_first_decay  # touch не меняет вес

    # dt=0 сразу после touch -> decay не должен ничего изменить
    mg.apply_decay(now=1000.0)
    row = mg.db.get_node(node_id)
    assert row["weight"] == weight_after_first_decay

    # инкрементальный decay должен считаться ОТ touch (1000), а не от
    # исходного создания узла (0)
    mg.apply_decay(now=1100.0)
    row = mg.db.get_node(node_id)
    expected = weight_after_first_decay * math.exp(-config.DECAY_RATE * 100.0 / config.AGE_T0)
    assert row["weight"] == pytest.approx(expected, rel=1e-6)

# ---------------------------------------------------------------------------
# 3. NULL-fallback тест: _decay_nodes/_decay_edges не должны падать, если
#    last_decayed_at IS NULL (пропущенный путь создания) — вместо этого
#    используют last_accessed/last_activated и лечат запись на этом же
#    проходе.
# ---------------------------------------------------------------------------
def test_decay_survives_null_last_decayed_at_on_node(mg):
    node_id = mg.db.insert_node(
        context="x", response="y", weight=1.0, timestamp=0.0, node_type="episodic"
    )

    cursor = mg.db._conn.cursor()
    cursor.execute("UPDATE nodes SET last_decayed_at = NULL WHERE id = ?", (node_id,))
    mg.db._conn.commit()

    mg.apply_decay(now=300.0)  # не должно бросить TypeError

    row = mg.db.get_node(node_id)
    assert row["last_decayed_at"] == 300.0, "NULL должен вылечиться на этом же проходе"


def test_decay_survives_null_last_decayed_at_on_edge(mg):
    a = mg.db.insert_node(context="a", response="a", weight=1.0, timestamp=0.0, node_type="episodic")
    b = mg.db.insert_node(context="b", response="b", weight=1.0, timestamp=0.0, node_type="episodic")
    mg.connect_nodes(a, b, weight_boost=0.5, timestamp=0.0)

    cursor = mg.db._conn.cursor()
    cursor.execute("UPDATE edges SET last_decayed_at = NULL")
    mg.db._conn.commit()

    mg.apply_decay(now=300.0)  # не должно бросить TypeError

    edges = mg.db.fetch_all_edges()
    assert edges[0]["last_decayed_at"] == 300.0

# ---------------------------------------------------------------------------
# 4. Все пути создания/обновления узлов должны проставлять last_decayed_at
#    (кроме мета-узлов — они пропускаются в decay ДО чтения этого поля,
#    поэтому NULL для них безопасен).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "create_fn",
    [
        lambda db: db.insert_node(context="x", response="y", weight=1.0, timestamp=0.0, node_type="episodic"),
        lambda db: db.upsert_concept_node("кошка", "животное", weight=0.7, timestamp=0.0)[0],
        lambda db: db.upsert_lexical_node("word", "привет", initial_weight=0.1, reinforce_step=0.05, timestamp=0.0)[0],
    ],
    ids=["insert_node", "upsert_concept_node", "upsert_lexical_node"],
)
def test_creation_paths_set_last_decayed_at(mg, create_fn):
    node_id = create_fn(mg.db)
    row = mg.db.get_node(node_id)
    assert row["last_decayed_at"] is not None, (
        "узел создан без last_decayed_at — apply_decay упадёт с TypeError "
        "на первом же проходе (см. баг upsert_concept_node/upsert_lexical_node, 2026-07-29)"
    )


def test_meta_node_decay_is_skipped_regardless_of_null(mg):
    """
    Мета-узлы (is_meta=1) пропускаются в decay ДО чтения last_decayed_at,
    поэтому у них это поле может быть NULL — это не баг, а норма.
    Главное — apply_decay не должен падать и не должен трогать их вес.
    """
    node_id = mg.db.upsert_meta_node(
        node_type="self_model", content="я — цифровой разум", weight=0.95, timestamp=0.0
    )
    row = mg.db.get_node(node_id)
    assert row["is_meta"] == 1

    mg.apply_decay(now=999999.0)  # огромный dt — не должно упасть и не должно съесть вес

    row = mg.db.get_node(node_id)
    assert row["weight"] == 0.95


def test_reinforce_paths_update_last_decayed_at(mg):
    """
    Повторное создание concept-узла (ветка UPDATE/reinforce) тоже должно
    продвигать last_decayed_at, а не оставлять старое значение.
    """
    node_id, _ = mg.db.upsert_concept_node("кошка", "животное", weight=0.7, timestamp=0.0)
    node_id2, was_created = mg.db.upsert_concept_node(
        "кошка", "домашнее животное", weight=0.7, timestamp=100.0
    )
    assert not was_created
    assert node_id2 == node_id

    row = mg.db.get_node(node_id)
    assert row["last_decayed_at"] == 100.0