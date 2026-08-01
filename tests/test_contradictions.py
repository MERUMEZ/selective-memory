"""
Тесты вытеснения устаревших фактов.

Контекст: память копила взаимоисключающие узлы и отдавала случайный.
Замер до правки — "мою собаку зовут Рекс", позже "мою собаку зовут Бобик":

    как зовут мою собаку?
        0.906  'мою собаку зовут Рекс'    <- УСТАРЕВШИЙ ВЫИГРЫВАЛ
        0.875  'мою собаку зовут Бобик'

Порядок выдачи решало сходство строк, а не время, поэтому бот уверенно
называл неверное имя.

ГЛАВНОЕ РЕШЕНИЕ: устаревшее ОСЛАБЛЯЕТСЯ, а не удаляется. Ложное
срабатывание тогда стоит дёшево и исправляется само — если факт остался
верным, пользователь упомянет его снова и узел восстановится. Половина
тестов здесь именно про то, чтобы механизм НЕ срабатывал где не надо.
"""
import pytest

import config
from decaymem.database import Database
from decaymem.graph_memory import MemoryGraph
from decaymem import embeddings

requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="вытеснение опирается на семантику; без модели оно отключено",
)


@pytest.fixture
def mg():
    return MemoryGraph(db=Database(db_path=":memory:"))


def remember(graph, text, timestamp=0.0, settled=True):
    """
    Кладёт факт в память. settled=True дополнительно "вспоминает" его —
    так и выглядит закрепившийся факт в жизни: его упоминали и к нему
    возвращались. Неявная поправка вытесняет только такое (см.
    CONTRADICTION_MIN_STABILITY), поэтому фикстура обязана это отражать.
    """
    node_id = graph.save_connection(text, "запомнил", weight=0.8, timestamp=timestamp)
    if settled:
        graph.touch_node(node_id, timestamp=timestamp + 1.0)
    return node_id


# ---------------------------------------------------------------------------
# Срабатывает там, где надо
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.parametrize(
    "old,new",
    [
        ("мою собаку зовут Рекс", "мою собаку зовут Бобик"),
        ("я живу в Москве", "я живу в Питере"),
    ],
    ids=["другое имя", "другой город"],
)
def test_new_version_supersedes_the_old(mg, old, new):
    remember(mg, old)
    assert mg.find_superseded(new), f"{new!r} должно вытеснять {old!r}"


@requires_model
def test_correction_puts_the_new_fact_first(mg):
    """Ровно тот случай, который наблюдался: устаревший факт выигрывал."""
    remember(mg, "мою собаку зовут Рекс", timestamp=0.0)
    remember(mg, "мою собаку зовут Бобик", timestamp=1000.0)

    found = mg.search("как зовут мою собаку", top_k=2, timestamp=2000.0,
                      with_associations=False)

    assert found, "хоть что-то должно найтись"
    assert "Бобик" in found[0].context, (
        f"первым обязан идти актуальный факт, а вернулось {found[0].context!r}"
    )


# ---------------------------------------------------------------------------
# НЕ срабатывает там, где не надо — этих тестов намеренно больше
# ---------------------------------------------------------------------------
@requires_model
def test_repetition_is_not_a_contradiction(mg):
    """Повтор того же самого обязан подкреплять, а не вытеснять."""
    remember(mg, "мою собаку зовут Рекс")
    assert mg.find_superseded("мою собаку зовут Рекс") == []


@requires_model
@pytest.mark.parametrize(
    "old,new",
    [
        ("у меня есть кошка", "у меня есть собака"),
        ("я живу в Москве", "я работаю программистом"),
        ("мою собаку зовут Рекс", "сегодня хорошая погода"),
    ],
    ids=["оба могут быть верны", "независимый факт", "другая тема"],
)
def test_independent_facts_are_left_alone(mg, old, new):
    """
    Самый опасный класс ошибок: ослабить воспоминание, которое осталось
    верным. Порог темы держится высоким именно ради этого.
    """
    remember(mg, old)
    assert mg.find_superseded(new) == [], (
        f"{new!r} не должно трогать {old!r} — это не поправка"
    )


# ---------------------------------------------------------------------------
# Ослабление, а не удаление
# ---------------------------------------------------------------------------
@requires_model
def test_superseded_node_is_weakened_not_deleted(mg):
    node_id = remember(mg, "мою собаку зовут Рекс", timestamp=0.0)
    before = mg.db.get_node(node_id)["weight"]

    remember(mg, "мою собаку зовут Бобик", timestamp=1000.0)

    row = mg.db.get_node(node_id)
    assert row is not None, "вытесненный узел не должен удаляться"
    assert row["weight"] < before
    assert row["stability"] <= config.STABILITY_INITIAL * 2


@requires_model
def test_false_positive_heals_itself(mg):
    """
    Если факт на самом деле остался верным, повторные упоминания
    возвращают его в оборот. Ради этого свойства и выбрано ослабление
    вместо удаления.
    """
    node_id = remember(mg, "мою собаку зовут Рекс", timestamp=0.0)
    remember(mg, "мою собаку зовут Бобик", timestamp=1000.0)

    weakened = mg.db.get_node(node_id)["stability"]
    for i in range(4):
        mg.touch_node(node_id, timestamp=2000.0 + i)

    assert mg.db.get_node(node_id)["stability"] > weakened * 2


@requires_model
def test_time_separates_stale_from_current(mg):
    """
    Настоящее разделение делает не штраф веса, а сброшенная стабильность:
    вытесненный угасает быстро, актуальный держится.
    """
    stale_id = remember(mg, "мою собаку зовут Рекс", timestamp=0.0)
    fresh_id = remember(mg, "мою собаку зовут Бобик", timestamp=100.0)
    for i in range(3):
        mg.touch_node(fresh_id, timestamp=200.0 + i)

    mg.apply_decay(now=300.0 + 3 * 86400 * config.TIME_ACCELERATION)

    assert mg.db.get_node(stale_id) is None, "устаревший факт должен забыться"
    assert mg.db.get_node(fresh_id) is not None, "актуальный должен остаться"


# ---------------------------------------------------------------------------
# Явная поправка и деградация
# ---------------------------------------------------------------------------
@requires_model
def test_explicit_correction_lowers_the_bar(mg):
    """
    Когда пользователь явно поправил ("нет", "неправильно"), это сильное
    свидетельство — порог темы снижается.
    """
    remember(mg, "мою собаку зовут Рекс")

    strict = len(mg.find_superseded("собака Бобик", explicit_correction=False))
    lenient = len(mg.find_superseded("собака Бобик", explicit_correction=True))

    assert lenient >= strict


def test_without_embeddings_nothing_is_superseded(mg, monkeypatch):
    """
    Без семантики отличить "другую версию" от "другой темы" нечем:
    строковое сходство одинаково высоко и там, и там. Молча ничего не
    вытесняем, а не гадаем.
    """
    monkeypatch.setattr(embeddings, "encode", lambda text: None)
    remember(mg, "мою собаку зовут Рекс")
    assert mg.find_superseded("мою собаку зовут Бобик") == []


def test_supersede_survives_missing_node(mg):
    mg.supersede_node(999999)  # не должно бросить


# ---------------------------------------------------------------------------
# Поправка должна ПОПАСТЬ в память, даже будучи неудивительной
#
# Найдено живым использованием: пользователь поправил имя собаки, а в
# памяти осталось старое. Причина оказалась не в вытеснении, а на шаг
# раньше — поправка вообще не записывалась. Замер на рабочей базе:
#     "мою собаку зовут бобик"    -> плотность 0.165 при пороге 0.35
#     "мою собаку зовут бобик!!!" -> 0.263, всё ещё мимо
# Поправка по природе НЕ УДИВИТЕЛЬНА: та же фраза, одно слово другое.
# Спайк-гейт видел знакомое и не пускал самое важное сообщение.
# ---------------------------------------------------------------------------
def _session(tmp_path):
    import sys
    sys.argv = ["x"]
    sys.path.insert(0, "tools")
    from simulate_learning import install_llm_stub

    install_llm_stub()
    from core.brain_session import BrainSession

    return BrainSession(db_path=str(tmp_path / "brain.db"))


@requires_model
def test_correction_is_written_even_without_a_spike(tmp_path):
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        # Поправка спокойная: ни капса, ни восклицаний, слова знакомые
        result = session.process_message("мою собаку зовут Бобик")

        assert result.debug.get("memory_written"), (
            "поправка обязана попасть в память, даже если спайк не сработал — "
            "иначе вытеснять устаревшее будет нечем"
        )
    finally:
        session.close()


@requires_model
def test_correction_outranks_the_stale_fact(tmp_path):
    """Ровно то, что наблюдалось у пользователя на живом боте."""
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        session.process_message("мою собаку зовут Бобик")

        found = session.memory.search(
            "как зовут мою собаку", top_k=2,
            timestamp=session.clock.get_brain_time(), with_associations=False,
        )

        assert found, "хоть что-то должно найтись"
        assert "обик" in found[0].context, (
            f"первым обязан идти актуальный факт, а вернулось {found[0].context!r}"
        )
    finally:
        session.close()


@requires_model
def test_correction_inherits_the_standing_of_what_it_replaces(tmp_path):
    """
    Вес узла обычно равен плотности сообщения, но поправка неудивительна и
    рождалась бы слабой (0.17 против 0.52 у устаревшего) — запись бы
    состоялась, а поиск всё равно выигрывал бы старый факт. Новая версия
    наследует вес предшественника: это тот же факт, обновлённый.
    """
    session = _session(tmp_path)
    try:
        session.process_message("мою собаку зовут Рекс")
        session.process_message("мою собаку зовут Бобик")

        rows = {
            r["context"]: r["weight"]
            for r in session.memory.db._conn.execute(
                "SELECT context, weight FROM nodes WHERE node_type='episodic'"
            )
        }
        fresh = next(v for k, v in rows.items() if "обик" in k)
        stale = next(v for k, v in rows.items() if "екс" in k)

        assert fresh > stale, "актуальная версия должна весить больше вытесненной"
    finally:
        session.close()
