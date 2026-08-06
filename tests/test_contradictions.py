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

# Ускорение субъективного времени — понятие ВИТРИНЫ: организм живёт
# быстрее настенных часов, чтобы демонстрация не тянулась неделями. В
# библиотеке время задаёт приложение через clock, поэтому здесь просто
# множитель для перевода суток в секунды стенда.
_TIME_ACCELERATION = 7.0

from selectivemem.settings import MemorySettings as _LibrarySettings

config = _LibrarySettings()
from selectivemem.database import Database
from selectivemem.graph_memory import MemoryGraph
from selectivemem import embeddings

requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="вытеснение опирается на семантику; без модели оно отключено",
)


def _handles_russian() -> bool:
    """
    Различает ли ДЕЙСТВУЮЩАЯ модель русские слова по смыслу.

    Проверка возможности, а не наличия. Тексты здесь русские, а модель по
    умолчанию английская (potion-base-8M) — сочетание, которое библиотека
    прямо не поддерживает и советует заменить на [semantic-ru]. Раньше
    такие проверки проходили случайно: перед кодированием из фразы
    выдирались служебные слова, и это сглаживало разницу. Когда отсев для
    модели уровня предложения убрали (он ронял английский бенчмарк с 96.0
    до 93.2), обман вскрылся: «у меня есть собака» и «у меня есть кошка»
    получили косинус 0.955 и стали неотличимы от поправки.
    """
    if not embeddings.is_available():
        return False
    near = embeddings.cosine(embeddings.encode("кот"), embeddings.encode("кошка"))
    far = embeddings.cosine(embeddings.encode("кот"), embeddings.encode("бетон"))
    return near is not None and far is not None and near > far + 0.1


requires_russian = pytest.mark.skipif(
    not _handles_russian(),
    reason="действующая модель не различает русские слова по смыслу; "
           "для русского нужен [semantic-ru]",
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


@requires_russian
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

    mg.apply_decay(now=300.0 + 3 * 86400 * _TIME_ACCELERATION)

    # МОДЕЛЬ СМЕНИЛАСЬ. Раньше здесь требовалось, чтобы устаревший факт
    # ИСЧЕЗ. Замер отменил: удаление по возрасту стоит 18.6 пункта полноты
    # на 500 вопросах LongMemEval — улики к вопросам о меняющихся фактах
    # старше вопроса на 16 дней и стирались все до единой.
    #
    # Разделение осталось, но выражается разницей ВЕСА: вытесненному
    # сбросили стабильность, поэтому он угасает быстро, а актуальный
    # держится. Это и есть "старый адрес всплывает реже нового" — как у
    # людей, где старое не стирается, а уступает.
    stale = mg.db.get_node(stale_id)
    fresh = mg.db.get_node(fresh_id)
    assert stale is not None and fresh is not None
    assert fresh["weight"] > stale["weight"] * 2, (
        "актуальный обязан быть заметно тяжелее устаревшего, иначе "
        "вытеснение ничего не разделяет"
    )


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
# ПЕРЕНАПРАВЛЕНИЕ ВМЕСТО ОСЛАБЛЕНИЯ
# ---------------------------------------------------------------------------
@requires_russian
def test_superseded_node_is_not_damaged(mg):
    """
    Поправка НЕ ТРОГАЕТ вес старого узла — она записывает связь.

    README обещает: «угасает путь к воспоминанию, а не само
    воспоминание». Код делал обратное — снижал вес и сбрасывал
    стабильность, то есть узел выпадал сразу из ВСЕХ запросов, а не
    только из того, где случилась поправка. При ошибочной поправке — а
    живой разговор дал семь ошибок из семи — верный факт повреждался
    навсегда.
    """
    mg.settings.contradiction_weight_penalty = 0.0
    remember(mg, "мою собаку зовут Рекс")
    stale = [r for r in mg.gate.episodic.searchable()
             if "Рекс" in (r["context"] or "")][0]
    before = stale["weight"]

    remember(mg, "мою собаку зовут Бобик")

    after = [r for r in mg.gate.episodic.searchable()
             if "Рекс" in (r["context"] or "")][0]
    assert after["weight"] == before, "вес устаревшего узла не должен меняться"


@requires_russian
def test_supersede_relation_is_recorded(mg):
    """Отношение «это заменило то» должно ХРАНИТЬСЯ, а не исчезать."""
    remember(mg, "мой рейс в четверг")
    remember(mg, "рейс перенесли на субботу")

    rows = mg.db._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE edge_type = 'supersedes'"
    ).fetchone()
    assert rows[0] >= 1, "связь замены не записана"


@requires_russian
def test_stale_version_is_not_returned_when_the_fresh_one_exists(mg):
    """
    Перенаправление: устаревшее не отдаётся, если замена под рукой.

    Отдельно проверяется, что достраивание не втащит его обратно — один
    раз именно это и случилось: связь «заменяет» была прочитана как
    обычное соседство.
    """
    mg.settings.contradiction_weight_penalty = 0.0
    remember(mg, "мою собаку зовут Рекс")
    remember(mg, "мою собаку зовут Бобик")

    found = [m.context for m in mg.search("как зовут собаку", top_k=5)]
    assert any("Бобик" in t for t in found), found
    assert not any("Рекс" in t for t in found), found


@requires_russian
def test_wrong_correction_does_not_hide_a_better_answer(mg):
    """
    Ошибочная поправка не должна прятать запись, которая отвечает лучше.

    Живой случай: «неа, омлет с молоком...» было признано поправкой к
    «тосты люблю, и омлет» — ошибочно, они делят только слово «омлет». На
    запрос «тосты» память отдавала запись про омлет, скрыв единственную
    запись со словом «тосты».

    Разрыв уместности и различает случаи: у настоящей поправки записи
    почти неразличимы, у ошибочной разрыв велик.
    """
    remember(mg, "тосты люблю, и омлет")
    remember(mg, "неа, омлет с молоком и тёртым сыром и зеленью")

    found = [m.context for m in mg.search("тосты", top_k=3)]
    assert any("тосты" in text for text in found), found
