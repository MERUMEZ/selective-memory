"""
Тесты семантического поиска по памяти.

Контекст: память искала ПО БУКВАМ — keyword-пересечение плюс
SequenceMatcher по целым строкам. Замер на реальном узле
"у меня есть кошка":

    "у меня есть кот"              -> 0.870  найдено
    "расскажи про кота"            -> не найдено
    "мой домашний питомец мяукает" -> не найдено
    "у меня есть кожа"             -> 0.884  НАЙДЕНО ЛУЧШЕ, ЧЕМ КОТ

То есть слово, отличающееся одной буквой, оказывалось релевантнее
синонима, а перифраз не находился вовсе.

Тесты помечены skipif: модель (~51 МБ) необязательна, и в окружении без
неё поиск обязан молча деградировать до строкового — это проверяется
отдельно, без пропуска.
"""
import pytest

import config
from decaymem.database import Database
from decaymem.graph_memory import MemoryGraph
from decaymem import embeddings

requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="модель эмбеддингов недоступна (необязательная зависимость)",
)


@pytest.fixture
def mg():
    graph = MemoryGraph(db=Database(db_path=":memory:"))
    graph.save_connection("у меня есть кошка", "кошки хорошие", weight=0.8, timestamp=0.0)
    return graph


def score(graph, query):
    found = graph.search(query, top_k=1, timestamp=1.0, with_associations=False)
    return found[0].similarity if found else 0.0


# ---------------------------------------------------------------------------
# Главная инверсия, ради которой всё делалось
# ---------------------------------------------------------------------------
@requires_model
def test_synonym_beats_lookalike(mg):
    """
    "кот" — синоним, "кожа" — случайное совпадение букв. Раньше побеждала
    "кожа" (0.884 против 0.870).
    """
    assert score(mg, "у меня есть кот") > score(mg, "у меня есть кожа")


@requires_model
def test_paraphrase_is_found_at_all(mg):
    """
    Раньше перифраз без общих значимых слов не находился в принципе:
    "расскажи про кота" не пересекается с "у меня есть кошка" ни одним
    содержательным словом.
    """
    assert score(mg, "расскажи про кота") >= config.MEMORY_SEARCH_THRESHOLD


@requires_model
def test_distant_paraphrase_is_beyond_the_light_model():
    """
    ЗАФИКСИРОВАННОЕ ОГРАНИЧЕНИЕ, а не забытый случай.

    "мой домашний питомец мяукает" и "у меня есть кошка" не имеют ни одного
    общего слова, и усреднённые статические векторы такую связь не тянут:
    сырая косинусная близость всего 0.268. Понижать ради этого общий порог
    нельзя — вместе с таким перифразом в выдачу полезет шум.

    Это цена лёгкой модели. Настоящие sentence-transformers справились бы,
    но тянут torch: ~2.5 ГБ на диске и 0.5-1 ГБ RAM на процесс, а на этой
    машине 3.7 ГБ всего и рядом работают ещё три сервиса.

    Тест закреплён намеренно: если однажды модель сменят на более сильную,
    он упадёт и напомнит пересмотреть это решение.
    """
    graph = MemoryGraph(db=Database(db_path=":memory:"))
    graph.save_connection("у меня есть кошка", "кошки хорошие", weight=0.8, timestamp=0.0)

    raw = embeddings.cosine(
        embeddings.encode("мой домашний питомец мяукает"),
        embeddings.encode("у меня есть кошка кошки хорошие"),
    )
    assert raw < 0.35, (
        f"близость выросла до {raw:.3f} — похоже, модель сменилась на более "
        "сильную, и порог поиска стоит пересмотреть"
    )


@requires_model
def test_lookalike_is_not_presented_as_confident(mg):
    """
    Случайное буквенное совпадение может попасть в выдачу, но обязано
    честно помечаться как смутное — иначе бот уверенно подмешает в ответ
    воспоминание не по теме.
    """
    assert score(mg, "у меня есть кожа") < config.MEMORY_INJECTION_CONFIDENT_THRESHOLD


@requires_model
def test_exact_match_still_wins(mg):
    exact = score(mg, "у меня есть кошка")
    assert exact > score(mg, "у меня есть кот")
    assert exact >= config.MEMORY_INJECTION_CONFIDENT_THRESHOLD


@requires_model
def test_unrelated_query_is_not_found(mg):
    assert score(mg, "квантовая хромодинамика") < config.MEMORY_SEARCH_THRESHOLD


@requires_model
def test_proper_names_survive_without_semantics(mg):
    """
    Строковые составляющие оставлены не для симметрии: имён собственных в
    векторной модели нет, и держатся они на keyword-пересечении.
    """
    mg.save_connection("меня зовут Паша Морозов", "приятно познакомиться",
                       weight=0.8, timestamp=0.0)
    assert score(mg, "как зовут Морозова") >= config.MEMORY_SEARCH_THRESHOLD


# ---------------------------------------------------------------------------
# Векторы считаются лениво и кэшируются
# ---------------------------------------------------------------------------
@requires_model
def test_embedding_is_computed_lazily_and_stored(mg):
    """
    Узлы, созданные до появления модели, приходят с embedding=NULL. Вместо
    разовой тяжёлой миграции вектор досчитывается при первом обращении.
    """
    node_id = mg.db.fetch_searchable_nodes()[0]["id"]
    mg.db.update_embedding(node_id, None)
    assert mg.db.get_node(node_id)["embedding"] is None

    mg.search("кошка", top_k=1, timestamp=1.0, with_associations=False)

    assert mg.db.get_node(node_id)["embedding"] is not None, (
        "вектор должен быть досчитан и сохранён при первом же поиске"
    )


@requires_model
def test_stored_vector_survives_a_round_trip():
    vector = embeddings.encode("проверка сохранности вектора")
    restored = embeddings.from_blob(embeddings.to_blob(vector))
    assert embeddings.cosine(vector, restored) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Без модели всё обязано продолжать работать
# ---------------------------------------------------------------------------
def test_search_works_without_embeddings(mg, monkeypatch):
    """
    Модель необязательна: 51 МБ может не оказаться на машине, библиотека
    может быть не установлена. Поиск обязан молча остаться строковым, а не
    упасть.
    """
    monkeypatch.setattr(embeddings, "encode", lambda text: None)

    assert score(mg, "у меня есть кошка") > 0.0, (
        "без семантики поиск должен работать на строковом сходстве"
    )


def test_encode_handles_degenerate_input():
    for text in ("", "   ", "!!!", "…"):
        assert embeddings.encode(text) is None


def test_cosine_is_safe_on_missing_vectors():
    assert embeddings.cosine(None, None) == 0.0
    assert embeddings.similarity("привет", None) == 0.0
