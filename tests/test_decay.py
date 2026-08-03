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

# Константы берутся из настроек БИБЛИОТЕКИ, а не из config витрины:
# витрина уезжает в свой репозиторий, а проверка ядра обязана остаться
# здесь и работать у всякого, кто поставил пакет.
from selectivemem.settings import MemorySettings as _LibrarySettings

config = _LibrarySettings()
from selectivemem.database import Database
from selectivemem.graph_memory import MemoryGraph


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

    expected = 1.0 * math.exp(-config.decay_rate * (n_ticks * tick) / config.age_t0)
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

    expected = 0.5 * math.exp(-config.edge_decay_rate * (n_ticks * tick) / config.age_t0)
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

    # Вспоминание не только сдвигает часы, но и УПРОЧНЯЕТ память —
    # эффект распределённого повторения (см. блок 7 ниже).
    stability_after_touch = row["stability"]
    assert stability_after_touch == pytest.approx(
        config.stability_initial * config.stability_growth_factor
    )

    # dt=0 сразу после touch -> decay не должен ничего изменить
    mg.apply_decay(now=1000.0)
    row = mg.db.get_node(node_id)
    assert row["weight"] == weight_after_first_decay

    # инкрементальный decay должен считаться ОТ touch (1000), а не от
    # исходного создания узла (0), и по УПРОЧНЁННОЙ шкале времени
    mg.apply_decay(now=1100.0)
    row = mg.db.get_node(node_id)
    expected = weight_after_first_decay * math.exp(
        -config.decay_rate * 100.0 / (config.age_t0 * stability_after_touch)
    )
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


# ---------------------------------------------------------------------------
# 5. ЛЕКСИКА ЖИВЁТ НА СВОЕЙ ШКАЛЕ ВРЕМЕНИ (config.lexical_age_t0).
#
# Контекст: с единым AGE_T0=1час освоенное слово (weight 0.20) падало ниже
# порога освоения за 6 часов паузы и УДАЛЯЛОСЬ из БД за ~28 часов. То есть
# бот забывал весь выученный язык за ночь, а за выходные — безвозвратно.
# Замер стендом: в режиме "20 сообщений + ночная пауза" словарь колебался
# 5->1, 13->3, 17->4 и не рос никогда, Stage 0 не проходился в принципе.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_type", ["word", "syllable"])
def test_lexical_nodes_decay_slower_than_episodic(mg, node_type):
    """Лексический узел при том же dt должен терять существенно меньше веса."""
    lexical_id, _ = mg.db.upsert_lexical_node(
        node_type, "кошка", initial_weight=0.5, reinforce_step=0.04, timestamp=0.0
    )
    episodic_id = mg.db.insert_node(
        context="как дела", response="нормально", weight=0.5, timestamp=0.0, node_type="episodic"
    )

    # Окно задаётся ОТНОСИТЕЛЬНО характерного времени эпизода, а не в
    # абсолютных сутках: иначе тест ломается при каждой перекалибровке
    # шкалы времени (так и случилось при переходе на когерентные часы).
    window = 20 * config.age_t0
    mg.apply_decay(now=window)

    lexical_weight = mg.db.get_node(lexical_id)["weight"]
    episodic_row = mg.db.get_node(episodic_id)
    episodic_weight = episodic_row["weight"] if episodic_row is not None else 0.0

    # Обе кривые считаются по своей шкале времени
    assert lexical_weight == pytest.approx(
        0.5 * math.exp(-config.decay_rate * window / config.lexical_age_t0), rel=1e-6
    )
    assert episodic_weight == pytest.approx(
        0.5 * math.exp(-config.decay_rate * window / config.age_t0), rel=1e-6
    )

    # Суть разделения: за 20 характерных времён эпизода слово почти не
    # трогается, а сам эпизод теряет больше половины веса.
    assert episodic_weight < 0.5 * 0.5, "эпизод должен заметно выцвести"
    assert lexical_weight > 0.49, "слово не должно заметно выцветать за то же время"
    assert lexical_weight > episodic_weight * 2


def test_mastered_word_survives_a_weekend(mg):
    """
    Регрессия на главный баг: слово, освоенное тремя повторениями, обязано
    остаться освоенным после паузы в выходные. Раньше оно удалялось из БД.
    """
    for i in range(3):
        word_id, _ = mg.db.upsert_lexical_node(
            "word", "мама",
            initial_weight=config.word_node_initial_weight,
            reinforce_step=config.word_node_reinforce_step,
            timestamp=float(i),
        )
    assert mg.get_vocabulary_size() == 1, "три повторения должны давать освоенное слово"

    mg.apply_decay(now=48 * 3600.0)  # ушёл на выходные

    assert mg.db.get_node(word_id) is not None, "слово не должно исчезнуть из БД за выходные"
    assert mg.get_vocabulary_size() == 1, "слово должно остаться ОСВОЕННЫМ после паузы"


# ---------------------------------------------------------------------------
# 6. Слоги иммунны к orphan-прунингу сна наравне со словами.
#
# Слог держится ребром SYLLABLE_WORD_EDGE_WEIGHT=0.45, но рёбра угасают
# быстрее узлов (EDGE_DECAY_RATE > DECAY_RATE), поэтому связь рано или
# поздно проседает ниже EDGE_ACTIVATION_THRESHOLD — и слог удалялся,
# подмывая субстрат лепета.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("node_type", ["word", "syllable"])
def test_lexical_nodes_survive_synaptic_pruning(mg, node_type):
    lexical_id, _ = mg.db.upsert_lexical_node(
        node_type, "ма", initial_weight=0.10, reinforce_step=0.03, timestamp=0.0
    )
    # Слабый эпизодический узел без связей — законная жертва прунинга,
    # нужен как контроль, что прунинг вообще работает.
    weak_episodic = mg.db.insert_node(
        context="шум", response="шум", weight=0.10, timestamp=0.0, node_type="episodic"
    )

    report = mg.run_synaptic_pruning()

    assert mg.db.get_node(lexical_id) is not None, (
        f"{node_type}-узел не должен удаляться orphan-прунингом: "
        "лексика — инфраструктура языка, а не эпизод"
    )
    assert mg.db.get_node(weak_episodic) is None, "контроль: слабый эпизод-сирота должен быть удалён"
    assert report.orphan_nodes_pruned >= 1


# ---------------------------------------------------------------------------
# 7. STABILITY — сопротивление забыванию растёт от вспоминания.
#
# Контекст: у живой памяти два параметра, а не один. weight — насколько
# ярко помнится сейчас, stability — насколько медленно забывается. Раньше
# был только weight, поэтому ВСЕ эпизоды старели по одной шкале AGE_T0=1час
# независимо от востребованности: замер показывал 11 эпизодических узлов до
# ночной паузы и 1-2 после. Это не забывание, а амнезия.
# ---------------------------------------------------------------------------
def test_new_node_starts_with_initial_stability(mg):
    node_id = mg.db.insert_node(
        context="x", response="y", weight=0.8, timestamp=0.0, node_type="episodic"
    )
    assert mg.db.get_node(node_id)["stability"] == pytest.approx(config.stability_initial)


def test_recall_grows_stability_up_to_the_cap(mg):
    node_id = mg.db.insert_node(
        context="x", response="y", weight=0.8, timestamp=0.0, node_type="episodic"
    )

    previous = mg.db.get_node(node_id)["stability"]
    for i in range(1, 6):
        mg.touch_node(node_id, timestamp=float(i))
        current = mg.db.get_node(node_id)["stability"]
        assert current > previous, "каждое вспоминание должно упрочнять память"
        previous = current

    # Потолок: вечной памяти у организма быть не должно
    for i in range(100):
        mg.touch_node(node_id, timestamp=float(100 + i))
    assert mg.db.get_node(node_id)["stability"] == pytest.approx(config.stability_max)


def test_recalled_memory_outlives_forgotten_one():
    """
    Главное свойство: два одинаковых эпизода расходятся в судьбе только
    потому, что к одному организм возвращался, а к другому нет.
    """
    # Тоже относительно характерного времени, а не в абсолютных неделях
    long_silence = 80 * config.age_t0

    forgotten = MemoryGraph(db=Database(db_path=":memory:"))
    cold_id = forgotten.db.insert_node(
        context="x", response="y", weight=0.8, timestamp=0.0, node_type="episodic"
    )
    forgotten.apply_decay(now=long_silence)

    recalled = MemoryGraph(db=Database(db_path=":memory:"))
    warm_id = recalled.db.insert_node(
        context="x", response="y", weight=0.8, timestamp=0.0, node_type="episodic"
    )
    for i in range(10):
        recalled.touch_node(warm_id, timestamp=float(i))
    recalled.apply_decay(now=long_silence)

    # МОДЕЛЬ СМЕНИЛАСЬ, и утверждение вместе с ней. Раньше здесь стояло
    # "невостребованный эпизод обязан ИСЧЕЗНУТЬ — экономия памяти это суть
    # проекта". Замер показал, что удаление по возрасту стоит 18.6 пункта
    # полноты на 500 вопросах LongMemEval: улики к вопросам о меняющихся
    # фактах старше вопроса на 16 дней и стирались все до единой.
    #
    # Экономия памяти никуда не делась, но живёт теперь в двух других
    # местах: спайк-гейт не пускает три четверти реплик, а переполнение
    # ёмкости вытесняет наименее заслуживших. Возраст судьбу не решает.
    #
    # Свойство, ради которого тест написан, СОХРАНЯЕТСЯ: два одинаковых
    # эпизода расходятся, потому что к одному возвращались. Только
    # выражается это разницей силы, а не наличием строки в таблице.
    cold_row = forgotten.db.get_node(cold_id)
    warm_row = recalled.db.get_node(warm_id)
    assert cold_row is not None, "узлы больше не удаляются по возрасту"
    assert warm_row is not None

    cold_strength = cold_row["strength"] or 0.0
    warm_strength = warm_row["strength"] or 0.0
    assert warm_strength > cold_strength, (
        "эпизод, к которому возвращались 10 раз, обязан быть СИЛЬНЕЕ "
        "невостребованного — иначе возвращение ничего не значит"
    )
    assert warm_row["weight"] > cold_row["weight"], (
        "и заметнее при извлечении"
    )


def test_stability_survives_null_for_legacy_rows(mg):
    """Узлы из БД, созданных до миграции, не должны ронять decay."""
    node_id = mg.db.insert_node(
        context="x", response="y", weight=0.8, timestamp=0.0, node_type="episodic"
    )
    cursor = mg.db._conn.cursor()
    cursor.execute("UPDATE nodes SET stability = NULL WHERE id = ?", (node_id,))
    mg.db._conn.commit()

    mg.apply_decay(now=3600.0)  # не должно бросить TypeError

    assert mg.db.get_node(node_id)["weight"] < 0.8


# ---------------------------------------------------------------------------
# 8. Триггер автосна считает воспоминания, а не лексику.
#
# Раньше здесь стоял подсчёт ВСЕХ узлов, а словарь набирает сотни узлов за
# первый десяток сообщений — порог пробивался на 9-м сообщении, и сон
# запускался на каждое следующее: STM очищался каждое сообщение, стресс
# сбрасывался, а в проде уходило два вызова LLM на сообщение.
# ---------------------------------------------------------------------------
def test_memory_node_count_ignores_lexical_infrastructure(mg):
    mg.db.insert_node(context="разговор", response="ответ", weight=0.8,
                      timestamp=0.0, node_type="episodic")
    mg.db.upsert_concept_node("кошка", "животное", weight=0.7, timestamp=0.0)
    # Мета-узлы создаются НАПРЯМУЮ через библиотечный upsert_meta_node, а не
    # через PersonaMemory витрины: проверяется поведение count_memory_nodes,
    # и тащить ради двух строк зависимость от приложения незачем.
    mg.db.upsert_meta_node(node_type="self_model", content="я организм", weight=1.0)
    mg.db.upsert_meta_node(node_type="user_model", content="собеседник", weight=1.0)
    for token in ("мама", "мыла", "раму", "папа", "чинил"):
        mg.db.upsert_lexical_node("word", token, initial_weight=0.12,
                                  reinforce_step=0.04, timestamp=0.0)
        mg.db.upsert_lexical_node("syllable", token[:2], initial_weight=0.10,
                                  reinforce_step=0.03, timestamp=0.0)

    assert mg.db.count_memory_nodes() == 2, (
        "считаться должны только эпизод и понятие: лексика и мета-узлы — не воспоминания"
    )
    assert mg.count_nodes() > 10, "контроль: всего узлов в графе заметно больше"


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