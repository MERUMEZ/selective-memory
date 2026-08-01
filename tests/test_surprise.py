"""
Тесты на СОБСТВЕННОЕ удивление организма (MemoryGraph.compute_surprise).

Контекст: спайк-память — сердце системы («записываем только то, что
удивило»). До этой правки «удивление» считалось энтропией Шеннона по
символам строки, то есть было свойством ТЕКСТА, а не отношением
ОРГАНИЗМА к тексту. Замер показывал:

    новорождённый мозг          «у меня есть кошка» -> 0.856
    мозг после 50 повторений    «у меня есть кошка» -> 0.856
    бессмысленный набор букв                        -> 0.819

Ни обучение, ни осмысленность не влияли ни на что. При этом perplexity
питает СРАЗУ ЧЕТЫРЕ механизма: спайк-гейт (Amygdala.evaluate), уверенность
(Cortex._estimate_confidence -> эхолалия), структурную консолидацию
(STM_STRUCTURAL_THRESHOLD) и любопытство в Mood.

Эти тесты фиксируют главное свойство: удивление ДОЛЖНО падать по мере
накопления опыта, иначе организм не живой, а просто считает буквы.
"""
import pytest

import config
from selectivemem.database import Database
from selectivemem.graph_memory import MemoryGraph

PHRASE = "мама мыла раму"


@pytest.fixture
def mg():
    """Свежий MemoryGraph на in-memory SQLite для каждого теста."""
    return MemoryGraph(db=Database(db_path=":memory:"))


def _teach(graph: MemoryGraph, text: str, times: int) -> None:
    for i in range(times):
        graph.process_language_input(text, timestamp=float(i))


# ---------------------------------------------------------------------------
# 1. Новорождённому всё ново
# ---------------------------------------------------------------------------
def test_newborn_is_maximally_surprised(mg):
    """
    Новорождённому ново всё — но только если реплике есть чем удивить.
    Удивление умножается на долю набранного содержания (см.
    surprise_full_content_tokens), поэтому проверять максимум надо на
    ПОЛНОЙ реплике, а не на трёх словах.
    """
    result = mg.compute_surprise("мама мыла раму белым мылом дважды")

    assert result.total == pytest.approx(1.0)
    assert result.known_words == 0
    assert result.known_pairs == 0


def test_short_utterance_cannot_surprise_much(mg):
    """
    Междометие ново ровно так же, как факт, но информации в нём нет.
    Без этой поправки библиотека записывала "ага" и "спасибо" наравне с
    "у меня аллергия на пенициллин" — замер на потоке, похожем на
    переписку с ассистентом, показал это на первом же дне.
    """
    interjection = mg.compute_surprise("ага").total
    fact = mg.compute_surprise("у меня аллергия на пенициллин").total

    assert interjection < fact
    assert fact == pytest.approx(1.0)

    # Порог сравнивается не с удивлением, а с ПЛОТНОСТЬЮ (эмоция +
    # удивление пополам) — на это и надо проверять, иначе тест меряет не
    # тот механизм.
    from selectivemem.plasticity import PlasticityGate

    gate = PlasticityGate()
    assert not gate.evaluate(emotion=0.0, surprise=interjection).is_spike
    assert gate.evaluate(emotion=0.0, surprise=fact).is_spike


# ---------------------------------------------------------------------------
# 2. Удивление СТРОГО падает с опытом — главное свойство правки
# ---------------------------------------------------------------------------
def test_surprise_decreases_monotonically_with_experience():
    previous = None
    for exposures in (0, 1, 2, 3):
        graph = MemoryGraph(db=Database(db_path=":memory:"))
        _teach(graph, PHRASE, exposures)
        current = graph.compute_surprise(PHRASE).total

        if previous is not None:
            assert current < previous, (
                f"после {exposures} повторений удивление ({current:.3f}) должно быть "
                f"строго меньше, чем после {exposures - 1} ({previous:.3f})"
            )
        previous = current

    assert previous == pytest.approx(0.0, abs=1e-9), "полностью привычная фраза не удивляет"


# ---------------------------------------------------------------------------
# 3. Опыт НЕ обесценивает настоящую новизну (иначе организм просто глохнет)
# ---------------------------------------------------------------------------
def test_experienced_brain_still_surprised_by_the_unknown(mg):
    _teach(mg, PHRASE, times=20)

    familiar = mg.compute_surprise(PHRASE).total
    # Тарабарщина той же длины, что и выученная фраза: сравнивать надо
    # при равном объёме содержания, иначе меряется длина, а не новизна.
    gibberish = mg.compute_surprise("ЫФВАПРОЛДЖ ЙЦУКЕН ЩЗХЪЭЖ").total

    assert familiar == pytest.approx(0.0, abs=1e-9)
    assert gibberish > 0.7, "незнакомые слова обязаны удивлять и опытный мозг"


def test_new_word_in_familiar_frame_is_partially_surprising(mg):
    _teach(mg, PHRASE, times=20)

    partial = mg.compute_surprise("мама мыла окно").total

    assert 0.0 < partial < 1.0, (
        "знакомая рамка с одним новым словом должна давать промежуточное удивление"
    )
    assert partial > mg.compute_surprise(PHRASE).total


# ---------------------------------------------------------------------------
# 4. Структурная составляющая: знакомые слова в невиданном сочетании
# ---------------------------------------------------------------------------
def test_unseen_word_combination_is_structurally_surprising(mg):
    # Два независимых контекста: слова знакомы, но никогда не встречались рядом
    _teach(mg, "мама мыла раму", times=10)
    _teach(mg, "папа чинил стул", times=10)

    seen_together = mg.compute_surprise("мама мыла раму")
    never_together = mg.compute_surprise("мама чинил")

    assert never_together.lexical == pytest.approx(0.0, abs=1e-9), (
        "оба слова хорошо знакомы -> лексического удивления быть не должно"
    )
    assert never_together.structural > seen_together.structural, (
        "невиданное сочетание знакомых слов обязано удивлять структурно"
    )
    assert never_together.total > 0.0


# ---------------------------------------------------------------------------
# 5. Краевые случаи — без деления на ноль и без исключений
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", "!!!", "5 7"])
def test_empty_or_tokenless_input_is_not_surprising(mg, text):
    """Нет токенов — удивляться нечему, а не «максимально неожиданно»."""
    result = mg.compute_surprise(text)
    assert result.total == 0.0
    assert result.total_words == 0


def test_single_token_uses_lexical_component_only(mg):
    """Одно слово: пар нет, итог определяется только лексикой."""
    result = mg.compute_surprise("привет")

    assert result.total_pairs == 0
    # lexical — чистая незнакомость слова, total — она же с поправкой на
    # объём содержания. Для одного токена поправка максимальная.
    assert result.lexical == pytest.approx(1.0)
    assert result.total < result.lexical, "одно слово не может удивить в полную силу"

    _teach(mg, "привет", times=5)
    assert mg.compute_surprise("привет").total < result.total


def test_surprise_stays_within_unit_range(mg):
    """Итог всегда в [0..1] — от него зависят пороги спайка и уверенности."""
    _teach(mg, PHRASE, times=3)
    for text in ["", PHRASE, "мама", "совершенно неизвестные слова тут", "!!! ??? ..."]:
        total = mg.compute_surprise(text).total
        assert 0.0 <= total <= 1.0, f"удивление вне диапазона на {text!r}: {total}"


# ---------------------------------------------------------------------------
# 6. Токенизация обучения и измерения обязана совпадать
# ---------------------------------------------------------------------------
def test_measured_tokens_match_learned_tokens(mg):
    """
    Организм должен удивляться ровно тем единицам, которым учится. Если
    process_language_input и compute_surprise разойдутся в токенизации,
    удивление будет считаться по словам, которые никогда не запоминаются,
    и никогда не упадёт до нуля.
    """
    text = "Мама, мыла раму 5 раз!"

    learned = mg.process_language_input(text, timestamp=0.0)
    measured = mg.compute_surprise(text)

    assert measured.total_words == learned.words_processed
    # Всё, чему научились с первого раза, теперь знакомо
    assert mg.compute_surprise(text).known_words == learned.words_processed
