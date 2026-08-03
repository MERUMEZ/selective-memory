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

from selectivemem import Memory, MemorySettings
from selectivemem import embeddings

# Проверка опирается на СЕМАНТИКУ: запрос сформулирован другими словами,
# чем сохранённый текст. Без модели поиск честно отвечает пустотой и сам
# об этом предупреждает в логе — это заявленная деградация, а не поломка.
# Без этой отметки чистая установка без extras давала девять красных
# тестов и создавала впечатление сломанного пакета.
requires_model = pytest.mark.skipif(
    not embeddings.is_available(),
    reason="запрос сформулирован иначе, чем запись: нужна семантическая модель",
)

YEAR = 365 * 86400


def _run(praises: int, floor: float = 0.25, years: float = 1.0):
    """Запоминает факт, хвалит его N раз за ВСПОМИНАНИЕ, ждёт."""
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

    memory.forget(now=now[0] + years * YEAR)
    row = memory.graph.db.get_node(obs.node_id)
    memory.close()
    return row


@requires_model
def test_unpraised_memory_yields_to_praised():
    """
    Пол не делает память бессмертной — но и уходит она теперь не сама.

    Прежнее утверждение было "без подкрепления узел ИСЧЕЗАЕТ, иначе
    библиотека превратится в свалку". Замер отменил его: удаление по
    возрасту стоит 18.6 пункта полноты на 500 вопросах LongMemEval,
    потому что улики к вопросам о меняющихся фактах старше вопроса на
    16 дней и стирались все до единой, 12 из 12.

    Свалку теперь предотвращают спайк-гейт (три четверти реплик не
    доходят до памяти вовсе) и вытеснение по ёмкости. А непохвалённое
    отличается от похвалённого тем, что ПРОИГРЫВАЕТ в силе, — то есть
    уступает место при конкуренции, а не пропадает по часам.
    """
    plain = _run(praises=0)
    praised = _run(praises=1)
    assert plain is not None, "узлы больше не удаляются по возрасту"
    assert praised is not None
    assert (praised["strength"] or 0.0) > (plain["strength"] or 0.0), (
        "похвала обязана давать перевес в силе — иначе она ни на что не влияет"
    )


def test_praised_memory_survives_a_year():
    """Одной похвалы достаточно, чтобы факт пережил год молчания."""
    row = _run(praises=1)
    assert row is not None
    assert row["weight"] > 0.0


@requires_model
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


@requires_model
def test_floor_can_be_switched_off():  # noqa: D401
    """
    Нулевой пол возвращает прежнее поведение: подкреплённое живёт дольше
    рутины, но конечное время.

    Горизонт здесь ТРИ года, а не один, и это не придирка. Похвала теперь
    реально усиливает вспомненный узел (вес плюс продвинутая метка
    доступа), поэтому за год он не успевает угаснуть даже без пола.
    Замер: без пола 0.196 через год и забыт через три; с полом 0.326
    через год, 0.171 через три, 0.164 через десять — то есть сходится к
    полу и не исчезает никогда.

    Первая версия теста проверяла годовой горизонт и упала, когда
    подкрепление стало доходить до вспомненного. Она кодировала прежнее,
    сломанное поведение.

    ВТОРАЯ ВЕРСИЯ УПАЛА ТОЖЕ, и по той же причине — кодировала поведение,
    которое мы заменили. Она требовала, чтобы без пола узел ИСЧЕЗ через
    три года. Теперь по умолчанию не исчезает ничего: удаление по возрасту
    стоило 18.6 пункта полноты на 500 вопросах, и его убрали.

    Пол при этом никуда не делся и по-прежнему различим — просто разница
    видна в ВЕСЕ, а не в наличии строки.
    """
    without = _run(praises=3, floor=0.0, years=3)
    with_floor = _run(praises=3, floor=0.25, years=3)
    assert without is not None and with_floor is not None
    assert with_floor["weight"] > without["weight"], (
        "пол обязан удерживать вес выше — иначе он ничего не делает"
    )


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


def test_praise_boosts_the_weight_of_recalled_nodes():
    """
    Похвала должна не только делать вспомненное ДОЛГОВЕЧНЕЕ (через пол),
    но и ЯРЧЕ прямо сейчас — то есть поднимать вес.

    Этот тест появился потому, что первую правку я сделал наполовину.
    Починил "либо-либо" в начислении ожидания награды и не заметил, что
    точно такое же "либо-либо" стоит в применении эффекта: прибавка веса
    доставалась только записанному узлу. Замер показал вес вспомненного
    0.500 -> 0.500 при похвале, хотя ожидание при этом росло.
    """
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])

    obs = memory.observe("мой телефон восемь девятьсот двенадцать", timestamp=now[0])
    now[0] += 300
    before = memory.graph.db.get_node(obs.node_id)["weight"]

    memory.observe("какой у меня телефон", timestamp=now[0])
    memory.recall("телефон", timestamp=now[0])
    memory.feedback(+1.0, timestamp=now[0])

    after = memory.graph.db.get_node(obs.node_id)["weight"]
    assert after > before, f"вес вспомненного не вырос: {before:.3f} -> {after:.3f}"
    memory.close()


def test_blame_reaches_recalled_nodes_too():
    """
    Симметрия обязательна. Если организм вспомнил не то и получил
    порицание, штраф должен дойти до вспомненного — иначе он повторит
    ошибку.
    """
    now = [1_700_000_000.0]
    memory = Memory(":memory:", clock=lambda: now[0])

    obs = memory.observe("мой телефон восемь девятьсот двенадцать", timestamp=now[0])
    now[0] += 300
    before = memory.graph.db.get_node(obs.node_id)["weight"]

    memory.observe("какой у меня адрес", timestamp=now[0])
    memory.recall("телефон", timestamp=now[0])
    memory.feedback(-1.0, timestamp=now[0])

    row = memory.graph.db.get_node(obs.node_id)
    assert row["weight"] < before, "штраф не дошёл до вспомненного"
    assert (row["reward_expectation"] or 0.0) < 0.0
    memory.close()
