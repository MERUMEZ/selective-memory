# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Кора отличает ТЕМУ от ОБОРОТА РЕЧИ.

Механизм выводит тему из повторяющегося. Беда в том, что повторяется и
канцелярит: замер поймал факты «рано утром», «прислал счёт», «записку
неделе оставил прошлой» — семь мусорных на два настоящих.

Разделяет не частота по записям (её проверка не убрала ни одного мусорного
факта), а привычность слова в собственном графе языка организма.

ПРОВЕРЯЕТСЯ СОСТОЯНИЕ ПОСЛЕ СНА, а не мгновенный снимок. Отсев идёт в
момент вывода темы, а тогда организм знает мало: в первые дни ему незнаком
и канцелярит. Оборот, подвернувшийся рано, оседает темой — и снимается
только пересмотром, который живое тоже делает не на входе, а потом.
Первая версия этих проверок падала через раз в зависимости от порядка
тестов ровно по этой причине.
"""

import random

import pytest

from selectivemem import Memory
from selectivemem.settings import MemorySettings

# СВОЙ КОДИРОВЩИК, А НЕ ОБЩИЙ НА ПРОЦЕСС.
#
# Обобщение живёт внутри поиска устаревшего, а тот без семантики молчит —
# значит кодировщик нужен. Но брать общий нельзя: проверки падали через
# раз в зависимости от порядка тестов. Причина оказалась не в них, а в
# соседе по прогону: он тянет код витрины (core.brain_session), а тот
# настраивает модуль моделей глобально. Когда модель оказывалась
# недоступна, включалось выращенное восприятие — у свежего организма
# словарь пуст, первые темы кодировать нечем, зато позднейший канцелярит
# обобщался прекрасно. Ровно то, что и показывали падения.
#
# Здесь нужна не модель, а способность различать похожее. Мешок слов даёт
# это точно, детерминированно и без единого общего состояния.
_DIM = 64


def word_bag_encoder(text: str):
    """Косинус между фразами примерно равен доле общих слов."""
    words = [w for w in (text or "").lower().split() if len(w) > 2]
    if not words:
        return None
    vector = [0.0] * _DIM
    for word in words:
        vector[hash_word(word) % _DIM] += 1.0
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector] if norm else None


def hash_word(word: str) -> int:
    """Устойчивый хеш: встроенный hash() солится при каждом запуске."""
    value = 0
    for char in word:
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return value


def make_memory() -> Memory:
    settings = MemorySettings()
    # ПОРОГ ТЕМЫ ПОНИЖЕН ПОД СЛАБЫЙ КОДИРОВЩИК. Мешок слов даёт двум
    # фразам про виолончель косинус около трети — они делят одно слово из
    # шести, — а умолчание 0.8 рассчитано на модель, которая понимает, что
    # обе фразы про одно. Проверяется извлечение темы, не величина порога.
    settings.contradiction_topic_threshold = 0.3
    return Memory(":memory:", settings=settings, encoder=word_bag_encoder)

WHO = ["курьер", "сосед", "мастер", "коллега", "администратор", "водитель"]
DID = ["перезвонил", "перенёс встречу", "прислал счёт",
       "предупредил заранее", "опоздал на час", "оставил записку"]
WHEN = ["во вторник", "к вечеру", "перед обедом",
        "на прошлой неделе", "рано утром", "в конце месяца"]

# ВОСЕМЬ, А НЕ ПЯТЬ. Обобщение включается, когда похожих набралось больше
# предела разделения образов (два) и не меньше порога темы (три). На пяти
# фразах счёт доходит едва до двух, и кора молчит — проверка падала не
# из-за отбора темы, а оттого что до отбора дело не доходило.
THEME_LINES = [
    "виолончель стоит в углу под чехлом",
    "я бросил виолончель после музыкальной школы",
    "смычок для виолончели пришлось менять дважды",
    "соседи жаловались на виолончель по вечерам",
    "виолончель досталась мне от бабушки",
    "виолончель настраивали дольше обычного",
    "виолончель увезли на дачу прошлым летом",
    "виолончель я протираю раз в неделю",
]


def live(memory, lines, clock=0.0):
    for line in lines:
        memory.observe(line, timestamp=clock)
        clock += 3600.0
    return clock


def test_boilerplate_does_not_become_a_theme():
    """
    Канцелярит повторяется чаще темы — и не должен становиться фактом.

    Наполнитель здесь ровно тот, на котором замер поймал беду: шесть
    оборотов, перемноженных между собой, дают слова с весом 1.000 при
    теме на 0.264.
    """
    memory = make_memory()
    rng = random.Random(11)
    clock = 0.0
    # Сначала долгий поток канцелярита: его слова становятся привычными.
    for _ in range(200):
        line = f"{rng.choice(WHO)} {rng.choice(DID)} {rng.choice(WHEN)}"
        memory.observe(line, timestamp=clock)
        clock += 3600.0

    memory.sleep(timestamp=clock)
    facts = [f["text"] for f in memory.graph.gate.semantic.facts(limit=50)]
    for junk in ("утром", "обедом", "счёт", "встречу"):
        assert not any(junk in f for f in facts), (
            f"канцелярит стал темой: {facts}"
        )
    memory.close()


def test_a_rare_word_still_becomes_a_theme():
    """
    Отсев канцелярита не должен заодно убить настоящие темы.

    Проверка парная к предыдущей: без неё «починка» сводилась бы к тому,
    чтобы не выводить тем вообще.
    """
    memory = make_memory()
    rng = random.Random(12)
    clock = live(memory, THEME_LINES)
    for _ in range(60):
        line = f"{rng.choice(WHO)} {rng.choice(DID)} {rng.choice(WHEN)}"
        memory.observe(line, timestamp=clock)
        clock += 3600.0
    # Тема повторяется ещё раз — теперь похожих накопилось довольно.
    clock = live(memory, THEME_LINES, clock)
    memory.sleep(timestamp=clock)

    facts = [f["text"] for f in memory.graph.gate.semantic.facts(limit=50)]
    assert any("виолончел" in f for f in facts), (
        f"редкая тема не выведена: {facts}"
    )
    memory.close()


def test_theme_keeps_the_word_order_of_the_text():
    """
    Тема собиралась из МНОЖЕСТВА, и в базу оседали перевёртыши:
    «обедом перед», «записку неделе оставил прошлой». Читать такое
    нельзя, а выдаётся оно в поиске наравне с эпизодами.
    """
    memory = make_memory()
    clock = live(memory, [
        "сегодня подписали трудовой договор с новым подрядчиком",
        "трудовой договор ушёл юристу на проверку вчера",
        "трудовой договор переделали из-за одной формулировки",
        "трудовой договор наконец вернулся подписанным",
        "трудовой договор лежит в верхнем ящике стола",
        "трудовой договор скопировали для бухгалтерии",
        "трудовой договор перечитали ещё раз внимательно",
        "трудовой договор подшили в общую папку",
    ])
    memory.sleep(timestamp=clock)
    facts = [f["text"] for f in memory.graph.gate.semantic.facts(limit=50)]
    reversed_order = [f for f in facts if "договор трудовой" in f]
    assert not reversed_order, f"слова темы переставлены: {facts}"
    memory.close()


def test_review_keeps_a_theme_that_has_one_rare_word():
    """
    Снимаем тему, только когда ВСЕ её слова примелькались.

    Одного редкого слова довольно: «трудовой договор» держится на
    «договоре», даже когда «трудовой» встречается всюду. Без этого условия
    пересмотр вычистил бы и настоящие темы вместе с оборотами.
    """
    memory = make_memory()
    # Тема подаётся ДВАЖДЫ: обобщение включается, когда похожих накопилось
    # довольно, и одного круга из восьми фраз для этого не хватает.
    clock = live(memory, THEME_LINES)
    clock = live(memory, THEME_LINES, clock)
    memory.sleep(timestamp=clock)
    facts = [f["text"] for f in memory.graph.gate.semantic.facts(limit=50)]
    assert any("виолончел" in f for f in facts), (
        f"пересмотр снял настоящую тему: {facts}"
    )
    memory.close()


def test_familiarity_ceiling_is_a_stated_setting():
    """Порог назван и подписан замером — менять его молча нельзя."""
    assert MemorySettings().cortex_theme_max_familiarity == pytest.approx(0.7)
