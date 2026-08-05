# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
"""
Память переживает перезапуск.

САМЫЙ ОБЫЧНЫЙ СЦЕНАРИЙ ИЗ ВСЕХ, И ОН БЫЛ СЛОМАН. Приложение закрывает
базу и открывает её при следующем запуске — после разъезда на два
хранилища `nodes` стала ВИДОМ, а `_init_schema` продолжала строить по ней
индекс. SQLite отвечает "views may not be indexed", и библиотека падала
прямо в конструкторе.

Ни один из 327 тестов этого не поймал: все они открывают ":memory:", то
есть всегда СВЕЖУЮ базу, где `nodes` ещё таблица. Поймал живой запуск
примера с ассистентом.

Отсюда правило, которое эти проверки и закрепляют: файловую базу нужно
открывать ДВАЖДЫ, иначе проверяется только создание схемы, а не работа с
ней.
"""

import pytest

from selectivemem import Memory


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "brain.db")


def test_database_reopens_after_close(db_path):
    """Открыть, записать, закрыть, открыть снова — и не упасть."""
    memory = Memory(db_path)
    memory.observe("меня зовут Паша", emotion=1.0)
    memory.close()

    memory = Memory(db_path)          # падало здесь
    assert memory.graph.gate.episodic.count() == 1
    memory.close()


def test_memories_survive_three_sessions(db_path):
    """
    Накопленное не теряется между запусками.

    Проверяется не только «не упало», но и что записи ДОСТУПНЫ: схема
    может пережить открытие и при этом потерять данные.
    """
    memory = Memory(db_path)
    memory.observe("у меня аллергия на пенициллин", emotion=1.0)
    memory.close()

    memory = Memory(db_path)
    memory.observe("я не ем мясо", emotion=1.0)
    memory.close()

    memory = Memory(db_path)
    assert memory.graph.gate.episodic.count() == 2
    found = [m.context for m in memory.recall("аллергия", top_k=3)]
    assert any("пенициллин" in text for text in found), found
    memory.close()


def test_reopening_does_not_multiply_nodes(db_path):
    """
    Открытие само по себе ничего не пишет.

    Миграции при открытии трогают схему, и ошибка в них проявилась бы
    именно так: узлы удвоились бы при каждом запуске, а заметили бы это
    через неделю эксплуатации.
    """
    memory = Memory(db_path)
    for text in ("первое событие про виолончель",
                 "второе событие про пенициллин"):
        memory.observe(text, emotion=1.0)
    before = memory.graph.gate.episodic.count()
    memory.close()

    for _ in range(3):
        memory = Memory(db_path)
        assert memory.graph.gate.episodic.count() == before
        memory.close()
