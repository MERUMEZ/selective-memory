"""
Корневой conftest.py.

ДВЕ ЗАДАЧИ, И ВТОРАЯ ПОЯВИЛАСЬ ДОРОГОЙ ЦЕНОЙ.

1. ПУТЬ ИМПОРТА. Пустой conftest в корне заставляет pytest вставить
   корень проекта в sys.path, чтобы тесты писали `from
   selectivemem.database import Database` без настройки PYTHONPATH.

2. ИЗОЛЯЦИЯ ГЛОБАЛЬНОГО СОСТОЯНИЯ embeddings — см. ниже.

ЧТО СЛУЧИЛОСЬ. `selectivemem/embeddings.py` держит настройки и загруженную
модель в переменных модуля (`_settings`, `_model`). Переконфигурирует их
`core/brain_session.py` вызовом `embeddings.configure(...)` — то есть код
ВИТРИНЫ, а тесты витрины лежат в .gitignore и в репозиторий не попадают.

Отсюда расхождение, которое пряталось месяцами:

    на машине автора  33 тестовых файла, витрина настраивает embeddings
                      под себя, следующие библиотечные тесты наследуют
                      эту настройку           -> 345 passed
    на GitHub         этих файлов нет, те же тесты идут с умолчанием
                      potion-base-8M          -> 11 failed

Проверяется прямо: тест, проходящий в полном наборе, ПАДАЕТ запущенный в
одиночку. Исход решал порядок, а не код.

Фикстура ниже возвращает модулю умолчания перед каждым тестом и после
него. Тест, которому нужна своя настройка, ставит её сам и больше никому
её не оставляет.
"""

import pytest

from selectivemem import embeddings
from selectivemem.settings import MemorySettings


@pytest.fixture(autouse=True)
def _isolate_embeddings():
    """Умолчания до и после каждого теста — исход не зависит от порядка."""
    def reset():
        previous = embeddings._settings.embedding_model_path
        embeddings._settings = MemorySettings()
        # Модель сбрасывается ТОЛЬКО при смене пути: она грузится с диска,
        # и обнулять её на каждом тесте значило бы платить загрузкой сотню
        # раз за прогон.
        if previous != embeddings._settings.embedding_model_path:
            embeddings._model = None

    reset()
    yield
    reset()


def handles_russian() -> bool:
    """
    Различает ли ДЕЙСТВУЮЩАЯ модель русские слова по смыслу.

    Проверка возможности, а не наличия. Тексты в части тестов русские, а
    модель по умолчанию английская (potion-base-8M) — сочетание, которое
    библиотека прямо не поддерживает и советует заменить на
    [semantic-ru]. Раньше такие проверки проходили не потому, что модель
    справлялась, а потому, что настройка натекала из витрины.
    """
    if not embeddings.is_available():
        return False
    near = embeddings.cosine(embeddings.encode("кот"), embeddings.encode("кошка"))
    far = embeddings.cosine(embeddings.encode("кот"), embeddings.encode("бетон"))
    return near is not None and far is not None and near > far + 0.1


requires_russian = pytest.mark.skipif(
    not handles_russian(),
    reason="действующая модель не различает русские слова по смыслу; "
           "для русского нужен [semantic-ru]",
)
