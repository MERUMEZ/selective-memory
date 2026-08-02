"""
================================================================================
 TEST_VERSION.PY — Номер версии должен быть записан ровно один раз
================================================================================
Номер жил в ДВУХ местах: `version` в pyproject.toml и `__version__` в
selectivemem/__init__.py. Совпадать они обязаны, но не сверялись ничем.

Разъехались бы они молча. Опасность не в самом расхождении, а в том, где
оно всплывает: у чужого человека, который прислал отчёт об ошибке. Пакет
из индекса называет себя 0.2.0, `__version__` внутри него говорит 0.1.0 —
и отладка идёт по неверной версии исходников.

Поэтому pyproject.toml берёт номер из пакета, а здесь проверяется, что
второй копии не завелось снова. Соблазн вернуть её реален: хардкод
короче на две строки и выглядит проще.
================================================================================
"""

import pathlib
import re
import tomllib

import selectivemem

PYPROJECT = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def test_version_is_not_hardcoded_in_pyproject():
    """Второй копии номера быть не должно."""
    project = _pyproject()["project"]
    assert "version" not in project, (
        "Версия снова записана в pyproject.toml. Теперь их две, и сверять "
        "их некому — уберите поле, номер берётся из selectivemem/__init__.py"
    )
    assert "version" in project.get("dynamic", [])


def test_pyproject_points_at_the_package():
    """Источник номера — тот самый атрибут, а не однофамилец."""
    dynamic = _pyproject()["tool"]["setuptools"]["dynamic"]
    assert dynamic["version"] == {"attr": "selectivemem.__version__"}


def test_version_stays_a_plain_literal():
    """
    setuptools достаёт значение РАЗБОРОМ СИНТАКСИСА, не импортируя пакет.
    Вычисляемый номер (из importlib.metadata, из файла, из git) он прочесть
    не сможет, и сборка упадёт — но не здесь, а на публикации.
    """
    source = pathlib.Path(selectivemem.__file__).read_text(encoding="utf-8")
    assignments = re.findall(r'^__version__\s*=\s*(.+)$', source, re.MULTILINE)
    assert len(assignments) == 1, "__version__ присвоен не один раз"
    assert re.fullmatch(r'"[^"]+"|\'[^\']+\'', assignments[0].strip()), (
        f"__version__ должен быть строковым литералом, а не {assignments[0]!r}: "
        "иначе setuptools не прочтёт его при сборке"
    )


def test_version_looks_like_a_release_number():
    assert re.fullmatch(r'\d+\.\d+\.\d+([.\-+].+)?', selectivemem.__version__), (
        f"непонятный номер версии: {selectivemem.__version__!r}"
    )
