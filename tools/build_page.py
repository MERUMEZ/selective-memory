"""
================================================================================
 TOOLS/BUILD_PAGE.PY — Страница проекта: всё, что иначе читается в терминале
================================================================================
Собирает самодостаточный HTML: что это, живая демонстрация, числа и как их
проверить, что измерено и отвергнуто, и полные документы целиком.

ЗАЧЕМ. Чтобы посмотреть проект, не запуская ничего. Открыть ссылку —
и увидеть решение ворот по каждой реплике, разницу в полтораста раз между
вспоминаемым и забытым, таблицу отвергнутых механизмов и весь аудит.

ПОЧЕМУ СБОРКОЙ, А НЕ РУКАМИ. Документы меняются, и страница, написанная
однажды, разойдётся с ними молча. Мы это уже проходили: числа облегчённого
набора простояли в README как заявка о качестве, пока прогон на полном не
показал разницу в тридцать пунктов. Здесь всё берётся из файлов.

ЗАВИСИМОСТЕЙ НЕТ, и преобразователь markdown написан здесь же. Он
намеренно минимален: заголовки, списки, таблицы, код, ссылки, выделение.
Ровно то, что встречается в наших документах, и ни строкой больше.

Запуск:
    python tools/build_page.py            # соберёт docs/index.html
    python tools/build_page.py --no-demo  # без прогона демонстрации
================================================================================
"""

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# МИНИМАЛЬНЫЙ MARKDOWN
# ---------------------------------------------------------------------------
def _inline(text: str) -> str:
    """Разметка внутри строки. Экранирование ПЕРВЫМ, иначе оно съест теги."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _table(rows):
    """Таблица markdown. Строка-разделитель уже отброшена вызывающим."""
    out = ["<table>"]
    for index, row in enumerate(rows):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        tag = "th" if index == 0 else "td"
        out.append("<tr>" + "".join(
            f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def markdown(source: str) -> str:
    """Достаточно для наших документов и ни строкой больше."""
    lines = source.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue

        if re.match(r"^\s*\|.*\|\s*$", line):
            rows = []
            while i < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                if not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i]):
                    rows.append(lines[i])
                i += 1
            out.append(_table(rows))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            anchor = re.sub(r"[^\w\-]+", "-", text.lower()).strip("-")[:60]
            out.append(f'<h{level} id="{anchor}">{_inline(text)}</h{level}>')
            i += 1
            continue

        if re.match(r"^\s*(---|===)\s*$", line):
            out.append("<hr>")
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            tag = "ol" if ordered else "ul"
            items = []
            pattern = r"^\s*\d+\.\s+" if ordered else r"^\s*[-*]\s+"
            while i < len(lines) and re.match(pattern, lines[i]):
                items.append(re.sub(pattern, "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(
                f"<li>{_inline(t)}</li>" for t in items) + f"</{tag}>")
            continue

        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(quote)) + "</blockquote>")
            continue

        if not line.strip():
            i += 1
            continue

        para = []
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(("#", "```", ">", "|"))
               and not re.match(r"^\s*[-*]\s+|^\s*\d+\.\s+|^\s*---\s*$", lines[i])):
            para.append(lines[i])
            i += 1
        out.append("<p>" + _inline(" ".join(para)) + "</p>")

    return "\n".join(out)


# ---------------------------------------------------------------------------
def run_demo() -> str:
    """Настоящий вывод демонстрации, снятый при сборке."""
    venv = ROOT / "venv" / "bin" / "python"
    python = str(venv) if venv.exists() else sys.executable
    try:
        result = subprocess.run(
            [python, str(ROOT / "tools" / "demo.py")],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT),
        )
    except Exception as error:  # noqa: BLE001
        return f"(демонстрацию запустить не удалось: {error})"
    # Загрузка модели печатает полосу прогресса — на странице она мусор.
    noise = ("Fetching", "Download complete", "it/s", "Reconstruction")
    return "\n".join(
        line for line in result.stdout.splitlines()
        if not any(mark in line for mark in noise)
    )


DOCUMENTS = [
    ("README.md", "README — с чего начинают", "en"),
    ("ARCHITECTURE.ru.md", "Как работает движок, шаг за шагом", "ru"),
    ("AUDIT.ru.md", "Аудит: как это мерилось", "ru"),
    ("ARCHITECTURE.md", "Architecture (English)", "en"),
    ("AUDIT.md", "Audit (English)", "en"),
    ("COMMERCIAL.ru.md", "Коммерческая лицензия", "ru"),
]

def build(demo_text: str) -> str:
    sections = []
    for name, title, lang in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            continue
        body = markdown(path.read_text(encoding="utf-8"))
        sections.append(
            f'<details class="doc"><summary>{html.escape(title)} '
            f'<span class="dim">· {name}</span></summary>'
            f'<div class="doc-body">{body}</div></details>'
        )
    return "\n".join(sections), demo_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать страницу проекта")
    parser.add_argument("--no-demo", action="store_true")
    parser.add_argument("-o", "--output", default="docs/index.html")
    args = parser.parse_args()

    demo_text = "(демонстрация не запускалась)" if args.no_demo else run_demo()
    docs_html, demo_text = build(demo_text)

    template = (ROOT / "tools" / "page_template.html").read_text(encoding="utf-8")
    page = template.replace("{{DEMO}}", html.escape(demo_text))
    page = page.replace("{{DOCUMENTS}}", docs_html)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"  страница: {out}  ({len(page) // 1024} КБ)")


if __name__ == "__main__":
    main()
