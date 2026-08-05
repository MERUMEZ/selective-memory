"""
================================================================================
 EXAMPLES/ASSISTANT.PY — Ассистент с памятью, в шестьдесят строк
================================================================================
Показывает единственное, что нужно знать для встраивания: ПОРЯДОК ВЫЗОВОВ.

    1. recall / context_for  — достать из памяти то, чем отвечать
    2. запрос к языковой модели с этим контекстом
    3. observe                — показать памяти, что произошло

Порядок именно такой, и он не косметический. Связи между воспоминаниями
рождаются из того, что было ВЫНУТО ИЗ ПАМЯТИ незадолго до записи: если
сначала писать, а доставать потом, ассоциативной сети не возникает вовсе.
Замер на LongMemEval: живой порядок даёт R@1 97.4% против 96.2% при той же
избирательности.

ЗАВИСИМОСТЕЙ НЕТ. Запрос к модели идёт через urllib из стандартной
библиотеки — как и сама память, пример ничего не тянет за собой.

Работает с любым OpenAI-совместимым интерфейсом:

    export ASSISTANT_API_KEY=sk-...
    export ASSISTANT_API_BASE=https://api.openai.com/v1     # по умолчанию
    export ASSISTANT_MODEL=gpt-4o-mini
    python examples/assistant.py

Для OpenRouter: ASSISTANT_API_BASE=https://openrouter.ai/api/v1

Без ключа пример всё равно запустится и покажет работу памяти — вместо
ответа модели будет заглушка. Память при этом настоящая.
================================================================================
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selectivemem import Memory

API_KEY = os.environ.get("ASSISTANT_API_KEY", "")
API_BASE = os.environ.get("ASSISTANT_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("ASSISTANT_MODEL", "gpt-4o-mini")

SYSTEM = (
    "Ты ассистент с долговременной памятью. Ниже — то, что ты помнишь об "
    "этом человеке. Если помнишь что-то относящееся к вопросу, опирайся на "
    "это и не переспрашивай. Если в памяти пусто, отвечай как обычно и не "
    "выдумывай воспоминаний."
)


def ask_model(context: str, question: str) -> str:
    """Запрос к модели. Без ключа — честная заглушка, а не выдумка."""
    if not API_KEY:
        return "(ключа нет — отвечает заглушка; память при этом настоящая)"

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": SYSTEM + ("\n\nПамять:\n" + context if context else
                                  "\n\nПамять пуста.")},
            {"role": "user", "content": question},
        ],
    }).encode()

    request = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
        return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as error:
        return f"(модель ответила ошибкой {error.code})"
    except Exception as error:  # noqa: BLE001
        return f"(не дозвонились до модели: {error})"


def main() -> None:
    memory = Memory("assistant.db")
    print("=" * 70)
    print(" АССИСТЕНТ С ПАМЯТЬЮ")
    print("=" * 70)
    print(f" {memory.describe_setup()}")
    print(" Команды: /память  /состояние  /забыть  /выход")
    print("-" * 70)

    while True:
        try:
            question = input("\nвы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/выход":
            break
        if question == "/состояние":
            print("  ", memory.feel().describe())
            continue
        if question == "/забыть":
            # Время передаётся снаружи, поэтому «прошёл месяц» — это просто
            # метка времени, а не ожидание месяца.
            import time
            memory.forget(now=time.time() + 30 * 24 * 3600)
            print("   прошёл месяц: слабое поблёкло, вспоминаемое осталось")
            continue

        # 1. ДОСТАТЬ. Пустая строка — законный ответ «мне нечего добавить»,
        #    и обрабатывать её надо: память, которая всегда что-то
        #    подмешивает, подмешивает шум.
        context = memory.context_for(question, top_k=3)
        if question == "/память":
            print("  ", context or "— пусто")
            continue

        # 2. ОТВЕТИТЬ.
        answer = ask_model(context, question)
        print(f"\nассистент: {answer}")

        # 3. ПОКАЗАТЬ ПАМЯТИ. Она сама решит, стоит ли это хранить.
        result = memory.observe(question, response=answer)
        mark = "записано" if result.node_id else "не записано"
        extra = " (поправка: старая версия ослаблена)" if result.superseded_ids else ""
        print(f"   [{mark}, новизна {result.surprise:.2f}]{extra}")
        if context:
            print(f"   [в ответ подмешано из памяти: {len(context.splitlines())} шт.]")

    memory.close()
    print("\nпамять сохранена в assistant.db — при следующем запуске всё на месте")


if __name__ == "__main__":
    main()
