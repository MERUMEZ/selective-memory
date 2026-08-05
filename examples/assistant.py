"""
================================================================================
 EXAMPLES/ASSISTANT.PY — Ассистент с памятью, в шестьдесят строк
================================================================================
Показывает единственное, что нужно знать для встраивания: ПОРЯДОК ВЫЗОВОВ.

    0. profile_text          — что о человеке известно ВСЕГДА
    1. recall / context_for  — достать то, что подходит к ЭТОМУ вопросу
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

# СЛОВА, КОТОРЫМИ ЧЕЛОВЕК ПРЯМО ПРОСИТ ЗАПОМНИТЬ.
#
# Библиотека обещает: emotion=1.0 значит «пользователь сказал запомни», и
# такая запись проходит мимо порога новизны. Но значимость приходит СНАРУЖИ
# — распознать просьбу обязано приложение, и пример этого не делал.
#
# Живой запуск показал цену: человек написал «запомни», память не записала
# ничего, и на следующей реплике он справедливо возмутился.
REMEMBER_WORDS = ("запомни", "запиши", "не забудь", "remember", "note that")


def looks_like_answer(previous_bot: str, text: str) -> bool:
    """
    Короткая реплика ПОСЛЕ вопроса ассистента — это ответ, и в нём смысл.

    Живой запуск: на вопрос «как зовут пса?» человек ответил «Леви».
    Новизна 0.06 — одно знакомое слово, — и кличка не записалась ни разу.
    То же с «шпиц» и «омлет». Односложный ответ несёт ВЕСЬ смысл обмена и
    при этом не удивляет вовсе.

    ЗДЕСЬ ЭВРИСТИКЕ МЕСТО, А В БИБЛИОТЕКЕ — НЕТ. Приложение знает свой
    разговор: оно само сформировало вопрос и видит, что человек ответил
    коротко. Библиотека видит только текст, и попытка научить её самой
    замечать брешь провалилась замером — поиск возвращал уверенные 0.776
    за неверный ответ и не подозревал, что не ответил.

    Признак грубый: ответ из шести слов не пройдёт, вопрос без знака не
    распознается. Приложение с моделью в цикле спросит у неё и обойдётся
    без этого вовсе — модель там уже стоит.
    """
    return (
        previous_bot.rstrip().endswith("?")
        and len(text.split()) <= 4
        and not text.startswith("/")
    )


SYSTEM = (
    "Ты ассистент с долговременной памятью. Ниже — то, что ты помнишь об "
    "этом человеке. Если помнишь что-то относящееся к вопросу, опирайся на "
    "это и не переспрашивай. Если в памяти пусто, отвечай как обычно и не "
    "выдумывай воспоминаний."
)


def ask_model(profile: str, context: str, question: str) -> str:
    """Запрос к модели. Без ключа — честная заглушка, а не выдумка."""
    if not API_KEY:
        return "(ключа нет — отвечает заглушка; память при этом настоящая)"

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            # ПРОФИЛЬ ИДЁТ В КАЖДЫЙ ЗАПРОС, а найденное по вопросу —
            # отдельно. Это разные вещи: профиль это то, что о человеке
            # известно ВСЕГДА, и держать его надо при себе постоянно.
            # Живой разговор показал, зачем: на «может ещё что-то?»
            # ассистент ответил «больше ничего не знаю» при двадцати
            # записях в памяти — запрос не совпал ни с чем.
            {"role": "system",
             "content": SYSTEM
                        + ("\n\nЧто известно о человеке:\n" + profile
                           if profile else "")
                        + ("\n\nПодходящее к вопросу:\n" + context
                           if context else "")},
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
    previous_answer = ""
    print("=" * 70)
    print(" АССИСТЕНТ С ПАМЯТЬЮ")
    print("=" * 70)
    print(f" {memory.describe_setup()}")
    print(" Команды: /профиль  /память  /состояние  /забыть  /выход")
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
        if question == "/профиль":
            print("  ", memory.profile_text() or "— пока ничего о вас не знаю")
            continue
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
        profile = memory.profile_text(limit=8)
        if question == "/память":
            print("  ", context or "— пусто")
            continue

        # 2. ОТВЕТИТЬ.
        answer = ask_model(profile, context, question)
        print(f"\nассистент: {answer}")

        # 3. ПОКАЗАТЬ ПАМЯТИ. Она сама решит, стоит ли это хранить, —
        #    но значимость обязано сообщить приложение: библиотека видит
        #    текст, а не разговор.
        lowered = question.lower()
        emotion, fills_gap = None, False
        if any(word in lowered for word in REMEMBER_WORDS):
            emotion = 1.0          # прямая просьба: пишем мимо порога
            why = " (сказано «запомни»)"
        elif looks_like_answer(previous_answer, question):
            # Библиотеке говорится ПРЯМО: это ответ на то, что я спросил.
            # Не «повысь значимость до 0.8», а факт о разговоре, которого
            # она знать не может.
            fills_gap = True
            why = " (ответ на вопрос ассистента)"
        else:
            why = ""               # обычная реплика: решает одна новизна

        result = memory.observe(question, response=answer,
                                emotion=emotion, fills_gap=fills_gap)
        previous_answer = answer
        mark = "записано" if result.node_id else "не записано"
        extra = " (поправка: старая версия ослаблена)" if result.superseded_ids else ""
        print(f"   [{mark}, новизна {result.surprise:.2f}]{why}{extra}")
        parts = []
        if profile:
            parts.append(f"профиль {len(profile.splitlines())}")
        if context:
            parts.append(f"по вопросу {len(context.splitlines())}")
        if parts:
            print(f"   [подмешано: {', '.join(parts)}]")

    memory.close()
    print("\nпамять сохранена в assistant.db — при следующем запуске всё на месте")


if __name__ == "__main__":
    main()
