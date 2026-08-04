"""
================================================================================
 TOOLS/BENCH_LONGMEMEVAL.PY — Внешний бенчмарк: recall@k на LongMemEval
================================================================================
Все числа проекта до сих пор получены на СВОЁМ корпусе. Это честно
измерено, но стоит ровно столько, сколько доверия к автору корпуса.
LongMemEval — внешний набор: 500 вопросов, у каждого «стог сена» из
сессий переписки с датами, внутри которого спрятаны реплики-улики.

МЕРЯЕТСЯ RECALL@K, а не правильность ответа. Так дешевле (LLM не
нужна вовсе) и так сопоставимо с тем, что публикуют соседи: Dhee
отчитывается R@1/R@3/R@5/R@10 на этом же наборе.

ЧЕГО ЖДАТЬ ЧЕСТНО. Бенчмарк меряет ПОЛНОТУ извлечения из стога — то
есть ровно тот режим, где мы уже измерили проигрыш случайной выборке
(87.2% против 88.8%, tools/compare_memory.py). Вдобавок наша память
забывает, а спайк-гейт часть стога вообще не записывает. Поэтому голое
число здесь бессмысленно, и стенд считает ТРИ конфигурации:

    lexical   без кодировщика — то, что получит человек после
              обычного pip install
    semantic  с кодировщиком (английская модель на английском наборе)
    archive   семантика + забывание выключено + пишем всё
              — это «мы как чистый поисковик»

Разница между semantic и archive и есть ЦЕНА нашей политики забывания,
выраженная числом. Без неё слабый общий результат нечем объяснить.

ORACLE ПРОТИВ S. В варианте oracle стог состоит ТОЛЬКО из сессий-улик,
поэтому recall там завышен и годится лишь для отладки. Сопоставимое с
соседями число даёт longmemeval_s, где улики спрятаны среди ~40 сессий.

Запуск:
    python tools/bench_longmemeval.py --limit 50
    python tools/bench_longmemeval.py --data storage/bench/longmemeval_s.json
    python tools/bench_longmemeval.py --mode archive --encoder potion
================================================================================
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--logs" not in sys.argv:
    os.environ["LOG_LEVEL"] = "ERROR"

DEFAULT_DATA = "storage/bench/longmemeval_oracle.json"
KS = (1, 3, 5, 10)


def parse_date(value: str) -> float:
    """Дата сессии -> unix-время. Непарсимое — ноль, счёт всё равно относительный."""
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except (ValueError, TypeError):
            continue
    return 0.0


def build_encoder(kind: str):
    """Кодировщик для стенда. None означает «только совпадение слов»."""
    if kind == "none":
        return lambda text: None
    if kind == "bare":
        # ГОЛАЯ УСТАНОВКА: `pip install selective-memory` без дополнений.
        # Ни model2vec, ни navec нет, скачивать нечего — организм
        # отращивает восприятие сам. Это и есть заявка «работает из
        # коробки», и её надо мерить, а не предполагать.
        from selectivemem import embeddings
        embeddings.is_available = lambda: False
        embeddings.encode = lambda text: None
        return None
    if kind == "builtin":
        return None                      # potion-base-8M, как в пакете
    if kind == "potion":
        from model2vec import StaticModel
        model = StaticModel.from_pretrained("minishlab/potion-base-8M")
        return lambda text: model.encode([text])[0]
    raise SystemExit(f"неизвестный кодировщик: {kind}")


def run_instance(instance: Dict, encoder, mode: str, settings_kwargs: Dict,
                 associations: bool = False) -> Optional[Dict]:
    """
    Прогоняет один вопрос: скармливает стог, спрашивает, считает попадания.

    Возвращает словарь с флагами попадания для каждого k, либо None, если
    у инстанции нет размеченных улик.
    """
    from selectivemem import Memory, MemorySettings

    evidence_sessions = set(instance.get("answer_session_ids") or [])
    if not evidence_sessions:
        return None

    sessions = instance["haystack_sessions"]
    session_ids = instance["haystack_session_ids"]
    dates = instance.get("haystack_dates") or []

    memory = Memory(":memory:", settings=MemorySettings(**settings_kwargs), encoder=encoder)
    node_session: Dict[int, str] = {}

    for index, session in enumerate(sessions):
        session_id = session_ids[index]
        when = parse_date(dates[index]) if index < len(dates) else float(index * 86400)

        # Реплики идут парами «пользователь -> ассистент»: узел памяти и
        # хранит обе половины, как в обычной работе библиотеки.
        pending_user = None
        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if turn.get("role") == "user":
                pending_user = content
                continue

            if pending_user is None:
                continue
            if mode == "archive":
                # «Мы как чистый поисковик»: гейт обойдён, пишем всё.
                node_id = memory.graph.save_connection(
                    context=pending_user, response=content, weight=0.6, timestamp=when,
                )
            else:
                node_id = memory.observe(
                    pending_user, response=content, timestamp=when,
                ).node_id
            if node_id is not None:
                node_session[node_id] = session_id
            pending_user = None

    # Вопрос задаётся ПОСЛЕ всего стога, датой вопроса
    asked_at = parse_date(instance.get("question_date", "")) or (
        max((parse_date(d) for d in dates), default=0.0) + 86400
    )
    if mode != "archive":
        memory.forget(now=asked_at)

    found = memory.recall(
        instance["question"], top_k=max(KS), timestamp=asked_at,
        # Растекание активации ВЫКЛЮЧЕНО по умолчанию — так стенд жил с
        # самого начала, и это оказалось слепой зоной: ни один из пяти
        # стендов проекта его не включал, то есть механизм с 1348
        # срабатываниями за разговор ни разу не был измерен.
        with_associations=associations,
    )
    ranked_sessions = [node_session.get(m.id) for m in found]

    result = {
        "type": instance["question_type"],
        "stored": len(node_session),
        "turns": sum(len(s) for s in sessions),
    }
    for k in KS:
        result[f"r@{k}"] = any(s in evidence_sessions for s in ranked_sessions[:k])
    memory.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval: recall@k")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=0, help="0 = все инстансы")
    parser.add_argument("--mode", choices=("normal", "archive"), default="normal")
    parser.add_argument("--encoder", choices=("none", "bare", "builtin", "potion"), default="none")
    parser.add_argument("--threshold", type=float, default=None,
                        help="порог поиска; по умолчанию из настроек")
    parser.add_argument("--content-tokens", type=int, default=None,
                        help="surprise_full_content_tokens: при скольких словах "
                             "реплика удивляет в полную силу")
    parser.add_argument("--plasticity", type=float, default=None,
                        help="порог ЗАПИСИ. Задаётся здесь, а не переменной "
                             "окружения: стенд строит Memory на MemorySettings, "
                             "а переменные идут в config — я дважды намерил "
                             "этим пустоту, прежде чем заметить")
    parser.add_argument("--cons-strength", type=float, default=None,
                        help="во сколько раз слабее рождается свёрнутый эпизод")
    parser.add_argument("--stm", type=int, default=None,
                        help="ёмкость кратковременного буфера; у человека ~4 чанка")
    parser.add_argument("--gate-gain", type=float, default=None,
                        help="эмоция УМНОЖАЕТ новизну вместо среднего")
    parser.add_argument("--intrinsic", action="store_true",
                        help="значимость события из внутренней среды организма")
    parser.add_argument("--interference", action="store_true",
                        help="модель интерференции: важность = доля накопленной "
                             "силы, часы на неё не влияют")
    parser.add_argument("--consolidate", action="store_true",
                        help="включить консолидацию эпизодов в фасаде")
    parser.add_argument("--associate", type=int, default=None,
                        help="со сколькими вспомненными связывать запись; "
                             "БЕЗ ЭТОГО растекание бессмысленно — рёбер не будет")
    parser.add_argument("--associations", action="store_true",
                        help="включить растекание активации при поиске")
    parser.add_argument("--capacity", type=int, default=None,
                        help="сколько воспоминаний держать; 0 = без предела")
    parser.add_argument("--keep-all", action="store_true",
                        help="не удалять по возрасту (delete_on_decay=False)")
    parser.add_argument("--imp-weight", type=float, default=None,
                        help="вклад веса узла в ВАЖНОСТЬ (не в релевантность)")
    parser.add_argument("--imp-connectivity", type=float, default=None,
                        help="вклад связности в важность")
    parser.add_argument("--imp-self-ref", type=float, default=None,
                        help="вклад самореференции в важность")
    parser.add_argument("--imp-use", type=float, default=None,
                        help="вклад использования (стабильности) в важность")
    parser.add_argument("--rerank-band", type=float, default=None,
                        help="полоса переупорядочивания по важности; 0 выключает")
    parser.add_argument("--spike-factor", type=float, default=None,
                        help="пол = сила спайка * этот множитель")
    parser.add_argument("--floor-base", type=float, default=None,
                        help="пол угасания для неподкреплённых узлов")
    parser.add_argument("--weight-influence", type=float, default=None,
                        help="вклад веса узла в оценку поиска (по умолчанию 0.15). "
                             "Вес падает от старости, поэтому это слагаемое "
                             "работает как приоритет свежести")
    parser.add_argument("--no-decay", action="store_true",
                        help="гейт работает, забывание выключено")
    parser.add_argument("--type", default=None,
                        help="мерить только этот тип вопроса")
    parser.add_argument("--topic-threshold", type=float, default=None,
                        help="порог темы для вытеснения; >1.0 выключает его")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="перемешать с этим сидом: набор отсортирован по типам")
    parser.add_argument("--logs", action="store_true")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        sys.exit(
            f"Нет файла {path}. Скачать:\n"
            "  curl -sL https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
            "/resolve/main/longmemeval_oracle.json -o storage/bench/longmemeval_oracle.json"
        )

    data = json.loads(path.read_text())
    if args.shuffle:
        # Набор отсортирован по типам вопросов: первые сорок инстансов —
        # сплошь temporal-reasoning. Без перемешивания срез по --limit
        # меряет одну способность и выдаёт её за общую.
        import random
        random.Random(args.shuffle).shuffle(data)
    if args.type:
        data = [i for i in data if i["question_type"] == args.type]
    if args.limit:
        data = data[: args.limit]

    encoder = build_encoder(args.encoder)
    settings_kwargs = {}
    if args.intrinsic:
        settings_kwargs["intrinsic_emotion"] = True
    if args.threshold is not None:
        settings_kwargs["memory_search_threshold"] = args.threshold
    if args.plasticity is not None:
        settings_kwargs["base_plasticity_threshold"] = args.plasticity
    if args.content_tokens is not None:
        settings_kwargs["surprise_full_content_tokens"] = args.content_tokens
    if args.cons_strength is not None:
        settings_kwargs["consolidated_strength_factor"] = args.cons_strength
    if args.stm is not None:
        settings_kwargs["stm_capacity"] = args.stm
    if args.gate_gain is not None:
        settings_kwargs["gate_emotion_gain"] = args.gate_gain
    if args.interference:
        settings_kwargs["use_relative_strength"] = True
    if args.consolidate:
        settings_kwargs["consolidate_from_stm"] = True
    if args.associate is not None:
        settings_kwargs["associate_recalled_limit"] = args.associate
    if args.capacity is not None:
        settings_kwargs["memory_capacity"] = args.capacity
    if args.keep_all:
        settings_kwargs["delete_on_decay"] = False
    for flag, field in (("imp_weight", "importance_weight_signal"),
                        ("imp_connectivity", "importance_connectivity"),
                        ("imp_self_ref", "importance_self_reference"),
                        ("imp_use", "importance_use")):
        value = getattr(args, flag)
        if value is not None:
            settings_kwargs[field] = value
    if args.rerank_band is not None:
        settings_kwargs["rerank_band"] = args.rerank_band
    if args.spike_factor is not None:
        settings_kwargs["memory_floor_spike_factor"] = args.spike_factor
    if args.floor_base is not None:
        settings_kwargs["memory_floor_base"] = args.floor_base
    if args.weight_influence is not None:
        settings_kwargs["memory_weight_influence"] = args.weight_influence
    if args.topic_threshold is not None:
        # Порог темы для вытеснения устаревшего. Значение выше 1.0 делает
        # вытеснение невозможным (косинус не превышает единицу) — это
        # способ померить, помогает механизм или мешает, не трогая код.
        settings_kwargs["contradiction_topic_threshold"] = args.topic_threshold
    if args.mode == "archive":
        # Забывание практически выключено: узлы живут тысячелетия.
        settings_kwargs.update({"age_t0": 1e12, "decay_rate": 1e-9})
    if args.no_decay:
        # Гейт РАБОТАЕТ, забывание выключено. Разница между этим режимом и
        # обычным — цена одного лишь угасания, без вклада отбора на входе.
        # Archive смешивает две причины и не даёт их разделить.
        settings_kwargs.update({"age_t0": 1e12, "decay_rate": 1e-9})

    totals = {f"r@{k}": 0 for k in KS}
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stored = turns = counted = 0

    for index, instance in enumerate(data, start=1):
        outcome = run_instance(instance, encoder, args.mode, settings_kwargs,
                               associations=args.associations)
        if outcome is None:
            continue
        counted += 1
        stored += outcome["stored"]
        turns += outcome["turns"]
        for k in KS:
            totals[f"r@{k}"] += outcome[f"r@{k}"]
            by_type[outcome["type"]][f"r@{k}"] += outcome[f"r@{k}"]
        by_type[outcome["type"]]["n"] += 1
        if index % 25 == 0:
            print(f"  ...{index}/{len(data)}", file=sys.stderr)

    print("=" * 78)
    print(f" LONGMEMEVAL — {path.name}, режим {args.mode}, кодировщик {args.encoder}")
    print("=" * 78)
    print(f" Вопросов: {counted}  |  реплик в стогах: {turns}  |  записано узлов: {stored}")
    if turns:
        print(f" Записано {stored / turns:.1%} реплик — остальное отсеял гейт")
    print("-" * 78)
    print("  " + "  ".join(f"R@{k}" for k in KS))
    print("  " + "  ".join(f"{totals[f'r@{k}'] / max(1, counted):>4.1%}" for k in KS))
    print("-" * 78)
    print(f" {'тип вопроса':<28} {'n':>4}  " + "  ".join(f"R@{k}" for k in KS))
    for question_type in sorted(by_type):
        row = by_type[question_type]
        n = row["n"]
        cells = "  ".join(f"{row[f'r@{k}'] / max(1, n):>4.0%}" for k in KS)
        print(f" {question_type:<28} {n:>4}  {cells}")
    print("=" * 78)


if __name__ == "__main__":
    main()
