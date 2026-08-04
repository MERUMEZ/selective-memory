# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# A commercial licence is available for use in closed products — see
# COMMERCIAL.md.
"""
================================================================================
 EMBEDDINGS.PY — Semantic vectors for meaning-based search
================================================================================
Before this module, memory searched BY LETTERS: keyword overlap plus
SequenceMatcher. Measurement showed where that ends —

    stored "I have a cat" (Russian: "у меня есть кошка"):
        "у меня есть кот"   (I have a cat, masc.)  -> 0.870  found
        "расскажи про кота" (tell me about the cat) -> not found
        "мой питомец мяукает" (my pet meows)        -> not found
        "у меня есть кожа"  (I have skin)           -> 0.884  FOUND BETTER

that is, "skin" beat "cat" because it differs by one letter. There was no
meaning in that search at all.

THE CHOICE OF MODEL was dictated by hardware, not taste.
sentence-transformers pulls in torch: ~2.5 GB on disk and 0.5-1 GB of RAM
per process. This machine has 3.7 GB in total with ~1.8 GB free and three
other services running — such a neighbour would take them down. Both
supported models are static: no torch, tens of megabytes, no GPU.

TWO MODELS, AND THE LANGUAGE DECIDES WHICH. `[semantic]` installs
model2vec/potion-base-8M — 30 MB, fetched on first use, no path to
configure, ENGLISH. `[semantic-ru]` installs navec (natasha project) —
51 MB quantised, ~370 MB of RAM, Russian, and it wants
embedding_model_path pointed at the downloaded file.

Getting this wrong is not a matter of degree. Measured over four related
and four unrelated word pairs:

    English text, potion-base-8M:  related 0.651  unrelated 0.013
    Russian text, potion-base-8M:  related 0.685  unrelated 0.661

On English it separates cleanly (cat/kitten 0.686 against cat/concrete
0.009). On Russian it puts кот/бетон at 0.803 — ABOVE кот/кошка at 0.643.
That is noise, and noise is worse than the lexical fallback, which at
least never invents a match. The docstring below still describes navec
because the phrase-averaging and function-word notes are specific to it.

FUNCTION WORDS ARE DROPPED, and that is not a detail. A phrase vector is
the mean over its words, so "у меня есть" ("I have") drowns out the one
word that carries meaning: without filtering, "skin" stayed at 0.863 and
still nearly matched "cat". After dropping function words it was 0.222
against 0.671 — the order finally became correct.

THOSE NUMBERS ARE NAVEC'S, AND WITH THE DEFAULT ENCODER THE PATHOLOGY IS
BACK. Re-measured on the same phrases:

    navec (Russian)          cat 0.671   skin 0.222   correct
    potion-base-8M (default) cat 0.643   skin 0.714   INVERTED

Filtering still helps — it is the model that cannot tell the words apart
on Russian text at all (see §2.15 of the audit: separation +0.638 in
English against +0.024 in Russian). Read this section as "why filtering
exists", not as "the example works today": with `[semantic]` on Russian
it does not, and that is an argument for `[semantic-ru]`, not against
filtering.

PERSON MARKERS ARE DROPPED TOO, AND THAT WAS CHECKED SEPARATELY. "мой",
"меня", "я" are in the list, so "my holiday" and "holiday" encode
identically while "у попутчика отпуск" keeps its owner — the discriminator
survives on one side and not the other. In personal memory "whose" is the
main thing that distinguishes near-duplicates, so this looked like the
cause of the one question the near-duplicate bench never solves.

MEASURED: RETURNING THEM CHANGES NOTHING — 83.3% R@1 either way. The
hypothesis was reasonable and wrong; the failing case is not fixed by
representing the owner.

TRAINED ON LITERARY FICTION (hudlit), and that matters more than its
size. Professional vocabulary is unrelated in such a model — measured
cosines:

    love ~ prefer                0.580   connected
    daughter ~ child             0.260   weakly
    allergy ~ medicine           0.303   weakly
    language ~ programming       0.114   NOT connected
    language ~ python            0.145   NOT connected

The words are in the vocabulary, but in fiction a "language" is a tongue
and a "python" is a snake. So a memory holding "I prefer Python, not
Java" is not found by "what language do I like", although the node is
intact and is found by "what programming language do I prefer".

For technical, medical, legal and any other professional domain this
model is unsuitable — bring your own encoder there (see
MemoryGraph(encoder=...)). This is not a memory defect: the node is in
place, only a phrasing from another domain fails to reach it.

DEGRADES GENTLY: if the model is missing, the library is not installed or
embeddings are disabled by settings, encode() returns None and
MemoryGraph.search keeps working on the previous string similarity. A bot
must not fall over because an optional 51 MB file is absent.

NUMPY IS OPTIONAL TOO, and that took a clean-environment run to notice.
to_blob/from_blob/cosine used to import numpy unconditionally, so the
very path the README invites people onto — bring your own encoder, keep
the zero-dependency install — died with ModuleNotFoundError. They now
fall back to array("f") and plain arithmetic: slower, present.

LOADS LAZILY: the model comes up on first use rather than at import, so a
process that does not need semantics (tests, benchmarks, the memory
inspector) does not pay for it in RAM.
================================================================================
"""

import math
import threading
from array import array
from typing import List, Optional, Sequence

from selectivemem.settings import MemorySettings
import logging

logger = logging.getLogger(__name__)

# Russian function words: their vectors dominate the mean and erase the
# meaning of a phrase. The set is deliberately wider than STOP_WORDS in
# graph_memory (there it filters noise for keyword search, here it fixes
# vector averaging). It is Russian because the bundled model is — with
# your own encoder this list is irrelevant.
_FUNCTION_WORDS = {
    "и", "в", "во", "на", "с", "со", "по", "к", "у", "из", "за", "от", "до",
    "для", "что", "как", "это", "то", "я", "ты", "он", "она", "мы", "вы",
    "они", "не", "но", "а", "же", "бы", "ли", "или", "тот", "его", "её", "ее",
    "их", "есть", "быть", "мой", "моя", "моё", "мне", "меня", "мной", "твой",
    "тебя", "тебе", "про", "о", "об", "при", "же", "уж", "вот", "так", "там",
    "тут", "ещё", "еще", "уже", "бы", "чтобы", "если", "когда", "где",
    # АНГЛИЙСКИЕ — ИХ ЗДЕСЬ НЕ БЫЛО ВОВСЕ, и это не мелочь: модель по
    # умолчанию английская (potion-base-8M), внешний бенчмарк английский,
    # а фраза уходила в кодировщик со всеми "the", "is", "my" внутри.
    # Русская половина списка при этом чистила фразу исправно — то есть
    # заявленный основной язык обслуживался хуже второстепенного.
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "as", "of", "at", "by", "for", "with", "about", "into", "to", "from",
    "in", "on", "up", "out", "over", "under", "is", "am", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "not", "no",
    "this", "that", "these", "those", "there", "here", "what", "when",
    "where", "who", "how", "why", "which",
}

# Module settings: an application may replace them via configure()
# before first use. The defaults are the library's own.
_settings = MemorySettings()


def configure(settings: MemorySettings) -> None:
    """Sets the embedding settings (model path, whether enabled)."""
    global _settings, _model, _load_failed
    _settings = settings
    _model, _load_failed = None, False


_model = None
_load_failed = False
_lock = threading.Lock()

# Кэш ответа на вопрос «есть ли numpy». _UNSET отличает «ещё не проверяли»
# от «проверили, нету»: без этого отсутствие numpy стоило бы неудачного
# импорта на каждом сравнении векторов.
_UNSET = object()
_NUMPY = _UNSET


def _load_model():
    """
    Brings the model up lazily. Repeated failures are not retried: a
    missing file will not appear on its own, and there is no point raising
    the same error on every message.
    """
    global _model, _load_failed

    if _model is not None or _load_failed:
        return _model

    with _lock:
        if _model is not None or _load_failed:
            return _model

        if not _settings.embeddings_enabled:
            _load_failed = True
            logger.info("[EMBEDDINGS] Disabled by settings — search stays lexical")
            return None

        # NO FILE CONFIGURED — try the model the [semantic] extra installs.
        #
        # Until now this branch simply gave up, so `pip install
        # selective-memory[semantic]` bought the user nothing: they still
        # had to find a model file, download it and write its path into
        # settings, which nobody does. The default path, meanwhile, pointed
        # at the author's own machine — semantics was off for everyone else
        # and said nothing about it.
        #
        # model2vec fetches potion-base-8M on first use (30 MB) and needs
        # no path. OFFLINE DEPLOYMENTS — games, embedded — must either
        # pre-fetch it or pass their own encoder to Memory(encoder=...);
        # the library will not reach for the network behind their back on
        # a machine that has none, it will just stay lexical and say so.
        if not _settings.embedding_model_path:
            try:
                from model2vec import StaticModel
            except ImportError:
                _load_failed = True
                logger.warning(
                    "[EMBEDDINGS] No model configured and model2vec is absent — "
                    "search matches by shared words only. "
                    "pip install selective-memory[semantic] — for Russian text "
                    "take [semantic-ru] instead, the default model is English "
                    "and scores unrelated Russian words as high as related ones"
                )
                return None
            try:
                _model = StaticModel.from_pretrained("minishlab/potion-base-8M")
            except Exception as exc:  # noqa: BLE001
                _load_failed = True
                logger.warning(
                    "[EMBEDDINGS] Could not load potion-base-8M (%s) — "
                    "search stays lexical", exc,
                )
                return None
            logger.info("[EMBEDDINGS] Model loaded: potion-base-8M")
            return _model

        try:
            from navec import Navec
        except ImportError:
            _load_failed = True
            logger.warning(
                "[EMBEDDINGS] navec is not installed — search stays lexical. "
                "pip install selective-memory[semantic-ru]"
            )
            return None

        try:
            _model = Navec.load(_settings.embedding_model_path)
        except Exception as exc:  # noqa: BLE001
            _load_failed = True
            logger.warning(
                "[EMBEDDINGS] Could not load the model (%s): %s — "
                "search stays lexical",
                _settings.embedding_model_path, exc,
            )
            return None

        logger.info("[EMBEDDINGS] Model loaded: %s", _settings.embedding_model_path)
        return _model


def is_available() -> bool:
    """Whether semantic search works (without loading the model as a side effect)."""
    return _load_model() is not None


def _content_words(text: str) -> List[str]:
    """
    The content words of a phrase. Function words are dropped — otherwise
    the mean of the vectors is decided by them rather than by meaning (see
    the module header).

    If no content words remain (a phrase made entirely of function words),
    all words are returned: a weak vector beats no vector.
    """
    words = [w.strip(".,!?;:()\"'«»").lower() for w in text.split()]
    words = [w for w in words if w]
    content = [w for w in words if w not in _FUNCTION_WORDS]
    return content or words


def encode(text: str):
    """
    The meaning vector of a phrase, or None when semantics is unavailable
    or not a single known word was found in the text.

    None is NOT an error: the caller must be able to work without
    semantics (see MemoryGraph.search), because the model is optional.
    """
    if not text or not text.strip():
        return None

    model = _load_model()
    if model is None:
        return None

    import numpy as np

    # ДВА РАЗНЫХ ИНТЕРФЕЙСА. navec ведёт себя как словарь "слово ->
    # вектор", и фраза собирается усреднением слов — оттого и отсев
    # служебных слов ниже. model2vec кодирует строку целиком и делает
    # усреднение сам.
    if hasattr(model, "encode"):
        # ФРАЗА ИДЁТ ЦЕЛИКОМ, БЕЗ ОТСЕВА СЛУЖЕБНЫХ СЛОВ.
        #
        # Здесь стоял тот же отсев, что и для navec, и он молча не работал:
        # в списке служебных слов не было ни одного английского, а модель
        # по умолчанию английская. Когда английские слова в список внесли,
        # отсев заработал — и бенчмарк упал с R@1 96.0% до 93.2%.
        #
        # Так и должно быть. Модель уровня ПРЕДЛОЖЕНИЯ обучена на
        # естественном тексте; выдирая из фразы предлоги и связки, мы
        # подаём ей то, чего она не видела. Отсев осмыслен только для
        # словарной модели ниже, где вектор фразы собирается усреднением и
        # служебные слова его размывают.
        return model.encode([text])[0]

    vectors = [model[w] for w in _content_words(text) if w in model]
    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def _numpy():
    """
    numpy if it is installed, otherwise None.

    Kept lazy on purpose: importing numpy costs about a tenth of a second,
    and a package that promises zero dependencies has no business paying
    that at import time for users who never touch semantics.
    """
    global _NUMPY
    if _NUMPY is _UNSET:
        try:
            import numpy as np
            _NUMPY = np
        except ImportError:
            _NUMPY = None
    return _NUMPY


def cosine(a, b) -> float:
    """Cosine similarity of two vectors; 0.0 for degenerate inputs."""
    if a is None or b is None:
        return 0.0

    np = _numpy()
    if np is not None:
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm <= 0.0:
            return 0.0
        return float(np.dot(a, b) / norm)

    # Pure-Python path. Slower, but the alternative is a crash on the very
    # thing the README invites people to do — plug in their own encoder.
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm = math.sqrt(sum(float(x) * float(x) for x in a)) * \
        math.sqrt(sum(float(y) * float(y) for y in b))
    if norm <= 0.0:
        return 0.0
    return dot / norm


def to_blob(vector) -> Optional[bytes]:
    """Vector -> BLOB for storage in SQLite."""
    if vector is None:
        return None
    np = _numpy()
    if np is not None:
        return np.asarray(vector, dtype=np.float32).tobytes()
    return array("f", (float(x) for x in vector)).tobytes()


def from_blob(blob: Optional[bytes]):
    """
    BLOB from SQLite -> vector. None when the field is absent or empty.

    Without numpy the result is an array("f"), which indexes, iterates and
    reports its length like a sequence of floats — everything the ranking
    code asks of a vector.
    """
    if not blob:
        return None
    np = _numpy()
    if np is not None:
        return np.frombuffer(blob, dtype=np.float32)
    vector = array("f")
    vector.frombytes(blob)
    return vector


def similarity(text: str, blob: Optional[bytes]) -> float:
    """Convenience wrapper: similarity of a text to a node's stored vector."""
    return cosine(encode(text), from_blob(blob))


def describe() -> str:
    """
    Чем эта установка различает смысл — одной строкой.

    ЗАЧЕМ. Семантика здесь необязательна и деградирует ТИХО: нет
    model2vec — поиск молча становится словарным, стоит не тот путь к
    модели — то же самое. Замер показывает разницу между этими режимами в
    три раза (9/16 против 3/16 на английском стенде), а узнать, в каком ты
    режиме, можно было только по логам при подходящем уровне.
    """
    if not _settings.embeddings_enabled:
        return "семантика выключена настройкой (embeddings_enabled=False)"
    if _settings.embedding_model_path:
        model = _load_model()
        if model is None:
            return f"модель не загрузилась: {_settings.embedding_model_path}"
        return f"navec: {_settings.embedding_model_path}"
    try:
        from model2vec import StaticModel  # noqa: F401
    except ImportError:
        return ("model2vec не установлен — организм отрастит восприятие сам "
                "(pip install selective-memory[semantic] даёт готовую модель)")
    return "potion-base-8M" if _load_model() is not None else "potion-base-8M не загрузилась"
