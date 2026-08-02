# selectivemem

**Memory that decides what to forget.**

Your assistant meets the user from scratch every time. Dumping everything
into a vector store means paying to keep noise and drowning in it at
retrieval. `selectivemem` does the opposite: it doesn't store everything,
it fades what goes unused, and it strengthens what turned out to matter.

```python
from selectivemem import Memory

memory = Memory("user_42.db")

memory.observe("I am allergic to penicillin")   # stored
memory.feedback(+1.0)                           # the user marked it important
memory.observe("thanks")                        # not stored, nothing to remember

memory.context_for("what am I allergic to")
# '- I am allergic to penicillin'
```

A month later the allergy is still there; "thanks" and the small talk
about weather are gone. That is not a setting — that is how the mechanism
works.

```
pip install selective-memory              # core, zero dependencies
pip install selective-memory[semantic]    # + meaning-based search
```

86 KB wheel, standard library and sqlite3. The name on PyPI is free; no
release has been published yet.

**If you work in English, read this before anything else.** The bundled
semantic model is Russian, so for English text there is effectively no
meaning-based search at all — matching falls back to shared words. The
example above works because "allergic" appears in both the query and the
note; ask for "allergy" instead and you get nothing.

This is not fine print. Plan on attaching your own encoder from day one
(see [below](#the-encoder-sets-the-language-and-the-domain)); everything
else in the library is language-agnostic, but retrieval is only as good
as the vectors you give it.

---

## Who it is for

| Use case | What you get |
|---|---|
| **Personal assistant** | remembers the user across sessions without inflating the context |
| **Game NPC** | remembers what the player did and forgets trivia. Offline, deterministic, no network |
| **Support bot** | keeps what the customer flagged as important, not the whole transcript |
| **Long-running agent** | context does not grow linearly with uptime |

Not for you if you need **completeness** — finding everything ever said.
Use a vector store for that; the numbers below show why.

---

## Five calls

```python
obs = memory.observe(text, response="", emotion=0.0)   # show an event
memory.feedback(+1.0)                                  # rate the last action
memory.recall("query", top_k=3)                        # find what fits
memory.context_for("query")                            # the same, ready for a prompt
memory.forget()                                        # let time pass
memory.stats()                                         # what is inside
```

`observe` returns not just whether it stored anything, but **why**:

```python
obs.written    # False
obs.reason     # 'routine, 0.130 short of the threshold'
obs.surprise   # 0.340
```

"Why didn't the bot remember that?" has an answer, and it is a number.

### Where `emotion` and `feedback` come from

The core does not infer emotion and knows no words of approval — that is
your application's job:

| Where | `emotion` on observe | `feedback` |
|---|---|---|
| Assistant | 0.0 normally; 1.0 when the user says "remember this" | 👍/👎 button, an explicit "right/wrong" |
| NPC | significance of the game event: death 1.0, picked up a blade of grass 0.0 | quest outcome, faction reaction |
| Support bot | ticket priority | customer's rating of the resolution |

You may pass nothing at all: with `emotion=0.0` writing is driven by
novelty alone, and that is a perfectly good default mode.

### Time comes from outside

```python
now = [1_700_000_000.0]
memory = Memory("brain.db", clock=lambda: now[0])
now[0] += 30 * 86400          # fast-forward a month
memory.forget()
```

Forgetting is computed on that scale. For games this also buys
**determinism**: the same input yields byte-identical results, which is
what replays, saves and QA require.

### The encoder sets the language and the domain

The bundled one is navec, Russian static vectors (51 MB, no torch). For
another language or another domain, pass your own function:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
memory = Memory("brain.db", encoder=lambda text: model.encode(text))
```

### The bundled model is worth replacing, and here is by how much

`tools/probe_semantic.py` holds sixteen pairs of "fact → how someone will
ask about it". Half of the pairs share no word with the fact ("where do I
live" against "we live at 12 Pushkin Street") — those are the ones that
actually test semantics.

| model | hits | RAM | disk |
|---|---|---|---|
| navec, bundled | 8/16 = 50% | 376 MB | 51 MB |
| potion-base-8M | 7/16 = 44% | 351 MB | 30 MB |
| potion-retrieval-32M | 7/16 = 44% | 351 MB | 130 MB |
| **potion-multilingual-128M** | **12/16 = 75%** | 1.6 GB | 1 GB |

Read the bundled model's number together with one fact: **eight pairs are
solvable by plain word overlap, and they are the same eight.** navec adds
nothing to retrieval. Weighting words by rarity was written and reverted
— it did not change a single hit. Tuning weights is pointless when there
is nothing to tell apart.

The cause is the domain, not the size: navec was trained on literary
fiction. "Love ~ prefer" scores 0.580, but "language ~ programming"
0.114 and "language ~ python" 0.145 — in fiction a "language" is a
tongue and a "python" is a snake.

A working replacement, if 1.6 GB of RAM is acceptable:

```python
from model2vec import StaticModel
from selectivemem import Memory

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
memory = Memory("brain.db", encoder=lambda text: model.encode([text])[0])
```

**Run `probe_semantic.py` on your own data before integrating.** Replace
`FACTS` and `PROBES` with yours — you will know your retrieval quality in
ten minutes instead of a month into production.

Without an encoder, search falls back to string similarity and keeps
working — but the library **warns once in the log**, and
`stats().semantic` is `False`. Check it at start-up if you would rather
not run blind.

The search threshold (`memory_search_threshold`) assumes a working
encoder, and with any of them it separates correctly. Measured in English
with `potion-base-8M`: relevant queries score 0.30–0.65, irrelevant ones
0.18–0.24, and the 0.3 threshold sits exactly between. Without an encoder
it is 0.178 against 0.167 — nothing to separate, and lowering the
threshold returns noise rather than answers.

### Settings are a dataclass, not a global config

```python
from selectivemem import Memory, MemorySettings
memory = Memory("brain.db", settings=MemorySettings(decay_rate=0.02, age_t0=3600))
```

---

## What has been measured

By benchmarks from this repository, over five seeds, byte-reproducible.

**Forgetting is selective by the importance the user assigns.** After two
weeks of silence, with volume held equal (`compare_retention.py`):

| Store | Praised | Ordinary | Gap |
|---|---|---|---|
| Sliding window | 77% | 70% | +7% |
| Random sample | 43% | 50% | −7% |
| **selectivemem** | **100%** | **60%** | **+40 pp** |

The naive stores show a gap near zero — they know nothing about the
user's opinion. "Volume held equal" means both groups produce the same
number of messages, so the gap cannot be explained by the important topic
simply coming up more often.

**The organism stops being surprised by the familiar.** Over a stream of
120 messages surprise falls 0.750 → 0.278 and spikes follow it 9 → 0. By
the thousandth message routine is no longer stored.

**Compression.** 38–51 episodes out of 120 exchanges.

**Speed.** Search is linear, but the expensive fuzzy comparison runs only
on the best candidates — profiling showed it ate 82% of the time.

| nodes | before | after |
|---|---|---|
| 500 | 32 ms | 7 ms |
| 10 000 | 655 ms | 177 ms |
| 30 000 | 2 074 ms | 503 ms |

**External benchmark: LongMemEval.** 60 questions, each with a haystack
of ~48 chat sessions with the evidence buried inside. Measured as
recall@k — the same figure our neighbours report.

| mode | stored | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| normal | 24.6% | 60.0% | **68.0%** | 72.0% |
| archive (gate bypassed, forgetting off) | 49.6% | 76.7% | **91.7%** | 96.7% |

The difference between the rows is the price of our forgetting policy,
stated as a number: 24 points. In exchange you store four times less and
get the selective retention from the table above.

The write threshold turned out to be the main lever. At the previous
default (0.35) one exchange in nine was stored and R@5 was 20%; measuring
against external data showed that 0.25 triples recall without touching
retention. The default was changed, and that is the first calibration of
this threshold **not against our own corpus**.

If you need maximum completeness, set `base_plasticity_threshold` lower
or disable the gate entirely.

### What selectivemem does NOT do

**It is not better than a random sample at uniform coverage.**
`compare_memory.py`, the same five seeds, equal budget in characters:

| Store | Found |
|---|---|
| Sliding window | 90.0% |
| Random sample | 90.8% |
| selectivemem | 92.4% |

**This used to be a negative result** — 87.2% against 88.8% for random
sampling — and it was honest at the previous write threshold. Lowering
it removed the deficit. But it is too early to claim a win: the spread
across seeds is 82–98%, so 92.4 against 90.8 means "level", not "ahead".

Do not expect more here either: against uniform questions an unbiased
sample is unbeatable by construction, and any meaningful selection is
biased.

---

## How it works

| Mechanism | What it does |
|---|---|
| **Spike gate** | Stores when `(emotion + surprise) / 2 >= threshold`. The threshold rises under load. |
| **Its own surprise** | Prediction error against its own graph of words and their pairings, not the entropy of a string. Experience lowers it; a short utterance surprises less, because it carries less information. |
| **Two memory parameters** | `weight` — how vividly it is remembered now; `stability` — how slowly it fades. Stability grows on every recall. |
| **Dopamine** | Reinforcement by reward prediction error (Rescorla–Wagner): expected praise barely moves the weight, unexpected praise moves it a lot. |
| **Supersession** | A contradiction with something known is itself a reason to store. The stale version is weakened, not deleted. |
| **Decay floor** | Reinforced memories dim but do not vanish: they decay towards a floor whose height is earned by approval. Unreinforced ones go as before. |

The floor deserves a note of its own, because it defines product
behaviour. Stability grows on recall — so a fact the user called
important but never returned to used to sink to zero along with the
routine. For an assistant that was told "I am allergic to penicillin",
that is unacceptable.

| times praised | after a year |
|---|---|
| 0 | forgotten |
| 1 | 0.075 |
| 2 | 0.128 |
| 5 | 0.343 |

Praise applies both to what was just written and to what was **recalled**
— with an assistant, approval usually lands on a good answer built from
memory. Switch it off with `MemorySettings(memory_floor_max=0.0)`.

---

## The showcase

Memory is invisible: the line "node stored, weight 0.7" moves nobody. So
this repository also holds a **showcase** — a Telegram bot that starts
out babbling `ma-ma-ma` and answers in sentences by the tenth message,
plus a mini-app that breaks down every turn.

The showcase is not the product and is not part of the package. Details
in [DEMO.md](DEMO.md) (in Russian).

```
selectivemem/  the memory package — the only thing pip installs
core/          showcase: persona, mood, speech stages
tools/         benchmarks and the memory inspector
tests/         287 tests
AUDIT.md       what was measured, what was refuted, what is left undone
RELEASING.md   how to cut a release
```

Documentation in Russian: [README.ru.md](README.ru.md).

---

## Licence

Copyright © 2026 MERUMEZ

The core is **AGPL-3.0** ([LICENSE](LICENSE)). Free for personal
projects, research and open products.

Closed games, closed SaaS and products shipped to customers need a
commercial licence: terms and price ranges in
[COMMERCIAL.md](COMMERCIAL.md), the agreement in
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Sending a patch? Read [CONTRIBUTING.md](CONTRIBUTING.md) first: one
signed line is what keeps dual licensing possible.
