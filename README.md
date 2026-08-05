# selectivemem

**Memory that decides what is worth keeping — and what to hand back first.**

Your assistant meets the user from scratch every time. Dumping everything
into a vector store means paying to keep noise and drowning in it at
retrieval. `selectivemem` writes about a quarter of what it is shown, and
ranks what it kept by what has actually earned its place.

Measured on LongMemEval, **the full set** (`longmemeval_s`: 500 questions,
246,750 turns, 47 sessions of haystack each), at stock settings:
**R@1 67.0%, R@10 81.0%** while storing **23.2%** of the turns.

That number is deliberately the hard one. Most published figures for this
benchmark — and every figure this repository quoted until recently — come
from `longmemeval_oracle`, where the haystack is a SINGLE session. The same
build scores 96.2% there. Thirty points of difference sit between those two
files, so ask which one any number came from, including ours.

Where it earns its keep is memory full of NEAR-DUPLICATES: one user, one
subject returned to for months. With 200 competing look-alikes in the
store, ranking by earned strength finds the right one first **83% of the
time against 50%** for the usual weight-and-recency scoring. On a store
of unrelated topics there is no such advantage, and this README will not
pretend otherwise.

Nothing is deleted by age. Long-term memory in people does not fill up
either; what fades there is the ROUTE to a memory, not the memory —
the classmate's name you cannot summon but recognise on sight.

**A working assistant in sixty lines** — `examples/assistant.py`. It shows
the only thing that matters for integration, the ORDER of calls: retrieve
what to answer with, ask the model, then show memory what happened. That
order is not cosmetic — links between memories are born from what was
pulled out shortly before a write, and reversing it produces no
associative network at all.

```bash
export ASSISTANT_API_KEY=sk-...        # any OpenAI-compatible endpoint
python examples/assistant.py
```

```python
memory.observe("Levi", response="nice to meet you", fills_gap=True)
```

`fills_gap` is the caller stating a fact the library cannot obtain: **this
answers something I asked for.** We tried to have memory notice its own
gaps — recall returns little, so the next event matters — and it never
fired once in thirty-six turns. Asked "what is the dog called", search
returned 0.776 for "I have a dog": memory FOUND something, confidently and
wrong. It measures similarity, not whether a need was met, and no
confidence threshold separates those.

It also shows the two things an application must decide FOR the memory,
because the library sees text and not a conversation: an outright "remember
this" (significance 1.0, written past the threshold), and a short reply to
a question the assistant itself asked. The second one is easy to miss and
expensive: asked "what is your dog called", a user answers "Levi" —
novelty 0.06, one familiar word, and the name is never stored. The reply
carries the whole point of the exchange and surprises no one.

Without a key it still runs and shows the memory working — the model
answer is stubbed, the memory is real. No dependencies: the HTTP call goes
through urllib.

**See it work in one command** — thirty-six turns of conversation, the
gate's decision on each, questions whose words are not in memory, and
forgetting:

```bash
python tools/demo.py
```

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
pip install selective-memory[semantic]    # + meaning-based search (English)
pip install selective-memory[semantic-ru] # + meaning-based search (Russian)
```

86 KB wheel, standard library and sqlite3. The name on PyPI is free; no
release has been published yet.

**Pick the extra that matches your language — the difference is not
subtle.** `[semantic]` brings potion-base-8M, which fetches itself on
first use (30 MB) and is an *English* model. Measured on four related and
four unrelated word pairs:

| | related pairs | unrelated pairs | separation |
|---|---|---|---|
| English | 0.651 | 0.013 | **+0.638** |
| Russian | 0.685 | 0.661 | **+0.024** |

In English the model tells `cat`/`kitten` (0.686) from `cat`/`concrete`
(0.009) cleanly. In Russian it rates `кот`/`бетон` at 0.803 — *higher*
than `кот`/`кошка` at 0.643. That is not weak performance, it is noise:
for Russian text this model is worse than no model, because the
lexical fallback at least never claims a false match. Use
`[semantic-ru]`, which brings navec, or attach your own encoder.

Without any extra, search matches by shared words and says so in the log.
The example above works because "allergic" appears in both the query and
the note; ask for "allergy" instead and you get nothing.

This is not fine print. Plan on attaching your own encoder from day one
(see [below](#the-encoder-sets-the-language-and-the-domain)); everything
else in the library is language-agnostic, but retrieval is only as good
as the vectors you give it.

**How the engine works inside** — a step-by-step walkthrough with the
biological correspondences: [ARCHITECTURE.md](ARCHITECTURE.md).

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

### What it needs from your application

`selectivemem` is a library your code calls directly. It has to sit where
you control the message loop:

    your app  ->  memory.observe(user_text)
              ->  memory.context_for(query)  ->  into the prompt
              ->  memory.feedback(+1.0)      when the user approves

**If you cannot intercept individual messages, no memory library will
help you** — and that is worth checking before you integrate anything.

This is not a hypothetical. A neighbouring project, PsychMem, built the
same ideas — dual-store STM/LTM, Ebbinghaus decay, learned importance —
as a plugin for coding agents, and abandoned it after concluding that
*"the memory model itself is sound; the architecture is wrong"*. Plugin
hooks exposed only coarse lifecycle events, with no way to intercept a
message or shape the context, so the only route left was injecting a fake
user turn that polluted the conversation. They rewrote the storage, the
write strategy and the retrieval, and hit the same ceiling every time.

So: check your integration point first. If your framework hands you the
turn, you are fine. If it only hands you `session.started`, no amount of
decay curves will save the design.

---

## Five calls

```python
obs = memory.observe(text, response="")               # show an event
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

The core does not read significance off the text and knows no words of
approval — that is your application's job when it knows better than the
core can:

| Where | `emotion` on observe | `feedback` |
|---|---|---|
| Assistant | 0.0 normally; 1.0 when the user says "remember this" | 👍/👎 button, an explicit "right/wrong" |
| NPC | significance of the game event: death 1.0, picked up a blade of grass 0.0 | quest outcome, faction reaction |
| Support bot | ticket priority | customer's rating of the resolution |

**You may pass nothing at all — and by default the organism answers for
itself.** Left out, `emotion` and `load` come from its own internal state:
how crowded its store is, how much of the world it currently understands,
how often it is being corrected. Two channels come out of that — urgency
makes writing easier, strain makes it harder — and they meet in the same
gate. See `interoception.py`.

Measured on LongMemEval, 500 questions: 43.9% -> **25.9%** of turns
written, R@1 96.8% -> 96.0% ON THE LIGHT SET. Two fifths fewer nodes for
eight tenths of a point. Like every configuration comparison in this file,
that pair comes from `longmemeval_oracle`; the claimed quality figure is
the full-set one above. Set `intrinsic_emotion=False` if completeness matters more than
volume.

Whatever you pass explicitly always wins: an application that read the
text with a model, or got a rating from a human, knows more than the core
can.

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

### The encoder sets the language — and a mismatched one CORRUPTS memory

The bundled one is navec, Russian static vectors (51 MB, no torch). For
another language or another domain, pass your own function:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
memory = Memory("brain.db", encoder=lambda text: model.encode(text))
```

**This is not only about search quality.** A model that does not match the
language does not merely fail to find things — it makes memory ACTIVELY
WRONG, and silently. Measured, Russian vectors over English text:

    "my dog is called Rex" against "the price of bread went up"   0.808

The supersession threshold is 0.8, so the memory sets about weakening the
fact about the dog because bread got more expensive. Over a LongMemEval
haystack that fired 3080 times across 79 writes — forty weakenings per
write, with nothing said about it.

A guard now blocks the worst of it: supersession also requires a quarter
of the content words to be shared, which cuts that to 372. The guard is a
floor, not a fix. **Pass an encoder for your language.**

### The configuration this library claims

**LongMemEval, the FULL set (`longmemeval_s`), 500 questions, 246,750
turns, 47 sessions per question.** Defaults, English, `[semantic]`
installed:

| install | written | R@1 | R@3 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|
| **`[semantic]`** (recommended) | **23.2%** | **67.0%** | 76.2% | 79.0% | **81.0%** |
| bare, no dependencies | 34.2% | 51.6% | 65.6% | 70.8% | 79.6% |

The model earns its 30 MB: fifteen points at k=1 while writing a third
less. The bare install is usable and needs nothing, but it pays for that
independence twice — in accuracy and in speed (the same run takes about
three hours against forty minutes, because grown perception computes its
vectors in pure Python).

```bash
python tools/bench_longmemeval.py --data storage/bench/longmemeval_s.json --encoder builtin
```

By question type, because the average hides a great deal:

| type | n | R@1 | R@10 |
|---|---:|---:|---:|
| knowledge-update | 78 | 83% | 91% |
| single-session-assistant | 56 | 82% | 91% |
| multi-session | 133 | 75% | 89% |
| temporal-reasoning | 133 | 64% | 80% |
| single-session-user | 70 | 49% | 60% |
| **single-session-preference** | 30 | **17%** | 53% |

Preferences are the weak spot and have been for a long time: "I would
rather not fly early" is a fact stated once, in passing, without emphasis —
the gate has nothing to catch it by.

**THE EASY SET FLATTERS BY THIRTY POINTS.** Most published measurements —
including every one made in this repository before now — use
`longmemeval_oracle`, where the haystack is a SINGLE session per question
instead of forty-seven:

| set | sessions per question | R@1 | R@10 |
|---|---:|---:|---:|
| oracle (light) | 1 | 96.2% | 96.4% |
| **s (full)** | 47 | **67.0%** | **81.0%** |

Read any number about this library, or any other, together with the file it
came from. The figures below in this section come from the light set and
are kept for comparison between configurations, not as a claim of quality.

### What the encoder buys, on sixteen questions

`tools/probe_semantic.py` holds sixteen pairs of "fact → how someone will
ask about it". Half share no word with the fact ("where do I live" against
"we live at 12 Pushkin Street") — those are the ones that test semantics.

| configuration | hits |
|---|---:|
| potion-base-8M (`[semantic]`) | **9/16** |
| grown perception (bare install) | 6/16 |
| shared words only | 6/16 |

```bash
python tools/probe_semantic.py --lang en           # with the model
python tools/probe_semantic.py --lang en --grown   # as a bare install
python tools/probe_semantic.py --lang ru           # Russian set
```

This bench and the one above disagree about grown perception, and the
disagreement is instructive: sixteen isolated facts give it nothing to
learn from, while a benchmark haystack is hundreds of turns in the very
vocabulary the questions use. Perception grown from experience is good at
the language it has actually seen and knows nothing of the rest.

**For Russian take `[semantic-ru]`.** The default model is English; on
Russian it scores "cat ~ concrete" 0.803 against "cat ~ kitty" 0.643, and
three tests in this repository skip themselves when they detect it.

A working replacement, if 1.6 GB of RAM is acceptable:A working replacement, if 1.6 GB of RAM is acceptable:

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

**Complete recall on two thirds of the storage.**
`compare_retention.py`, 3 seeds, 14 days of silence, volume held equal,
**through the library** rather than the showcase:

| Store | Nodes | Important | Ordinary |
|---|---:|---:|---:|
| Random sample | 43 | 78% | 56% |
| Sliding window | 44 | 89% | 89% |
| **selectivemem** | **33** | **100%** | **100%** |



A random sample holding half again as much finds about six answers in
ten. The advantage comes from the write gate, not from deleting:
selectivemem simply never took the noise in.

AN EARLIER VERSION OF THIS TABLE CLAIMED A GAP of +40 pp between praised
and ordinary topics, and that claim has been withdrawn. Investigation
showed it measured DELETION: the memory was not ranking praised material
higher, it was erasing the rest. Once age-based deletion was removed —
which cost it nothing, see the LongMemEval numbers below — the gap
collapsed to zero and the store simply keeps everything it wrote.

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

**What the assistant volunteers is what the user marked.** When memory is
asked a general question — "what do you remember about me" — and fills a
prompt on its own, `compare_ordering.py` measures which memories get in:

| | selectivemem | random order |
|---|---:|---:|
| share of praised material in the top 5 | **100.0%** | 50% |
| MRR gap | +0.046 | |

Eighteen topical memories reached the answers across three seeds, so the
share is a measurement rather than a coin toss — an earlier version of
this bench reported 40% on a denominator of 0.6 and was rightly ignored.

**WHAT "IMPORTANT COMES FIRST" MEANS HERE, AND IT IS A REAL
QUALIFICATION.** Importance MULTIPLIES fitness to the question rather than
adding to it. So what the user marked overtakes what fits EQUALLY WELL —
and does not overtake what fits better. Praise settles disputes; it does
not override relevance.

It used to be the other way round: importance was added to fitness and
could outweigh it. The MRR gap was slightly larger for that (+0.052
against +0.046), but a heavy node floated to the top of every query,
including the ones it answered wrongly — and that cost fifteen points of
recall on the external set.

**External benchmark: LongMemEval.** All 500 questions, each with a
haystack of ~48 chat sessions with the evidence buried inside. Measured
as recall@k — the same figure our neighbours report. Stock settings, no
flags.

| mode | stored | R@1 | R@5 |
|---|---|---|---|
| stock | 24.7% | 66.0% | **83.4%** |
| archive (gate bypassed, forgetting off) | 49.6% | 76.7% | **93.2%** |

The gap between the rows is the price of writing selectively: **9 points
of recall for three quarters of the storage.**

By question type, because the average hides the interesting part
(**light set** — for the claimed figures see the full-set table above):

| type | n | R@5 | archive |
|---|---:|---:|---:|
| single-session-assistant | 56 | 96.4% | 98.2% |
| multi-session | 133 | 92.6% | 94.2% |
| knowledge-update | 78 | 88.4% | 98.9% |
| temporal-reasoning | 133 | 82.6% | 91.0% |
| single-session-user | 70 | 67.1% | 92.9% |
| single-session-preference | 30 | 46.6% | 76.8% |

`knowledge-update` — questions where a fact CHANGED and the current value
is wanted — used to score 20.8% here. The cause was measured and it was
not the write filter: evidence for those questions is on average 16 days
older than the question, and age-based deletion erased all of it, 12
nodes out of 12 in every instance examined. Removing that deletion is
what moved the overall figure from 64.8% to 83.4%.

Preferences remain the weak row and are not dressed up.

The write threshold turned out to be the main lever. At the previous
default (0.35) one exchange in nine was stored and R@5 was 20%; measuring
against external data showed that 0.25 triples recall without touching
retention. The default was changed, and that is the first calibration of
this threshold **not against our own corpus**.

If you need maximum completeness, set `base_plasticity_threshold` lower
or disable the gate entirely.

### What selectivemem does NOT do

**A third of the storage, the coverage of keeping everything.**
`compare_memory.py`, equal budget in characters, three seeds:

| Store | Nodes | All | **Rare** | Frequent |
|---|---:|---:|---:|---:|
| Everything (upper bound) | 120 | 94% | 100% | 92% |
| Sliding window | 37 | 72% | 50% | 83% |
| Random sample | 37 | 72% | **33%** | 83% |
| **selectivemem** | 42 | **94%** | **100%** | 92% |

Forty-two nodes answer as well as all hundred and twenty. On material
mentioned ONCE the gap is decisive — 100% against 33% for a random sample
of the same size — and that is exactly what the write gate is for: what
is rare is surprising, so it gets in, while selection by recency or
chance loses it.

Across three seeds selectivemem holds 94% on "all" with rare between 83%
and 100%; the random sample swings 61–78% and 33–50%. Questions are
phrased as a person would ask them, not as the stored sentence — an
earlier version probed with the stored text verbatim and produced a
meaningless 100% everywhere.


---

## How it works

| Mechanism | What it does |
|---|---|
| **Spike gate** | Stores when `surprise * (1 + emotion) / 2 >= threshold`. Emotion MULTIPLIES the plasticity novelty has opened, rather than being averaged with it. The threshold rises under load. |
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
in [DEMO.md](https://github.com/MERUMEZ/selective-memory/blob/master/DEMO.md) (in Russian).

```
selectivemem/  the memory package — the only thing pip installs
core/          showcase: persona, mood, speech stages
tools/         benchmarks and the memory inspector
tests/         287 tests
AUDIT.md       what was measured, what was refuted, what is left undone
RELEASING.md   how to cut a release
```

Documentation in Russian: [README.ru.md](https://github.com/MERUMEZ/selective-memory/blob/master/README.ru.md).

---

## Licence

Copyright © 2026 MERUMEZ

The core is **AGPL-3.0** ([LICENSE](https://github.com/MERUMEZ/selective-memory/blob/master/LICENSE)). Free for personal
projects, research and open products.

Closed games, closed SaaS and products shipped to customers need a
commercial licence: terms and price ranges in
[COMMERCIAL.md](https://github.com/MERUMEZ/selective-memory/blob/master/COMMERCIAL.md), the agreement in
[LICENSE-COMMERCIAL.md](https://github.com/MERUMEZ/selective-memory/blob/master/LICENSE-COMMERCIAL.md).

Sending a patch? Read [CONTRIBUTING.md](https://github.com/MERUMEZ/selective-memory/blob/master/CONTRIBUTING.md) first: one
signed line is what keeps dual licensing possible.
