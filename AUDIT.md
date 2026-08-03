# Audit

**Date:** 2026-08-02 · **Tests:** 286 · **Lines of Python:** ~18,400

Everything asserted here comes from **runs**, not from reading code. The
numbers are reproducible: `tools/simulate_learning.py`,
`tools/compare_memory.py`, `tools/compare_retention.py`,
`tools/probe_semantic.py`, `tools/bench_longmemeval.py`.

---

## 1. What this is

Not a chatbot and not an LLM wrapper. It is a **controller of memory and
attention** that governs a language model.

The difference from RAG is fundamental. In RAG memory is an index: put
everything in, take out what looks similar. Here memory is **an economy
of scarcity**: a node spends weight, fades exponentially, competes for
the right to be recalled, and dies if it was never useful.

One principle holds the construction together — **prediction error
governs plasticity** — and it is implemented twice:

| Where | What it is | Measured |
|---|---|---|
| On input | surprise = divergence from its own graph of language | 0.750 → 0.278 as it matures |
| On action | dopamine = the unexpectedness of reward, not the reward | joy increment 0.32 → 0.08 |

---

## 2. What was found and fixed

None of the defects below **could have been seen by reading the code**.
Every one was found by measurement. Constant values are explained here so
they do not get "tidied up" back.

### 2.1 Vocabulary was erased every night

`AGE_T0 = 1 hour` applied to all nodes, vocabulary included. A mastered
word (0.20) fell below the threshold after a 6-hour pause and was
**deleted from the database** within 28. Teaching the bot was physically
impossible: a month of a live database held 2 mastered words out of 40
heard.

**Fix:** `LEXICAL_AGE_T0 = 30 days`. Vocabulary is the infrastructure of
language, not yesterday's conversation.

### 2.2 The spike gate was driven by counting exclamation marks

"Surprise" was Shannon entropy over the characters of a string. A newborn
mind and a mind that had seen a phrase fifty times both produced 0.856;
meaningless gibberish scored 0.819.

**Fix:** `MemoryGraph.compute_surprise` — prediction error against its own
graph (lexical novelty plus structural).

`BASE_PLASTICITY_THRESHOLD` had to be calibrated **three times**, and
every time by measurement:

- **0.65 → 0.45.** The old threshold was implicitly tuned to
  always-high perplexity: character entropy stayed near 0.85–0.9 for any
  text and quietly held density near the threshold. With honest surprise,
  the measurement showed **zero spikes over an entire run**.
- **0.45 → 0.35** later, when the arousal axes were merged (§2.5):
  arousal used to be dead and contributed nothing; now it is live and
  genuinely raises the threshold, so the base had to drop by the size of
  a typical contribution.
- **0.35 → 0.25** against an external benchmark — the first calibration
  **not** against our own corpus. At 0.35 one exchange in nine was
  stored and R@5 on LongMemEval was 18.3%; at 0.25 it became 68% at the
  same retention. Current value: **0.25**.

  This also forced `SURPRISE_FULL_CONTENT_TOKENS` from 2 to 3: at a
  threshold of 0.25 a one-word utterance produced a density of exactly
  0.250 and slipped through, bringing back the interjection clutter that
  had been removed the day before. The two parameters turned out to be
  coupled and must be chosen as a pair.

### 2.3 Short-term memory did not exist

STM was empty after **39 messages out of 40**. Three causes:

- auto-sleep fired on every message from the ninth onwards (the
  threshold counted *all* nodes, and vocabulary accumulates 145 within
  nine messages) — plus **two LLM calls per message** in production;
- a spike wiped the buffer even though the exchange had already been
  stored as its own node;
- `STM_CAPACITY = 8` counts entries, and a message produces two — so the
  context was four exchanges.

**Fix:** trigger on `count_memory_nodes()`, consolidation decoupled from
spikes, capacity 16. Result: empty 5 times out of 40, auto-sleep 0
out of 40.

### 2.4 Memory was amnesia, not forgetting

Every episode aged on one scale regardless of usefulness: 11 nodes before
an overnight pause, 1–2 after.

**Fix:** a `stability` column — resistance to forgetting, growing
multiplicatively on every recall.

| Recalls | After 2 days | After 2 weeks |
|---|---|---|
| 0 | 0.073 | forgotten |
| 3 | 0.393 | forgotten |
| 10 | 0.753 | 0.526 |

### 2.5 The self-protection layer never once engaged

The Stress Protection declared in the manifesto was **dead code**: stress
accumulated at +0.025 per message against a recovery of −6.00 per tick,
so it collapsed to zero instantly. `is_overloaded` never fired.

On top of that, the same state existed a second time as `Mood.anxiety`.

**Fix:** a single arousal axis in `Mood`, fed by appraisals rather than
punctuation. `InstinctSystem` no longer holds state. `MOOD_AROUSAL_GAIN`
was chosen by measurement: at 0.35 arousal settled at 0.84 against a
threshold of 0.8 — **permanent overload**, 14 LLM calls instead of 86.

### 2.6 Forgetting switched itself off silently after a restart

`brain_time` was composite: +120 per message (the clock ran **seven
times** faster than the wall) and a reset to `time.time()` at session
start. Node marks ended up in the future and `if dt <= 0: continue`
skipped the decay.

| Messages in a session | Forgetting after returning |
|---|---|
| 10 | works |
| 60 | stalled another 1.0 h |
| 100 | **stalled another 2.3 h** |

**Fix:** one scale, `brain_time = epoch + (wall − epoch) × 7`, with the
epoch stored in the database. `AGE_T0` recomputed 3600 → 25200: theory
predicted growth by exactly `TIME_ACCELERATION`, and measurement
confirmed it.

### 2.7 "Skin" was more relevant than "cat"

Search went by letters. On the node "I have a cat" (in Russian): "skin"
scored 0.884, "cat" 0.870, and paraphrases were not found at all.

**Fix:** semantic search (navec, 51 MB, no torch — this machine has
3.7 GB). Dropping function words proved decisive: without it "skin" held
at 0.863.

### 2.8 A thread race

`process_message` and `run_idle_tick` worked on one session object from
different threads without synchronisation. The only defect capable of
**corrupting data**.

**Fix:** a lock with priority skew — a message waits, the background tick
yields. The test's meaningfulness was demonstrated: bypassing the lock
lets up to 8 threads inside.

### 2.9 A dead CLI

`main.py` did not start (a `TypeError` at launch) and logged that as
`CRITICAL`. Its 581 lines duplicated `BrainSession` and lagged an entire
session of fixes behind. **Deleted**, along with `async_console.py`,
`debug_formatting.py` and two broken manual scripts.

### 2.10 Contradictions were never resolved

"My dog is called Rex", a month later "my dog is called Buddy" — both
nodes lived on as equals, and the stale one actually ranked **better**
(0.906 against 0.875), because order is decided by string similarity
rather than by time.

**Fix:** `find_superseded` plus `supersede_node`. A contradiction is a
reason to store in its own right — a correction is unsurprising by
nature and would never clear the density threshold on its own. The stale
version is weakened, never deleted: if the supersession was a false
positive, the user will mention the fact again and it will recover.

### 2.11 Praise never reached what was recalled

Through the facade, feedback only reached the node just WRITTEN. With an
assistant, praise usually lands on a good answer built from what was
RECALLED — so the main case reinforced nothing at all. Measured: eight
consecutive praises produced the same reward expectation of 0.300 as one.

**Fix:** `recall` now marks the nodes it used as involved in the turn,
and the reinforcement loop rewards all of them. The either/or was present
in **two** places — reward expectation and effect application — and
fixing only the first left praise making memory more durable but not
brighter.

### 2.12 What was marked important still vanished

Stability grows through recall, so a fact the user called important but
never returned to sank to zero along with the routine. Measured: six
months after a conversation, nothing survived — including the penicillin
allergy.

**Fix:** a decay floor whose height comes from `reward_expectation`. A
node that was never reinforced has a floor of zero and fades as before,
so this does not make memory immortal.

| Times praised | After a year |
|---|---|
| 0 | forgotten |
| 1 | 0.075 |
| 2 | 0.128 |
| 5 | 0.343 |

---

### 2.13 Seven mechanisms that were described, tested, and never ran

Found over two days, all of them covered by green tests and named in the
README. This is the single most useful finding in the file, because it is
a PATTERN rather than seven accidents.

**No edges were created between memories at all.** 201 nodes, 0 edges.
The library never linked episodes; the edges visible in the demo are
created by the showcase, which orchestrates recall and write together. So
spreading activation — advertised as multi-hop retrieval and occupying a
fair share of the search code — had nothing to travel along.

**A fresh edge was born inactive.** Weight 0.150 against an activation
threshold of 0.3: zero edges of twenty-seven above the bar. To take part
in retrieval a link had to be reinforced; to be reinforced it had to fire.

**Surviving that, an edge went quiet after five days** while remaining in
the database — below the activation threshold, above the deletion one.
The same disease, deferred.

**Deletion by age erased the evidence** for every knowledge-update
question, 12 nodes of 12, while removing only a tenth of memory.

**Consolidation and sleep were unreachable from the library.** Both are
written IN the package and were called only by the showcase.

**Supersession only caught near-verbatim restatements.** It fired on the
examples in its own docstring (0.923, 0.952) and missed how people
actually report a change: "I moved to Piter" against "I live in Moscow"
scores 0.369.

**What they have in common** is that none is a coding error. Each is an
interaction between a threshold and a decay rate that nobody had measured
end to end. Ordinary tests cannot catch this class: a test checks that a
mechanism behaves correctly when called with suitable data, not that such
data ever arises in live use.

`tools/check_liveness.py` exists for exactly that, and it counts OUTCOMES
rather than calls — because a mechanism can be invoked faithfully and
decide "do nothing" every single time, which consolidation did in 100% of
cases while the first version of the bench called it alive.

### 2.14 Forgetting was a clock; it should have been competition

`weight` was doing three jobs: recency (it decayed), importance (praise
raised it) and retrieval strength (it was a term in the score). Nearly
every defect above grew out of that conflation. Re-ranking BY IMPORTANCE
turned out to be re-ranking BY AGE, and widening its band dropped R@1
from 32% to 18%.

Split into three: `strength` accumulates from approval and from proving
useful and NO CLOCK TOUCHES IT; `weight` means recency; the share of
strength among candidates decides retrieval. Forgetting becomes losing a
competition rather than running out of time — which is also the better
supported account of human forgetting: we lose memories mostly because
new learning competes with them.

Nothing is deleted by age any more. The limit sits where biology puts it:
on the way in (the gate takes a quarter) and on the connections, which
grow over when unused.

## 3. What remains

### 3.1 Defects

| What | Severity | Comment |
|---|---|---|
| Search is linear | medium | Still linear, but four times cheaper: 30,000 nodes in 0.5 s instead of 2.1 s. What remains is reading every row from SQLite (30% of the time). Going sub-linear needs an index and a new dependency — against the promise of zero dependencies |
| The bundled semantics adds NOTHING beyond word overlap | **high for selling** | `tools/probe_semantic.py`: 8 hits out of 16, and they are exactly the 8 solvable by word overlap. A modern multilingual model gives 12/16 but costs 1.6 GB of RAM. The cause is the domain: navec was trained on literary fiction ("language ~ programming" 0.114). Weighting words by rarity was written and reverted — zero change |
| Distant paraphrase is out of reach | low | the limit of a light model, pinned by a test |

### 3.2 What is missing compared to the neighbours

- **Structured facts.** A node is a concatenation of an STM window. Fact
  extractors would give `name = Pasha` as its own entity. This memory is
  associative but not structured.
- **Grounding.** The bot says "hello" because you repeated it, not
  because it understands a greeting. Co-occurrence edges exist and feed
  only the surprise computation.
- **Speech of its own.** Competence lives in an LLM behind HTTP. The
  graph governs attention and admission, but not the ability to build a
  sentence. A deliberate decision: the LLM is a speech organ under the
  graph's control.

---

## 4. Measured claims

### 4.1 Confirmed

**Complete recall on two thirds of the storage.**
`compare_retention.py`, 3 seeds, 14 days of silence, volume held equal,
**through the library** rather than the showcase:

| Store | Nodes | Important | Ordinary |
|---|---:|---:|---:|
| Random sample | 43 | 78% | 56% |
| Sliding window | 44 | 89% | 89% |
| **selectivemem** | **33** | **100%** | **100%** |


The advantage comes from the WRITE GATE, not from deleting: a quarter of
the turns are taken in, and that quarter turns out to contain the
answers. A random sample holding half again as much finds six in ten.

**A CLAIM WITHDRAWN, AND THIS IS THE MOST IMPORTANT ENTRY IN THIS FILE.**
This table used to read 100% praised against 60% ordinary — a gap of
+40 pp — and that gap was the project's headline argument for a year.

It was measured honestly and it meant something other than advertised.
Investigation showed the memory was NOT ranking praised material higher.
It was deleting the rest. The moment deletion by age was prevented — by
any means, floor or capacity — the gap collapsed to zero while retrieval
IMPROVED by 18.6 points on 500 external questions.

So the number was real and the interpretation was wrong: it measured the
cost of erasure, not the value of selection. Anyone who reruns
`compare_retention.py` today sees +0, and the honest claim is the table
above: fewer nodes, complete recall.

**Also confirmed:**
- surprise falls with experience: 0.750 → 0.278
- the write rate regulates itself: 9 → 3 → 0 spikes
- reinforcement does not saturate: joy increment 0.32 → 0.08
- exit from the pre-verbal stage: 10 messages at demo pace (was "never")
- compression: 38–51 episodes out of 120 exchanges

### 4.2 External benchmark: LongMemEval

The first number not from our own corpus. **All 500 questions**, a
haystack of ~48 sessions each, recall@k — comparable with what the
neighbours publish. Stock settings, no flags.

| Mode | Stored | R@1 | R@5 |
|---|---|---|---|
| threshold 0.35, deletion on (a year ago) | 5.5% | 13.3% | **18.3%** |
| threshold 0.25, deletion on | 24.8% | 50.8% | **64.8%** |
| **stock today** | 24.7% | 66.0% | **83.4%** |
| archive (no gate, no forgetting) | 49.6% | 76.7% | **93.2%** |

Two calibrations moved this number, and both were forced by measurement
rather than chosen.

**The write threshold, 0.35 → 0.25.** Tripled recall while retention did
not move (0.20/0.25/0.30/0.35 gave +47/+50/+53/+50 — noise).

**Deletion by age, removed entirely: +18.6 points.** The diagnosis is in
§2.13. Note what this means about the earlier framing: for a year the
project described the 28-point gap to `archive` as "the price of our
forgetting policy". Two thirds of that price was not selectivity at all,
it was age-based erasure, and removing it cost nothing.

By question type, since the average hides the interesting part:

| Type | n | R@5 | archive |
|---|---:|---:|---:|
| single-session-assistant | 56 | 96.4% | 98.2% |
| multi-session | 133 | 92.6% | 94.2% |
| knowledge-update | 78 | 88.4% | 98.9% |
| temporal-reasoning | 133 | 82.6% | 91.0% |
| single-session-user | 70 | 67.1% | 92.9% |
| single-session-preference | 30 | 46.6% | 76.8% |

**What this benchmark CANNOT show, and it took four empty measurements to
learn.** LongMemEval loads a haystack with `observe()` and calls `recall`
exactly once, at the end. Nothing is ever reinforced and nothing is
recalled during ingestion. So associations never form, spreading
activation has nothing to travel along, accumulated strength never grows
above its birth value, and consolidation is never exercised. Four
ablations in a row returned byte-identical numbers before the cause was
understood: the mechanism was not firing, and "no difference" was read as
"no use".

Its haystacks are also topically DIVERSE, so retrieval interference
barely occurs — see §4.6, where the same system gains 33 points once the
store is full of near-duplicates. This benchmark systematically
understates selectivity because it measures it where it is not needed.

### 4.3 Previously REFUTED, now level

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


A methodological error caught along the way: the budget was counted **in
nodes**, while `consolidate_from_stm` merges an STM window into a single
node. By nodes the compression looked like 9×, by text 3.1×, and the
naive stores were being given three times less memory.

**Conclusion:** this is not a system for exhaustive search. Against
uniform questions an unbiased sample is unbeatable by construction. The
domain of application is "hold on to what matters", not "lose nothing".

### 4.6 Where selectivity actually pays: near-duplicates

`tools/compare_interference.py`. The same six facts, buried under a
growing pile of NEAR-DUPLICATES — same words, different subject: "my dog
is called Rex" against "the neighbour's dog is called Rex".

| Distractors | Nodes | R@1 | R@5 |
|---:|---:|---:|---:|
| 0 | 6 | 100.0% | 100.0% |
| 50 | 56 | 50.0% | 100.0% |
| 200 | 206 | 50.0% | 91.7% |
| 800 | 806 | 50.0% | 83.3% |

**Storage is not free.** R@1 halves at fifty look-alikes: the answer is
still in the store — R@10 barely moves — but it is pushed off the top by
neighbours sharing its vocabulary. In memory research this is cue
overload, and it is why forgetting exists in people at all: not to free
space, which long-term memory does not lack, but to keep retrieval
possible.

The FIRST version of this bench nearly said the opposite. Its distractors
shared one or two words with the facts and produced a flat 100% at every
level; the conclusion "storage is free, forgetting cannot be sold" was
one sentence away from being written into the strategy. What saved it was
noticing that 100% at ZERO distractors is a ceiling, not a result.

**Ranking by accumulated strength recovers most of it.** Replacing node
weight — which is dominated by age, because decay is built into it —
with strength, which no clock touches:

| Importance from | R@1 at 200 | R@1 at 800 |
|---|---:|---:|
| node weight | 50.0% | 50.0% |
| **accumulated strength** | **83.3%** | **83.3%** |

+33 points. On LongMemEval the same change moves nothing at all (76.0%
both ways), because there are no near-duplicates there to tell apart.
That contrast is the product's niche stated as a measurement: one user,
one subject returned to for months — not a store of unrelated topics.

### 4.4 Reproducibility of the measurements

Every number above was obtained after the benchmarks stopped wandering.
Before that, two runs of the same code over the same data diverged by
several percentage points — meaning a precision like "+43 pp" was
self-deception: the difference between two variants of the system could
fit entirely inside the measurement jitter.

Three causes, all found by measurement rather than by reading code:

1. **The wall clock leaked into subjective time.** `brain_time` is
   `epoch + (wall − epoch) × TIME_ACCELERATION`, so the real duration of
   a run entered node ages multiplied by seven. Nodes sitting at the
   retrieval threshold crossed it depending on machine load. The
   benchmarks got a substitutable clock (`ManualWallClock`) and set
   pauses explicitly.

2. **The brain epoch came from `time.time()`** when the database was
   created — before the benchmark's clock was substituted. Found like
   this: the graph, the node count and the RNG state matched bit for bit
   (`n=25 w=5.743750000 rnd=0.590492512`) while the replies diverged on
   the very first message; exactly one quantity differed, the epoch.

3. **`ORDER BY RANDOM()` in SQL.** The syllable pool for babbling was
   drawn by SQLite's generator, which is seeded by nothing and takes
   entropy from the system. The sampling moved into Python, where the
   generator is under control; the distribution is unchanged.

Verification: `simulate_learning`, `compare_retention` and
`compare_memory` produce byte-identical output over three consecutive
runs.

### 4.5 What the mechanisms rest on

Not one of the mechanisms here was invented from nothing — they all
rediscover things long described in the psychology of memory. The
references are not for respectability but to separate the borrowed from
the invented: the first has been tested for decades, the second only by
our benchmarks.

| Mechanism | Basis |
|---|---|
| Exponential decay | **Ebbinghaus (1885)** — the forgetting curve |
| Two stores, STM -> LTM | **Atkinson & Shiffrin (1968)** — the dual-store model |
| STM capacity of 16 entries (8 exchanges) | **Miller (1956)**, refined by **Cowan (2001)** — the limit of working memory |
| `stability` grows on recall | **the spacing effect**; depth of processing — **Craik & Lockhart (1972)** |
| Superseding versions of a fact | **Nader et al. (2000)** — reconsolidation: a memory becomes plastic when retrieved |
| Consolidation during sleep | **McGaugh (2000)** |
| Importance earned through use | **Anderson & Schooler (1991)** — rational analysis: the accessibility of a memory reflects the probability that it will be needed |
| Dopamine as prediction error | **Rescorla & Wagner (1972)**; the neuroscience — **Schultz et al. (1997)** |

In fairness, this list came from a neighbour: `muratg98/psychmem`
published it in its README, and it turned out to be the most useful thing
that project left behind. Anderson & Schooler matters most here — it is
literally the theoretical grounding of the project's central bet, and
until now it had never been named.

**What is not in those papers, and was devised here:** the spike gate
driven by the organism's OWN prediction error — deciding whether to store
at all, based on how far the input diverges from the accumulated graph of
language. Psychology describes how to forget what has been stored; here
the question is whether to store it.

---

## 5. Position among other approaches

The axis is **who decides what matters, and when**.

| Approach | Who decides | When | Forgetting |
|---|---|---|---|
| Obsidian, Roam, Logseq | a human, by hand | at write time | none |
| Vector store + RAG | nobody | never | none |
| Fact extractors | an LLM judge | at write time | by rules |
| Context pagination | a policy | at read time | none |
| **This project** | **nobody in advance** | **over time, through use** | **the central mechanism** |

Obsidian is an opposite, not a competitor: there the graph is a human's
protocol of thought — inert and complete. Fact extractors come closest,
but importance there is **declared at write time**, whereas here it is
**earned afterwards**.

**What can be borrowed from this:** importance without labelling; a write
rate that regulates itself; a forgetting curve instead of eviction by
quota; reinforcement that does not saturate.

---

## 6. Where to go next

In descending order of return:

1. **Publish the package.** It is ready: the wheel builds, the README
   examples are verified by tests, the licence is in place. Nothing is
   worth much while nobody can install it.
2. **The full benchmark run.** The numbers above come from 60 of 500
   questions. Honest, and covering all six question types, but a public
   claim deserves the whole set.
3. **An English semantic model out of the box.** The bundled one adds
   nothing beyond word overlap, and that is the first thing a buyer will
   hit.
4. **Grounding through use.** Feed co-occurrence edges into word choice.
   Intellectually the most interesting, commercially the vaguest.
5. **Structured facts.** Would make this a competitor to fact
   extractors, but requires LLM calls on write. Postpone until it is
   clear whether this is a product or research.

---

## 7. Operation

**In production the code is older than the repository.** The
`mindnumbness.service` process has not been restarted since the latest
commits.

```bash
sudo systemctl restart mindnumbness
free -h    # RSS grows ~167 MB -> ~550 MB (the embedding model)
```

The machine has 3.7 GB and three neighbouring services. If it gets
tight, semantics can be switched off without touching code:
`EMBEDDINGS_ENABLED=false` in `.env` — search quietly reverts to string
matching.

**Verified working:** the Telegram bot (an end-to-end run against a copy
of the live database), the memory inspector, the mini-app (403 without a
Telegram signature), the benchmarks, 286 tests. Schema migrations are
idempotent and were verified against a copy of the live database.

---

Русская версия: [AUDIT.ru.md](AUDIT.ru.md).
