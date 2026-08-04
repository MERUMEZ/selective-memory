# Audit

**Date:** 2026-08-02 · **Tests:** 286 · **Lines of Python:** ~18,400

Everything asserted here comes from **runs**, not from reading code. The
numbers are reproducible: `tools/simulate_learning.py`,
`tools/compare_memory.py`, `tools/compare_retention.py`,
`tools/probe_semantic.py`, `tools/bench_longmemeval.py`.

---

A step-by-step walkthrough of the engine — which function does what
and what corresponds to it in the brain — lives in
[ARCHITECTURE.md](ARCHITECTURE.md).

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

### 2.15 The package broke where it promised to work without dependencies

Found by installing into a CLEAN environment — a check that did not exist
before. The suite was green on a machine where everything is installed.

**numpy was secretly required.** `to_blob`, `from_blob` and `cosine`
imported it unconditionally. The README meanwhile invites you to attach
YOUR OWN encoder and promises a zero-dependency install — and precisely
that path died with `ModuleNotFoundError` on the first write. The promise
was false for the one case it was made for. There is now a fallback on
`array("f")` and plain arithmetic: slower, present. Ranking in the same
situation silently zeroed semantics with a working encoder and live
vectors; it now computes pairwise.

**`[semantic]` installed the wrong model.** The extra named navec, left
over from when the library was Russian-only, while the code had long
defaulted to model2vec. Anyone following the hint printed by our own
warning got a package that is never used, and no way to tell.

**The English model on Russian is noise.** Measured over four related and
four unrelated pairs: separation +0.638 in English, +0.024 in Russian.
`кот`/`бетон` scores 0.803, `кот`/`кошка` 0.643. That is worse than no
model, because the lexical path at least never invents a match. Split
into `[semantic]` (English) and `[semantic-ru]` (navec), with the numbers
written into both READMEs. The `embeddings.py` header described navec as
the bundled model and was wrong — rewritten.

Nine tests failed rather than skipped on a clean install: they rely on
semantics but carried no marker. A new user would have seen nine red
tests and concluded the package was broken.

### 2.16 Two defects covering for each other

The work started from a simple question: why does reinforcement without
an explicit `feedback()` call achieve nothing. The answer took a day and
came in a pair.

**First: the threshold is out of reach.** Strength grows by 0.05 per
successful retrieval, but the effect switches on at a THRESHOLD of about
+0.6 accumulated strength rather than gradually — below it nothing moves:

| revisits to a topic | gain | R@1 at 200 near-duplicates |
|---:|---:|---:|
| 3 | 0.15 | 38.9% |
| 6 | 0.30 | 44.4% |
| 12 | 0.60 | **83.3%** |

Twelve revisits to one topic is not something conversation produces. The
mechanism existed, fired, showed up in the counters — and could not
affect anything.

**Second: two thirds of the scale did not exist.** Candidate scoring took
`min(1.0, own)` while `strength_max` was declared 3.0. A node at strength
1.0 and a node at 3.0 ranked identically.

**They covered for each other, and that is the point here.** While
reinforcement was too weak, only praised nodes reached the ceiling and
ordinary ones fell short — so the discrimination of important memories
rested ON THE MECHANISM NOT WORKING. Measured by group (24 competing
nodes):

| step | praised at ceiling | ordinary at ceiling | MRR gap |
|---:|---:|---:|---:|
| 0.05 | 100% | 25% | +0.050 |
| 0.15 | 100% | **67%** | +0.028 |

Raise the step and both groups hit the ceiling, leaving nothing to tell
apart. The drop looked like the price of the change; it was a symptom of
the second defect.

**How it ended: both defaults were put back.** A soft shoulder instead of
a hard ceiling (`strength_headroom`) fixes the second defect and lifts
the share of important items in the top 5 from 83.3% to 100.0% — but on
the near-duplicate bench it drops R@1 from 83.3% to 61.1%, because it
amplifies wrongly retrieved nodes as well. The near-duplicate bench is
the product's stated niche; trading it for top-5 share is not on.

The larger step won on three benches out of four, but only while the hard
ceiling held — its benefit rested on a defect.

So behaviour is unchanged and the knowledge grew:

- the threshold is measured and reproducible via `--sweep-use-step`;
- the ceiling now has a name (`strength_headroom`, default 0.0 = as
  before) — it used to be an unnamed `min(1.0, own)` deep in the formula;
- a negative result on competitor suppression: the pairing of a large
  step with inhibition of losers, predicted from the biological analogy,
  does not work (+0.028 -> +0.028 -> +0.025).

**And the real cause is named.** Reinforcement without `feedback()` is
blocked not by the size of the step but by the fact that the WHOLE result
set is strengthened, not the answer that turned out to be right. While
that holds, any speed-up of reinforcement also speeds up the entrenchment
of errors. The cure is credit by consequence — the conversation moved on,
no correction followed — which lives in `reinforcement.py`, not in a
number in the settings.

### 2.17 Sleep had no first two stages, and the bench lied about it

Sleep pruned weak edges and folded a dense cluster into an abstraction —
the third and fourth stages of consolidation. The first two, the ones
sleep exists for in biology, were missing entirely.

**Reactivation.** In slow-wave sleep the hippocampus replays daytime
sequences in bursts, and does so selectively: what led to reward replays
more often. That is what moves a trace into cortex.

**Homeostatic downscaling.** Synapses strengthen on average through the
day, and sleep scales them ALL down proportionally (the synaptic
homeostasis hypothesis, Tononi and Cirelli). Relative order survives,
while whatever was hanging by a thread drops out on its own — pruning
stops being a separate policy with an absolute threshold and becomes a
consequence of the general shrinkage. The same principle ranking was
already moved to: the share matters, not the absolute.

Both stages are implemented (`MemoryGraph.replay`,
`MemoryGraph.downscale_edges`) and LEFT OFF: no benefit could be measured.
Replay strengthens edges rather than node strength — deliberately, since
growing strength blindly was already tried (see 2.16) and entrenches
wrongly retrieved nodes.

**WHILE LOOKING FOR THEM, A DEFECT TURNED UP IN THE BENCH ITSELF.**
`tools/compare_sleep.py` judged sleep by node count and printed "sleep
compressed nothing: no clusters found". The check showed the opposite: a
cluster is found and folded, the abstract node is created, sources are
archived — from 18.3 to 20.3 archived nodes.

Node counts do not fall for a different, architectural reason. Archiving
only LOWERS the weight and strength of sources, while deletion by age is
gone — switched off for the sake of recall, which it raised by 18.6
points. Two decisions taken at different times contradict each other:
sleep files things into an archive nobody empties.

The bench now takes its numbers from `SleepReport` instead of guessing
from node counts, and prints the stages separately. An instrument denying
the work of a mechanism that did work is exactly the class of error this
project keeps catching — this time it was in the measuring device.

### 2.18 The correction mechanism was a shredder in a crowd of look-alikes

The near-duplicate bench is the product's stated niche, and for a day and
a half it was fixed from the RETRIEVAL side: importance moved to
accumulated strength, the ceiling sorted out, the reinforcement step
tuned. The cause was at the input.

Broken down over 200 near-duplicates:

    relevance only (no strength, no penalties)   5 hits out of 6
    the full search                              3 hits out of 6

So retrieval found the right thing, and what spoiled it happened AT WRITE
TIME.

**6078 supersessions over 200 writes.** The mechanism written for
corrections ("my dog is called Rex" -> "... Buddy") reads a neighbour as a
correction: a near-duplicate satisfies both its conditions — high semantic
similarity and incomplete word overlap. Every new write "corrected" thirty
other nodes.

**And it hits the true facts hardest.** The original resembles all of its
duplicates at once, so it collects the most penalties:

    strength of the six facts       0.0000
    strength of their duplicates    0.0292

The original ended up weaker than its own copies.

**THE RELATION IS NON-MONOTONE, WHICH IS INSTRUCTIVE:**

| supersessions | R@1 at 200 |
|---:|---:|
| 47 | **73.3%** |
| 98 | 56.7% |
| 265 | 33.3% |
| 1033 | 16.7% |
| 29246 | 53.3% |

When EVERYTHING is damaged, ranking does not suffer: the harm is uniform
and the shares survive. When damage is selective, it lands on the most
typical item — the true fact. The worst place is in between. The same
principle as in ranking and in sleep's homeostatic downscaling: the share
matters, not the absolute.

**WHAT WAS DONE — PATTERN SEPARATION.** A correction replaces ONE specific
memory. If a dozen nodes respond equally well to the new text, "which one
is being corrected" has no answer, and the honest answer is to touch none
of them and store separately. `pattern_separation_limit = 2`.

In the brain the dentate gyrus does this, and does it AT THE INPUT:
similar inputs get dissimilar codes BEFORE storage, so they will not
interfere later. We had been fighting the consequence.

Measured over five seeds with the default encoder:

| distractors | R@1 before | R@1 after | R@5 before | R@5 after |
|---:|---:|---:|---:|---:|
| 50 | 46.7% | **73.3%** | 100.0% | 86.7% |
| 200 | 53.3% | **73.3%** | 83.3% | 73.3% |
| 800 | 56.7% | **73.3%** | 83.3% | 73.3% |

Alongside: the share of important items in the top 5 on the ordering bench
went 83.3% -> 89.3%, the MRR gap +0.052 -> +0.056, retention 30 nodes ->
25 at the same 100%/100%. LongMemEval did not move by a point, including
the knowledge-update type (70%) — real corrections did not break.

**THE PRICE IS STATED PLAINLY: R@5 drops by 10 points.** The old recall
was bought by damaging everything around — weakened duplicates left room
in the top five. Now they keep their strength and crowd the answer out.
And R@1 became EQUAL to R@3: with all strengths equal, pure relevance
decides, and there are no intermediate positions.

This is half of a biological pair. The dentate gyrus separates; area CA3
COMPLETES — restoring the whole from a partial cue. The lost recall is
exactly a completion failure: the answer is in memory, but a fragment of a
cue does not reach it. Our spreading activation runs AFTER the winners are
chosen, so it completes from what was found rather than converging to it.
The second half is separate work.

THIS WAS FOUND BECAUSE THE CODE WAS SPLIT BY OWNER. While it sat in one
file, the question "where is our pattern separation" never arose: there
was no place where its absence showed. A split by function would not have
helped — there supersession legitimately lives next to search, and nothing
hints that biology has a separate mandatory step before storage.

### 2.19 Three mechanisms rejected by measurements that could not measure them

The costliest methodological error of the project, and it overturns three
recorded conclusions at once.

**What they had in common.** Candidate scoring was computed as

    combined = min(1.0, relevance + importance × share)

The hard clamp at one means importance HAS NOWHERE TO GROW: the heavier it
weighs, the more often the sum crosses one and candidates collapse into a
single score. Measured directly:

| importance share | R@1 |
|---:|---:|
| 0.15 | 73.3% |
| 0.40 | 30.0% |
| 0.80 | 13.3% |

**Three conclusions drawn under those conditions:**

1. "A larger reinforcement step from retrieval does not pay off" (2.16) —
   it won on three benches and lost on the fourth.
2. "The soft shoulder on the strength ceiling must stay off" (2.16) — it
   restored discrimination but dropped the near-duplicate bench from 83.3%
   to 61.1%.
3. "Credit by consequence is useless" — the negative branch moved one
   question out of thirty with no dose-response at all.

Every measurement was honest. Every conclusion was wrong: the mechanism was
tested where it could not act, by construction of the formula.

**WHAT WAS FIXED IS THE FORM, NOT THE NUMBERS.** Importance now MULTIPLIES
relevance:

    combined = relevance × (1 + importance × share)

No ceiling, no collapse. The ordering by relevance survives and is merely
stretched: a strong node overtakes one that matches equally well, but not
one that matches better. This is the third time in the project that
modulation beats addition — after the write gate and the strength ceiling.

**The retrieval threshold changed together with the form** (0.3 → 0.20),
and they cannot be changed apart. The first run at the old threshold showed
the external benchmark regressing from 75.0% to 60.0% on R@5, and the
verdict "the form is bad" was one step from being recorded here as a fourth
error. What saved it was that R@10 fell just as far: recall at ten is not
about ordering but about whether the answer reaches the result set at all —
so entries were being filtered out, not out-competed.

**The strength ceiling was lifted** (`strength_headroom` 0.0 → 1.0):
strength now counts linearly across its whole declared range up to 3.0.
Paired with multiplication, the very knob that dropped the near-duplicate
bench yesterday behaves monotonically and at no cost:

| shoulder | share of important in top 5 | MRR gap | near-duplicates R@1 |
|---:|---:|---:|---:|
| 0.00 | 86.3% | +0.010 | 83.3% |
| 0.60 | 96.1% | +0.036 | 83.3% |
| 1.00 | **100.0%** | +0.046 | 83.3% |

**Result of the three changes, counted as one:**

| bench | before | after |
|---|---:|---:|
| LongMemEval R@3 | 71.7% | **75.0%** |
| LongMemEval R@5 | 75.0% | **83.3%** |
| LongMemEval R@10 | 75.0% | **85.0%** |
| near-duplicates R@1 | 44.4% | **83.3%** |
| share of important in top 5 | 83.3% | **100.0%** |
| MRR gap | +0.052 | +0.046 |
| retention | 25 nodes, 100%/100% | unchanged |

By question type: multi-session 83% → 96%, preference 60% → 80%, temporal
83% → 92%, corrections unchanged.

**THE PRICE IS STATED.** The MRR gap is slightly below the old one, and
that is a consequence rather than an oversight: importance NO LONGER
OVERTAKES relevance. Praise settles disputes between equally fitting
entries; it does not override fitness to the question. "Important comes
first" now means "important comes first among equally fitting", and the
documents say so.

**THE LESSON IS WORTH MORE THAN THE CHANGE.** The project rule was: before
believing in a difference or its absence, check that the mechanism fired.
It turns out a second layer is needed — a mechanism can fire perfectly and
still have no way to act, if the channel is saturated or the scales are not
comparable.

Check not only that the mechanism fired, but that its action has somewhere
to go.

### 2.20 A common turn of phrase beat the one word that mattered

The fifth mechanism rejected by a measurement that could not measure it —
and this time the rejection was written into the README itself: "rarity
weighting was written and reverted, it did not change a single hit".

It was measured on `probe_semantic` — SIXTEEN FACTS. There, word frequency
is meaningless by construction: nearly every word occurs once, and rarity
distinguishes nobody from nobody. The conclusion was true for that bench
and inapplicable to haystacks of hundreds of turns.

**WHAT THE MEASUREMENT ON REAL DATA SHOWED.** Two failing LongMemEval
questions, taken apart:

    question  "I'm planning a trip to DENVER..."
    first     "I'm planning a trip with friends"        0.425
    second    "I'm planning a trip to California"       0.402
    evidence  "During my previous visit to Denver..."   0.363

    question  "I've been SNEEZING... my LIVING SPACE?"
    the whole top five starts with "I've been...", evidence not in ten

Keyword overlap counted "planning" and "Denver" alike. The shared opening
won.

**FIXED:** a matched word's weight falls with the number of entries that
contain it. The frequency comes from the FULL-TEXT INDEX — one COUNT per
word rather than a scan.

| | without weighting | with weighting |
|---|---:|---:|
| **preference R@1** | **20%** | **60%** |
| multi-session R@1 | 70% | 74% |
| overall R@1 | 60.0% | **65.0%** |
| R@3, R@5 | 75.0%, 83.3% | unchanged |
| R@10 | 85.0% | 83.3% |

Preferences — the worst question type in the project's history — got three
times better. The other benches did not move: near-duplicates 83.3%, share
of important in top 5 100.0%, retention 25 nodes at 100%/100%, liveness
green.

**A DEFECT ALONG THE WAY, CAUGHT BY A TEST.** The first version computed
frequency over the PRE-FILTER CANDIDATES, which made scoring non-local: a
node's score depended on who else got into the pre-filter, so the
pre-filter stopped agreeing with exhaustive search.
`test_prefilter_agrees_on_top_three` caught it — the same test that once
caught normalising strength by the maximum among candidates. The same
temptation: to compute something "among those who made it".

Computing frequency over the whole store fixed correctness but cost the
pre-filter's benefit (1805 → 2206 ms on 3 000 nodes). Taking it from the
full-text index costs about 4% instead.

**THE PRICE:** R@10 down 1.7 points, the "user" type from 62% to 50% at
ten. A rare word may be a typo or an accident, and then the weight goes to
the wrong place.

**BIOLOGICALLY** this is the second half of pattern separation — at the
level of REPRESENTATION rather than the decision to supersede. The dentate
gyrus makes similar inputs dissimilar by emphasising what differs and
damping what is shared. "What everyone has stops deciding" is exactly that
in our terms. The first half went in as 2.18.

### 2.21 The full set corrected the claim about preferences

Every change of the session was measured on a hard 60-question subset. A
control run over all 500 confirmed the overall gain and **refuted one
specific claim**.

**Confirmed:**

| | before | after |
|---|---:|---:|
| R@1 | 66.0% | **70.8%** |
| R@5 | 83.4% | **84.6%** |
| R@10 | — | 87.8% |
| turns stored | 24.7% | 24.7% |

By type (R@5): knowledge-update 88.4% → 90%, multi-session 92.6% → 92%,
assistant 96.4% → 95%, temporal 82.6% → 84%, user 67.1% → 70%, preference
46.6% → 50%.

**REFUTED: "preferences 20% → 60%".** That number came from FIVE questions
of the subset. Over the thirty preference questions of the full set, R@1
is still 20%, and the whole gain shows only at R@5: 46.6% to 50%.

The claim has been removed from the documents. The cause is ordinary and
familiar: at n = 5 a single question is worth twenty percentage points,
and three "lucky" questions produce exactly the jump I took for a result.
The rule "look at the denominator" is already written down in this project
— it was applied to the ordering bench, where 40% turned out to be a share
of 0.6 — and here I failed to apply it.

**Preferences remain the worst type and an open problem.** Rarity
weighting does not fix them: the evidence is found (R@5 50%) but does not
come first.

### 2.22 Two stores: the hippocampus and the cortex moved apart

Measurement had found no gain in splitting: search sped up by 5%, inside
the noise, while the cost was real (a shared id counter, edges crossing the
boundary). **It was done anyway, on a different ground.** The goal of this
project is fidelity to the machinery, not throughput. The hippocampus and
the cortex are different tissue with different trace lifetimes; holding
them in one table with a type column modelled a distinction the medium did
not have.

    episodes  (id, context, response, weight, strength, spike_strength, ...)
    cortex    (id, kind, text, meaning, occurrences, is_meta, ...)
    node_seq  (id INTEGER PRIMARY KEY AUTOINCREMENT)
    CREATE VIEW nodes AS SELECT ... UNION ALL SELECT ...

Three of my assumptions turned out wrong, and only the migration caught
them: `reward_expectation` belongs to the cortex too, decay applies to it
too, and `is_meta` must be stored rather than derived.

**What it bought, in substance:** capacity now limits the hippocampus
only. Cortical facts are never evicted — which is exactly why forgetting
became compression rather than plain loss (see 2.25).

### 2.23 Perception the organism grows for itself

Random indexing: each word gets a random sparse fingerprint, and its
meaning vector is the sum of its neighbours' fingerprints. Words in similar
company converge without ever meeting. Pure Python, no dependencies, no
downloads.

**It turns on as a fallback** — no encoder supplied AND no bundled model
present.

| | hits out of 16 |
|---|---:|
| bundled model | 9 |
| grown, 973+ exposures | 7 |
| no semantics at all | 5 |

Where a model exists, the grown one would REPLACE it (mixing two vector
spaces makes cosine meaningless) and lose four hits. Where none exists, the
choice is 7 against 5. Weightier than the probe: the whole test suite on a
machine with no model gives **22 failures without perception, 2 with it**.

**TWO DEFECTS FOUND BY CHECKING MY OWN WORK.** `encode` summed word vectors
UNNORMALISED, and vector length grows with how often a word was seen — so a
word's contribution to a phrase was proportional to its frequency, meaning
function words decided the phrase. Measured on live speech: in "я не ем
мясо уже три года" the contribution of "уже" was 13.5%, "не" 12.4%, and
"мясо" only 9.4%. Second: a neighbour left its trace with weight 1
regardless of whether it occurs everywhere or rarely, so every word
accumulated a common clerical component — similarity between deliberately
UNRELATED words crept from +0.04 to +0.12 over 2700 exposures.

Both are fixed by habituation to the frequent — the very conclusion already
proven by measurement elsewhere in this library (rarity weighting in
search). Noise between unrelated phrases fell 0.174 -> 0.082.

**A PREMISE WAS RETIRED.** "The bundled model adds nothing over word
overlap" rested on a substitution of quantities: the number of pairs
sharing a word (8) was reported as the number of pairs word search
actually solves (5). The real count is 9 against 5. The probe was fixed and
now computes its baseline with a second run instead of counting pairs.

### 2.24 The internal environment: significance stopped arriving from outside

Significance used to arrive as an argument and defaulted to zero — the
gate's emotional input was dead for the ordinary library user. There is
nothing wrong with the split itself: the hippocampus does not compute fear
or pleasure either; significance reaches it from the amygdala and the brain
stem. What was wrong is that those parts did not exist AT ALL.

**Three drives, all of them real:** crowding (at capacity the weakest
traces are lost for good), comprehensibility (how far the world yields to
prediction — a TWO-SIDED variable, since both chaos and routine are bad)
and coherence (how often the new contradicts the stored).

**MY MISTAKE, AND IT WAS STRUCTURAL.** The first version folded all three
into one arousal that MADE WRITING EASIER. For crowding that is a loop
eating itself: the fuller the store, the more eagerly it writes, the more
it evicts. In living tissue it is the opposite — acute stress raises
plasticity, chronic stress suppresses it; glucocorticoids inhibit LTP. So
two channels:

| channel | grows from | effect |
|---|---|---|
| URGENCY | incoherence | writing EASIER |
| STRAIN | crowding + incomprehensibility | writing HARDER |

Valence is computed from CHANGE, not level — constant crowding stops being
bad news; growing crowding is. The same machinery as reward prediction
error.

**A second mistake: mean instead of sum.** Strain was a weighted average,
so a drive at its limit yielded only 0.375 because the other one was
satisfied. Stressors summate; being fed does not cancel pain.

**MEASURED.** LongMemEval, 500 questions: 43.9% -> **25.9%** of turns
written, R@1 96.8% -> 96.0%. Two fifths fewer nodes for eight tenths of a
point. **On by default** since selectivity is the whole point of this
library.

**THE SYNTHETIC STAND MISSED IT, and that is the lesson.** Its
environments are EXTREME — surprise is either 0.00 or 1.00. Where it is
zero nothing was written anyway; where it is one, strain is not enough (see
below). Real conversation lives in the MIDDLE of the scale, and that is
where a threshold shift decides.

**THE CEILING OF SELF-PRESERVATION.** Density at zero emotion is
surprise/2, at most 0.5; the threshold under stress is 0.25 + strain·0.25
and hits the same 0.5. Strain can never overpower maximal novelty, only
match it: at modifier 0.55 the threshold reaches 0.58 and an incoherent
stream drops from 98.4% written to 0.0%. The channel works, but it switches
rather than grades. The default was left alone: it also affects those who
report overload themselves.

### 2.25 Forgetting became compression — but saves one theme in five

A stand was built for the question that had never been asked: when the
details are evicted, does anything remain?

| capacity | with cortex | without |
|---:|---:|---:|
| 25 | 5/5 | 4/5 |
| 10 | 3/5 | 2/5 |
| 3 | **2/5** | 1/5 |

At capacity 3 the hippocampus holds three episodes and the organism can
still answer two themes — the cortex answers for what is no longer there.
The gain, however, is exactly one theme and does not grow with pressure.

**The stand caught three defects in itself.** Eviction lives inside
`forget()` and the stand never called it — 42 episodes at capacity 30,
i.e. measurement over a memory where eviction had never once run. Theme
episodes were fed with `emotion=0.9`, so eviction spared them and all three
conditions gave the same number. And questions sharing no words with the
theme are not answerable by this system at all, so they would have measured
the encoder's weakness rather than the loss from forgetting.

### 2.26 The cortex tells a theme from a turn of phrase

Generalisation produced junk more readily than sense: at capacity 25, two
real themes out of nine facts; the rest were clerical fragments of the
filler ("рано утром", "прислал счёт"), with the words scrambled because the
theme was built from a SET intersection.

**The first criterion was inert.** A rarity check existed but applied ONLY
to a single word: two or more passed unchallenged, as if many shared words
were stronger evidence. It is exactly backwards — a real theme usually
shares ONE word, while a repeated turn of phrase shares many, because it is
the same phrase. Extending the check to every word did not help either:
share-of-episodes does not separate, since "рано утром" sits in five
percent of records.

**What separates is how familiar the word is in the organism's own graph of
language:**

| real themes | | clerical | |
|---|---:|---|---:|
| виолончель | 0.264 | утром | 1.000 |
| пенициллин | 0.264 | обедом | 1.000 |

Junk 7 -> 3 at capacity 25 and 4 -> 0 at capacity 10, with the compression
curve unchanged — the same answers from five times fewer facts.
LongMemEval: R@1 96.8% -> 96.6%, other columns unchanged.

**REVIEW DURING SLEEP.** Selection happens AT THE MOMENT the theme is
derived, and the organism knows little then: in its first days even
boilerplate is novel, so a turn of phrase that came early settled in
forever. Living tissue solves this by revisiting rather than by filtering
at the door — a consolidated memory becomes labile again on reactivation.
`review_cortex_facts` drops a theme once ALL of its words have become
familiar; one rare word is enough to keep it.

### 2.27 Spreading activation: 2787 firings, zero effect

The most frequent mechanism in the system — 2787 firings in an
eighty-message conversation — and its effect on quality had never been
measured by any of the five stands.

**LongMemEval with `--associations` returns numbers identical to the byte.**
The cause was not that the mechanism is useless:

    episode-to-episode edges after loading a haystack:   0
    word-to-word edges (the graph of language):        640

Links between memories are born in `_associate_with_recalled`: the new one
attaches to what was PULLED OUT of memory shortly before. The benchmark
loads the haystack and only then asks — no recall happens along the way, so
no associative network forms. **The same is true of all five stands in this
project**: every one of them is "write everything, then ask".

A stand with the living order was built — retrieve first, then store:

| condition | edges | k=1 | k=3 | k=5 |
|---|---:|---|---|---|
| load then ask | 0 | 3->3 | 4->4 | 4->4 |
| recall as you go | 45 | 1->1 | 1->1 | 4->4 |
| + competing for slots | 45 | 1->1 | 1->1 | 4->4 |

**Not one number moved.** The reason is twofold and structural. Pulled-in
nodes are APPENDED past `top_k` — at `top_k=3` the appended node was
consistently fourth. And letting them compete (`associations_compete`,
added and left off) does not help either, which cuts deeper: an
association's score is its SOURCE's score times the edge decay, hence
always below the source. It can only overtake what already ranked lower.

**Incidental and unpleasant:** recalling as you go MAKES later search
worse, 3/6 -> 1/6 at k=1. Retrieval touches nodes and shifts their
freshness. The living order of work costs quality in itself — a separate
job and a separate measurement.

**What this means.** A mechanism occupying a visible share of the search
code and advertised as multi-hop retrieval retrieves nothing. It is not
broken — it is wired in the wrong place: in living tissue, completing a
pattern is not an addendum to a list but part of recall itself. For it to
work, an association must earn its own score from connectivity rather than
inherit a fraction of its source's.

### 2.28 The out-of-the-box configuration: claimed, measured, reproducible

The project lacked **one claimed configuration with a number**. Measurements
were taken in different modes, figures wandered from 70.8% to 96.8%, and
anyone reading the README would have got something other than what it said.
There are now three named configurations, each with the command that
reproduces it.

| install | meaning | written | R@1 | R@10 |
|---|---|---:|---:|---:|
| bare, no dependencies | grown perception | 41.1% | 97.4% | 97.6% |
| the same, matched selectivity | " | 25.8% | 92.6% | 92.8% |
| `[semantic]` | potion-base-8M | 26.3% | **96.2%** | 96.4% |
| semantics off | shared words only | 18.3% | 88.2% | 88.2% |

**The first row cannot be read without the second.** The bare install looks
better only because it writes 1.6 times as much: novelty is judged against
a perception that starts empty. Held to the same selectivity it lands 3.6
points BELOW the model.

**THREE DEFECTS FOUND ALONG THE WAY, ALL OF THEM "DOES NOT WORK OUT OF THE
BOX".**

**1. The search threshold was calibrated for the semantic mode only.**
Without an encoder the score lives on a different scale, and the default
0.20 cut off almost everything: R@1 0.0% at 0.20, 31.2% at 0.12, 91.2% at
0.06 (80 questions). On the full set the library without a model returned
**9.2%** — it looked broken while it could search perfectly well; it simply
never handed the results over. `memory_search_threshold_lexical = 0.06` now
applies when there is no query vector at all: **9.2% -> 88.2%**. The
threshold is flat between 0.04 and 0.10, so this is not curve-fitting.

**2. Neither word list contained a single English function word.**
`_FUNCTION_WORDS` held 64 Russian entries and zero English ones — while the
default model is English and the external benchmark is English. The primary
language was served worse than the secondary one.

**3. But they could not simply be added, and that matters more than the fix
itself.** Extending the lists dropped the benchmark from 96.0% to 93.2%.
The culprit was not the keyword list but the **filtering before the
encoder**: a SENTENCE-level model is trained on natural text, and stripping
prepositions and copulas hands it something it has never seen. Filtering
only makes sense for a dictionary-style model (navec), where the phrase
vector is an average of word vectors.

The phrase now goes into potion whole — and the result beat the original:
96.0% -> **96.2%**, because the model also gets the punctuation that
rebuilding from tokens used to lose.

**INCIDENTAL: ONE TEST HAD BEEN PASSING BY ACCIDENT.** "у меня есть собака"
against "у меня есть кошка" stopped being distinguishable from a correction
(cosine 0.955) the moment filtering was removed. The test is Russian and the
model is English — a combination the library explicitly does not support,
and `requires_model` could not tell the difference. It was replaced with a
CAPABILITY check: the active model must rank "cat ~ kitty" above "cat ~
concrete". Three Russian tests now skip themselves honestly.

**`describe_setup()` WAS ADDED.** Semantics degrades silently and the gap
between modes reaches tenfold in R@1. The line answers "why did retrieval
get worse" before anyone asks.

### 2.29 The live order of work: measured at last, and the fear did not hold

All five stands in this project are built the same way: load the haystack,
then ask. A live application works differently — for every turn it FIRST
pulls from memory whatever it needs to answer with, and only then stores
what was said. Every number we had was therefore measured in a mode the
library will not actually run in.

A small stand (`compare_association`, six chains) suggested the live order
made retrieval WORSE: 3/6 -> 1/6 at k=1. That conclusion made it into the
README as a caveat: "read the table as an upper bound".

**The full set overturned it.** LongMemEval, 500 questions, `--live`:

| order | install | written | R@1 |
|---|---|---:|---:|
| load, then ask | `[semantic]` | 26.3% | 96.2% |
| **retrieve before each write** | `[semantic]` | 26.0% | **97.4%** |
| load, then ask | bare | 41.1% | 97.4% |
| retrieve before each write | bare | 41.7% | 96.8% |

With the model the live order is **better** by 1.2 points at the same
selectivity, and the explanation is at hand and biological: recall raises
the stability of what was recalled, so what gets used stops being
forgotten — the spacing effect, and it pays.

On the bare install the live order costs 0.6 points. Both differences are
five or six questions out of five hundred, so the honest reading is not
"the live order is better" but "the live order costs you nothing".

**A LESSON ABOUT STAND SIZE, AND AN EXPENSIVE ONE.** The gap "3/6 against
1/6" is two questions. The same six numbers shifted again when an unrelated
change to how phrases are fed to the encoder landed. A small stand is good
for showing that a mechanism EXISTS (edges are there or they are not) —
quality cannot be judged on it, and the conclusion that stood in the README
was exactly that kind.

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

To reproduce:

```
python tools/bench_longmemeval.py --data storage/bench/longmemeval_s.json \
    --encoder potion --threshold 0.0
```

**The flags are mandatory, and that is not a detail.** "Stock" refers to
the LIBRARY settings, which are indeed untouched. The bench's own
defaults differ: `--encoder none` and the `longmemeval_oracle.json`
dataset. Running it bare yields 6.0% (search with no semantics) or 90.7%
(the oracle set, whose haystack is almost entirely evidence sessions, so
recall does not vary with k) — and neither number is comparable with the
table above. Both mistakes were made while re-checking this very table.

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
barely occurs — see §4.6, where the same system gains 44 to 50 points once the
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

**What the assistant volunteers is what the user marked.** When memory is
asked a general question — "what do you remember about me" — and fills a
prompt on its own, `compare_ordering.py` measures which memories get in:

| | selectivemem | random order |
|---|---:|---:|
| share of praised material in the top 5 | **88.9%** | 50% |
| MRR, praised topics | 0.889 | |
| MRR, ordinary topics | 0.833 | |

Eighteen topical memories reached the answers across three seeds, so the
share is a measurement rather than a coin toss — an earlier version of
this bench reported 40% on a denominator of 0.6 and was rightly ignored.

The MRR gap is small (+0.056) and consistent across seeds. Ranking by
earned strength is a nudge on targeted questions and decisive on open
ones: nine of ten memories the assistant volunteers are the ones the user
called important.

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

**Ranking by accumulated strength recovers all of it — but only when
strength has something to tell apart.** Replacing node weight, which is
dominated by age, with strength, which no clock touches, on a run where
the six facts have been reinforced (`--compare-strength`):

| Importance from | R@1 at 50 | R@1 at 200 | R@1 at 800 |
|---|---:|---:|---:|
| node weight | 44.4% | 44.4% | 61.1% |
| **accumulated strength** | **88.9%** | **94.4%** | **100.0%** |

With navec, the Russian model, the separation is cleaner still: 50.0%
against 100.0% at all three levels. The mechanism does not depend on
which encoder is installed.

THIS ENTRY USED TO READ "+33 points, 83.3% against 50.0%", and that
number has been withdrawn as unmeasured: strength ranking switched on and
off produced BYTE-IDENTICAL numbers.

The reason turned out to be subtler than first written, and is worth
stating precisely. It is not that the nodes hold equal strengths — they
do not: near-duplicates push each other down through the supersession
penalty, and by the end of a run the six facts sit at 0.000 while the
distractors average 0.029. It is that STRENGTH AND WEIGHT HOLD THE SAME
VALUE. The penalty hits both equally, and nothing else in this bench ever
moved them apart. Choosing between two equal numbers cannot change
anything.

Weight and strength diverge from three things, and the bench did none of
them before measuring:

| mode | share of nodes where weight == strength | mean gap |
|---|---:|---:|
| writes only | 100% | 0.000 |
| writes + decay | 9% | 0.090 |
| writes + retrieval | 48% | 0.234 |
| writes + approval | 25% | 0.300 |
| all together | **0%** | **0.567** |

That table also answers whether the two variables should be merged into
one: no. In ordinary operation they diverge for EVERY node, and coincide
only in the degenerate mode the bench happened to sit in.

The fifth instance of the false "no difference" from 2.13, caught by the
same habit — check whether the mechanism fired before believing a result.

The bench is fixed: the six facts now earn strength the way an
application would mark them, and the difference became both reproducible
and twice as large.

On LongMemEval the same change moves nothing at all (76.0% both ways),
because there are no near-duplicates there to tell apart and nothing is
reinforced during ingestion. That contrast is the product's niche stated
as a measurement: one user, one subject returned to for months — not a
store of unrelated topics.

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
