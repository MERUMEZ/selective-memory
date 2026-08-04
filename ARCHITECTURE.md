# How the engine works, step by step

A walkthrough of the current code: every function, every variable, and at
each step what corresponds to it in the brain and where we diverge.

Russian pair: [ARCHITECTURE.ru.md](ARCHITECTURE.ru.md).

---

## PART I. A message on its way in

Entry point: `memory.py:observe(text, response, emotion=0.0, load=0.0, timestamp=None)`.

The order of steps mirrors the living one: **be surprised first, learn
afterwards**. The other way round would mean being surprised by what you
memorised a moment ago — that is, never being surprised at all.

### Step 1. How new is this

`neocortex.py: compute_surprise(text) → SurpriseResult`

```
tokens         = words at least lexical_min_token_length (2) long
known          = {word: (id, weight)} — one query for all of them
mastery        = vocabulary_mastery_min_weight = 0.18

familiarities[i]      = min(1, word_weight / 0.18)  or 0 if unseen
lexical_surprise      = 1 − mean(familiarities)

pair_familiarities[i] = min(1, edge_weight / edge_activation_threshold 0.3)
structural_surprise   = 1 − mean(pair_familiarities)

total = 0.6·lexical + 0.4·structural
total *= min(1, word_count / 3)      (surprise_full_content_tokens)
```

That last line corrects for HOW MUCH content an utterance carries. Without
it, "uh-huh" and "my daughter Lisa is allergic to peanuts" both scored 1.0
on an empty memory, and the library filled up with interjections.

**In the brain.** Novelty detection is area CA1's job: it compares what
arrived against what cortex predicted from context. Novelty drives the
hippocampus → ventral tegmental area → dopamine loop, and that loop
enhances plasticity in the hippocampus itself.

**Divergence.** Our novelty is derived entirely from cortical word
statistics rather than from a failed prediction. The organism is surprised
by unfamiliar letter sequences; a paraphrase of a familiar thought
surprises it more than it should.

### Step 2. Let it in or not

`plasticity.py: evaluate(emotion, surprise, load) → PlasticityDecision`

```
density   = 0.5·emotion + 0.5·surprise
threshold = min(1, base_plasticity_threshold 0.25
                   + load·plasticity_stress_modifier 0.25)
is_spike  = density ≥ threshold
```

**In the brain.** The induction threshold for long-term potentiation.
Noradrenaline and dopamine shift it; under overload cortisol suppresses
plasticity — that is our `load`.

**Divergence, and it hurts the ordinary user.** The library does not
compute emotion; the application supplies it, defaulting to `0.0`. So half
the formula is dead and the threshold rests on surprise alone.

The form is wrong too: in the brain noradrenaline does not add to novelty,
it MULTIPLIES the plasticity novelty has already opened. The alternative is
written — `gate_emotion_gain`, `density = surprise·(1 + emotion·gain)` —
and switched off, because at zero emotion the product would shut the gate
entirely.

### Step 3. Is this a correction

`hippocampus.py: find_superseded(text) → List[SupersededNode]`

A separate entrance into memory, **bypassing the gate**. A correction is
unsurprising by nature — the same words, one of them different — and
measured on a live database it scored 0.165 against a threshold of 0.35.
The single most important message of a conversation was the one the gate
turned away.

```
candidates = db.fetch_candidates_by_text(words, supersede_scan_limit 200)
for each:
    similarity = cosine of vectors
    overlap    = share of shared significant words
    keep if  similarity ≥ 0.8   (contradiction_topic_threshold)
         and overlap  < 0.85    (contradiction_repeat_threshold)
         and overlap ≥ 0.25     (contradiction_min_overlap)
```

Then pattern separation, `_separate_patterns`:

```
if len(found) > pattern_separation_limit (2):  return []
```

**In the brain.** The comparison is area CA1. Pattern separation is the
dentate gyrus: it makes similar inputs dissimilar BEFORE storage, so two
close events are stored apart and do not interfere at retrieval.

**What was broken.** Without separation the correction mechanism read a
neighbour as a correction: 6078 supersessions over 200 writes, and the
strength of the true facts fell to 0.0000 against 0.0292 for their copies —
the original ended up weaker than its own duplicates. After the fix,
near-duplicates give R@1 83.3% against 44.4%.

### Step 4. The write

`hippocampus.py: save_connection(context, response, weight, timestamp) → node_id`

```
weight = decision.density
if anything is superseded:  weight = max(weight, superseded weight)
db.insert_node(context, response, weight, node_type="episodic")
vector = _encode(context + " " + response)  →  db.update_embedding
for each superseded: supersede_node(id)
```

Inheriting the weight is mandatory: otherwise the write would happen and
the stale version would still win the search.

On insert `strength = weight`. From there the two diverge, and **they must
not be merged** — measured: in live operation they differ for every node
and coincide only when nothing but writing happens.

**In the brain.** Hippocampal one-shot encoding. The hippocampus stores not
the details but a pointer to cortical traces.

**Divergence.** Our node stores the text itself; there is no split between
index and content.

### Step 5. Binding to what was active

`memory.py: _associate_with_recalled(node_id, ts)`

The new node is linked to whatever was retrieved **before** it appeared —
up to `associate_recalled_limit = 3` of them, with edges stepped by
`edge_boost_step = 0.15`.

**In the brain.** The recurrent network of area CA3, Hebb's rule: what
fires together wires together.

### Step 6. The short-term buffer

`prefrontal.py: WorkingMemory.add_message(...)`, capacity `stm_capacity = 16`.

When full it calls `consolidation.py: consolidate_from_stm(entries)`, which
judges the episode by emotion and mean surprise and reaches one of three
verdicts: emotional node, structural node, routine noise.

**Off by default** (`consolidate_from_stm = False`).

**In the brain.** Prefrontal cortex holds what is happening; the
hippocampus takes it at an episode boundary.

**Divergence.** For us this is described as part of memory and does not run
for the user.

### Step 7. Learning the language

`neocortex.py: process_language_input(text)`

Words and syllables become `word`/`syllable` nodes with growing weights,
and edges of co-occurrence are placed between adjacent words. This is the
statistic that step 1 measured surprise against.

**In the brain.** Cortical statistical learning — slow, overlapping, not
tied to any single occasion.

**A serious divergence.** The vocabulary lives **in the same table** as
memories. This has already caused a live defect: counting nodes included
the lexicon, the threshold was crossed at the ninth message, and sleep ran
on every message after that — two language-model calls per turn in
production.

### Step 8. Recording the action for reinforcement

`reinforcement.py: record_action(user_input, bot_output, node_id, action_type)`

Remembers the pairing "what was done → which node was touched", so the next
rating knows what it applies to. Without it `feedback()` has nothing to find.

**In the brain.** An eligibility trace: the synapse is tagged and waits for
a dopamine signal.

---

## PART II. A question on its way out

`memory.py: recall(...)` → `retrieval.py: search(query, top_k, timestamp,
with_associations, touch)`

### Step 1. The cue

```
query_keywords = _extract_keywords(query)     ← WORD_PATTERN, len ≥ 3,
                                                 not a stop word
query_vector   = _encode(query)               ← computed ONCE per search
rows           = db.fetch_searchable_nodes()  ← no lexicon, no summaries
```

If there is no vector, a warning is logged **once per graph lifetime**.
Staying silent is not an option: the caller gets emptiness with no hint as
to why, and half a day of benchmark debugging once went that way.

**In the brain.** The entorhinal cortex is the sole gateway into the
hippocampus and back.

**Divergence.** There is no gateway; every region reaches into the database
itself.

### Step 2. The pre-filter

`retrieval.py: _prefilter(rows, query_keywords, query_vector, top_k)`

Computes only the cheap parts — words and semantics — picks candidates, and
runs the expensive part on the survivors alone.

Profiled over 10 000 nodes: `SequenceMatcher` ate **82% of search time**
(4.49 s out of 5.44), semantics took 6%. The first search after the fix:
781 ms → 14 ms.

### Step 3. Scoring a candidate

```
idf            = db.document_frequency(query words)   ← from the index
keyword_score  = share of matched DISCRIMINATING POWER, not of words
fuzzy_score    = SequenceMatcher(query, context).ratio()
semantic_score = cosine(query_vector, node_vector)

relevance = 0.3·keyword + 0.1·fuzzy + 0.5·semantic

own       = strength (or weight if strength is null)
base      = _importance_base(own, strength_headroom 1.0)  ← no ceiling
combined  = relevance × (1 + base·memory_weight_influence 0.15)

keep if combined ≥ memory_search_threshold (0.20)
```

**IMPORTANCE MULTIPLIES RATHER THAN ADDS.** The old form
`min(1, relevance + importance·share)` ran into a ceiling: the heavier
importance weighed, the more often the sum crossed one and candidates
collapsed into a single score. A share of 0.15 gave R@1 73.3%, a share of
0.80 gave 13.3%. Importance had nowhere to grow, and three useful
mechanisms were rejected in a row under those conditions.

Multiplication preserves the ordering by relevance and merely stretches it:
a strong node overtakes one that matches equally well, but not one that
matches better. **"Important comes first" now means "first among equally
fitting".**

The threshold changed TOGETHER with the form (0.3 → 0.20): the two forms
produce different scales, and at a single threshold the first run showed a
false 15-point regression.

**WORDS ARE WEIGHTED BY RARITY.** Keyword overlap counted "planning" and
"Denver" alike, so an entry sharing only the opening phrase beat the one
sharing the single word that mattered. A word's weight now falls with the
number of entries containing it; the frequency comes from the full-text
index, one COUNT per word (about 4% of search time). Preference questions
went from 20% to 60% R@1.

The threshold of 0.20 separates **only with a working encoder**: with one,
relevant queries score 0.30–0.65 and irrelevant 0.18–0.24. Without one it
is 0.178 against 0.167 — nothing to separate, and lowering the threshold
returns noise instead of answers.

`_importance_base` is where a defect was found: `min(1, own)` meant
accumulated strength above one did not exist, while `strength_max` was
declared as 3.0. The ceiling is lifted (`strength_headroom = 1.0`), and
paired with multiplication that raised the share of important material in
the top five from 83.3% to 100.0%.

**In the brain.** Trace strength plus fit to the cue.

**Divergence.** The brain does not score traces independently: there,
candidates compete for activation and the strong one SUPPRESSES its
neighbours. We score each candidate alone.

### Step 4. Order and selection

```
scored.sort(by combined, descending)
scored = _rerank_by_importance(scored)   ← rerank_band 0.0, i.e. off
top_matches = scored[:top_k]
```

### Step 5. Retrieval reinforces

`touch_node(id)` → `db.update_last_accessed(id)` — one UPDATE, four fields:

```
last_accessed   = now
last_decayed_at = now                          ← decay origin moves forward
stability      *= stability_growth_factor 1.5  up to stability_max 40
strength       += strength_use_step 0.05       up to strength_max 3.0
```

The `touch=False` flag exists because the supersession check also calls
search, and without it **every write** quietly reinforced the neighbours.
A measurement caught nodes at strength 2.90 earned on internal checks.

**In the brain.** The testing effect: successful retrieval strengthens a
trace more than restudy does.

**Divergence, and this is our main unsolved problem.** The WHOLE result set
is strengthened, not the answer that turned out to be right. While that
holds, any speed-up of reinforcement also speeds up the entrenchment of
errors.

### Step 6. Spreading activation

`retrieval.py: get_associated_nodes(node_id)`

Neighbours of the winners are pulled in through edges of weight ≥
`edge_activation_threshold = 0.3` and join the result with a weakened
score. Over an 80-message conversation this fires about 2800 times.

**In the brain.** Area CA3 — completing a whole from a part.

**Divergence.** Ours runs **after** the winners are chosen, so it completes
from what was found. Real CA3 works the other way: a partial cue converges
on the nearest stable state by itself.

---

## PART III. What happens when nobody asks

### Forgetting

`synapses.py: apply_decay(now)`

```
stability     = row["stability"] or stability_initial
effective_t0  = age_t0(type) × stability   ← 25200 s episode, 2592000 word
decay_factor  = exp(−decay_rate 0.05 × dt / effective_t0)
new_weight    = max(floor, weight × decay_factor)
```

Edges decay separately, `edge_decay_rate = 0.08`. Capacity eviction is off
(`memory_capacity = 0`).

**Nothing is deleted by age**: `delete_on_decay = False`, and that raised
recall on the external set by 18.6 points.

**In the brain.** Forgetting is mostly interference, not decay by the clock.

**Divergence.** The migration is half done. Ranking lives by interference —
importance is a share of accumulated strength, which no clock touches.
Decay itself is still an exponential of elapsed time. Two theories in one
engine.

### Sleep

`memory.py: sleep(timestamp, summarise)`

```
1. replay(...)             ← sleep_replay_nodes 0, off
2. downscale_edges(...)    ← sleep_downscale_factor 1.0, off
3. run_synaptic_pruning()  ← cuts weak edges, then nodes without edges
4. find_hub_clusters() + create_abstract_node()
```

**In the brain.** In slow-wave sleep the hippocampus replays the day's
sequences in bursts, selectively — what led to reward replays more often.
In parallel comes homeostatic downscaling: all synapses weaken
proportionally, order survives, and whatever hung by a thread drops out.

**Divergence.** The first two stages did not exist at all: sleep could
clean but not consolidate. Both are written and OFF — they fire, but no
benefit could be measured.

A separate contradiction: consolidation ARCHIVES its sources by lowering
their weight and strength, but deletion by age is gone. Sleep files things
into an archive nobody empties.

---

## PART IV. What can be brought to a biological form, and what cannot

### Boundaries: three things that cannot be reproduced

**Emotion cannot be computed.** The amygdala evaluates an event relative to
the organism's goals: a threat to what, a benefit to what. A library has no
body, no needs and no stakes — so there is nothing to evaluate against.
Emotion can only be RECEIVED. That is a boundary between memory and
organism, not a gap to close.

What needs fixing here is not the missing amygdala but the formula: emotion
is currently averaged with novelty, whereas in the brain it multiplies the
plasticity novelty has already opened. At zero, an average gives half; a
product gives nothing.

**Internal time is impossible for a library.** The brain builds time from a
slowly drifting context, but drift requires continuous experience. A
library is called in bursts; between calls nothing happens and there is
nothing to drift. Internal time makes sense for a continuously living
organism — the showcase — not for the package. An external clock is
unavoidable here.

**Neurogenesis cannot be imitated.** The dentate gyrus adds neurons
throughout life, and that is what gives it spare discriminating capacity.
Our representation of a node IS its text and vector; there is nothing to
grow. Patterns can be separated by rules, not by growing capacity.

### Contradictions that must be resolved by choosing

**Deferred writing against the product's headline.** Our main measured
claim — "writes 24.7% of turns, costs nine points of recall" — is about
selection AT THE INPUT. Deferred consolidation means the opposite: write
everything cheaply, then fail to keep it. More faithful biologically, but
the product's story changes and every number needs deriving again.

**Interference-only forgetting against the retention bench.**
`compare_retention` measures what survives two weeks of silence. Pure
interference says silence forgets nothing; what forgets is new material
crowding out the old. Finishing that migration would leave the bench
measuring nothing and make "memory fades with time" false. The nastiest
conflict on the list: the biology we chose contradicts the story we tell.

**Separation against completion.** Measured: pattern separation gave R@1
+20 and R@5 −10. Completion would restore recall but pull in more
competitors — exactly what separation defends against. The two are
antagonists by design; in the brain their balance is anatomical, for us it
would have to be found by measurement.

**Credit by consequence against archival use.** A consequence is visible
only where the conversation continues: no correction arrived, the question
was not repeated. If the library is used as a store — write today, read in
a month — consequences do not exist. The mechanism serves dialogue and
stays silent for archives, and that should be said plainly.

### What the plan produced

Eight items were planned from divergences with biology. Within a day it
became clear that **a divergence with biology and a bottleneck are
different things**, and the plan showed it on the very first item.

| # | work | outcome |
|---|---|---|
| 1 | credit by consequence | **measured, no benefit.** One question in thirty, no dose-response. But it led to the scoring-form finding, worth 8 points on the benchmark |
| 2 | completion from a partial cue | **dropped.** The debt it was meant to repay was settled by the scoring form: R@5 returned to 83.3% on its own |
| 3 | deferred consolidation | **measured, harmful.** R@1 65.0% → 46.7%: capture writes everything near a spike and floods the store. Its premise ("preferences are lost at write time") was also wrong |
| 4 | two stores | not started |
| 5 | semantic surprise | not started |
| 6 | the gateway | not started |
| 7 | sleep: replay and downscaling | written, off, no benefit measurable |
| 8 | forgetting by competition | not started, the conflict with the retention bench is unresolved |

**WHAT WORKED INSTEAD.** Three changes, none of them in the plan. All three
came from measurements rather than from the list of divergences, and all
three are about telling similar things apart:

| change | where it came from | result |
|---|---|---|
| pattern separation at the input | analysis: the correction mechanism was penalising originals, 6078 supersessions per 200 writes | near-duplicates R@1 44.4% → 83.3% |
| importance multiplies relevance | analysis: importance had nowhere to grow, `min(1, ...)` collapsed candidates | R@5 75.0% → 83.3%, R@10 → 85.0% |
| words weighted by rarity | analysis: a common turn of phrase beat the one word that mattered | preferences R@1 20% → 60% |

All three are the same biological place: the dentate gyrus makes similar
inputs dissimilar. The first acts on the decision to supersede, the second
gives differences room to grow, the third damps what is shared and
emphasises what discriminates.

### The rule the plan should have been built on

A divergence with biology says what we DO NOT HAVE. It does not say that
the missing piece will solve our problem — for that you need a measurement
showing where answers are actually lost.

Three times in one day the measurement pointed at the same place, and all
three times it was not the place the list of divergences pointed to.

---

## Summary of divergences

| place | ours | the brain's | state |
|---|---|---|---|
| pattern separation | **at input and in representation** | dentate gyrus | done |
| novelty | word statistics | prediction error | open |
| emotion | a parameter, 0 by default | amygdala, **multiplies** | boundary: cannot be computed |
| writing | irreversible decision | labile trace, consolidated later | **measured, harmful** |
| reinforcement | the whole result set | by consequence | **measured, no benefit** |
| forgetting | exponential in hours | competition | open, conflicted |
| storage | one table | hippocampus and cortex apart | open |
| gateway | none | entorhinal cortex | open |
| completion | after selection | convergence from a cue | dropped: debt settled |
| sleep | cleans | replays and downscales | written, off |

Four of the nine were closed in a day: one built, two measured and
rejected, one dropped as unnecessary. Five remain, and two of those are
boundaries rather than tasks.
