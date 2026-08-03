# Commercial licence for selectivemem

The memory core is distributed under **AGPL-3.0** (see [LICENSE](LICENSE)).
That is a full free-software licence: study it, change it, use it —
including commercially — subject to one condition.

## The condition in AGPL that this page exists for

AGPL requires that **derivative work be released under AGPL as well**.
Unlike ordinary GPL, this covers not only distributing the program but
also providing access to it over a network (§13). In practice:

| What you are doing | Does AGPL work for you |
|---|---|
| A personal project, research, learning | yes |
| An open product under an AGPL-compatible licence | yes |
| An internal tool not reachable by third parties over a network | yes |
| **A closed game with selectivemem embedded** | no |
| **A closed SaaS where the memory runs on your server** | no |
| **A closed product shipped to customers** | no |

Those bottom three rows are what a commercial licence is for.

## What a commercial licence gives you

- The right to embed selectivemem in a closed product without disclosing
  your source.
- The right to serve it over a network without falling under §13 of AGPL.
- Priority support and a say in the roadmap.

Separately, as add-ons:

- **Game development.** A deterministic mode (identical input yields
  identical output — required for replays, saves and QA), offline
  operation, no heavy dependencies. Plus paid integration for your
  engine.
- **B2B.** Multi-tenancy, observability and a dashboard, adapters for
  production embedding models, guarantees around export and migration.

## What it costs

The exact figure is agreed case by case: it depends on scope and scale.
But bargaining blind is unpleasant for both sides, so here are the
reference points.

| Who | Model | Order of magnitude |
|---|---|---|
| Indie developer, product revenue below €50,000 a year | **free**, no licence needed | 0 |
| Indie above that threshold | per product, one-off | €200–1,000 |
| Studio, commercial title | per title + support | €3,000–15,000 |
| Closed SaaS | annual, by scale | €5,000–30,000 / year |
| Embedded in a product shipped to customers | OEM, annual | from €10,000 / year |
| Custom integration for an engine or stack | per day | separate |

**The revenue threshold is arithmetic, not generosity.** An indie who is
not yet earning will not pay in any case: they will either use something
else or use nothing. And if the product takes off, the licence is
discussed in a very different conversation. The game industry works this
way (FMOD, Wwise), and it works.

What the price includes: the right to use (see
[scope](LICENSE-COMMERCIAL.md)), updates for a year, a channel for
questions. What it does not: work customised for you and a guaranteed
response time — those are separate.

The agreement is [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md). It is a
draft: a lawyer reviews it before signing.

## What to say honestly, up front

selectivemem's measured advantage is **telling apart memories that look
alike**. Bury six facts under 200 near-duplicates — same words, different
subject — and ranking by earned strength puts the right one first **83%
of the time against 50%** for the usual weight-and-recency scoring.

That is the niche stated plainly: one user, one subject returned to over
months. Support histories, a player and one NPC, a personal assistant
after a year. **On a store of unrelated topics the same change moves
nothing at all** (76.0% either way on LongMemEval) — there, an ordinary
vector store is no worse, and we will say so before you buy.

**What it costs.** On the external LongMemEval benchmark, all 500
questions at stock settings: R@5 **84.0%** while storing 24.7% of the
turns. The same engine with the write filter removed scores 93.2%. So
selective writing costs about nine points of recall and saves three
quarters of the storage.

**A claim we withdrew, because you would find it anyway.** Until recently
this page led with "+40 pp selective retention". That number was measured
honestly and meant something other than advertised: the memory was not
ranking important material higher, it was deleting the rest. Removing
that deletion raised recall by 18.6 points and collapsed the gap to zero.
The audit records this in full (§4.1), and `compare_retention.py` shows
+0 today. What replaced it is arguably the better claim: **34 nodes
against 52 for a random sample, with complete recall against its 56–67%.**

All of this is measured by benchmarks that sit in the open repository.
You are welcome to rerun them; they are the reason we found the mistake
above ourselves.

If what you need is completeness of search, use a vector store.
selectivemem is about which memory surfaces first when many look alike.

## Contact

Write to selectivemem@gmail.com describing your project and how you
intend to use it. The price depends on scope and delivery model; for
indie games and small teams the terms are separate.

Русская версия: [COMMERCIAL.ru.md](COMMERCIAL.ru.md).
