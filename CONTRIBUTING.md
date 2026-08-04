# Contributing

Before describing the process, one requirement needs explaining, because
it usually raises questions.

## Why a contributor licence agreement is needed

`selectivemem` lives under a dual licence: AGPL-3.0 for everyone, and a
commercial licence for those AGPL does not suit — closed games, closed
SaaS, products shipped to customers. That is possible precisely because
the rights to all of the code belong to one person.

Accept someone's patch without an agreement and its author remains the
rightsholder of their part. From that moment on **a commercial licence
can no longer be granted for the project** — you cannot license what you
do not own. This is usually discovered a year later, when there is
nothing left to rewrite.

So: for your code to enter the project, one signed statement is needed.
That is not a formality and not greed — without it the project loses the
ability to exist in its present form.

## The agreement

By opening a pull request you confirm the following. Writing this once in
the PR is enough:

> I have read CONTRIBUTING.md and grant the rights to my contribution on
> the terms described there.

What that means:

1. **You are the author.** The code was written by you and contains no
   fragments of someone else's work that you have no rights to. If you
   wrote it under contract or at work, you have made sure your employer
   makes no claim to it.
2. **You grant the project's rightsholder a non-exclusive, perpetual,
   irrevocable, worldwide right** to use your contribution, modify it and
   license it to third parties **on any terms, including commercial and
   closed ones**.
3. **You keep your own rights.** This is not a waiver of authorship: you
   remain free to use your code anywhere and however you like.
4. **You give no warranties.** The contribution is provided as is.

If those terms do not suit you, that is entirely fine. Open an issue
instead: an idea, a measurement or a reproduction of a defect is valuable
in itself and requires no agreement at all.

## What counts as a good contribution here

The project has one methodological quirk, and it is stricter than the
usual style rules.

**A claim without a measurement is not accepted.** "It got faster", "this
is more correct", "this will improve quality" are not arguments. Numbers
before and after, produced by a run, are. The repository has benchmarks
for exactly that:

```bash
python tools/check_liveness.py          # does every mechanism still FIRE?
python tools/compare_interference.py    # the niche: near-duplicates
python tools/compare_retention.py --balanced
python tools/compare_dialogue.py        # recall and write interleaved
python tools/compare_memory.py --seed 42
python tools/simulate_learning.py --messages 120 --session-length 20 --gap-hours 8
python tools/probe_semantic.py
```

**Run `check_liveness.py` first, and take a zero seriously.** Seven
mechanisms in this project were described in the README, covered by green
tests and never ran once in live use — none of them a coding error, each
an interaction between a threshold and a decay rate that nobody had
measured end to end. An ordinary test checks that a mechanism behaves
correctly when called with suitable data; it does not check that such
data ever arises.

That bench counts OUTCOMES, not calls, and the distinction is not
pedantic: consolidation was invoked faithfully and decided "do nothing"
in 100% of cases, and the first version of the bench called it alive.

**Before measuring any mechanism, check that it fires at all.** Four
ablations in this project returned byte-identical numbers in a row
because the mechanism under test was never triggered by the bench —
"no difference" was read as "no use" every time.

**A negative result is a result too, and it does not get thrown away.**
`tools/compare_memory.py` used to show that the organism was no better
than a random sample on uniform questions, and it sat in the repository
for exactly that reason. If your change made something worse, write that
in the PR rather than picking a more flattering measurement.

**A comment explains WHY, not WHAT.** What the code does is visible from
the code. A comment is the place for what is not: which measurement led
to this number, what defect existed before, why the obvious approach did
not work.

## Practicalities

- Tests must pass: `python -m pytest tests/ -q` (currently 309). Green
  BEFORE the commit, not in the same command as it — a commit with two red
  tests has already reached the branch that way.
- Changes inside `selectivemem/` must not add dependencies: the package
  lives on the standard library, and
  `test_memory_package_is_self_contained` guards that.
- If you change a number recorded in README or AUDIT, re-measure it and
  update both. `tests/test_readme_examples.py` runs the README examples
  verbatim.
- Changes to `core/`, `bot.py` and `tools/` belong to the showcase and are
  not shipped in the package: it lives in a separate repository.
- The package is in English, including comments. The showcase is in
  Russian — it is a Russian-language demo.

## Three rules for a measurement

The project has twice withdrawn a published number and five times rejected
a useful mechanism on the strength of a bad measurement. Every one of those
measurements was honest and every conclusion was wrong. Hence three checks,
without which a number means nothing.

**1. The mechanism must FIRE.** Count firings; do not infer them from the
result. `tools/check_liveness.py` exists for this: it once declared four
ablation measurements useless when in fact the mechanism had never run. The
tell-tale sign is byte-identical numbers across different configurations.

**2. Its action must have SOMEWHERE TO GO.** A mechanism can fire perfectly
and still change nothing if the channel is saturated. Candidate scoring was
`min(1.0, relevance + importance·share)`, and the hard ceiling collapsed
candidates into one score as soon as importance grew heavier: a share of
0.15 gave R@1 73.3%, a share of 0.80 gave 13.3%. Three useful mechanisms
were rejected in a row under those conditions.

**3. The setting must ARRIVE.** Patching classes and settings from outside
silently fails to apply — when the import happens inside a function, when
the parameter is passed explicitly, when the name is rebound. That happened
four times in a single session. The one reliable method is to change the
default in `settings.py` and run the bench normally. If you do patch, print
a column showing what was ACTUALLY applied: twice, only that column
prevented a false conclusion.

**And in general.** When comparing two scoring forms, check that the
retrieval threshold is comparable for both. Multiplication and addition
produce different scales, and at a single threshold the first run showed a
15-point regression — the verdict "the form is bad" was one step from the
audit. What saved it was the drop in R@10: recall at ten is not about
ordering but about whether the answer reaches the result set at all.

Timing measurements belong on an idle machine and must be compared WITHIN a
single run: the baseline on identical code wanders by a factor of two.

Русская версия: [CONTRIBUTING.ru.md](CONTRIBUTING.ru.md).
