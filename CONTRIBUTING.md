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
python tools/simulate_learning.py --messages 120 --session-length 20 --gap-hours 8
python tools/compare_retention.py --balanced
python tools/compare_memory.py --seed 42
python tools/probe_semantic.py
```

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

- Tests must pass: `python -m pytest tests/ -q` (currently 286).
- Changes inside `selectivemem/` must not add dependencies: the package
  lives on the standard library, and
  `test_memory_package_is_self_contained` guards that.
- If you change a number recorded in README or AUDIT, re-measure it and
  update both. `tests/test_readme_examples.py` runs the README examples
  verbatim.
- Changes to `core/`, `bot.py` and `tools/` belong to the showcase (see
  [DEMO.md](DEMO.md)) and are not shipped in the package.
- The package is in English, including comments. The showcase is in
  Russian — it is a Russian-language demo.

Русская версия: [CONTRIBUTING.ru.md](CONTRIBUTING.ru.md).
