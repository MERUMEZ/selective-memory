# Cutting a release

## The rule that makes care worthwhile

**A version number on PyPI is burned forever.** Upload `0.1.0` and that
number is taken, even if the release is later deleted. A mistake in
anything at all — a typo in the README included — means shipping
`0.1.1`.

**A project cannot be renamed.** Not at all. You can only publish under a
new name and leave the old one sitting there marked "do not install".

Therefore: the name and the public API must settle BEFORE the first
release to the real PyPI.

## One-time setup

**We publish with no token at all.** Trusted Publishing: PyPI trusts a
specific workflow in a specific repository rather than a secret, verified
through an OIDC credential GitHub mints for that run and which lives for
minutes. There is nothing to revoke, because no secret exists — not in the
repository, not in the settings, not on the author's machine.

1. Create accounts on https://pypi.org and on https://test.pypi.org —
   these are **different** sites with different passwords.
2. Enable two-factor authentication (PyPI will not let you upload
   without it).
3. On each site: Your projects → Publishing → Add a new pending
   publisher, filled in exactly like this:

   | field | value |
   |---|---|
   | PyPI Project Name | `selective-memory` |
   | Owner | `MERUMEZ` |
   | Repository name | `selective-memory` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` (`testpypi` on TestPyPI) |

   A "pending publisher" is a rule for a project that does not exist
   yet — the first publish is what creates it.
4. In GitHub: Settings → Environments → create `pypi` and `testpypi`.
   The names must match what you entered on PyPI, or publishing is
   refused.

From then on [.github/workflows/publish.yml](.github/workflows/publish.yml)
does the publishing: a manual run targets TestPyPI, and only a `v*` tag
reaches the real PyPI.

A token remains possible as a fallback (`twine upload`); if you have one
it looks like `pypi-AgEIcHlwaS5vcmc...` and it must never enter the
repository in any form.

## Checks before every release

```bash
./venv/bin/python -m pytest tests/ -q
./venv/bin/python tools/check_liveness.py       # exits non-zero on a dead mechanism
./venv/bin/python tools/compare_interference.py
./venv/bin/python tools/compare_retention.py --balanced
./venv/bin/python tools/probe_semantic.py
```

`check_liveness.py` exits non-zero when a mechanism stops firing OR keeps
firing while doing nothing. Both have happened here and neither was
caught by the test suite; treat a red run as a release blocker.

**Check what a stranger gets, not what your machine gets.** The default
path to the semantic model once pointed at the author's own disk, so
semantics was off for every user and 23 tests passed only on one
computer. Install the wheel into a clean virtualenv and confirm
`stats().semantic` is True with the `[semantic]` extra and False without.

If any numbers changed, update them in README, README.ru, AUDIT,
COMMERCIAL and the docstrings. `tests/test_readme_examples.py` runs the
README examples verbatim, but it does not verify the measurement tables —
a human is responsible for those.

Bump the version in **one** place — `__version__` in
[selectivemem/__init__.py](selectivemem/__init__.py). `pyproject.toml`
reads it from there, so there is no second copy to keep in step. It has to
stay a plain string literal: setuptools extracts it by parsing the syntax,
without importing the package, and cannot evaluate anything computed.
`tests/test_version.py` guards both properties.

The rule is simple:

| What changed | Version |
|---|---|
| A fix, same behaviour | `0.1.0` → `0.1.1` |
| A new capability, old code still works | `0.1.0` → `0.2.0` |
| Someone else's code breaks (a renamed method, field or package) | `0.x` → `1.0.0` |

## Building

```bash
rm -rf build dist
./venv/bin/python -m build
./venv/bin/python -m twine check dist/*
```

`twine check` must say `PASSED` for both files.

## TestPyPI first

A separate server. Names and versions there reserve **nothing** on the
real PyPI, so experiment freely.

GitHub → Actions → Publish → Run workflow, `target = testpypi`. The build,
the tests and `twine check` all run there; the upload is done by a job
that holds the signing permission and no secrets at all.

The fallback, if Actions are unavailable:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TESTPYPI_TOKEN
./venv/bin/python -m twine upload --repository testpypi dist/*
```

**`403 Forbidden` almost always means the token is from the other site.**
pypi.org and test.pypi.org are separate systems with separate accounts;
registering on one does nothing for the other. Check in this order: an
account exists on TestPyPI, the token was created there, the username is
exactly `__token__`, two-factor is on. If `~/.pypirc` exists it wins over
the environment variables. `--verbose` prints the server's actual reason.

**To look at the page a second time you need a new version number** — a
version burns on TestPyPI too. Use a `.dev` suffix rather than the next
real number:

```bash
sed -i 's/0\.1\.0/0.1.0.dev1/' selectivemem/__init__.py   # test only
```

Then put it back before the real upload, so the first release goes out as
a clean `0.1.0`.

Check the result in a clean environment — **and not from the repository
directory**:

```bash
python -m venv /tmp/check && /tmp/check/bin/pip install \
  -i https://test.pypi.org/simple/ selective-memory
cd /tmp && /tmp/check/bin/python -c \
  "import selectivemem; print(selectivemem.__file__); print(selectivemem.__version__)"
```

**`cd /tmp` IS THE POINT OF THE COMMAND, not decoration.** Python puts the
working directory on the import path, so run this next to the
`selectivemem/` folder and you import the source tree on disk instead of
the package you just published. It has already happened here: an install
failed with "No matching distribution found", the very next line printed a
healthy-looking banner, and nothing at all had been installed.

Printing `__file__` is what tells the two apart. It must point inside
`/tmp/check/lib/`, never into the repository.

And look at the project page with your own eyes: how the README rendered,
whether the links work, whether the licence is in place.

## Then the real PyPI

Here the tag IS the release — there is no separate upload command:

```bash
git tag -a v0.1.0 -m "First release"
git push origin v0.1.0
```

A `v*` tag triggers the same workflow, which targets the real PyPI. Then
install from there into a clean environment and confirm it works.

The order matters: the tag CREATES the release rather than marking one
already made. Otherwise the tag and the uploaded artefact drift apart, and
later nobody can tell what the package people installed was built from.

## When something goes wrong

**A typo in the README, the metadata or the code.** Ship the next
version. There is no rollback.

**You published something broken.** Mark the release as `yank` in the
PyPI web interface: it stays available to anyone who already pinned it,
but new users will not get it. That is gentler than deletion and does not
break other people's builds.

**A token leaked.** Revoke it in PyPI settings immediately and create a
new one.

**You need a different name.** Publish under the new one, then release a
final version of the old name that simply depends on the new one, and
`yank` it. Ugly, but it works.

## What counts as breaking after publication

- Renaming the `selectivemem` package — breaks everyone's imports.
- Renaming `Memory`, `observe`, `feedback`, `recall`, `context_for`,
  `forget` or `stats`.
- Renaming fields of `Observation` or `MemoryStats`.
- Removing a field from `MemorySettings` (adding new ones is safe).

Changing a setting's default is formally not a breaking change, but it
changes behaviour for everyone at once. Do that in a minor version and
write it in the changelog — as was done when the write threshold moved
from 0.35 to 0.25.

Русская версия: [RELEASING.ru.md](RELEASING.ru.md).
