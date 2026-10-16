# Explaining sepsis case outcomes with interox

A [marimo](https://marimo.io) notebook that walks through interactive, rule-based
outcome explanation on the sepsis event log, using
[interox](https://pypi.org/project/interox/) for the rules and widgets and
[eclear](https://pypi.org/project/eclear/) to turn the event log into a case log.

The sepsis log records how sepsis patients are treated in a hospital, from
registration at the emergency room to release. By default the notebook explains
which cases end **without a recorded release**.

## Setup

You need [uv](https://docs.astral.sh/uv/). No Node or JavaScript toolchain: the
interox widgets come pre-built from PyPI.

```bash
uv sync
```

Then put the event log where the notebook expects it. Logs are not
redistributed in this repo (`logs/*.xes` is gitignored):

| File | Notes |
|---|---|
| `logs/SepsisCases2020EventLog.xes` | the sepsis cases event log; the `file_name = ...` cell in the notebook is the authority if the name changes |

## Run

```bash
uv run marimo edit rule_induction_sepsis_marimo_iox.py
```

The notebook has four parts: prepare the data, define the outcome, explain it
interactively (store suggested rules or build your own), and inspect the cases
behind a rule. Cells marked ✏️ are meant to be edited.

## What is here

| File | Role |
|---|---|
| `rule_induction_sepsis_marimo_iox.py` | the notebook |
| `case_explorer.py` | case browsing for part 4; reads interox's public case-log API |
| `case_timeline.py` | the event timeline plot used by the case explorer |

The two helpers sit next to the notebook rather than in interox on purpose:
rule induction never touches events or plotting.

```bash
./code-check.sh     # pylint + mypy on the helpers
```

## Working against a local interox checkout

To try an unreleased interox change in this notebook, swap the PyPI dependency
for an editable checkout, and swap it back before committing:

```bash
uv add --editable ../interox
uv add "interox>=0.1.0,<0.2"
```
