# Explaining process outcomes with interox

Two [marimo](https://marimo.io) notebooks that walk through interactive,
rule-based outcome explanation on an event log, using
[interox](https://pypi.org/project/interox/) for the rules and widgets and
[eclear](https://pypi.org/project/eclear/) to turn the event log into a case log.
Both follow the same four parts; only the log and the outcome differ.

- **Sepsis cases** — how sepsis patients are treated in a hospital, from
  registration at the emergency room to release. By default the notebook
  explains which cases end **without a recorded release**.
- **Road traffic fines** — how an Italian local police force manages fines,
  from creation to payment, appeal or credit collection. By default the notebook
  explains which cases are **not sent for credit collection**; its cell 2.2
  carries a commented-out baseline (drop fines that stop after *Send Fine*,
  fines dismissed by a judge or the prefecture, and fully paid ones). It runs on
  the full log (150,370 cases) without sampling, so expect each induction in
  Part 3 to take from a few seconds to about a minute.

## Setup

You need [uv](https://docs.astral.sh/uv/). No Node or JavaScript toolchain: the
interox widgets come pre-built from PyPI.

```bash
uv sync
```

Then put the event logs where the notebooks expect them. Logs are not
redistributed in this repo (`logs/*.xes` is gitignored):

| File | Notebook | Notes |
|---|---|---|
| `logs/SepsisCases2020EventLog.xes` | sepsis | the sepsis cases event log |
| `logs/Road_Traffic_Fine_Management_Process.xes` | road fines | from 4TU.ResearchData, [doi:10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5); download the `.xes.gz` and unzip it (`gunzip Road_Traffic_Fine_Management_Process.xes.gz`) |

The `file_name = ...` cell in each notebook is the authority if a name changes.

## Run

```bash
uv run marimo edit rule_induction_sepsis_marimo_iox.py
```

```bash
uv run marimo edit rule_induction_road_fines_marimo_iox.py
```

Each notebook has four parts: prepare the data (load the log, add event
attributes in code, build the case log), define the outcome and choose the cases
to analyse, explain it interactively (store suggested rules or build your own),
and inspect the cases behind a rule. Cells marked ✏️ are meant to be edited.

## What is here

| File | Role |
|---|---|
| `rule_induction_sepsis_marimo_iox.py` | the sepsis notebook |
| `rule_induction_road_fines_marimo_iox.py` | the road traffic fines notebook |
| `case_explorer.py` | case browsing for part 4; reads interox's public case-log API |
| `case_timeline.py` | the event timeline plot used by the case explorer |

The two helpers sit next to the notebook rather than in interox on purpose:
rule induction never touches events or plotting. Both notebooks share them.

```bash
./code-check.sh     # pylint + mypy on the helpers
```

## Working against a local interox or eclear checkout

To try an unreleased interox or eclear change in these notebooks, swap the PyPI
dependency for an editable checkout, and swap it back before committing:

```bash
uv add --editable ../interox
uv add "interox>=0.1.0,<0.2"
```

```bash
uv add --editable ../eclear
uv add "eclear>=0.1.0,<0.2"
```
