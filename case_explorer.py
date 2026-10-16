"""Case browsing and event-timeline scaffolding for the notebooks.

This lives outside the `interox` package on purpose: none of it is rule
induction. `InterOX` is a case-level, ruleset-oriented class and has no notion
of events, so plumbing an event log through it would drag a plotting dependency
into the core for something it never uses.

The dependency therefore points one way -- this module reads InterOX's existing
public case-log API, and InterOX knows nothing about this module:

    notebook --+--> interox.InterOX     (rules, case log)
               +--> case_explorer       (selection + event log; this file)
               +--> case_timeline       (the Plotly figure builder)

`resolve_selection` turns one case *category* into a case log. `CaseExplorer`
owns the event log and draws the cases of such a category. The two meet only in
the notebook, over case ids.

A `Selection` -- which base, which outcome, which rule filter -- is the unit the
notebook's selectors commit and this module resolves. It is built and checked
here too (`build_rule_filter_options`, `build_selection`, `validate_selection`),
so that what a selector can offer, and what InterOX will accept, are decided in
one place: `base` chooses the ruleset a rule id is resolved against, and pairing
it with the other ruleset's rule is an error rather than a different answer.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go

import case_timeline
from case_timeline import ACTIVITY, CASE_ID, COMPLETION_TIME, REL_TIME, CaseGroup

logger = logging.getLogger(__name__)

#: Either a case log (as returned by `resolve_selection`) or plain case ids.
CaseSelection = Union[pd.DataFrame, Iterable[str]]

#: One rule-filter menu entry: which rule, and whether to keep the cases passing
#: it. `(None, ...)` applies no rule at all.
RuleFilter = Tuple[Optional[int], bool]

#: One case-category selection, keyed to match `resolve_selection`'s keyword
#: arguments so that a holder of a selection can pass it straight through as
#: `resolve_selection(iox, **selection)`.
Selection = Dict[str, Any]

#: The rule-filter entry that applies no rule at all. Present in every menu, for
#: every base and every ruleset -- including an empty one -- so that a selector
#: always has a valid option to fall back on.
NO_RULE_FILTER = 'no matter which rules they pass'


def resolve_selection(iox,
                      base: str = 'whole',
                      outcome: str = 'any',
                      iox_rule_id: Optional[int] = None,
                      pass_: bool = True) -> pd.DataFrame:
    """Resolve one case-category selection to a case log.

    A single entry point over the four case-log getters `InterOX` already
    exposes, so that a caller holding the three "browse" choices -- which base,
    which outcome, which rule -- does not have to branch over them itself.

    Args:
        iox: The `InterOX` instance to read the selection from.
        base: `whole` for the full case log, `remaining` for the unclaimed
            cases (no stored rule fires on them), negatives included.
        outcome: Ground-truth outcome filter (`any`, `positive`, `negative`).
        iox_rule_id: Stable id of the rule to filter by, or `None` for no rule
            filter. Resolved against the stored ruleset when `base` is `whole`
            and against the suggested ruleset when it is `remaining`, matching
            where each ruleset is defined.
        pass_: With a rule, whether to keep the cases that pass it.

    Returns:
        The selected rows of the case log, indexed by case id.

    Raises:
        ValueError: If `base` is not `whole` or `remaining`, if the outcome
            filter is invalid, or if no rule carries `iox_rule_id`.
    """
    if base not in ('whole', 'remaining'):
        raise ValueError(
            f"Invalid base: {base}. Expected one of {{'whole', 'remaining'}}."
        )
    if base == 'remaining':
        if iox_rule_id is None:
            return iox.get_remaining_unexpl_case_log(outcome=outcome)
        return iox.get_data_suggested_rule_by_id(
            iox_rule_id, outcome=outcome, pass_=pass_
        )
    if iox_rule_id is None:
        return iox.get_whole_case_log(outcome=outcome)
    return iox.get_data_stored_rule_by_id(iox_rule_id, outcome=outcome, pass_=pass_)


def build_rule_filter_options(iox, base: str = 'whole') -> Dict[str, RuleFilter]:
    """Build the rule-filter menu for a base, as `{label: (rule_id, pass_)}`.

    Which rule and whether to keep the cases that pass it are one choice, not
    two: a rule id without a pass/fail is not a filter, and a pass/fail without
    a rule id filters nothing. Offering them as a single menu keeps every entry
    a legal filter, and keeps `NO_RULE_FILTER` on the menu whatever the ruleset
    holds -- so a selector's default can never fall out of its own options.

    The menu is drawn from the ruleset `base` resolves against, matching
    `resolve_selection`: stored rules for `whole`, suggested ones for
    `remaining`. A cross-ruleset pair is therefore never offered.

    Args:
        iox: The `InterOX` instance to read the rules from.
        base: `whole` or `remaining`, as in `resolve_selection`.

    Returns:
        Menu entries in display order, `NO_RULE_FILTER` first.

    Raises:
        ValueError: If `base` is not `whole` or `remaining`.
    """
    if base not in ('whole', 'remaining'):
        raise ValueError(
            f"Invalid base: {base}. Expected one of {{'whole', 'remaining'}}."
        )
    rules = (iox.get_stored_rule_options() if base == 'whole'
             else iox.get_suggested_rule_options())
    options: Dict[str, RuleFilter] = {NO_RULE_FILTER: (None, True)}
    for name, iox_rule_id in rules.items():
        options[f'that pass {name}'] = (iox_rule_id, True)
        options[f'that fail {name}'] = (iox_rule_id, False)
    return options


def build_selection(base: str, outcome: str, rule_filter: RuleFilter) -> Selection:
    """Assemble one selection from the choices a selector holds."""
    iox_rule_id, pass_ = rule_filter
    return {'base': base, 'outcome': outcome,
            'iox_rule_id': iox_rule_id, 'pass_': pass_}


def validate_selection(iox, selection: Selection) -> Optional[str]:
    """Check a selection still resolves, returning why it does not.

    A selection captured earlier can go stale: rule ids are stable, but a rule's
    *ruleset* is not. Storing a suggested rule keeps its id and moves it into the
    stored ruleset, at which point a `remaining` selection pointing at it no
    longer resolves -- and re-induction drops suggestions altogether. Callers
    check here rather than letting `resolve_selection` raise, so that a stale
    selection costs a message instead of every dependent cell.

    Args:
        iox: The `InterOX` instance to check the selection against.
        selection: A selection as built by `build_selection`.

    Returns:
        `None` if the selection resolves, else a reason fit to show the user.
    """
    base = selection['base']
    if base not in ('whole', 'remaining'):
        return f"Invalid base: {base}."
    iox_rule_id = selection['iox_rule_id']
    if iox_rule_id is None:
        return None
    options = (iox.get_stored_rule_options() if base == 'whole'
               else iox.get_suggested_rule_options())
    if iox_rule_id in options.values():
        return None
    ruleset = 'stored' if base == 'whole' else 'suggested'
    return (f"Rule '{iox.get_rule_name(iox_rule_id)}' is no longer in the "
            f"{ruleset} ruleset -- it may have been stored, removed, or dropped "
            f"by re-induction. Pick a filter again and press Show.")


def describe_selection(iox, selection: Selection) -> str:
    """Phrase a selection as a plot label, e.g. `whole log, in the outcome,
    passing fervent-vole`."""
    base_word = ('whole log' if selection['base'] == 'whole'
                 else 'unclaimed cases')
    outcome_word = {'positive': 'in the outcome',
                    'negative': 'not in the outcome',
                    'any': 'any outcome'}[selection['outcome']]
    iox_rule_id = selection['iox_rule_id']
    if iox_rule_id is None:
        return f'{base_word}, {outcome_word}'
    verb = 'passing' if selection['pass_'] else 'failing'
    return (f'{base_word}, {outcome_word}, '
            f'{verb} {iox.get_rule_name(iox_rule_id)}')


class CaseExplorer:
    """Draws the event sequences behind a set of cases.

    Holds the event log, indexed once by case, and hands case selections to
    `case_timeline` for rendering. Knows nothing about rules: a "category" of
    cases reaches it as nothing more than a set of case ids.
    """

    def __init__(self, event_log: pd.DataFrame,
                 case_index: Optional[pd.Index] = None):
        """Index an event log for drilling into individual cases.

        Args:
            event_log: One row per event, carrying at least `case:concept:name`,
                `concept:name` and `time:timestamp`.
            case_index: Optional index of the case log this event log backs.
                Supplying it enables a warning about cases that carry no events.

        Raises:
            ValueError: If a required event-log column is missing.
        """
        for column in (CASE_ID, ACTIVITY, COMPLETION_TIME):
            if column not in event_log.columns:
                raise ValueError(
                    f"Event log is missing the required column '{column}'."
                )

        events = event_log
        # Derived here rather than relied upon, so that an event log which never
        # had `rel_time` added -- one that did not come out of eclear's
        # `EnrichmentWidget.event_log` -- works.
        if REL_TIME not in events.columns:
            events = events.copy()
            events[REL_TIME] = events.groupby(CASE_ID)[COMPLETION_TIME].transform(
                lambda times: times - times.min()
            )

        # Sorting once here is what lets every later lookup be an index hit
        # rather than a full scan -- the road fines log is 185 MB.
        self._events_by_case = events.set_index(CASE_ID).sort_index()
        self._events_by_case.index = self._events_by_case.index.astype(str)
        self._activity_order = case_timeline.activity_order(self._events_by_case)

        # The cases this explorer can actually draw: those of the case log that
        # carry events. `case_events` raises for the rest, so keeping the two
        # apart here is what lets a selector be fed straight from `case_ids`.
        event_cases = set(self._events_by_case.index)
        if case_index is None:
            self._case_ids = list(dict.fromkeys(self._events_by_case.index))
        else:
            case_log_ids = [str(case_id) for case_id in case_index]
            self._case_ids = [c for c in case_log_ids if c in event_cases]
            missing = [c for c in case_log_ids if c not in event_cases]
            if missing:
                logger.warning(
                    '%d of %d cases in the case log have no events in the event '
                    'log (e.g. %s); they will be absent from the timeline.',
                    len(missing), len(case_log_ids), sorted(missing)[:3]
                )

    @property
    def case_ids(self) -> List[str]:
        """Every case that can be drawn or read out, in case-log order.

        The whole case log, not any one category of it: a case reached through
        the data browser is as readable as one seen in the timeline. Cases the
        event log has nothing for are left out, so a selector fed from here can
        never offer a case that `case_events` would then refuse.
        """
        return list(self._case_ids)

    def _as_case_ids(self, selection: CaseSelection) -> List[str]:
        """Accept either a case-log DataFrame or a plain iterable of case ids."""
        if isinstance(selection, pd.DataFrame):
            return [str(case_id) for case_id in selection.index]
        return [str(case_id) for case_id in selection]

    def timeline(self,  # pylint: disable=too-many-arguments,too-many-positional-arguments
                 group_a: CaseSelection,
                 group_b: Optional[CaseSelection] = None,
                 label_a: str = 'A',
                 label_b: str = 'B',
                 sample_size: int = 30,
                 seed: int = 0) -> go.Figure:
        """Return a dotted chart of a random sample of one or two case groups.

        Each case becomes a row and each of its events a marker, placed at its
        offset from that case's own first event so that all cases start at 0 and
        are directly comparable. With a second group, the first is drawn above
        the divider and the second below, sharing one time axis.

        Args:
            group_a: Cases to draw on top -- either a case log (as returned by
                `resolve_selection`) or an iterable of case ids.
            group_b: Optional cases to draw below the divider, for comparison.
            label_a: Name of the first group, shown above its block.
            label_b: Name of the second group.
            sample_size: Cases to draw per group; a smaller group is drawn whole.
            seed: Sampling seed. Changing it redraws a different random sample.

        Returns:
            A Plotly figure, rendered directly by marimo.
        """
        groups = [CaseGroup(label_a, self._as_case_ids(group_a))]
        if group_b is not None:
            groups.append(CaseGroup(label_b, self._as_case_ids(group_b)))
        return case_timeline.build_case_timeline(
            self._events_by_case, groups,
            sample_size=sample_size, seed=seed, activities=self._activity_order,
        )

    def case_events(self, case_id: str) -> pd.DataFrame:
        """Return the events of a single case, in time order.

        Args:
            case_id: Id of the case, as it appears in the case log index.

        Returns:
            One row per event, carrying the case's relative time (`rel_time`)
            alongside the original event-log columns.

        Raises:
            ValueError: If the case has no events.
        """
        case_id = str(case_id)
        if case_id not in self._events_by_case.index:
            raise ValueError(f"No events found for case '{case_id}'.")
        return self._events_by_case.loc[[case_id]].sort_values(COMPLETION_TIME)
