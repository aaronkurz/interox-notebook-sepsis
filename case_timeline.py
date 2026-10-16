"""Dotted-chart figure builder for temporal case comparison.

Renders a sample of cases as one row each, with their events placed on a shared
time axis normalized to 0 at each case's own start. Two case groups can be
stacked above and below a divider to compare the event sequences of, say, the
cases a rule fires on against the cases it misses.

This module is deliberately free of any `InterOX` dependency and holds all of
the Plotly-specific code in the package, so that swapping the rendering layer
(e.g. for a custom anywidget) touches this file only.

Color
-----
The categorical palette is 8 hues, and real logs carry more activities than that
(sepsis has 16). Rather than cycling hues -- which would leave two activities
indistinguishable -- activities are encoded compositely: the hue cycles over the
8 slots while the marker *symbol* advances every 8 activities, so identity is
carried by (hue, symbol) together and stays unique up to 32 activities.

Folding the rare activities into an "Other" bucket, the usual answer to having
more than 8 series, would be wrong here: in the sepsis log the rarest activities
are `Release B`..`Release E`, which are precisely the ones that *define* the
outcome under explanation.

The palette sits in the CVD floor band and three of its hues fall below 3:1 on
the chart surface. Both are legal only alongside secondary encoding, which this
chart ships: the marker symbol, an always-present legend, per-point hover naming
the activity, and the raw-event table view next to it.
"""
from collections import namedtuple
from random import Random
from typing import AbstractSet, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CASE_ID = 'case:concept:name'
ACTIVITY = 'concept:name'
COMPLETION_TIME = 'time:timestamp'
REL_TIME = 'rel_time'

#: A named set of cases to draw as one block of rows.
CaseGroup = namedtuple('CaseGroup', ['label', 'case_ids'])

# Categorical slots, in the fixed order that makes them CVD-separable. Do not
# reorder: the order is the safety mechanism, not decoration.
_HUES = [
    '#2a78d6',  # blue
    '#1baf7a',  # aqua
    '#eda100',  # yellow
    '#008300',  # green
    '#4a3aa7',  # violet
    '#e34948',  # red
    '#e87ba4',  # magenta
    '#eb6834',  # orange
]
# The second, non-color channel. Each symbol carries one full pass over the hues.
_SYMBOLS = ['circle', 'diamond', 'square', 'triangle-up']

_SURFACE = '#fcfcfb'
_INK_PRIMARY = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_INK_MUTED = '#898781'
_GRIDLINE = '#e1e0d9'
_BASELINE = '#c3c2b7'
_SPINE = '#d8d7d0'

_ROW_HEIGHT_PX = 19
_CHROME_HEIGHT_PX = 150
_MIN_HEIGHT_PX = 300

ActivityStyle = Tuple[str, str]  # (color, symbol)


def activity_order(events: pd.DataFrame) -> List[str]:
    """Return every activity in the log, most frequent first.

    Must be computed over the *whole* log, never over the cases currently on
    screen: a style map derived from the visible sample would repaint activities
    whenever the sample changed, and the A/B comparison would stop meaning
    anything. Ties break on the name so the order is stable across runs.
    """
    counts = events[ACTIVITY].value_counts()
    return sorted(counts.index, key=lambda name: (-counts[name], str(name)))


def activity_styles(activities: Sequence[str]) -> Dict[str, ActivityStyle]:
    """Map each activity to a stable (color, symbol) pair.

    The hue cycles over the 8 categorical slots; the symbol advances every 8
    activities, so no two activities share both channels until the 33rd.
    """
    return {
        str(name): (_HUES[i % len(_HUES)],
                    _SYMBOLS[(i // len(_HUES)) % len(_SYMBOLS)])
        for i, name in enumerate(activities)
    }


def _sample_case_ids(case_ids: Sequence[str], sample_size: int, seed: int) -> List[str]:
    """Draw up to `sample_size` case ids, reproducibly for a given seed."""
    unique = sorted({str(cid) for cid in case_ids})
    if len(unique) <= sample_size:
        return unique
    return Random(seed).sample(unique, sample_size)


def _events_for(events_by_case: pd.DataFrame, case_ids: Sequence[str]) -> pd.DataFrame:
    """Return the events of `case_ids`, hitting the sorted index rather than scanning."""
    present = [cid for cid in case_ids if cid in events_by_case.index]
    if not present:
        return events_by_case.iloc[0:0]
    return events_by_case.loc[present]


def _rel_hours(events: pd.DataFrame) -> pd.Series:
    return events[REL_TIME].dt.total_seconds() / 3600.0


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False,
                          xref='paper', yref='paper', x=0.5, y=0.5,
                          font={'color': _INK_SECONDARY, 'size': 13})
    figure.update_layout(
        paper_bgcolor=_SURFACE, plot_bgcolor=_SURFACE, height=_MIN_HEIGHT_PX,
        xaxis={'visible': False}, yaxis={'visible': False},
    )
    return figure


def _ordered_cases(events_by_case: pd.DataFrame,
                   group: CaseGroup,
                   sample_size: int,
                   seed: int) -> List[str]:
    """Sample a group's cases and order them by duration, longest first.

    A dotted chart whose rows are in arbitrary order is far harder to read than
    one sorted into a wedge.
    """
    events = _events_for(events_by_case, _sample_case_ids(group.case_ids, sample_size, seed))
    if len(events) == 0:
        return []
    durations = _rel_hours(events).groupby(level=0).max()
    return [str(cid) for cid in durations.sort_values(ascending=False).index]


def build_case_timeline(events_by_case: pd.DataFrame,
                        groups: Sequence[CaseGroup],
                        sample_size: int = 30,
                        seed: int = 0,
                        activities: Optional[Sequence[str]] = None) -> go.Figure:
    """Build a dotted chart of the sampled cases of one or two groups.

    Each case is a row; each event a marker placed at its offset from that
    case's own first event, so all cases start at 0 and are directly comparable.
    With two groups, the first is drawn above the divider and the second below,
    sharing one time axis.

    Args:
        events_by_case: Event log indexed by case id and sorted, carrying at
            least the `concept:name` and `rel_time` columns.
        groups: One or two `CaseGroup`s. The first is drawn on top.
        sample_size: Cases to draw per group; a group with fewer is drawn whole.
        seed: Sampling seed. Bumping it reshuffles which cases are shown.
        activities: Every activity of the *whole* log, in the order that fixes
            the colors. Defaults to deriving them from `events_by_case`, which
            is only correct when that frame is the whole log.

    Returns:
        A Plotly figure with a linear x axis of hours since case start.
    """
    groups = [g for g in groups if g is not None]
    if not groups:
        return _empty_figure('No case group selected.')

    styles = activity_styles(
        activities if activities is not None else activity_order(events_by_case)
    )
    blocks = [(group, _ordered_cases(events_by_case, group, sample_size, seed))
              for group in groups]
    blocks = [(group, cases) for group, cases in blocks if cases]
    if not blocks:
        return _empty_figure('No cases match this selection.')

    figure = make_subplots(
        rows=len(blocks), cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[len(cases) for _, cases in blocks],
        subplot_titles=[f'{group.label}  (n={len(cases)})' for group, cases in blocks],
    )

    drawn: Set[str] = set()
    for row, (group, cases) in enumerate(blocks, start=1):
        drawn |= _draw_block(figure, events_by_case, cases, styles, row)
        figure.update_yaxes(
            row=row, col=1,
            tickmode='array',
            tickvals=list(range(len(cases))),
            # y grows upward, so reverse to get the longest case on top.
            ticktext=list(reversed(cases)),
            tickfont={'color': _INK_MUTED, 'size': 9},
            showgrid=False, zeroline=False, showline=False, automargin=True,
            range=[-0.8, len(cases) - 0.2],
        )

    _add_legend(figure, styles, drawn)

    total_rows = sum(len(cases) for _, cases in blocks)
    # A small negative pad keeps the markers of the first event, which sit
    # exactly at 0, from colliding with the case-id tick labels.
    span = max((float(_rel_hours(_events_for(events_by_case, cases)).max())
                for _, cases in blocks), default=1.0) or 1.0
    figure.update_xaxes(
        gridcolor=_GRIDLINE, zeroline=True, zerolinecolor=_BASELINE,
        linecolor=_BASELINE, tickfont={'color': _INK_MUTED, 'size': 10},
        range=[-0.02 * span, span * 1.02], showticklabels=True,
    )
    figure.update_xaxes(
        row=len(blocks), col=1,
        title={'text': 'hours since case start', 'font': {'size': 11}},
    )
    figure.update_layout(
        height=max(_MIN_HEIGHT_PX, total_rows * _ROW_HEIGHT_PX + _CHROME_HEIGHT_PX),
        paper_bgcolor=_SURFACE, plot_bgcolor=_SURFACE,
        font={'family': 'system-ui, -apple-system, "Segoe UI", sans-serif',
              'color': _INK_SECONDARY, 'size': 11},
        margin={'l': 8, 'r': 16, 't': 76, 'b': 48},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.03,
                'xanchor': 'left', 'x': 0, 'font': {'size': 10}},
        hovermode='closest',
    )
    for note in figure.layout.annotations:
        note.update(x=0, xanchor='left',
                    font={'color': _INK_PRIMARY, 'size': 11})
    return figure


def _add_legend(figure: go.Figure,
                styles: Dict[str, ActivityStyle],
                drawn: AbstractSet[str]) -> None:
    """Give every drawn activity exactly one legend entry, in canonical order.

    The entries are their own marker-only traces rather than a flag on the data
    traces: an activity may well appear in the second block but not the first
    (`Release A` shows up only in resolved cases), and hanging the legend off
    the data would silently drop it. Each entry shares its activity's
    `legendgroup`, so clicking it still isolates that activity in every block.
    """
    for name, (color, symbol) in styles.items():
        if name not in drawn:
            continue
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', name=name, legendgroup=name,
            showlegend=True, hoverinfo='skip',
            marker={'color': color, 'symbol': symbol, 'size': 9,
                    'line': {'color': _SURFACE, 'width': 1}},
        ), row=1, col=1)


def _draw_block(figure: go.Figure,  # pylint: disable=too-many-locals
                events_by_case: pd.DataFrame,
                cases: Sequence[str],
                styles: Dict[str, ActivityStyle],
                row: int) -> Set[str]:
    """Draw one group's rows into subplot `row`.

    Returns the activities it actually drew, so the caller can build a legend
    covering every block.
    """
    # y grows upward, so the first (longest) case takes the highest y.
    y_of_case = {case_id: len(cases) - 1 - i for i, case_id in enumerate(cases)}
    events = _events_for(events_by_case, cases)
    hours = _rel_hours(events)

    # One trace for all spines, using None separators, rather than one trace per
    # case: this keeps the figure small enough for Plotly to stay responsive.
    spine_x: List[Optional[float]] = []
    spine_y: List[Optional[float]] = []
    for case_id, span in hours.groupby(level=0):
        y = y_of_case[str(case_id)]
        spine_x += [float(span.min()), float(span.max()), None]
        spine_y += [y, y, None]
    figure.add_trace(go.Scatter(
        x=spine_x, y=spine_y, mode='lines', showlegend=False, hoverinfo='skip',
        line={'color': _SPINE, 'width': 2},
    ), row=row, col=1)

    # One trace per activity, so a legend click isolates that activity.
    drawn = set()
    for name, (color, symbol) in styles.items():
        matching = events[events[ACTIVITY].astype(str) == name]
        if len(matching) == 0:
            continue
        drawn.add(name)
        case_ids = [str(cid) for cid in matching.index]
        figure.add_trace(go.Scatter(
            x=[float(h) for h in _rel_hours(matching)],
            y=[y_of_case[cid] for cid in case_ids],
            mode='markers', name=name, legendgroup=name, showlegend=False,
            customdata=case_ids,
            marker={'color': color, 'symbol': symbol, 'size': 9,
                    'line': {'color': _SURFACE, 'width': 1}},
            hovertemplate=(
                '<b>%{customdata}</b><br>'
                f'{name}<br>'
                '%{x:.2f} h after case start<extra></extra>'
            ),
        ), row=row, col=1)
    return drawn
