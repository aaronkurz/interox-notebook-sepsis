import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explaining Process Outcomes with Rules — Sepsis Cases

    This notebook helps you find out *why* some cases of a process end the way
    they do. You describe every case by a set of features, choose an outcome, and
    then build a set of rules — with suggestions from a rule induction algorithm —
    that tells the cases with this outcome apart from the others.

    The sepsis log records how sepsis patients are treated in a hospital, from
    their registration at the emergency room (ER) to their release. By default,
    the notebook explains which cases end **without a recorded release**.

    The notebook has four parts:

    1. **Prepare the data** — load an event log and turn it into a case log.
    2. **Define what to explain** — decide which cases have the/belong to the outcome we want to explain.
    3. **Explain the outcome interactively** — store suggested rules or build your own.
    4. **Inspect cases** — look at the cases behind a rule to check it and to find new ideas.

    **Working with the notebook.** marimo notebooks are *reactive*: when you run a
    cell, every cell that depends on it runs again automatically. Cells introduced
    with ✏️ are meant to be edited. The code of all other cells is hidden — you can
    show it, but you don't need to read or edit it.

    /// attention | Changes in Part 1 and 2 restart Part 3.
    Applying an enrichment or running an edited cell in Part 1 or 2 rebuilds the
    explanation from scratch, and **all stored rules are lost**. Hiding and
    unhiding features in Part 3 keeps them.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import pm4py

    return mo, pd, pm4py


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Prepare the data

    An *event log* has one row per event. Rule induction needs a *case log*
    instead: one row per case, and one column — a *feature* — per property of the
    whole case, such as how often an activity occurred or the highest lab value
    measured.

    ### 1.1 Load the event log

    ✏️ Edit this cell to load a different XES event log.
    """)
    return


@app.cell
def _(pm4py):
    file_name = "logs/SepsisCases2020EventLog.xes"
    event_log = pm4py.read_xes(file_name, variant="rustxes")
    ACTIVITY = 'concept:name'  # the column holding the activity name

    event_log.head(2)
    return ACTIVITY, event_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Build the case log

    The widget below computes new features (*enrichments*). It has two tabs:

    - **Case log** — features of a whole case. These are what rule induction uses.
    - **Event log** — new attributes of single events. To use one for rule
      induction, track it on the case tab, which turns it into a case feature.

    On the case tab, features come in four families:

    - **Activity counts** — how often an activity occurs in the case, e.g. `CRP::count`
    - **First/last occurrence** — when an activity first (`::start`) or last (`::end`) happened, measured from the start of the case
    - **Activity delays** — the time between the first occurrences of two activities
    - **Attribute tracking** — an event attribute aggregated over the case, e.g. `Leucocytes.Leucocytes::max` is the highest leucocyte value; `all_events.concept:name::append` is the sequence of activities (the *variant*)

    On the event tab, **binning** and **arithmetic** read an event's own attributes;
    **time delta** and **running aggregation** read the other events of its case.
    For example, add a *Δt since previous event* on `ER Triage` events, then track
    its `max` on the case tab. The operand pickers only offer attributes that the
    chosen event type actually carries.

    To add features, stage them, check their values on a few real cases with
    **Preview sample cases** or **Preview sample events**, then press **Apply**.
    Applying an event enrichment does not recompute case features that already
    exist. The case log starts with one count per activity and the variant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    enrichment_applied, set_enrichment_applied = mo.state(False)
    return enrichment_applied, set_enrichment_applied


@app.cell(hide_code=True)
def _(ACTIVITY, event_log, set_enrichment_applied):
    from eclear.enrichment import counts_spec, tracking_spec
    from eclear import EnrichmentWidget

    enricher = EnrichmentWidget(
        event_log,
        initial_specs=[
            *[counts_spec(_activity) for _activity in event_log[ACTIVITY].unique()],
            tracking_spec('concept:name', 'append'),
        ],
        on_apply=lambda: set_enrichment_applied(True),
    )
    enricher
    return (enricher,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Add your own case features

    ✏️ Edit this cell for anything the widget can't do, e.g. a feature derived
    from three others. The cell runs again
    whenever you press **Apply** in the enrichment widget, so no need to re run it after you ran it first and then make changes in the widget. Removing attributes in the widget that you depend on here may cause failures. `enriched_event_log` holds the
    events with your event enrichments, `case_log` the case features.
    """)
    return


@app.cell
def _(enricher, enrichment_applied):
    enrichment_applied  # run this cell again whenever the widget applies

    enriched_event_log = enricher.event_log
    case_log = enricher.case_log.dropna(axis=1, how='all')  # drop empty columns

    # Example: a feature derived from two others.
    # case_log['lab_tests'] = case_log['CRP::count'] + case_log['Leucocytes::count']

    case_log.head(2)
    return case_log, enriched_event_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Encode features for rule induction

    The rule induction algorithm only accepts text (categorical) and decimal
    (numeric) features. This cell converts everything else:

    - booleans become text, integers become decimals
    - timestamps and durations become numbers in `time_unit` (default: days)
    - missing text values become the category `missing_value`, so a rule can
      refer to them; missing numbers stay missing

    ✏️ Edit this cell when a default doesn't fit — e.g. a feature that is a sum
    should be 0, not missing, for a case where there was nothing to sum.
    """)
    return


@app.cell
def _(case_log, pd):
    normalized_case_log = case_log.copy()

    time_unit = pd.Timedelta(days=1)

    for column in normalized_case_log.select_dtypes(include=['bool']).columns:
        normalized_case_log[column] = normalized_case_log[column].astype('object')
    for column in normalized_case_log.select_dtypes(include=['int64']).columns:
        normalized_case_log[column] = normalized_case_log[column].astype('float64')
    # Timestamps are counted from 1970-01-01.
    for column in normalized_case_log.select_dtypes(include=['datetimetz']).columns:
        normalized_case_log[column] = (normalized_case_log[column] - pd.Timestamp(0, tz='UTC')) / time_unit
    for column in normalized_case_log.select_dtypes(include=['timedelta']).columns:
        normalized_case_log[column] = normalized_case_log[column] / time_unit
    # Replace the original missing values: after astype('str') they would be
    # the text 'nan', which a rule would read as a real category.
    STRING_IMPUTE_VAL = 'missing_value'
    for column in normalized_case_log.select_dtypes(include=['object', 'str']).columns:
        _values = normalized_case_log[column]
        normalized_case_log[column] = _values.astype('str').where(
            _values.notna(), STRING_IMPUTE_VAL
        )

    # Example: a sum over nothing is 0.
    # normalized_case_log['lab_tests'] = normalized_case_log['lab_tests'].fillna(0)

    normalized_case_log.info(verbose=True)
    return (normalized_case_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Define what to explain

    ### 2.1 The outcome

    The *outcome* is a yes/no property of a case. The rules in Part 3 describe
    the cases where it is *yes* — the *positive* cases. Here, a case is positive
    when it contains none of the release activities `Release_A` to `Release_E`.

    ✏️ Edit this cell to explain a different outcome: `TARGET` can be any
    condition on a row of the case log.
    """)
    return


@app.cell
def _(normalized_case_log):
    RELEASES = ['Release_A', 'Release_B', 'Release_C', 'Release_D', 'Release_E']

    def release_type(case):
        """The first release activity the case contains, or 'unresolved'."""
        for release in RELEASES:
            if case[release + '::count'] > 0:
                return release
        return 'unresolved'

    TARGET = lambda case: release_type(case) == 'unresolved'

    normalized_case_log.apply(release_type, axis=1).value_counts()
    return (TARGET,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Exclude cases that are already explained (optional)

    You may already know an explanation for some of the positive cases — from
    domain knowledge or from an earlier session. Removing those cases focuses the
    search on the rest. Unlike a rule stored in Part 3, excluded cases are gone
    completely: they count neither as positive nor as negative.

    ✏️ Write the known explanation as a rule and uncomment the example lines.
    """)
    return


@app.cell
def _(TARGET, normalized_case_log):
    filtered_case_log = normalized_case_log
    # rule1 = lambda case: (case['Admission_NC::count'] == 0) & (case['Admission_IC::count'] == 0)
    # explanation1 = filtered_case_log.apply(rule1, axis=1)
    # filtered_case_log = filtered_case_log[~explanation1]
    print(len(filtered_case_log), 'cases out of', len(normalized_case_log), 'remain')

    outcome = filtered_case_log.apply(TARGET, axis=1)
    outcome.value_counts()
    return filtered_case_log, outcome


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Explain the outcome interactively

    Here you build an *explanation*: a set of rules. A case is *claimed* when at
    least one stored rule fires on it, and *unclaimed* otherwise — whether or not
    the claim is right. Compared with the outcome, cases fall into three groups
    that matter:

    - **Explained** — claimed and positive
    - **Wrongly claimed** — claimed but negative
    - **Remaining** — positive but not claimed (yet)

    The remaining cases are the positive part of the unclaimed ones; the negative
    part is simply left alone.

    A rule's *precision* is the share of the cases it claims that are positive;
    its *recall* is the share of the positive cases it explains.

    A typical way to work: check the status (3.1), keep features out of the
    rules if needed (3.2), look through the suggested rules (3.3) or build one
    yourself (3.4), and store a rule that makes sense. Status and suggestions
    then update for the cases still unclaimed. Use Part 4 to check what a rule
    actually claims, then repeat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    iox_state_updated, set_iox_state_updated = mo.state(False)
    return iox_state_updated, set_iox_state_updated


@app.cell(hide_code=True)
def _(filtered_case_log, outcome):
    from interox import InterOX

    iox = InterOX(outcome, filtered_case_log)
    return (iox,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Explanation status

    How many positive cases the stored rules explain, how many cases they wrongly
    claim, and how many positive cases remain.
    """)
    return


@app.cell(hide_code=True)
def _(iox, iox_state_updated):
    iox_state_updated

    iox.get_exp_status()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Choose which features rules may use

    Some features make rules that are true but uninformative, or that give the
    outcome away. List them here to keep them out of the suggested rules (3.3)
    and the rule builder (3.4): add features to hide them, remove them to bring
    them back, then press **Apply**. Your stored rules stay as they are.
    """)
    return


@app.cell(hide_code=True)
def _(iox, iox_state_updated, mo):
    iox_state_updated

    # One list holding the whole hidden set, so a single Apply can hide and
    # unhide at once -- and suggestions are re-induced only once.
    hidden_features_select = mo.ui.multiselect(
        iox.get_unhidden_features() + iox.get_hidden_features(),
        value=list(iox.get_hidden_features()),
        label="Hidden features",
        full_width=True,
    )
    apply_hidden_features_button = mo.ui.run_button(label="Apply")

    mo.hstack(
        [hidden_features_select, apply_hidden_features_button],
        align="end",
        widths=[1, 0],
    )
    return apply_hidden_features_button, hidden_features_select


@app.cell(hide_code=True)
def _(
    apply_hidden_features_button,
    hidden_features_select,
    iox,
    set_iox_state_updated,
):
    if (apply_hidden_features_button.value
            and set(hidden_features_select.value) != set(iox.get_hidden_features())):
        iox.set_hidden_features(hidden_features_select.value)
        set_iox_state_updated(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Suggested and stored rules

    **Stored** lists the rules of your explanation; **Suggested** lists the rules
    the algorithm proposes for the unclaimed cases. To look at a suggested rule
    more closely, send it to the rule builder (3.4) with its edit button: there
    you can see how its features rank and adjust its conditions.
    """)
    return


@app.cell(hide_code=True)
def _(iox, iox_state_updated, mo, set_iox_state_updated):
    iox_state_updated

    tabs = mo.ui.tabs({
        "Stored": iox.get_stored_ruleset(on_refresh=lambda: set_iox_state_updated(True)),
        "Suggested": iox.get_suggested_ruleset(on_refresh=lambda: set_iox_state_updated(True)),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.4 Build your own rule

    Drag or click features onto the canvas — they are ranked by how useful a
    decision tree finds them. Set bounds on the distribution of a numeric feature,
    or pick values of a categorical one. The rule is evaluated against the
    unclaimed cases; store it once you are satisfied.
    """)
    return


@app.cell(hide_code=True)
def _(iox, iox_state_updated, set_iox_state_updated):
    iox_state_updated

    rule_builder_widget = iox.get_rule_builder(
        scope="remaining",
        on_refresh=lambda: set_iox_state_updated(True),
    )
    rule_builder_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Inspect cases

    Numbers alone can mislead. Look at the cases behind a rule to check that it
    makes sense, and at the wrongly claimed and remaining cases to see what it
    misses. What you find there is often an idea for a new feature to add in
    Part 1 — keep in mind that this resets your stored rules.
    """)
    return


@app.cell(hide_code=True)
def _(enriched_event_log, filtered_case_log):
    # Case browsing and the event timeline sit outside InterOX: they read its
    # public case-log API but rule induction never touches events.
    from case_explorer import (
        NO_RULE_FILTER,
        CaseExplorer,
        build_rule_filter_options,
        build_selection,
        describe_selection,
        resolve_selection,
        validate_selection,
    )

    explorer = CaseExplorer(enriched_event_log, case_index=filtered_case_log.index)
    return (
        NO_RULE_FILTER,
        build_rule_filter_options,
        build_selection,
        describe_selection,
        explorer,
        resolve_selection,
        validate_selection,
    )


@app.cell(hide_code=True)
def _(mo):
    # Everything in Part 4 that is never rebuilt, in one cell. It depends on
    # nothing but `mo`, so it never re-runs -- and since a re-run is the only
    # thing that resets an element, nothing here can be reset. The menus that
    # *are* rebuilt -- the three rule menus -- each keep a cell of their own
    # below, or a change to one would reset the others.

    # The wording of the two menus that are the same everywhere a selection is
    # made -- the data browser, and each of the two timeline groups. The base
    # doubles as its own value, so a selection never has to be translated.
    BASE_OPTIONS = {"the whole": "whole", "the unclaimed": "remaining"}
    OUTCOME_OPTIONS = {
        "cases that are part of the outcome": 'positive',
        "cases that are not part of the outcome": 'negative',
        "cases no matter the outcome": 'any',
    }

    # A selection is committed by pressing Show, and kept here. Rebuilding a
    # menu -- because the base changed, or a rule was stored -- leaves the last
    # committed selection exactly as it was. The views re-resolve it whenever
    # InterOX changes, so they show current data, or say the selection is stale.
    browse_selection, set_browse_selection = mo.state(None)
    group_a_selection, set_group_a_selection = mo.state(None)
    group_b_selection, set_group_b_selection = mo.state(None)
    tl_seed, set_tl_seed = mo.state(0)

    # 4.1 data browser
    el_base = mo.ui.dropdown(BASE_OPTIONS, value="the whole")
    el_outcome = mo.ui.dropdown(OUTCOME_OPTIONS,
                                value="cases that are part of the outcome")

    # 4.2 timeline
    ga_base = mo.ui.dropdown(BASE_OPTIONS, value="the whole")
    ga_outcome = mo.ui.dropdown(OUTCOME_OPTIONS,
                                value="cases that are part of the outcome")
    gb_base = mo.ui.dropdown(BASE_OPTIONS, value="the whole")
    gb_outcome = mo.ui.dropdown(OUTCOME_OPTIONS,
                                value="cases that are not part of the outcome")
    compare_toggle = mo.ui.checkbox(value=False,
                                    label="Compare against a second group")
    tl_sample_size = mo.ui.slider(5, 100, value=25, step=5,
                                  label="Cases per group", show_value=True)
    tl_resample_button = mo.ui.run_button(
        label="Re-sample",
        on_change=lambda _: set_tl_seed(lambda current: current + 1),
    )
    return (
        browse_selection,
        compare_toggle,
        el_base,
        el_outcome,
        ga_base,
        ga_outcome,
        gb_base,
        gb_outcome,
        group_a_selection,
        group_b_selection,
        set_browse_selection,
        set_group_a_selection,
        set_group_b_selection,
        tl_resample_button,
        tl_sample_size,
        tl_seed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 Browse the case log

    Choose a set of cases and press **Show** to list their features. The explorer
    below the table lets you chart them.
    """)
    return


@app.cell(hide_code=True)
def _(
    NO_RULE_FILTER,
    build_rule_filter_options,
    el_base,
    iox,
    iox_state_updated,
    mo,
):
    # Which rules exist depends on the base (and on InterOX), so this menu is
    # rebuilt. `NO_RULE_FILTER` is on every one of those menus, so the default
    # below is always one of its own options.
    iox_state_updated
    el_rule = mo.ui.dropdown(build_rule_filter_options(iox, el_base.value),
                             value=NO_RULE_FILTER)
    return (el_rule,)


@app.cell(hide_code=True)
def _(build_selection, el_base, el_outcome, el_rule, mo, set_browse_selection):
    # The commit gate. A run button's `on_change` fires only when it is pressed,
    # so the menus above stay live -- they can still cascade -- while nothing is
    # resolved or drawn until Show reads their values here.
    el_show = mo.ui.run_button(
        label="Show",
        on_change=lambda _: set_browse_selection(
            build_selection(el_base.value, el_outcome.value, el_rule.value)
        ),
    )

    mo.hstack([mo.md("Explore"), el_base, mo.md("case log, with"),
               el_outcome, el_rule, el_show], justify="start")
    return


@app.cell(hide_code=True)
def _(
    browse_selection,
    describe_selection,
    iox,
    iox_state_updated,
    mo,
    resolve_selection,
    validate_selection,
):
    # Re-resolved whenever InterOX changes, so a store or remove either updates
    # the rows or flags the selection as stale. The rule menu above is reset on
    # every such change, so the caption -- not the menus -- says what is shown.
    iox_state_updated
    _selection = browse_selection()
    _stale = None if _selection is None else validate_selection(iox, _selection)

    if _selection is None:
        browse_view = mo.md("Pick a selection above and press **Show**.")
    elif _stale is not None:
        browse_view = mo.callout(mo.md(_stale), kind="warn")
    else:
        _df = resolve_selection(iox, **_selection)
        browse_view = mo.vstack([
            mo.md(f"Showing **{describe_selection(iox, _selection)}** "
                  f"({len(_df)} cases)."),
            _df,
            mo.ui.data_explorer(_df),
        ])

    browse_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Compare case timelines

    Each case is one row and each of its events a marker, placed by its time since
    the first event of the case — so every case starts at 0 and the cases are
    directly comparable. Turn on a second group to compare, e.g. the cases a rule
    claims against those it misses. Click an activity in the legend to show only
    that activity; drag across the plot to zoom in.
    """)
    return


@app.cell(hide_code=True)
def _(
    NO_RULE_FILTER,
    build_rule_filter_options,
    ga_base,
    iox,
    iox_state_updated,
    mo,
):
    # Group A's rule menu. It never shares a cell with Group B's: rebuilding
    # one would reset the other.
    iox_state_updated
    ga_rule = mo.ui.dropdown(build_rule_filter_options(iox, ga_base.value),
                             value=NO_RULE_FILTER)
    return (ga_rule,)


@app.cell(hide_code=True)
def _(
    NO_RULE_FILTER,
    build_rule_filter_options,
    gb_base,
    iox,
    iox_state_updated,
    mo,
):
    iox_state_updated
    gb_rule = mo.ui.dropdown(build_rule_filter_options(iox, gb_base.value),
                             value=NO_RULE_FILTER)
    return (gb_rule,)


@app.cell(hide_code=True)
def _(
    build_selection,
    compare_toggle,
    ga_base,
    ga_outcome,
    ga_rule,
    gb_base,
    gb_outcome,
    gb_rule,
    mo,
    set_group_a_selection,
    set_group_b_selection,
    tl_resample_button,
    tl_sample_size,
):
    # One Show for both groups: the second group is another part of the same
    # picture, so it is drawn in the same stroke. The Show button is the only
    # element this cell defines, and a run button carries nothing a rebuild
    # could lose -- which is why it may read both groups' menus without
    # coupling them to each other. Rendering an element defined in another
    # cell does not rebuild it, so the compare toggle disturbs neither group.
    def _commit(_):
        set_group_a_selection(
            build_selection(ga_base.value, ga_outcome.value, ga_rule.value)
        )
        set_group_b_selection(
            build_selection(gb_base.value, gb_outcome.value, gb_rule.value)
            if compare_toggle.value else None
        )

    tl_show = mo.ui.run_button(label="Show", on_change=_commit)

    mo.vstack([
        mo.hstack([mo.md("**Group A** —"), ga_base, mo.md("case log, with"),
                   ga_outcome, ga_rule], justify="start"),
        compare_toggle,
        mo.hstack([mo.md("**Group B** —"), gb_base, mo.md("case log, with"),
                   gb_outcome, gb_rule], justify="start")
        if compare_toggle.value else mo.md(""),
        mo.hstack([tl_show, tl_sample_size, tl_resample_button], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(
    compare_toggle,
    describe_selection,
    explorer,
    group_a_selection,
    group_b_selection,
    iox,
    iox_state_updated,
    mo,
    resolve_selection,
    tl_sample_size,
    tl_seed,
    validate_selection,
):
    iox_state_updated
    _notes = []

    def _resolve(selection, name):
        """Resolve one committed selection, or say why it no longer resolves."""
        if selection is None:
            return None, name
        _stale = validate_selection(iox, selection)
        if _stale is not None:
            _notes.append(
                mo.callout(mo.md(f"**Group {name}** — {_stale}"), kind="warn")
            )
            return None, name
        return resolve_selection(iox, **selection), describe_selection(iox, selection)

    _group_a, _label_a = _resolve(group_a_selection(), "A")
    _group_b, _label_b = (_resolve(group_b_selection(), "B")
                          if compare_toggle.value else (None, "B"))

    _timeline = mo.md(
        "Choose a Group A selection above and press **Show** to draw the timeline."
    )
    if _group_a is not None:
        _timeline = explorer.timeline(
            _group_a,
            _group_b,
            label_a=_label_a,
            label_b=_label_b,
            sample_size=tl_sample_size.value,
            seed=tl_seed(),
        )
    mo.vstack([*_notes, _timeline])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Read a single case

    Pick any case — e.g. one you found in 4.1 or 4.2 — to see all its events as a
    timeline and as a table.
    """)
    return


@app.cell(hide_code=True)
def _(explorer, mo):
    # The whole case log, not the timeline's Group A: a case found in the data
    # browser above should be readable here directly, and the timeline only ever
    # draws a sample of its group anyway. Searchable whatever the size -- marimo
    # only turns search on by itself past a thousand options, and a few hundred
    # opaque case ids are already unscannable.
    case_picker = mo.ui.dropdown(
        options=explorer.case_ids,
        searchable=True,
        label=f"Case ({len(explorer.case_ids)} in the case log)",
    )
    case_picker
    return (case_picker,)


@app.cell(hide_code=True)
def _(case_picker, explorer, mo):
    case_detail = mo.md("Pick a case above.")
    if case_picker.value is not None:
        case_events = explorer.case_events(case_picker.value)
        case_detail = mo.vstack([
            explorer.timeline([case_picker.value],
                              label_a=f"case {case_picker.value}"),
            mo.ui.table(case_events.reset_index(), selection=None),
        ])
    case_detail
    return


if __name__ == "__main__":
    app.run()
