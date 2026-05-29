"""
Lakers Offseason Scenario Explorer -- Streamlit Dashboard (Phase 2B scaffold).

Run: streamlit run dashboard/app.py

Three pages:
  1. Player Explorer  -- Browse all 582 players with filters and sorting
  2. Lakers War Room  -- Roster, cap situation, top targets by archetype
  3. Trade Simulator  -- Interactive trade builder with real-time CBA check
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable when running from repo root or dashboard/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.pick_value import lakers_tradeable_picks_summary
from src.models.trade_evaluator import evaluate_single_trade, packages_to_dataframe, build_trade_packages
from src.utils.cba_rules import available_exceptions, lakers_cap_scenarios, salary_match_options
from src.utils.constants import (
    LAKERS_DRAFT_PICKS,
    LAKERS_EXTENSION_ELIGIBLE,
    LAKERS_TRADEABLE,
    RESTRICTED_FREE_AGENTS,
    SALARY_CAP_2026_27,
)

# ---------------------------------------------------------------------------
# Colors and config
# ---------------------------------------------------------------------------

PURPLE = "#552583"
GOLD = "#FDB927"
LIGHT_GOLD = "#FFE08A"
DARK_BG = "#1a1a2e"

st.set_page_config(
    page_title="Lakers Trade Engine",
    page_icon="basketball",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Lakers branding
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0f0f1a; }}
    .main-header {{
        background: linear-gradient(135deg, {PURPLE}, {DARK_BG});
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid {GOLD};
        margin-bottom: 1.5rem;
    }}
    .main-header h1 {{ color: {GOLD}; margin: 0; font-size: 1.8rem; }}
    .main-header p {{ color: #ccc; margin: 0.3rem 0 0; font-size: 0.9rem; }}
    .metric-card {{
        background: #1e1e3a;
        border: 1px solid {PURPLE};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    .legal-badge {{ color: #00cc66; font-weight: bold; }}
    .illegal-badge {{ color: #ff4444; font-weight: bold; }}
    .archetype-rim_runner {{ background-color: #1a3a5c; border-left: 4px solid #4da6ff; }}
    .archetype-three_and_d {{ background-color: #2a1a3a; border-left: 4px solid #cc44ff; }}
    .archetype-shooter {{ background-color: #3a2a1a; border-left: 4px solid {GOLD}; }}
    .archetype-secondary_creator {{ background-color: #1a3a1a; border-left: 4px solid #44ff88; }}
    div[data-testid="stSidebarNav"] {{ display: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

DATA_PATH = _REPO_ROOT / "data" / "cache" / "player_valued_2025-26.parquet"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Valued parquet not found at {DATA_PATH}. Run the Phase 1 pipeline first.")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    # Round display columns
    for col in ["ON_COURT_VALUE", "SURPLUS_VALUE", "AGE_ADJUSTED_SURPLUS",
                "ARCHETYPE_DISTANCE", "ROLL_GRAVITY_SCORE", "RIM_PROTECTION_SCORE"]:
        if col in df.columns:
            df[col] = df[col].round(3)
    return df


def fmt_salary(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "FA / N/A"
    return f"${val:,.0f}"


def fmt_val(val: float | None, decimals: int = 3) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val:+.{decimals}f}" if val != 0 else "0.000"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    f"<h2 style='color:{GOLD}; margin-bottom:0.2rem;'>Lakers Trade Engine</h2>"
    f"<p style='color:#aaa; font-size:0.8rem;'>2026 Offseason Analysis</p>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Player Explorer", "Lakers War Room", "Trade Simulator"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    "<p style='color:#888; font-size:0.75rem;'>"
    "Data: 2025-26 season. CBA rules: 2026-27 projected thresholds. "
    "Source: ESPN Bobby Marks / Sports Business Classroom, May 2026."
    "</p>",
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE 1: Player Explorer
# ===========================================================================

def page_player_explorer(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='main-header'>"
        "<h1>Player Explorer</h1>"
        "<p>Browse all 582 players. Filter by team, position, age, archetype, or value metrics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        teams = sorted(df["TEAM_ABBREVIATION"].dropna().unique())
        selected_teams = st.multiselect("Team", teams, placeholder="All teams")

    with col2:
        archetypes = ["rim_runner", "three_and_d", "secondary_creator", "shooter"]
        selected_archetypes = st.multiselect("Archetype", archetypes, placeholder="All archetypes")

    with col3:
        age_range = st.slider(
            "Age range",
            int(df["AGE"].min(skipna=True)),
            int(df["AGE"].max(skipna=True)),
            (18, 35),
        )

    with col4:
        min_value = st.slider(
            "Min ON_COURT_VALUE",
            float(df["ON_COURT_VALUE"].min(skipna=True)),
            float(df["ON_COURT_VALUE"].max(skipna=True)),
            -1.0,
            step=0.1,
        )

    col5, col6 = st.columns(2)
    with col5:
        fa_only = st.checkbox("Free agents only")
    with col6:
        sort_col = st.selectbox(
            "Sort by",
            ["COMBINED_TARGET_RANK", "ON_COURT_VALUE", "SURPLUS_VALUE",
             "AGE_ADJUSTED_SURPLUS", "ARCHETYPE_DISTANCE", "ROLL_GRAVITY_SCORE"],
        )

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    mask = pd.Series(True, index=df.index)
    if selected_teams:
        mask &= df["TEAM_ABBREVIATION"].isin(selected_teams)
    if selected_archetypes:
        mask &= df["BEST_ARCHETYPE"].isin(selected_archetypes)
    mask &= (df["AGE"] >= age_range[0]) & (df["AGE"] <= age_range[1])
    mask &= (df["ON_COURT_VALUE"] >= min_value) | df["ON_COURT_VALUE"].isna()
    if fa_only:
        mask &= df["IS_FREE_AGENT"] == True

    filtered = df[mask].copy()

    # Sort
    asc = sort_col in ("ARCHETYPE_DISTANCE", "COMBINED_TARGET_RANK")
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=asc, na_position="last")

    # ------------------------------------------------------------------
    # Display columns
    # ------------------------------------------------------------------
    display_cols = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
        "ON_COURT_VALUE", "SURPLUS_VALUE", "AGE_ADJUSTED_SURPLUS",
        "SALARY_FORWARD", "IS_FREE_AGENT",
        "BEST_ARCHETYPE", "ARCHETYPE_DISTANCE",
        "ROLL_GRAVITY_SCORE", "COMBINED_TARGET_RANK",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.caption(f"Showing {len(filtered)} players")

    display_df = filtered[display_cols].copy()
    display_df["SALARY_FORWARD"] = display_df["SALARY_FORWARD"].apply(
        lambda x: fmt_salary(x) if pd.notna(x) else "FA"
    )

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "PLAYER_NAME": st.column_config.TextColumn("Player", width=170),
            "TEAM_ABBREVIATION": st.column_config.TextColumn("Team", width=70),
            "AGE": st.column_config.NumberColumn("Age", width=55, format="%d"),
            "ON_COURT_VALUE": st.column_config.NumberColumn("Value", format="%.3f"),
            "SURPLUS_VALUE": st.column_config.NumberColumn("Surplus", format="%.3f"),
            "AGE_ADJUSTED_SURPLUS": st.column_config.NumberColumn("Age-Adj Surplus", format="%.3f"),
            "SALARY_FORWARD": st.column_config.TextColumn("2026-27 Salary"),
            "IS_FREE_AGENT": st.column_config.CheckboxColumn("FA?", width=55),
            "BEST_ARCHETYPE": st.column_config.TextColumn("Archetype"),
            "ARCHETYPE_DISTANCE": st.column_config.NumberColumn("Arch. Dist.", format="%.2f"),
            "ROLL_GRAVITY_SCORE": st.column_config.NumberColumn("Roll Gravity", format="%.2f"),
            "COMBINED_TARGET_RANK": st.column_config.NumberColumn("Target Rank", format="%d"),
        },
    )

    # ------------------------------------------------------------------
    # Player detail card
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Player Detail")
    player_names = sorted(filtered["PLAYER_NAME"].dropna().unique())
    if player_names:
        selected_player = st.selectbox("Select a player to inspect", player_names)
        prow = filtered[filtered["PLAYER_NAME"] == selected_player].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ON_COURT_VALUE", fmt_val(prow.get("ON_COURT_VALUE")))
        c2.metric("SURPLUS_VALUE", fmt_val(prow.get("SURPLUS_VALUE")))
        c3.metric("AGE_ADJUSTED_SURPLUS", fmt_val(prow.get("AGE_ADJUSTED_SURPLUS")))
        c4.metric("2026-27 Salary", fmt_salary(prow.get("SALARY_FORWARD")))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Best Archetype", str(prow.get("BEST_ARCHETYPE", "-")))
        c6.metric("Arch. Distance", fmt_val(prow.get("ARCHETYPE_DISTANCE"), 2))
        c7.metric("Roll Gravity", fmt_val(prow.get("ROLL_GRAVITY_SCORE"), 2))
        c8.metric("Rim Protection", fmt_val(prow.get("RIM_PROTECTION_SCORE"), 2))

        with st.expander("All metrics"):
            stat_cols = ["BPM", "WS_48", "NET_RATING", "TS_PCT", "USG_PCT",
                         "DREB_PCT", "OREB_PCT", "AST_PCT", "FG3_PCT", "DURABILITY_SCORE"]
            stat_cols = [c for c in stat_cols if c in prow.index and pd.notna(prow[c])]
            if stat_cols:
                st.dataframe(
                    pd.DataFrame({"Metric": stat_cols, "Value": [round(float(prow[c]), 3) for c in stat_cols]}),
                    hide_index=True, use_container_width=True,
                )


# ===========================================================================
# PAGE 2: Lakers War Room
# ===========================================================================

def page_lakers_war_room(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='main-header'>"
        "<h1>Lakers War Room</h1>"
        "<p>2026 offseason overview: roster, cap situation, and top targets by archetype.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    lal = df[df["TEAM_ABBREVIATION"] == "LAL"].copy()

    # ------------------------------------------------------------------
    # Cap situation overview
    # ------------------------------------------------------------------
    st.subheader("Cap Scenarios (2026-27)")
    scenarios = lakers_cap_scenarios()
    scen_cols = st.columns(len(scenarios))
    for col, (key, scen) in zip(scen_cols, scenarios.items()):
        with col:
            committed = scen.get("approx_committed", 0)
            st.markdown(
                f"<div style='background:#1e1e3a; border:1px solid {PURPLE}; "
                f"border-radius:8px; padding:0.8rem;'>"
                f"<b style='color:{GOLD}'>{scen['description']}</b><br>"
                f"<span style='color:#aaa; font-size:0.85rem;'>{scen['apron_status'].replace('_', ' ').title()}</span><br>"
                f"<b style='font-size:1.1rem;'>~${committed/1e6:.0f}M committed</b><br>"
                f"<span style='color:#ccc; font-size:0.8rem;'>{scen['note']}</span>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Draft pick inventory
    # ------------------------------------------------------------------
    st.subheader("Draft Pick Inventory")
    picks_summary = lakers_tradeable_picks_summary()
    if picks_summary:
        picks_df = pd.DataFrame(picks_summary)
        picks_df["estimated_value_M"] = picks_df["estimated_value_M"].apply(
            lambda x: f"${x:.1f}M" if x else "-"
        )
        picks_df["expected_range"] = picks_df["expected_range"].apply(
            lambda r: f"#{r[0]}-{r[1]}" if r[0] != r[1] else f"#{r[0]}"
        )
        st.dataframe(
            picks_df[["year", "round", "expected_range", "estimated_value_M", "note"]],
            hide_index=True, use_container_width=True,
            column_config={
                "year": "Year",
                "round": "Round",
                "expected_range": "Expected Pick",
                "estimated_value_M": "Est. Value",
                "note": "Note",
            },
        )
    st.caption(
        "Non-tradeable: 2027 1st (top-4 protected to Memphis), 2029 1st (owed to Dallas unprotected). "
        "Stepien Rule: cannot trade 2031+2032 or 2032+2033 in same deal."
    )

    # ------------------------------------------------------------------
    # Current roster valuations
    # ------------------------------------------------------------------
    st.subheader("Current Lakers Roster")
    roster_cols = [
        "PLAYER_NAME", "AGE", "ON_COURT_VALUE", "SURPLUS_VALUE",
        "SALARY_FORWARD", "IS_FREE_AGENT", "BEST_ARCHETYPE", "DURABILITY_SCORE",
    ]
    roster_cols = [c for c in roster_cols if c in lal.columns]
    roster_display = lal[roster_cols].copy()
    roster_display["SALARY_FORWARD"] = roster_display["SALARY_FORWARD"].apply(
        lambda x: fmt_salary(x) if pd.notna(x) else "FA"
    )
    roster_display = roster_display.sort_values("ON_COURT_VALUE", ascending=False, na_position="last")

    st.dataframe(
        roster_display.reset_index(drop=True),
        hide_index=True, use_container_width=True,
        column_config={
            "PLAYER_NAME": st.column_config.TextColumn("Player", width=170),
            "AGE": st.column_config.NumberColumn("Age", format="%d"),
            "ON_COURT_VALUE": st.column_config.NumberColumn("Value", format="%.3f"),
            "SURPLUS_VALUE": st.column_config.NumberColumn("Surplus", format="%.3f"),
            "SALARY_FORWARD": st.column_config.TextColumn("2026-27 Salary"),
            "IS_FREE_AGENT": st.column_config.CheckboxColumn("FA?"),
            "BEST_ARCHETYPE": st.column_config.TextColumn("Archetype"),
            "DURABILITY_SCORE": st.column_config.NumberColumn("Durability", format="%.2f"),
        },
    )

    # Extension-eligible callout
    with st.expander("Extension-eligible players (June 2026 deadlines)"):
        ext_rows = []
        for name, info in LAKERS_EXTENSION_ELIGIBLE.items():
            ext_rows.append({"Player": name, "Max Extension": info["max_extension"], "Deadline": info["deadline"]})
        st.dataframe(pd.DataFrame(ext_rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Top targets by archetype
    # ------------------------------------------------------------------
    st.subheader("Top Trade / FA Targets by Archetype")

    non_lakers = df[df["TEAM_ABBREVIATION"] != "LAL"].copy()

    archetype_colors = {
        "rim_runner": "#4da6ff",
        "three_and_d": "#cc44ff",
        "shooter": GOLD,
        "secondary_creator": "#44ff88",
    }

    arch_tabs = st.tabs(["Rim Runner", "3-and-D Wing", "Shooter", "Secondary Creator"])
    arch_map = {
        0: "rim_runner",
        1: "three_and_d",
        2: "shooter",
        3: "secondary_creator",
    }

    for tab_idx, tab in enumerate(arch_tabs):
        archetype = arch_map[tab_idx]
        color = archetype_colors[archetype]
        with tab:
            subset = non_lakers[
                non_lakers["BEST_ARCHETYPE"] == archetype
            ].sort_values("AGE_ADJUSTED_SURPLUS", ascending=False, na_position="last").head(15)

            if subset.empty:
                st.info(f"No qualifying players found for archetype: {archetype}")
                continue

            display_cols = [
                "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
                "ON_COURT_VALUE", "AGE_ADJUSTED_SURPLUS", "SURPLUS_VALUE",
                "SALARY_FORWARD", "IS_FREE_AGENT", "ARCHETYPE_DISTANCE",
            ]
            display_cols = [c for c in display_cols if c in subset.columns]
            tab_df = subset[display_cols].copy()
            tab_df["SALARY_FORWARD"] = tab_df["SALARY_FORWARD"].apply(
                lambda x: fmt_salary(x) if pd.notna(x) else "FA"
            )
            # Flag RFAs
            tab_df["RFA"] = tab_df["PLAYER_NAME"].isin(RESTRICTED_FREE_AGENTS)

            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 0.75rem; margin-bottom:0.75rem;'>"
                f"<b style='color:{color};'>{archetype.replace('_', ' ').title()}</b> - "
                f"Top {len(subset)} candidates ranked by Age-Adjusted Surplus"
                "</div>",
                unsafe_allow_html=True,
            )

            st.dataframe(
                tab_df.reset_index(drop=True),
                hide_index=True, use_container_width=True,
                column_config={
                    "PLAYER_NAME": st.column_config.TextColumn("Player", width=170),
                    "TEAM_ABBREVIATION": st.column_config.TextColumn("Team", width=70),
                    "AGE": st.column_config.NumberColumn("Age", format="%d"),
                    "ON_COURT_VALUE": st.column_config.NumberColumn("Value", format="%.3f"),
                    "AGE_ADJUSTED_SURPLUS": st.column_config.NumberColumn("Age-Adj Surplus", format="%.3f"),
                    "SURPLUS_VALUE": st.column_config.NumberColumn("Surplus", format="%.3f"),
                    "SALARY_FORWARD": st.column_config.TextColumn("2026-27 Salary"),
                    "IS_FREE_AGENT": st.column_config.CheckboxColumn("FA?"),
                    "ARCHETYPE_DISTANCE": st.column_config.NumberColumn("Arch. Dist.", format="%.2f"),
                    "RFA": st.column_config.CheckboxColumn("RFA?"),
                },
            )


# ===========================================================================
# PAGE 3: Trade Simulator
# ===========================================================================

def page_trade_simulator(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='main-header'>"
        "<h1>Trade Simulator</h1>"
        "<p>Build trade scenarios interactively. Real-time CBA legality check and value assessment.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Salary matching rules reflect the 2026-27 CBA thresholds. "
        "Select your apron scenario, pick a target, choose Lakers assets to send, and hit Evaluate."
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Incoming: Target Player")

        # Target selection
        all_players = sorted(
            df[df["TEAM_ABBREVIATION"] != "LAL"]["PLAYER_NAME"].dropna().unique()
        )
        target_name = st.selectbox("Select target player", all_players, key="target_select")

        if target_name:
            trow = df[df["PLAYER_NAME"] == target_name].iloc[0]
            t_salary = trow.get("SALARY_FORWARD")
            t_value = trow.get("ON_COURT_VALUE")
            t_archetype = trow.get("BEST_ARCHETYPE", "None")
            t_arch_dist = trow.get("ARCHETYPE_DISTANCE")
            t_is_fa = bool(trow.get("IS_FREE_AGENT", False))

            is_rfa = target_name in RESTRICTED_FREE_AGENTS

            m1, m2, m3 = st.columns(3)
            m1.metric("2026-27 Salary", fmt_salary(t_salary))
            m2.metric("ON_COURT_VALUE", fmt_val(t_value))
            m3.metric("Archetype", str(t_archetype))

            if t_is_fa:
                st.info("This player is a **free agent** (no trade needed - sign via cap room or MLE)")
            if is_rfa:
                st.warning("RFA: Current team can match any offer sheet. Trade/offer-sheet risk.")
            if t_arch_dist and pd.notna(t_arch_dist):
                st.caption(f"Archetype distance from nearest centroid: {t_arch_dist:.2f}")

    with col_right:
        st.subheader("Outgoing: Lakers Assets")

        apron_status = st.selectbox(
            "Lakers apron scenario",
            ["first_apron", "second_apron", "below_aprons", "below_cap"],
            format_func=lambda x: {
                "first_apron": "Above First Apron (100% match, can aggregate)",
                "second_apron": "Above Second Apron (100% match, NO aggregation)",
                "below_aprons": "Below Both Aprons (generous matching)",
                "below_cap": "Cap Space Team (cap room available)",
            }.get(x, x),
        )

        # Player checkboxes
        st.markdown("**Players to include in trade:**")
        selected_players_out = []
        for player, salary in LAKERS_TRADEABLE.items():
            checked = st.checkbox(f"{player} (${salary:,.0f})", key=f"out_{player}")
            if checked:
                selected_players_out.append(player)

        # Pick checkboxes
        st.markdown("**Picks to include:**")
        selected_picks = []
        tradeable_picks = {k: v for k, v in LAKERS_DRAFT_PICKS.items() if v.get("tradeable")}
        for pick_key, pick_info in tradeable_picks.items():
            label = f"{pick_info['year']} {'1st' if pick_info['round']==1 else '2nd'} Round"
            if "pick_number" in pick_info:
                label += f" (#{pick_info['pick_number']})"
            label += f" - est. ${pick_info.get('expected_range', (20,30))[0]}-{pick_info.get('expected_range', (20,30))[1]}"
            if st.checkbox(label, key=f"pick_{pick_key}"):
                selected_picks.append(pick_key)

    # ------------------------------------------------------------------
    # Evaluate button
    # ------------------------------------------------------------------
    st.divider()
    evaluate_clicked = st.button("Evaluate Trade", type="primary", use_container_width=True)

    if evaluate_clicked and target_name:
        if not selected_players_out:
            st.warning("Select at least one Lakers player to send out.")
        else:
            pkg = evaluate_single_trade(
                outgoing_players=selected_players_out,
                incoming_player=target_name,
                df=df,
                picks_included=selected_picks,
                apron_status=apron_status,
            )

            # ----------------------------------------------------------
            # Results display
            # ----------------------------------------------------------
            st.subheader("Trade Evaluation Results")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if pkg.is_legal:
                    st.markdown(f"<h3 class='legal-badge'>LEGAL TRADE</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 class='illegal-badge'>ILLEGAL TRADE</h3>", unsafe_allow_html=True)

                st.markdown(f"**Legality note:** {pkg.legality_reason}")

                out_sal_str = f"${pkg.outgoing_salary:,.0f}"
                in_sal_str = f"${pkg.target_salary:,.0f}"
                st.markdown(
                    f"**Salary:** Sending {out_sal_str} | Receiving {in_sal_str} | "
                    f"Net ${pkg.outgoing_salary - pkg.target_salary:+,.0f}"
                )

                if selected_picks:
                    st.markdown(
                        f"**Picks included:** {', '.join(selected_picks)} "
                        f"(est. total value ${pkg.picks_value_M:.1f}M surplus)"
                    )

            with res_col2:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Feasibility",
                    f"{pkg.feasibility_score:.0%}",
                    help="Likelihood the other team accepts. 70%+ = realistic offer.",
                )
                m2.metric(
                    "Lakers Net Value",
                    fmt_val(pkg.lakers_net_value),
                    help="Positive = Lakers gain on-court value in this trade.",
                )
                m3.metric(
                    "Other Team Premium",
                    f"{pkg.other_team_value_premium:.2f}x",
                    help="1.10x+ = team selling the player has reason to say yes.",
                )

            st.progress(min(1.0, pkg.feasibility_score), text=f"Feasibility: {pkg.feasibility_score:.0%}")

            # ----------------------------------------------------------
            # Apron implications
            # ----------------------------------------------------------
            with st.expander("Apron implications and available exceptions"):
                exceptions = available_exceptions(apron_status)
                exc_rows = [{"Mechanism": k.replace("_", " ").title(), "Available": str(v)} for k, v in exceptions.items()]
                st.dataframe(pd.DataFrame(exc_rows), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Auto-package builder section
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Auto-Package Builder")
    st.caption("Let the engine find all salary-legal combinations for a target player.")

    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        auto_target = st.selectbox(
            "Target player for auto-builder",
            sorted(df[df["TEAM_ABBREVIATION"] != "LAL"]["PLAYER_NAME"].dropna().unique()),
            key="auto_target",
        )
    with col_ab2:
        auto_apron = st.selectbox(
            "Apron scenario",
            ["first_apron", "second_apron", "below_aprons"],
            key="auto_apron",
            format_func=lambda x: x.replace("_", " ").title(),
        )

    auto_picks = st.multiselect(
        "Include picks in all packages",
        [k for k, v in LAKERS_DRAFT_PICKS.items() if v.get("tradeable")],
        key="auto_picks",
    )

    if st.button("Find All Legal Packages", key="auto_build"):
        packages = build_trade_packages(
            target_player_name=auto_target,
            df=df,
            apron_status=auto_apron,
            include_picks=auto_picks or None,
            max_packages=15,
        )

        if not packages:
            st.warning(
                f"No salary-legal packages found for {auto_target} under {auto_apron} rules. "
                "Try a different apron scenario or check that the player has a 2026-27 salary on record."
            )
        else:
            pkg_df = packages_to_dataframe(packages)
            st.success(f"Found {len(packages)} legal package(s) for **{auto_target}**.")
            st.dataframe(pkg_df, hide_index=True, use_container_width=True)


# ===========================================================================
# Router
# ===========================================================================

def main() -> None:
    df = load_data()

    if page == "Player Explorer":
        page_player_explorer(df)
    elif page == "Lakers War Room":
        page_lakers_war_room(df)
    elif page == "Trade Simulator":
        page_trade_simulator(df)


if __name__ == "__main__":
    main()
