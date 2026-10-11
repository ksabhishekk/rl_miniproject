import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from experiments.run_comparative_benchmark import (
    PROJECT_ROOT,
    _load_shared_demo_agents,
    _shared_demo_checkpoint_paths,
    load_or_train_shared_demo_agents,
    run_shared_scenario_comparison,
)
from recommendation.data_preprocessing import load_movies, load_processed_matrix


st.set_page_config(
    page_title="RecShield | Attack Observatory",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #101315; }
    [data-testid="stSidebar"] {
        background: #15191c;
        border-right: 1px solid #2a3034;
    }
    [data-testid="stHeader"] { background: #101315; }
    h1, h2, h3, h4 { letter-spacing: -0.025em; }
    h1 { font-size: 2.2rem !important; }
    [data-testid="stMetric"] {
        background: #171c20;
        border: 1px solid #2a3034;
        border-radius: 6px;
        padding: 12px 14px;
    }
    [data-testid="stMetricValue"] { color: #c6d8a4; }
    .eyebrow {
        color: #91a99b;
        font-size: .72rem;
        font-weight: 650;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .profile-label {
        border-left: 2px solid #879e91;
        background: #171c20;
        padding: 10px 13px;
        margin: 8px 0;
    }
    .profile-label.attack { border-color: #cf8e73; }
    .profile-label.genuine { border-color: #9fbd8c; }
    .profile-meta { color: #a7b0b2; font-size: .85rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #2a3034; }
    div.stButton > button { border-radius: 5px; }
    hr { border-color: #2a3034; }
    </style>
    """,
    unsafe_allow_html=True,
)


TARGETS = {
    40: "Billy Madison",
    77: "Free Willy",
    15: "French Twist",
}
AGENT_ORDER = ("Q-Learning", "DQN", "PPO")
ATTACK_TYPES = ("random", "average", "bandwagon")
ATTACK_LABELS = {
    "random": "Random fillers",
    "average": "Average fillers",
    "bandwagon": "Bandwagon",
}
COMPARISON_MODES = {
    "shared_scenario": {
        "label": "Shared-scenario replay",
        "description": (
            "The same generated stream is replayed through three isolated policy environments. "
            "This keeps the benchmark fair and makes matching values possible when policies converge."
        ),
        "interpretation": (
            "Same stream replay: identical values are expected when the policies learn the same defensive response to that exact scenario."
        ),
        "shared_scenario": True,
    },
    "independent_runs": {
        "label": "Independent runs",
        "description": (
            "Each policy gets its own generated scenario, so the comparison reflects separate random draws rather than a single replayed stream."
        ),
        "interpretation": (
            "Independent runs: different random draws help reveal variation between policies, but they are not directly comparable to a single shared replay."
        ),
        "shared_scenario": False,
    },
}


def get_comparison_mode_details(mode: str | None) -> Dict[str, Any]:
    normalized_mode = mode if mode in COMPARISON_MODES else "shared_scenario"
    return dict(COMPARISON_MODES[normalized_mode])


def calculate_stream_counts(episode_length: int, attack_share_pct: float) -> tuple[int, int]:
    """Return the integer attack and genuine counts for a stream length and attack share."""
    episode_length = max(0, int(episode_length))
    attack_share_pct = max(0.0, min(100.0, float(attack_share_pct)))
    attack_count = int(round(episode_length * (attack_share_pct / 100.0)))
    attack_count = max(0, min(episode_length, attack_count))
    return attack_count, episode_length - attack_count


def should_auto_refresh(
    last_signature: Dict[str, Any] | None,
    current_signature: Dict[str, Any],
    agents: Dict[str, Any] | None,
    loaded_target_idx: int | None,
    target_idx: int,
) -> bool:
    return bool(
        agents
        and loaded_target_idx == target_idx
        and last_signature is not None
        and last_signature != current_signature
    )


def prepare_demo_agents_for_target(target_idx: int, *, silent: bool = False) -> bool:
    agents = st.session_state.get("demo_agents")
    loaded_target_idx = st.session_state.get("loaded_target_idx")
    if agents and loaded_target_idx == target_idx:
        return True

    if not manifest:
        try:
            with st.spinner("Preparing policies for the selected target..."):
                agents, trained_manifest = load_or_train_shared_demo_agents(
                    training_seed=42,
                    target_idx=target_idx,
                )
            st.session_state["demo_agents"] = agents
            st.session_state["demo_manifest"] = trained_manifest
            st.session_state["loaded_target_idx"] = target_idx
            return True
        except Exception as exc:
            if not silent:
                st.error(f"The policy set could not be prepared for the selected target: {exc}")
            return False

    try:
        seed = int(manifest.get("training_seed", 42))
        agents, loaded_manifest = _load_shared_demo_agents(paths, target_idx, seed)
        st.session_state["demo_agents"] = agents
        st.session_state["demo_manifest"] = loaded_manifest
        st.session_state["loaded_target_idx"] = target_idx
        return True
    except Exception:
        try:
            with st.spinner("Policy set for this target is missing; preparing it automatically..."):
                agents, trained_manifest = load_or_train_shared_demo_agents(
                    training_seed=42,
                    target_idx=target_idx,
                )
            st.session_state["demo_agents"] = agents
            st.session_state["demo_manifest"] = trained_manifest
            st.session_state["loaded_target_idx"] = target_idx
            return True
        except Exception as exc:
            if not silent:
                st.error(f"Could not prepare the policy set for the selected target: {exc}")
            return False


@st.cache_resource
def get_matrices():
    return (
        load_processed_matrix("data/processed/train_matrix.npy"),
        load_processed_matrix("data/processed/test_matrix.npy"),
    )


@st.cache_data
def get_movie_catalog() -> pd.DataFrame:
    return load_movies()


def checkpoint_manifest() -> Dict[str, Any] | None:
    path = _shared_demo_checkpoint_paths()["manifest"]
    if not os.path.isfile(path):
        return None
    with open(path, "r") as manifest_file:
        return json.load(manifest_file)


def movie_title_map() -> Dict[int, str]:
    return {
        int(row.item_id) - 1: str(row.title)
        for row in get_movie_catalog().itertuples(index=False)
    }


def ratings_table(
    ratings: List[Dict[str, Any]], titles: Dict[int, str]
) -> pd.DataFrame:
    rows = [
        {
            "Movie": titles.get(
                int(item["item_index"]), f"Item {int(item['item_index']) + 1}"
            ),
            "Rating": float(item["rating"]),
        }
        for item in ratings
    ]
    return pd.DataFrame(rows, columns=["Movie", "Rating"]).sort_values(
        "Rating", ascending=False, ignore_index=True
    )


def event_log_table(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for event in events:
        features = event["features"]
        rows.append(
            {
                "Step": event["step"],
                "Profile ID": event["profile_index"] + 1,
                "Source": "Injected" if event["is_attacker"] else "MovieLens",
                "Simulation label": "Attack" if event["is_attacker"] else "Genuine",
                "Decision": event["action_name"],
                "Weight": event["defense_weight"],
                "Reward": event["reward"],
                "Target score": event["target_prediction_after"],
                "RDMA": features["rdma"],
                "Target affinity": features["target_affinity"],
                "Activity": features["activity_ratio"],
                "System shift": features["system_prediction_shift"],
                "Ratings": len(event["ratings"]),
            }
        )
    return pd.DataFrame(rows)


def build_profile_history(result: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for policy_name in AGENT_ORDER:
        for event in result["agents"][policy_name]["events"]:
            rows.append(
                {
                    "Policy": policy_name,
                    "Step": event["step"],
                    "Profile ID": event["profile_index"] + 1,
                    "Is attacker": bool(event["is_attacker"]),
                    "Source": "Injected" if event["is_attacker"] else "MovieLens",
                    "Decision": event["action_name"],
                    "Weight": float(event["defense_weight"]),
                    "Reward": float(event["reward"]),
                    "Target before": float(event["target_prediction_before"]),
                    "Target after": float(event["target_prediction_after"]),
                    "Target shift": float(event["target_prediction_after"] - event["target_prediction_before"]),
                    "Ratings": len(event["ratings"]),
                }
            )
    return pd.DataFrame(rows)


def build_run_summary(result: Dict[str, Any], scenario_signature: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for policy_name in AGENT_ORDER:
        metrics = result["agents"][policy_name]
        summary[policy_name] = {
            "Total reward": float(metrics["total_reward"]),
            "Target score shift": float(metrics["target_prediction_shift"]),
            "Target Hit@10 change (pp)": 100.0 * (
                metrics["target_hit_ratio_at_10"] - metrics["clean_target_hit_ratio_at_10"]
            ),
            "Genuine RMSE change": float(metrics["test_rmse"] - metrics["clean_test_rmse"]),
            "Catalog coverage change (pp)": 100.0 * (
                metrics["catalog_coverage"] - metrics["clean_catalog_coverage"]
            ),
            "Genuine Hit@10 change (pp)": 100.0 * (
                metrics["genuine_hit_rate_at_10"] - metrics["clean_genuine_hit_rate_at_10"]
            ),
            "Attack blocked": float(metrics["strict_block_metrics"]["recall"])
            * float(result["scenario"]["num_attackers_in_stream"]),
            "Injected profiles": float(result["scenario"]["num_attackers_in_stream"]),
            "False positives": float(metrics["strict_block_metrics"]["false_positive_rate"])
            * float(result["scenario"]["num_genuine_in_stream"]),
        }
    return {
        "scenario": scenario_signature,
        "summary": summary,
    }


def append_run_history(result: Dict[str, Any], scenario_signature: Dict[str, Any]) -> Dict[str, Any]:
    """Store a single successful run only once per actual button-triggered execution."""
    history = st.session_state.setdefault("run_history", [])
    next_run_id = int(st.session_state.get("run_id_counter", 0)) + 1
    st.session_state["run_id_counter"] = next_run_id

    run_record = build_run_summary(result, scenario_signature)
    run_record["run_id"] = next_run_id
    run_record["timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
    history.append(run_record)
    st.session_state["run_history"] = history[-30:]
    return run_record


train_matrix, _ = get_matrices()
titles = movie_title_map()
paths = _shared_demo_checkpoint_paths()
manifest = checkpoint_manifest()

st.title("RecShield · Attack Observatory")
st.caption(
    "You operate the defense. The simulator generates and injects fake profiles; "
    "each policy decides whether to accept, downweight, or block them."
)

with st.sidebar:
    st.markdown("### Scenario")
    target_idx = st.selectbox(
        "Target movie",
        options=list(TARGETS),
        format_func=lambda index: f"{TARGETS[index]} · item {index + 1}",
        help="The movie the simulated attacker tries to promote.",
    )
    attack_type = st.selectbox(
        "Attack pattern",
        options=ATTACK_TYPES,
        format_func=lambda name: ATTACK_LABELS[name],
        index=2,
        help=(
            "Controls how the simulator builds fake rating profiles. "
            "You are choosing a test attack, not creating profiles yourself."
        ),
    )
    attack_share_pct = st.slider(
        "Injected attack share",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Percent of the total profile stream generated as fake attack profiles.",
    )
    episode_length = st.slider(
        "Profiles in audit stream",
        min_value=20,
        max_value=300,
        value=100,
        step=10,
        help=(
            "Total profiles reviewed, including both genuine users and injected "
            "profiles. Example: 20 profiles at 30% attack share gives 6 attacks."
        ),
    )
    comparison_mode = st.radio(
        "Comparison mode",
        options=("shared_scenario", "independent_runs"),
        format_func=lambda mode: get_comparison_mode_details(mode)["label"],
        index=0,
        help="Choose whether all policies are judged on the same generated stream or on separate random scenarios.",
    )
    comparison_details = get_comparison_mode_details(comparison_mode)
    st.caption(comparison_details["description"])
    st.caption(f"Interpretation: {comparison_details['interpretation']}")
    scenario_seed = st.number_input(
        "Scenario seed",
        min_value=0,
        max_value=999999,
        value=10042,
        step=1,
        help=(
            "Reproduces the generated ratings, sampled genuine users, and profile order "
            "for the selected mode."
        ),
    )
    playback_delay = st.slider(
        "Animation delay per profile (seconds)",
        min_value=0.0,
        max_value=0.15,
        value=0.02,
        step=0.01,
        help="Only slows down the on-screen replay. It does not change any decisions or metrics.",
    )

    st.markdown("---")
    st.markdown("### Saved policies")
    if manifest:
        training = manifest.get("training_budget", {})
        trained_target_idx = int(manifest.get("target_index_used_for_training", 40))
        trained_target = TARGETS.get(
            trained_target_idx, f"MovieLens item {trained_target_idx + 1}"
        )
        st.caption(
            f"Checkpoint training target: {trained_target} · "
            f"{training.get('dqn_timesteps', 0):,} DQN / "
            f"{training.get('ppo_timesteps', 0):,} PPO steps"
        )
        if target_idx != trained_target_idx:
            st.caption(
                "Selected target differs from the training target; the app will prepare the policy set for the active target automatically."
            )
        if st.button("Load policies for selected target", width="stretch"):
            if prepare_demo_agents_for_target(target_idx):
                st.rerun()
    else:
        st.warning("No saved policies found.")
        if st.button("Train and save the three policies", width="stretch"):
            try:
                with st.spinner(
                    "Training Q-Learning, DQN, and PPO. First-time training may take several minutes."
                ):
                    agents, trained_manifest = load_or_train_shared_demo_agents(
                        training_seed=42,
                        target_idx=target_idx,
                    )
                st.session_state["demo_agents"] = agents
                st.session_state["demo_manifest"] = trained_manifest
                st.session_state["loaded_target_idx"] = target_idx
                st.rerun()
            except Exception as exc:
                st.error(f"Training or saving the policies failed: {exc}")

    st.markdown("---")
    st.markdown("### Reference user")
    selected_user = st.selectbox(
        "MovieLens user",
        options=list(range(train_matrix.shape[0])),
        format_func=lambda user_index: f"User {user_index + 1:03d}",
        help=(
            "For inspecting one real user's ratings only. The audit independently "
            "samples genuine users; this choice does not select or inject this profile."
        ),
    )

scenario_signature = {
    "target_idx": target_idx,
    "attack_type": attack_type,
    "attack_share_pct": attack_share_pct,
    "episode_length": episode_length,
    "scenario_seed": int(scenario_seed),
    "comparison_mode": comparison_mode,
}

attack_count, genuine_count = calculate_stream_counts(episode_length, attack_share_pct)

overview = st.columns(3)
overview[0].metric("MovieLens users", f"{train_matrix.shape[0]:,}")
overview[1].metric("Movie catalog", f"{train_matrix.shape[1]:,}")
overview[2].metric("Attack profiles in this run", f"{attack_count}")

profile_col, scenario_col = st.columns([1.1, 0.9], gap="large")
with profile_col:
    st.subheader("Reference profile")
    user_ratings = np.flatnonzero(train_matrix[selected_user] > 0)
    user_table = pd.DataFrame(
        [
            {
                "Movie": titles.get(int(item_idx), f"Item {int(item_idx) + 1}"),
                "Rating": float(train_matrix[selected_user, item_idx]),
            }
            for item_idx in user_ratings
        ]
    ).sort_values("Rating", ascending=False, ignore_index=True)
    st.caption(
        f"User {selected_user + 1} · {len(user_table)} known ratings · "
        "inspection only, not the attack target or selected stream user"
    )
    st.dataframe(user_table.head(10), hide_index=True, width="stretch")
with scenario_col:
    st.subheader("What this run does")
    st.write(
        f"The simulator injects **{attack_count}** "
        f"synthetic {ATTACK_LABELS[attack_type].lower()} profiles targeting "
        f"**{TARGETS[target_idx]}** into a stream of **{episode_length}** profiles."
    )
    mode_details = get_comparison_mode_details(comparison_mode)
    st.write(
        f"The remaining **{genuine_count}** profiles are genuine MovieLens users. "
        f"{mode_details['description']}"
    )
    st.caption(
        "The policies see profile features, not the simulator's attack/genuine labels. "
        "Those labels are used afterward to score detection."
    )

agents = st.session_state.get("demo_agents")
loaded_target_idx = st.session_state.get("loaded_target_idx")
if not agents or loaded_target_idx != target_idx:
    with st.spinner("Preparing the active policy set for the selected target so each scenario can run without manual loading..."):
        prepare_demo_agents_for_target(target_idx, silent=True)
    agents = st.session_state.get("demo_agents")
    loaded_target_idx = st.session_state.get("loaded_target_idx")

if agents and loaded_target_idx != target_idx:
    st.caption("The policy set is being refreshed for the current target before a new simulation starts.")

run_demo = st.button(
    "Run defense simulation",
    disabled=not agents or loaded_target_idx != target_idx,
    type="primary",
    width="stretch",
)

result = st.session_state.get("last_demo_result")
last_signature = st.session_state.get("last_demo_signature")
if result and last_signature != scenario_signature:
    st.info("Scenario settings changed. Click the run button to execute a new analysis with the updated parameters.")

if run_demo:
    st.markdown("---")
    st.subheader("Live audit")
    progress = st.progress(0, text="Preparing the three isolated environments")
    step_label = st.empty()
    current_profile = st.empty()
    agent_columns = st.columns(3)
    chart_area = st.empty()
    history: List[Dict[str, float]] = []

    def show_step(step: int, step_events: Dict[str, Dict[str, Any]]) -> None:
        progress.progress(
            step / episode_length,
            text=f"Reviewing profile {step} of {episode_length}",
        )
        step_label.markdown(f"**Profile {step} / {episode_length}**")

        representative = step_events["Q-Learning"]
        is_attack = representative["is_attacker"]
        profile_kind = "attack" if is_attack else "genuine"
        profile_name = "Injected attack profile" if is_attack else "Genuine MovieLens profile"
        profile_table = ratings_table(representative["ratings"], titles)
        target_rating = next(
            (
                float(item["rating"])
                for item in representative["ratings"]
                if int(item["item_index"]) == target_idx
            ),
            None,
        )
        target_rating_text = (
            f"{target_rating:g} / 5" if target_rating is not None else "not rated"
        )
        with current_profile.container():
            st.markdown(
                f"""
                <div class="profile-label {profile_kind}">
                  <strong>{profile_name}</strong>
                  <div class="profile-meta">
                    Profile {representative['profile_index'] + 1} ·
                    {len(profile_table)} ratings ·
                    target rating: {target_rating_text}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                profile_table.head(8),
                hide_index=True,
                width="stretch",
                height=260,
            )

        for column, name in zip(agent_columns, AGENT_ORDER):
            event = step_events[name]
            column.markdown(f"**{name}**")
            column.metric(
                "Decision",
                event["action_name"],
                f"weight {event['defense_weight']:.2f}",
            )
            column.caption(
                f"Reward {event['reward']:+.2f} · target score "
                f"{event['target_prediction_before']:.3f} → "
                f"{event['target_prediction_after']:.3f}"
            )

        history.append(
            {
                "Profile": step,
                **{
                    name: float(step_events[name]["target_prediction_after"])
                    for name in AGENT_ORDER
                },
            }
        )
        if len(history) > 1:
            chart_area.line_chart(
                pd.DataFrame(history).set_index("Profile"),
                height=220,
                y_label="Predicted target rating",
            )
        else:
            chart_area.caption("Prediction trace will appear after the next profile.")
        if playback_delay:
            time.sleep(playback_delay)

    try:
        result = run_shared_scenario_comparison(
            agents,
            seed=int(scenario_seed),
            target_idx=target_idx,
            attack_type=attack_type,
            episode_length=episode_length,
            genuine_ratio=1.0 - attack_share_pct / 100.0,
            comparison_mode=comparison_mode,
            progress_callback=show_step,
        )
        result["training"] = st.session_state.get("demo_manifest", manifest)
        result["ui_controls"] = {
            **scenario_signature,
            "playback_delay_seconds": playback_delay,
        }
        result_dir = os.path.join(PROJECT_ROOT, "results", "ui_runs")
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "shared_scenario_comparison.json")
        with open(result_path, "w") as result_file:
            json.dump(result, result_file, indent=2)
        st.session_state["last_demo_result"] = result
        st.session_state["last_demo_signature"] = scenario_signature
        run_record = append_run_history(result, scenario_signature)
        st.success(
            f"Audit complete. Detailed events saved to `{result_path}`. "
            f"Run #{run_record['run_id']} has been recorded in the history."
        )
    except Exception as exc:
        st.error(f"The audit run failed: {exc}")

if result:
    st.markdown("---")
    st.subheader("After-action comparison")
    result_agents = result["agents"]
    profiles = result["scenario"]["profile_stream"]
    attack_count = sum(profile["is_attacker"] for profile in profiles)
    genuine_count = len(profiles) - attack_count
    policy_actions = {
        name: [event["action"] for event in result_agents[name]["events"]]
        for name in AGENT_ORDER
    }
    same_decisions = all(
        policy_actions[name] == policy_actions[AGENT_ORDER[0]]
        for name in AGENT_ORDER[1:]
    )
    if same_decisions:
        mode_details = get_comparison_mode_details(comparison_mode)
        st.info(
            f"All three policies made the same decision sequence on this {mode_details['label'].lower()} benchmark. "
            f"{mode_details['description']}"
        )
    else:
        disagreements = sum(
            len(
                {
                    policy_actions[name][index]
                    for name in AGENT_ORDER
                }
            )
            > 1
            for index in range(len(profiles))
        )
        st.caption(
            f"Policies disagreed on {disagreements} of {len(profiles)} profile decisions."
        )

    summary_rows = []
    decision_chart_rows = []
    for name in AGENT_ORDER:
        metrics = result_agents[name]
        events = metrics["events"]
        attack_events = [event for event in events if event["is_attacker"]]
        genuine_events = [event for event in events if not event["is_attacker"]]
        attack_blocks = sum(event["action"] == 2 for event in attack_events)
        genuine_blocks = sum(event["action"] == 2 for event in genuine_events)
        attack_recall = attack_blocks / len(attack_events) if attack_events else 0.0
        false_positive_rate = (
            genuine_blocks / len(genuine_events) if genuine_events else 0.0
        )
        summary_rows.append(
            {
                "Policy": name,
                "Attack blocked": f"{attack_blocks} / {len(attack_events)}",
                "Attack recall": attack_recall,
                "Genuine blocked": f"{genuine_blocks} / {len(genuine_events)}",
                "False-positive rate": false_positive_rate,
                "Target score shift (stars)": metrics["target_prediction_shift"],
                "Target Hit@10 change (pp)": 100
                * (
                    metrics["target_hit_ratio_at_10"]
                    - metrics["clean_target_hit_ratio_at_10"]
                ),
                "Genuine RMSE change": metrics["test_rmse"]
                - metrics["clean_test_rmse"],
                "Catalog coverage change (pp)": 100
                * (metrics["catalog_coverage"] - metrics["clean_catalog_coverage"]),
                "Genuine Hit@10 change (pp)": 100
                * (
                    metrics["genuine_hit_rate_at_10"]
                    - metrics["clean_genuine_hit_rate_at_10"]
                ),
                "Total reward": metrics["total_reward"],
            }
        )
        decision_chart_rows.append(
            {
                "Policy": name,
                "Injected blocked (%)": 100 * attack_recall,
                "Genuine blocked (%)": 100 * false_positive_rate,
            }
        )

    for column, name in zip(st.columns(3), AGENT_ORDER):
        row = next(row for row in summary_rows if row["Policy"] == name)
        with column:
            st.markdown(f"#### {name}")
            st.metric("Injected profiles blocked", row["Attack blocked"])
            st.metric("Genuine blocked (false positives)", row["Genuine blocked"])
            st.metric("Total reward", f"{row['Total reward']:.2f}")

    st.markdown("#### Comparison charts")
    st.caption(
        "These charts show the same run from two angles: how strongly each policy suppresses attacks and how much it preserves genuine-user quality."
    )
    comparison = pd.DataFrame(summary_rows).set_index("Policy")

    total_reward_chart = comparison[["Total reward"]].copy()
    st.bar_chart(total_reward_chart, height=240)

    target_shift_chart = comparison[["Target score shift (stars)"]].copy()
    st.line_chart(target_shift_chart, height=240)

    if not decision_chart_rows:
        decision_chart = pd.DataFrame()
    else:
        decision_chart = pd.DataFrame(decision_chart_rows).set_index("Policy")
    if not decision_chart.empty:
        st.bar_chart(decision_chart[["Injected blocked (%)", "Genuine blocked (%)"]], height=240)

    st.markdown("#### Recommendation impact vs clean baseline")
    st.caption(
        "Lower RMSE is better, and smaller changes in genuine-user quality are preferred when attack suppression is strong."
    )
    st.dataframe(
        comparison[
            [
                "Target score shift (stars)",
                "Target Hit@10 change (pp)",
                "Genuine RMSE change",
                "Catalog coverage change (pp)",
                "Genuine Hit@10 change (pp)",
            ]
        ].style.format(
            {
                "Target score shift (stars)": "{:+.4f}",
                "Target Hit@10 change (pp)": "{:+.2f}",
                "Genuine RMSE change": "{:+.5f}",
                "Catalog coverage change (pp)": "{:+.2f}",
                "Genuine Hit@10 change (pp)": "{:+.2f}",
            }
        ),
        width="stretch",
    )
    if attack_count and all(
        event["action"] == 2
        for name in AGENT_ORDER
        for event in result_agents[name]["events"]
        if event["is_attacker"]
    ):
        st.caption(
            "The policies blocked every injected profile in this run, so the "
            "recommender's genuine-user accuracy is expected to match its clean baseline."
        )

    profile_history = build_profile_history(result)
    if not profile_history.empty:
        st.markdown("#### Profile-by-profile target prediction trace")
        st.line_chart(
            profile_history.pivot_table(
                index="Step",
                columns="Policy",
                values="Target after",
                aggfunc="mean",
            ),
            y_label="Target prediction after decision",
            height=260,
        )
        st.dataframe(
            profile_history.sort_values(["Step", "Policy"], ignore_index=True),
            width="stretch",
            height=320,
        )

    run_history = st.session_state.get("run_history", [])
    if run_history:
        run_history_df = pd.DataFrame(
            [
                {
                    "Run": item.get("run_id", idx + 1),
                    "Scenario seed": item["scenario"]["scenario_seed"],
                    "Target": TARGETS.get(item["scenario"]["target_idx"], item["scenario"]["target_idx"]),
                    "Attack type": item["scenario"]["attack_type"],
                    "Attack share %": item["scenario"]["attack_share_pct"],
                    "Episode length": item["scenario"]["episode_length"],
                    "Comparison mode": item["scenario"].get("comparison_mode", "shared_scenario"),
                    "Q reward": item["summary"]["Q-Learning"]["Total reward"],
                    "DQN reward": item["summary"]["DQN"]["Total reward"],
                    "PPO reward": item["summary"]["PPO"]["Total reward"],
                    "Q target shift": item["summary"]["Q-Learning"]["Target score shift"],
                    "DQN target shift": item["summary"]["DQN"]["Target score shift"],
                    "PPO target shift": item["summary"]["PPO"]["Target score shift"],
                }
                for idx, item in enumerate(run_history)
            ]
        )
        st.markdown("#### Run history")
        st.caption("Run history records only successful benchmark executions started by the Run defense simulation button.")
        st.dataframe(run_history_df, width="stretch", height=260)

    with st.expander("Inspect profile decisions and ratings"):
        selected_policy = st.selectbox("Policy", AGENT_ORDER, key="event_policy")
        events = result_agents[selected_policy]["events"]
        st.caption(
            "Simulation labels are ground truth for evaluation only; the policies "
            "do not receive them as input."
        )
        st.dataframe(
            event_log_table(events),
            hide_index=True,
            width="stretch",
            height=320,
        )
        selected_step = st.selectbox(
            "Profile to inspect",
            options=list(range(len(events))),
            format_func=lambda index: (
                f"Step {events[index]['step']} · "
                f"{'injected attack' if events[index]['is_attacker'] else 'genuine user'} · "
                f"{events[index]['action_name']}"
            ),
            key="event_step",
        )
        selected_event = events[selected_step]
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("**Profile ratings**")
            st.dataframe(
                ratings_table(selected_event["ratings"], titles),
                hide_index=True,
                width="stretch",
                height=280,
            )
        with right:
            st.markdown("**Policy signal and decision**")
            st.json(
                {
                    "features_seen_by_policy": selected_event["features"],
                    "decision": selected_event["action_name"],
                    "trust_weight": selected_event["defense_weight"],
                    "reward": selected_event["reward"],
                    "target_prediction_before": selected_event[
                        "target_prediction_before"
                    ],
                    "target_prediction_after": selected_event[
                        "target_prediction_after"
                    ],
                },
                expanded=True,
            )

st.divider()
st.caption(
    "Research simulation, not a production moderation service. The Real-user "
    "reference is for inspection; the audit independently samples genuine MovieLens "
    "profiles and generated attacks."
)
