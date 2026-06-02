from __future__ import annotations

import hashlib
from io import StringIO
import math
import re
from typing import Literal, cast

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.core.config import *  # noqa: F403
from esg_dashboard.core.taxonomy import get_peer_group
from esg_dashboard.data.processing import build_issue_summary, build_results, load_training_peer_rows
from esg_dashboard.core.scoring import calculate_esg_trust_scores, calculate_overall_trust_score, format_score_metric, peer_score_comment, severity_from_trust
from esg_dashboard.ui.components import inject_responsive_styles, render_section_heading


if get_script_run_ctx() is None:
    print("This is a Streamlit app. Start it with: python -m streamlit run app.py")
    raise SystemExit(0)


# Streamlit setup
st.set_page_config(page_title="ESG Sentinal 綠色哨兵", page_icon="📊", layout="wide")
inject_responsive_styles()





# Formatting helpers
def localize_value(value: object) -> object:
    value_key = str(value)
    return VALUE_LABELS.get(value_key, CATEGORY_LABELS.get(value_key, value))


def localize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    for column in localized.columns:
        if column in {"esg_category", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"}:
            localized[column] = localized[column].map(localize_value)
    return localized.rename(columns=COLUMN_LABELS)



def build_issue_summary_display(issue_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    columns = [
        "file_name",
        "esg_category",
        "topic",
        "overall_trust_score",
        "evidence_count",
    ]
    display_source = issue_df[columns].reset_index(drop=True).copy()
    file_min_trust = display_source.groupby("file_name")["overall_trust_score"].transform("min")
    is_file_min = display_source["overall_trust_score"].eq(file_min_trust)

    localized = localize_dataframe(display_source)
    file_column = COLUMN_LABELS["file_name"]
    esg_column = COLUMN_LABELS["esg_category"]
    trust_column = COLUMN_LABELS["overall_trust_score"]

    repeated_file = localized[file_column].eq(localized[file_column].shift())
    repeated_esg = repeated_file & localized[esg_column].eq(localized[esg_column].shift())
    localized.loc[repeated_file, file_column] = ""
    localized.loc[repeated_esg, esg_column] = ""

    def highlight_file_min(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        row_position = row.name if isinstance(row.name, int) else 0
        if bool(is_file_min.iloc[row_position]):
            styles[list(row.index).index(trust_column)] = f"color: {TRUST_COLOR_LOW}; font-weight: 800;"
        return styles

    return localized.style.apply(highlight_file_min, axis=1).format(
        {
            COLUMN_LABELS["overall_trust_score"]: "{:.1f}",
            COLUMN_LABELS["evidence_count"]: "{:.0f}",
        }
    )




# Trust score display

def render_trust_gauge(score: float) -> None:
    severity, color = severity_from_trust(score)
    angle = 180 + max(0, min(100, score)) * 1.8
    red_end = TRUST_LOW_THRESHOLD * 1.8
    yellow_end = TRUST_STABLE_THRESHOLD * 1.8
    blend = TRUST_GAUGE_BLEND_DEGREES
    red_solid_end = max(0, red_end - blend)
    yellow_start = min(180, red_end + blend)
    yellow_solid_end = max(yellow_start, yellow_end - blend)
    green_start = min(180, yellow_end + blend)
    st.markdown(
        f"""
        <div style="max-width: 460px; margin: 0 0 0.75rem 0;">
          <div style="
              position: relative;
              width: 100%;
              aspect-ratio: 2 / 1;
              overflow: visible;
            ">
            <div style="
                position: absolute;
                inset: 0;
                overflow: hidden;
                border-radius: 460px 460px 0 0;
              ">
              <div style="
                width: 100%;
                height: 100%;
                border-radius: 460px 460px 0 0;
              background: conic-gradient(from 270deg at 50% 100%,
                {TRUST_COLOR_LOW} 0deg {red_solid_end:.1f}deg,
                #df6b40 {red_end - blend * 0.45:.1f}deg,
                #eba945 {red_end + blend * 0.45:.1f}deg,
                {TRUST_COLOR_MEDIUM} {yellow_start:.1f}deg {yellow_solid_end:.1f}deg,
                #cfba4d {yellow_end - blend * 0.45:.1f}deg,
                #86b35a {yellow_end + blend * 0.45:.1f}deg,
                {TRUST_COLOR_HIGH} {green_start:.1f}deg 180deg,
                transparent 180deg 360deg);
              -webkit-mask: radial-gradient(ellipse at 50% 100%, transparent 0 58%, #000 58.5% 100%);
              mask: radial-gradient(ellipse at 50% 100%, transparent 0 58%, #000 58.5% 100%);
              box-shadow: inset 0 0 0 1px rgba(20, 31, 43, 0.12);
              "></div>
            </div>
            <div style="
                position: absolute;
                left: 50%;
                bottom: 0;
                width: 40%;
                height: 10px;
                background: {color};
                transform-origin: 0% 50%;
                transform: rotate({angle:.1f}deg);
                clip-path: polygon(0 50%, 84% 12%, 100% 50%, 84% 88%);
                filter: drop-shadow(0 2px 5px rgba(20, 31, 43, 0.55));
                z-index: 5;
              "></div>
            <div style="
                position: absolute;
                left: calc(50% - 12px);
                bottom: -12px;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                background: {color};
                box-shadow: 0 0 0 5px color-mix(in srgb, Canvas 82%, transparent), 0 2px 6px rgba(20, 31, 43, 0.35);
                z-index: 6;
              "></div>
            <div style="position: absolute; left: 0; bottom: -1.15rem; color: inherit; font-size: 0.95rem; font-weight: 700;">0</div>
            <div style="position: absolute; left: 50%; top: -1.25rem; transform: translateX(-50%); color: inherit; font-size: 0.95rem; font-weight: 700;">50</div>
            <div style="position: absolute; right: 0; bottom: -1.15rem; color: inherit; font-size: 0.95rem; font-weight: 700;">100</div>
          </div>
          <div style="text-align: center; margin-top: 1.35rem;">
            <div style="font-size: 2rem; font-weight: 700; line-height: 1;">信任分數: {score:.1f}</div>
            <div style="color: {color}; font-weight: 700;">{severity}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Peer comparison
def get_peer_rows_for_issue(issue: pd.Series) -> pd.DataFrame:
    company = str(issue.get("company", "unknown")).lower()
    training_rows = load_training_peer_rows()
    peer_group = get_peer_group(company)
    if peer_group:
        peer_rows = training_rows[training_rows["company"].isin(peer_group) & ~training_rows["company"].eq(company)]
    else:
        peer_rows = training_rows

    return training_rows if peer_rows.empty else peer_rows


def display_company_name(company: str) -> str:
    aliases = COMPANY_ALIASES.get(str(company).lower(), [])
    chinese_aliases = [alias for alias in aliases if re.search(r"[\u4e00-\u9fff]", alias)]
    return f"{chinese_aliases[0]} ({company})" if chinese_aliases else str(company).upper()




def build_peer_issue_score_rows(peer_rows: pd.DataFrame) -> pd.DataFrame:
    if peer_rows.empty:
        return pd.DataFrame(columns=["company", "esg_category", "topic", "overall_trust_score", "evidence_count"])

    return (
        peer_rows.groupby(["company", "esg_category", "topic"], as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "mean"),
            evidence_count=("overall_trust_score", "count"),
        )
        .reset_index(drop=True)
    )


def build_peer_company_summary(peer_rows: pd.DataFrame) -> pd.DataFrame:
    peer_score_rows = build_peer_issue_score_rows(peer_rows)
    if peer_score_rows.empty:
        return pd.DataFrame()

    summary = (
        peer_score_rows.groupby("company", as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "mean"),
            evidence_count=("overall_trust_score", "count"),
        )
        .sort_values(["overall_trust_score", "company"], ascending=[False, True])
    )

    for category in PEER_CATEGORY_LABELS:
        category_scores = (
            peer_score_rows[peer_score_rows["esg_category"].eq(category)]
            .groupby("company")["overall_trust_score"]
            .mean()
        )
        summary[category] = summary["company"].map(category_scores)

    return summary.reset_index(drop=True)


def render_peer_company_panel(peer_rows: pd.DataFrame) -> None:
    peer_summary = build_peer_company_summary(peer_rows)
    render_section_heading(
        "比較同業名單",
        "列出同業公司的平均信任分數；展開公司名稱可查看 E、S、G 分項分數與簡短評價。",
        level=4,
    )

    if peer_summary.empty:
        st.info("目前沒有可顯示的同業資料。")
        return

    with st.container(height=460, border=True):
        for _, row in peer_summary.iterrows():
            company = str(row["company"])
            score = float(row["overall_trust_score"])
            severity, color = severity_from_trust(score)
            label = display_company_name(company)
            with st.expander(f"{label} · {score:.1f}", expanded=False):
                st.markdown(
                    f'<span style="color:{color}; font-weight:800;">{severity}</span> · 信任分數 `{score:.1f}`',
                    unsafe_allow_html=True,
                )
                st.write(peer_score_comment(score))
                score_cols = st.columns(3)
                score_cols[0].metric("E 環境", format_score_metric(row.get("Environment")))
                score_cols[1].metric("S 社會", format_score_metric(row.get("Social")))
                score_cols[2].metric("G 治理", format_score_metric(row.get("Governance")))


def build_peer_radar_data(issue: pd.Series, report_score_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    file_name = issue["file_name"]
    report_rows = report_score_rows[report_score_rows["file_name"].eq(file_name)]
    peer_rows = get_peer_rows_for_issue(issue)
    baseline_note = f"同業比較基準：{peer_rows['company'].nunique()} 家公司"
    peer_scores = calculate_esg_trust_scores(build_peer_issue_score_rows(peer_rows))
    report_scores = calculate_esg_trust_scores(report_rows)

    axes = list(PEER_CATEGORY_LABELS.items())
    rows: list[dict[str, object]] = []
    for series_name, scores in {"同業平均": peer_scores, "本報告平均": report_scores}.items():
        for order, (category, axis_label) in enumerate(axes):
            angle = (math.pi / 2) - (2 * math.pi * order / len(axes))
            value = float(scores[category] or 0)
            rows.append(
                {
                    "series": series_name,
                    "axis": axis_label,
                    "order": order,
                    "score": value,
                    "x": math.cos(angle) * value,
                    "y": math.sin(angle) * value,
                }
            )
        first = rows[-len(axes)].copy()
        first["order"] = len(axes)
        rows.append(first)

    return pd.DataFrame(rows), peer_rows, baseline_note

def build_radar_grid(axis_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score in [25, 50, 75, 100]:
        for order in range(axis_count + 1):
            angle = (math.pi / 2) - (2 * math.pi * (order % axis_count) / axis_count)
            rows.append({"score": score, "order": order, "x": math.cos(angle) * score, "y": math.sin(angle) * score})
    return rows


def build_radar_axis_labels(axes: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order, axis in enumerate(axes):
        angle = (math.pi / 2) - (2 * math.pi * order / len(axes))
        label_x = math.cos(angle) * RADAR_AXIS_LABEL_RADIUS
        label_y = math.sin(angle) * RADAR_AXIS_LABEL_RADIUS
        rows.append(
            {
                "axis": axis,
                "axis_label": axis.replace(" ", "\n", 1),
                "label_x": label_x,
                "label_y": label_y,
                "label_align": "right" if label_x < -1 else "left" if label_x > 1 else "center",
                "label_baseline": "bottom" if label_y > 1 else "top" if label_y < -1 else "middle",
            }
        )
    return rows


def render_peer_comparison(issue: pd.Series, report_score_rows: pd.DataFrame) -> None:
    peer_data, peer_rows, baseline_note = build_peer_radar_data(issue, report_score_rows)
    axes = peer_data[peer_data["order"].lt(3)]["axis"].drop_duplicates().tolist()
    chart_domain = [-160, 160]
    radar_scale = alt.Scale(domain=chart_domain, nice=False, zero=False)
    radar_axis_font_size = alt.ExprRef(
        expr=(
            "containerSize()[0] ? "
            f"max({RADAR_AXIS_LABEL_FONT_SIZE_MIN}, "
            f"min({RADAR_AXIS_LABEL_FONT_SIZE_MAX}, containerSize()[0] / 26)) "
            f": {RADAR_AXIS_LABEL_FONT_SIZE_FALLBACK}"
        )
    )
    grid_chart = (
        alt.Chart(pd.DataFrame(build_radar_grid(axis_count=len(axes))))
        .mark_line(color="#cbd5e1", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", axis=None, scale=radar_scale, sort=None),
            y=alt.Y("y:Q", axis=None, scale=radar_scale, sort=None),
            detail="score:N",
            order="order:Q",
        )
        .properties(width=RADAR_CHART_SIZE, height=RADAR_CHART_SIZE)
    )
    axis_label_rows = build_radar_axis_labels(axes)
    axis_label_data = pd.DataFrame(axis_label_rows)
    label_charts = []
    for (label_align, label_baseline), label_rows in axis_label_data.groupby(
        ["label_align", "label_baseline"]
    ):
        label_align_value = cast(Literal["left", "center", "right"], label_align)
        label_baseline_value = cast(Literal["top", "middle", "bottom"], label_baseline)
        label_charts.append(
            alt.Chart(label_rows)
            .mark_text(
                align=label_align_value,
                baseline=label_baseline_value,
                fontSize=radar_axis_font_size,
                fontWeight="bold",
                lineBreak="\n",
                lineHeight=RADAR_AXIS_LABEL_LINE_HEIGHT,
                color="currentColor",
            )
            .encode(
                x=alt.X("label_x:Q", axis=None, scale=radar_scale, sort=None),
                y=alt.Y("label_y:Q", axis=None, scale=radar_scale, sort=None),
                text="axis_label:N",
            )
            .properties(width=RADAR_CHART_SIZE, height=RADAR_CHART_SIZE)
        )
    label_chart = alt.layer(*label_charts)
    peer_line_chart = (
        alt.Chart(peer_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("x:Q", axis=None, scale=radar_scale, sort=None),
            y=alt.Y("y:Q", axis=None, scale=radar_scale, sort=None),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["同業平均", "本報告平均"],
                    range=["#2563eb", "#f97316"],
                ),
                legend=alt.Legend(title="比較對象", orient="bottom"),
            ),
            detail="series:N",
            order="order:Q",
            tooltip=[
                alt.Tooltip("series:N", title="比較對象"),
                alt.Tooltip("axis:N", title="向度"),
                alt.Tooltip("score:Q", title="分數", format=".1f"),
            ],
        )
        .properties(width=RADAR_CHART_SIZE, height=RADAR_CHART_SIZE)
    )
    chart = (
        grid_chart + peer_line_chart + label_chart
    ).properties(
        autosize={"type": "none"},
        padding=RADAR_CHART_PADDING,
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=CHART_LABEL_FONT_SIZE,
        titleFontSize=CHART_TITLE_FONT_SIZE,
    ).configure_legend(
        labelFontSize=CHART_LEGEND_FONT_SIZE,
        titleFontSize=CHART_TITLE_FONT_SIZE,
        symbolSize=CHART_LEGEND_SYMBOL_SIZE,
    )

    comparison_cols = st.columns([1.65, 0.8])
    with comparison_cols[0]:
        st.markdown(
            f"""
            <div style="color: inherit; background: transparent; padding: 0 0 0.35rem 0; margin-bottom: 0.6rem; font-size: 1rem;">
              {baseline_note}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, use_container_width=False)
    with comparison_cols[1]:
        render_peer_company_panel(peer_rows)


# Audit and issue detail data
def build_audit_feed(issue: pd.Series, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    items: list[dict[str, str]] = []
    low_quality = evidence_rows[evidence_rows["evidence_quality"].isin(["Not Clear", "Misleading"])]
    no_evidence = evidence_rows[evidence_rows["evidence_status"].eq("No")]
    long_timeline = evidence_rows[evidence_rows["verification_timeline"].eq("longer_than_5_years")]

    if not low_quality.empty:
        items.append(
            {
                "status": "需複核",
                "audit_item": "確認品質不足或可能誤導的證據。",
                "owner": "ESG 稽核",
                "basis": f"{len(low_quality)} 句證據品質偏弱",
            }
        )
    if not no_evidence.empty:
        items.append(
            {
                "status": "需補件",
                "audit_item": "請公司補充量化證據與來源文件。",
                "owner": "揭露團隊",
                "basis": f"{len(no_evidence)} 句缺少佐證",
            }
        )
    if not long_timeline.empty:
        items.append(
            {
                "status": "需追蹤",
                "audit_item": "請補充長期目標的階段性進度與近期檢查點。",
                "owner": "永續專案辦公室",
                "basis": f"{len(long_timeline)} 句屬長期時程",
            }
        )
    if not items:
        items.append(
            {
                "status": "通過",
                "audit_item": "證據、時程與承諾訊號一致。",
                "owner": "ESG 稽核",
                "basis": "未偵測到高優先待辦",
            }
        )

    return pd.DataFrame(items)


def audit_status_style(status: str) -> tuple[str, str]:
    if status in {"需複核", "需補件"}:
        return TRUST_COLOR_LOW, "高優先"
    if status == "需追蹤":
        return TRUST_COLOR_MEDIUM, "有一定風險"
    return TRUST_COLOR_HIGH, "完成"


def render_audit_actions(issue: pd.Series, evidence_rows: pd.DataFrame) -> None:
    audit_rows = build_audit_feed(issue, evidence_rows)
    if audit_rows.empty:
        st.info("此議題目前沒有可整理的稽核待辦。")
        return

    st.caption("以下為依據承諾、證據品質與驗證時程自動整理的審查建議。")
    columns = st.columns(min(3, len(audit_rows)))
    for index, (_, audit_row) in enumerate(audit_rows.iterrows()):
        status = str(audit_row["status"])
        color, priority = audit_status_style(status)
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid {color};
                    padding: 0.85rem 0.95rem;
                    margin-bottom: 0.8rem;
                    background: color-mix(in srgb, {color} 8%, Canvas 92%);
                    border-radius: 8px;
                ">
                    <div style="font-size: 0.98rem; color: {color}; font-weight: 800;">{status} · {priority}</div>
                    <div style="font-size: 1.12rem; font-weight: 700; margin-top: 0.35rem;">{audit_row["audit_item"]}</div>
                    <div style="font-size: 1rem; margin-top: 0.55rem;">判定依據：{audit_row["basis"]}</div>
                    <div style="font-size: 1rem; margin-top: 0.2rem;">建議負責：{audit_row["owner"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_related_paragraphs(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    if evidence_rows.empty:
        return pd.DataFrame()

    paragraph_rows: list[dict[str, object]] = []
    sort_columns = ["page", "paragraph_id", "sentence_id"] if "page" in evidence_rows.columns else ["paragraph_id", "sentence_id"]
    group_columns = ["file_name", "page", "paragraph_id"] if "page" in evidence_rows.columns else ["file_name", "paragraph_id"]
    grouped = evidence_rows.sort_values(sort_columns).groupby(
        group_columns,
        as_index=False,
        sort=True,
    )

    for _, paragraph_df in grouped:
        sentence_ids = paragraph_df["sentence_id"].astype(int).tolist()
        matched_sentences = " / ".join(paragraph_df["sentence"].astype(str).tolist())
        paragraph_rows.append(
            {
                "file_name": paragraph_df["file_name"].iloc[0],
                "page": int(paragraph_df["page"].iloc[0]) if "page" in paragraph_df.columns else "",
                "paragraph_id": int(paragraph_df["paragraph_id"].iloc[0]),
                "sentence_id": f"{min(sentence_ids)}-{max(sentence_ids)}" if len(sentence_ids) > 1 else str(sentence_ids[0]),
                "overall_trust_score": round(float(paragraph_df["overall_trust_score"].min()), 2),
                "promise_status": "Yes" if paragraph_df["promise_status"].eq("Yes").any() else "No",
                "verification_timeline": paragraph_df["verification_timeline"].mode().iat[0],
                "evidence_status": "Yes" if paragraph_df["evidence_status"].eq("Yes").any() else paragraph_df["evidence_status"].mode().iat[0],
                "evidence_quality": paragraph_df["evidence_quality"].mode().iat[0],
                "matched_sentences": matched_sentences,
                "paragraph_context": paragraph_df["paragraph_context"].iloc[0],
            }
        )

    return pd.DataFrame(paragraph_rows).sort_values(
        ["overall_trust_score"],
        ascending=[True],
    )


# Issue detail UI
def render_ai_analysis(issue: pd.Series) -> None:
    trust = float(issue["overall_trust_score"])
    promise_rate = float(issue.get("promise_rate", 0.0))
    clear_evidence_rate = float(issue.get("clear_evidence_rate", 0.0))
    evidence_count = int(issue.get("evidence_count", 0) or 0)

    if trust < TRUST_LOW_THRESHOLD:
        verdict = "綜合判斷為高優先複核。此議題雖被辨識出相關揭露，但承諾、證據品質或時程訊號之間的一致性不足，信任分數偏低。"
        action = "建議先回到原文確認承諾是否具體，並補查是否有第三方驗證、量化成果或明確年度目標。"
    elif trust < TRUST_STABLE_THRESHOLD:
        verdict = "綜合判斷為建議追蹤。此議題已有一定揭露基礎，但仍可能存在證據不夠清楚、時程不完整，或承諾與成果沒有完全對齊的情況。"
        action = "建議追蹤後續報告是否補充量化指標、完成進度與外部查證資訊。"
    else:
        verdict = "綜合判斷為相對穩健。此議題的承諾、佐證與時程訊號整體較一致，揭露內容具備較高可信度。"
        action = "後續可持續檢查年度進展是否符合既定目標，並留意是否維持清楚的量化揭露。"

    st.write(verdict)
    st.write(
        f"本議題共彙整 `{evidence_count}` 句相關內容；承諾比例為 `{promise_rate:.1f}%`，"
        f"清楚證據比例為 `{clear_evidence_rate:.1f}%`，整體信任分數為 `{trust:.1f}`。"
    )
    st.write(action)


# Topic selector and task distribution
def render_topic_selector(issue_df: pd.DataFrame) -> tuple[str, str] | None:
    available_topics = {
        (str(row["esg_category"]), str(row["topic"]))
        for _, row in issue_df.iterrows()
    }
    if not available_topics:
        return None

    topic_scores = (
        issue_df.groupby(["esg_category", "topic"], as_index=False)["overall_trust_score"]
        .mean()
        .set_index(["esg_category", "topic"])["overall_trust_score"]
        .to_dict()
    )
    selected = st.session_state.get("selected_esg_topic")
    if selected not in available_topics:
        for category in ("Environment", "Social", "Governance"):
            for topic in TOPIC_KEYWORDS.get(category, {}).keys():
                if (category, topic) in available_topics:
                    selected = (category, topic)
                    st.session_state["selected_esg_topic"] = selected
                    break
            if selected in available_topics:
                break

    st.markdown(
        """
        <style>
        div[data-testid="stPopoverBody"] button:disabled {
            opacity: 0.34;
            cursor: not-allowed;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    selector_cols = st.columns(3)
    for col, (category, topics) in zip(selector_cols, TOPIC_KEYWORDS.items()):
        category_available = sum((category, topic) in available_topics for topic in topics)
        label = f"{ESG_TOPIC_GROUP_LABELS.get(category, category)} ({category_available}/{len(topics)})"
        with col:
            with st.popover(label, use_container_width=True):
                for topic in topics:
                    exists = (category, topic) in available_topics
                    is_selected = selected == (category, topic)
                    score = topic_scores.get((category, topic))
                    button_label = f"✓ {topic}" if is_selected else topic
                    row_cols = st.columns([18, 1])
                    with row_cols[0]:
                        if st.button(
                            button_label,
                            key=f"topic_selector_{category}_{topic}",
                            disabled=not exists,
                            use_container_width=True,
                        ):
                            st.session_state["selected_esg_topic"] = (category, topic)
                            st.rerun()
                    with row_cols[1]:
                        if exists and score is not None:
                            _, dot_color = severity_from_trust(float(score))
                            st.markdown(
                                f'<div title="信任分數 {float(score):.1f}" style="text-align:right; padding-top:0.48rem;">'
                                f'<span style="display:inline-block; width:0.62rem; height:0.62rem; border-radius:50%; background:{dot_color};"></span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown("&nbsp;", unsafe_allow_html=True)

    return st.session_state.get("selected_esg_topic")


def localize_task_pie_value(column: str, value: object) -> object:
    column_labels = TASK_PIE_LABELS.get(column, {})
    return column_labels.get(str(value), localize_value(value))


def build_task_pie_data(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    if rows.empty or column not in rows:
        return pd.DataFrame(columns=["label", "count", "order"])

    counts = (
        rows[column]
        .fillna("N/A")
        .replace("", "N/A")
        .map(lambda value: localize_task_pie_value(column, value))
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    order = TASK_PIE_ORDER.get(column, counts["label"].tolist())
    order_map = {label: index for index, label in enumerate(order)}
    counts["order"] = counts["label"].map(order_map).fillna(len(order)).astype(int)
    return counts.sort_values(["order", "label"]).reset_index(drop=True)


def task_pie_domain(labels: list[str], column: str) -> list[str]:
    order = TASK_PIE_ORDER.get(column, labels)
    ordered_labels = [label for label in order if label in labels]
    ordered_labels.extend(label for label in labels if label not in ordered_labels)
    return ordered_labels


def render_task_pie_charts(evidence_rows: pd.DataFrame) -> None:
    for row_start in range(0, len(TASK_PIE_CHARTS), 2):
        chart_cols = st.columns(2)
        for chart_col, (column, title, help_text) in zip(chart_cols, TASK_PIE_CHARTS[row_start : row_start + 2]):
            pie_data = build_task_pie_data(evidence_rows, column)
            color_domain = task_pie_domain(pie_data["label"].tolist(), column)
            color_range = [TASK_PIE_COLORS.get(label, "#64748b") for label in color_domain]
            with chart_col:
                st.markdown(f"**{title}**")
                st.caption(help_text)
                if pie_data.empty:
                    st.info("無資料")
                    continue

                chart = (
                    alt.Chart(pie_data)
                    .mark_arc(innerRadius=54, outerRadius=100)
                    .encode(
                        theta=alt.Theta("count:Q", stack=True),
                        order=alt.Order("order:Q"),
                        color=alt.Color(
                            "label:N",
                            title="分類",
                            scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(
                                orient="bottom",
                                columns=2,
                                labelLimit=260,
                                symbolSize=TASK_PIE_LEGEND_SYMBOL_SIZE,
                                titleFontSize=TASK_PIE_LEGEND_FONT_SIZE,
                                labelFontSize=TASK_PIE_LEGEND_FONT_SIZE,
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("label:N", title="分類"),
                            alt.Tooltip("count:Q", title="句數", format=".0f"),
                        ],
                    )
                    .properties(height=430)
                    .configure_view(stroke=None)
                )
                st.altair_chart(chart, use_container_width=True)


def render_issue_detail(issue: pd.Series, result_df: pd.DataFrame) -> None:
    evidence_rows = result_df[
        (result_df["file_name"] == issue["file_name"])
        & (result_df["esg_category"] == issue["esg_category"])
        & (result_df["topic"] == issue["topic"])
    ].sort_values("overall_trust_score", ascending=True)

    overview_cols = st.columns([1.05, 1.35])
    with overview_cols[0]:
        render_trust_gauge(float(issue["overall_trust_score"]))
    with overview_cols[1]:
        st.markdown("**綜合評估**")
        render_ai_analysis(issue)

    task_chart_tab, audit_timeline_tab, related_tab = st.tabs(
        [
            "四項判定",
            "稽核與時程",
            "相關文句",
        ]
    )

    with task_chart_tab:
        st.caption("用四個分布圖檢視此議題相關句子的承諾、驗證時程、證據狀態與證據品質。")
        render_task_pie_charts(evidence_rows)

    with audit_timeline_tab:
        st.caption("依據低信任訊號自動整理需複核、補件或追蹤的稽核待辦。")
        render_audit_actions(issue, evidence_rows)

    with related_tab:
        render_section_heading(
            "相關文句",
            "列出模型命中的原文段落與判定欄位，方便回到報告書脈絡複核；表格可透過表格工具下載。",
            level=4,
        )
        related_paragraphs = build_related_paragraphs(evidence_rows)
        st.dataframe(
            localize_dataframe(
                related_paragraphs[
                    [
                        "page",
                        "overall_trust_score",
                        "promise_status",
                        "verification_timeline",
                        "evidence_status",
                        "evidence_quality",
                        "matched_sentences",
                        "paragraph_context",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


# Export and upload helpers
def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    localize_dataframe(df).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def uploaded_file_signature(uploaded_files) -> tuple[tuple[str, int, str], ...]:
    signature: list[tuple[str, int, str]] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()[:16]
        signature.append((uploaded_file.name, len(file_bytes), digest))
    return tuple(signature)


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def render_upload_confirmation(uploaded_files, show_file_table: bool = True) -> tuple[tuple[str, int, str], ...]:
    signature = uploaded_file_signature(uploaded_files)
    total_size = format_file_size(sum(size for _, size, _ in signature))

    st.caption(f"已選擇 `{len(uploaded_files)}` 份 PDF，總大小 `{total_size}`。")
    if show_file_table:
        render_uploaded_file_table(signature)

    confirmed_signature = st.session_state.get("confirmed_upload_signature")
    if confirmed_signature != signature:
        if st.button("確認並開始分析", type="primary"):
            st.session_state["confirmed_upload_signature"] = signature
            st.rerun()
        st.caption("請確認上傳檔案無誤後，再開始分析。")
        st.stop()

    st.success("PDF 已確認，開始進行分析。")
    return signature


def render_uploaded_file_table(signature: tuple[tuple[str, int, str], ...]) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {"檔案": file_name, "大小": format_file_size(size)}
                for file_name, size, _ in signature
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# App entrypoint
st.title("ESG Sentinal 承諾驗證")
st.caption("上傳 ESG 或永續報告 PDF，系統會先判斷句子的 E/S/G 語意，再細分議題並彙總評估。")

upload_col, file_table_col, report_col = st.columns([1.25, 2.2, 1.25])
with upload_col:
    uploaded_files = st.file_uploader(
        "上傳 ESG / 永續報告 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.session_state.pop("confirmed_upload_signature", None)
    with upload_col:
        st.info("請先上傳一份或多份 PDF，系統會擷取 ESG 相關句子並評估承諾、證據與信任分數。")
    st.stop()

with upload_col:
    upload_signature = render_upload_confirmation(uploaded_files, show_file_table=False)

with file_table_col:
    render_uploaded_file_table(upload_signature)

with st.spinner("正在擷取 PDF 文字並分析 ESG 訊號..."):
    # Cache analysis results in session state by upload signature so
    # switching files or changing the topic selector won't re-run analysis.
    cached_sig = st.session_state.get("analysis_signature")
    if cached_sig != upload_signature or "analysis_results" not in st.session_state:
        result_df = build_results(uploaded_files)
        st.session_state["analysis_results"] = result_df
        st.session_state["analysis_signature"] = upload_signature
    else:
        result_df = st.session_state["analysis_results"]

required_columns = {"overall_trust_score", "esg_category", "topic", "sentence", "sentence_id"}
if result_df.empty or not required_columns.issubset(result_df.columns):
    st.warning("未偵測到 E / S / G 相關句子。請確認 PDF 可選取文字，或檢查 OCR 品質。")
    st.stop()

all_issue_df = build_issue_summary(result_df)
available_files = all_issue_df["file_name"].drop_duplicates().tolist()

if len(available_files) > 1:
    with report_col:
        selected_file_name = st.selectbox(
            "選擇要分析的報告書",
            available_files,
            index=0,
        )
else:
    selected_file_name = available_files[0]
    with report_col:
        st.caption(f"目前分析報告書：{selected_file_name}")

selected_file_name_text = str(selected_file_name)
result_df = result_df[result_df["file_name"].eq(selected_file_name_text)].copy()
issue_df = all_issue_df[all_issue_df["file_name"].eq(selected_file_name_text)].copy()

st.markdown("<hr>", unsafe_allow_html=True)

render_section_heading(
    "信任分數總覽",
    "快速查看目前報告的整體信任分數，以及 E、S、G 三大面向的平均信任分數；分數由承諾明確度、證據狀態、證據品質與驗證時程加權計算；低於 35：高優先複核；35 到 70：建議追蹤；70 以上：相對穩健。",
    level=2,
)
metric_cols = st.columns(4)
report_esg_scores = calculate_esg_trust_scores(issue_df)
metric_cols[0].metric("整體信任分數", format_score_metric(calculate_overall_trust_score(issue_df)))
metric_cols[1].metric("E 信任分數", format_score_metric(report_esg_scores["Environment"]))
metric_cols[2].metric("S 信任分數", format_score_metric(report_esg_scores["Social"]))
metric_cols[3].metric("G 信任分數", format_score_metric(report_esg_scores["Governance"]))

summary_heading_col, summary_download_col = st.columns([1.0, 0.22])
with summary_heading_col:
    render_section_heading(
        "議題摘要",
        "列出此報告命中的 ESG 議題、所屬類別、平均信任分數與相關句數；表中紅色信任分數代表該報告整體最低分，須優先注意；此表格可下載為 CSV。",
        level=2,
    )
with summary_download_col:
    st.markdown('<div style="height: 1.7rem;"></div>', unsafe_allow_html=True)
    st.download_button(
        "下載議題摘要 CSV",
        data=to_csv_download(issue_df),
        file_name="esg_hybrid_trust_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
st.dataframe(
    build_issue_summary_display(issue_df),
    use_container_width=True,
    hide_index=True,
)

render_section_heading(
    "同業比較",
    "比較本報告與同業平均在環境、社會、治理三個面向的信任分數差異。",
    level=2,
)
render_peer_comparison(issue_df.iloc[0], issue_df)

st.markdown("<hr>", unsafe_allow_html=True)

render_section_heading(
    "10 項 ESG 議題",
    "從固定 ESG 議題分類中選擇要深入檢視的主題，系統只顯示此報告實際命中的議題。",
    level=2,
)
selected_topic = render_topic_selector(issue_df)

if selected_topic is None:
    st.info("此報告書沒有命中 10 項 ESG 議題。")
else:
    selected_category, selected_topic_name = cast(tuple[str, str], selected_topic)
    selected_issues = issue_df[
        issue_df["esg_category"].eq(selected_category)
        & issue_df["topic"].eq(selected_topic_name)
    ].sort_values(["overall_trust_score", "file_name"], ascending=[True, True])

    topic_title = f"{ESG_TOPIC_GROUP_LABELS.get(selected_category, selected_category)} / {selected_topic_name}"
    topic_description = TOPIC_DESCRIPTIONS.get(selected_topic_name, "")
    if topic_description:
        topic_title = f"{topic_title} | {topic_description}"
    st.markdown(f'<div class="esg-topic-title">{topic_title}</div>', unsafe_allow_html=True)

    for index, issue in selected_issues.reset_index(drop=True).iterrows():
        if len(selected_issues) > 1:
            with st.expander(
                f"{issue['file_name']} - 信任分數 {issue['overall_trust_score']:.1f}",
                expanded=index == 0,
            ):
                render_issue_detail(issue, result_df)
        else:
            render_issue_detail(issue, result_df)
