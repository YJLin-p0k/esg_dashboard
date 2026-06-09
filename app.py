from __future__ import annotations

import hashlib
from io import StringIO
from typing import cast

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.core.config import *  # noqa: F403
from esg_dashboard.core.taxonomy import get_peer_group
from esg_dashboard.data.processing import build_issue_summary, build_results, load_training_peer_rows, normalize_task_columns
from esg_dashboard.core.scoring import (
    calculate_greenwashing_risk,
)
from esg_dashboard.ui.components import inject_responsive_styles, render_section_heading


RISK_LEVEL_ORDER = ["High", "Medium", "Low", "Neutral"]
RISK_LEVEL_LABELS = {
    "High": "高風險",
    "Medium": "中度風險",
    "Low": "低風險",
    "Neutral": "無明顯風險",
}
RISK_LEVEL_LABEL_ORDER = [RISK_LEVEL_LABELS[level] for level in RISK_LEVEL_ORDER]
HIGH_RISK_LEVELS = ["High"]


if get_script_run_ctx() is None:
    print("This is a Streamlit app. Start it with: python -m streamlit run app.py")
    raise SystemExit(0)


# Streamlit setup
st.set_page_config(page_title="ESG Sentinal 綠色哨兵", page_icon="📊", layout="wide")
inject_responsive_styles()


def is_dark_theme() -> bool:
    theme_base = st.get_option("theme.base")
    if theme_base is None:
        return True
    return str(theme_base).lower() != "light"


def chart_theme_colors() -> dict[str, str]:
    if is_dark_theme():
        return {
            "text": "#f8fafc",
            "muted": "#e2e8f0",
            "subtle": "#cbd5e1",
            "panel": "#111827",
            "panel_alt": "#0f172a",
            "border": "#64748b",
            "cell_stroke": "#e2e8f0",
            "cell_divider": "#f8fafc",
        }
    return {
        "text": "#172033",
        "muted": "#334155",
        "subtle": "#64748b",
        "panel": "#f8fafc",
        "panel_alt": "#ffffff",
        "border": "#cbd5e1",
        "cell_stroke": "#ffffff",
        "cell_divider": "#475569",
    }





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


def ensure_greenwashing_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or {"risk_level", "risk_score", "risk_reason"}.issubset(df.columns):
        return df

    enriched = df.copy()
    risk_rows = enriched.apply(calculate_greenwashing_risk, axis=1).apply(pd.Series)
    for column in ["risk_level", "risk_score", "risk_reason"]:
        enriched[column] = risk_rows[column]
    return enriched


def localize_risk_level(value: object) -> str:
    return RISK_LEVEL_LABELS.get(str(value), str(value))


def style_high_risk_rows(row: pd.Series) -> list[str]:
    risk_level = row.get("risk_level", row.get("風險等級", ""))
    if str(risk_level) in HIGH_RISK_LEVELS or str(risk_level) == RISK_LEVEL_LABELS["High"]:
        return ["background-color: rgba(216, 74, 58, 0.13); font-weight: 700;"] * len(row)
    return [""] * len(row)


def build_risk_result_table(rows: pd.DataFrame) -> pd.DataFrame:
    column_labels = {
        "topic": "主題",
        "sentence": "原文句子",
        "promise_status": "承諾狀態",
        "evidence_status": "證據狀態",
        "evidence_quality": "證據品質",
        "verification_timeline": "驗證時程",
        "risk_level": "風險等級",
        "risk_reason": "判定原因",
    }
    columns = [
        "topic",
        "sentence",
        "promise_status",
        "evidence_status",
        "evidence_quality",
        "verification_timeline",
        "risk_level",
        "risk_reason",
    ]
    risk_rank = {level: index for index, level in enumerate(RISK_LEVEL_ORDER)}
    display_source = rows[columns].copy()
    display_source["_risk_rank"] = display_source["risk_level"].map(risk_rank).fillna(len(risk_rank))
    display_source = display_source.sort_values(["_risk_rank", "topic", "sentence"]).drop(columns="_risk_rank").reset_index(drop=True)
    display_source["risk_level"] = display_source["risk_level"].map(localize_risk_level)
    return display_source.rename(columns=column_labels)


def build_risk_result_display(rows: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        build_risk_result_table(rows)
        .style.apply(style_high_risk_rows, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "tbody tr:hover",
                    "props": [
                        ("background-color", "rgba(59, 130, 246, 0.08)"),
                    ],
                },
            ]
        )
    )


def render_risk_statistics_and_results(rows: pd.DataFrame) -> None:
    heading_col, download_col = st.columns([1.0, 0.28])
    with heading_col:
        render_section_heading(
            "風險統計&結果",
            "彙整本報告所有 ESG 相關句子的風險等級分布與模型判定結果；High / Medium / Low / Neutral 分別代表高風險、中度風險、低風險、無明顯風險；結果表保留主題、原文句子、承諾狀態、證據狀態、證據品質、驗證時程、風險等級與判定原因。",
            level=2,
        )
    with download_col:
        st.markdown('<div style="height: 1.7rem;"></div>', unsafe_allow_html=True)
        st.download_button(
            "下載風險統計&結果 CSV",
            data=to_csv_download(build_risk_result_table(rows)),
            file_name="esg_risk_statistics_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    risk_counts = rows["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0)
    high_count = int(rows["risk_level"].isin(HIGH_RISK_LEVELS).sum())
    metric_cols = st.columns(2)
    metric_cols[0].metric("高風險句數", f"{high_count}")
    metric_cols[1].metric("分析總句數", f"{len(rows)}")

    overview_cols = st.columns([0.62, 1.38])
    with overview_cols[0]:
        st.dataframe(
            pd.DataFrame({"風險等級": [localize_risk_level(level) for level in risk_counts.index], "句數": risk_counts.values}),
            use_container_width=True,
            hide_index=True,
        )
    with overview_cols[1]:
        available_levels = [level for level in RISK_LEVEL_ORDER if level in set(rows["risk_level"])]
        selected_levels = st.multiselect(
            "依風險等級篩選",
            available_levels,
            default=available_levels,
            format_func=localize_risk_level,
        )
        filtered_rows = rows[rows["risk_level"].isin(selected_levels)]
        st.caption(f"目前顯示 `{len(filtered_rows)}` 句。")
        st.dataframe(
            build_risk_result_display(filtered_rows),
            use_container_width=True,
            hide_index=True,
        )


def build_risk_share_rows(
    rows: pd.DataFrame,
    comparison_group: str,
    category_column: str | None = None,
    average_by_company: bool = False,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["comparison_group", "category", "risk_level", "share"])

    source = ensure_greenwashing_risk_columns(rows).copy()
    if category_column:
        source["category"] = source[category_column].astype(str)
        categories = [category for category in PEER_CATEGORY_LABELS if category in set(source["category"])]
    else:
        source["category"] = "整體"
        categories = ["整體"]

    if average_by_company and "company" in source.columns:
        group_columns = ["company", "category"]
        counts = source.groupby(group_columns + ["risk_level"]).size().reset_index(name="count")
        company_category_grid = source[group_columns].drop_duplicates().reset_index(drop=True)
        company_category_grid["row_id"] = range(len(company_category_grid))
        risk_grid = pd.MultiIndex.from_product(
            [company_category_grid["row_id"], RISK_LEVEL_ORDER],
            names=["row_id", "risk_level"],
        ).to_frame(index=False)
        full_counts = risk_grid.merge(company_category_grid, on="row_id", how="left").drop(columns="row_id")
        full_counts = full_counts.merge(counts, on=group_columns + ["risk_level"], how="left")
        full_counts["count"] = full_counts["count"].fillna(0)
        totals = full_counts.groupby(group_columns)["count"].sum().reset_index(name="total")
        shares = full_counts.merge(totals, on=group_columns, how="left")
        shares["share"] = shares["count"] / shares["total"].where(shares["total"].ne(0), 1)
        result = shares.groupby(["category", "risk_level"])["share"].mean().reset_index()
    else:
        counts = source.groupby(["category", "risk_level"]).size().reset_index(name="count")
        totals = counts.groupby("category")["count"].sum().reset_index(name="total")
        result = counts.merge(totals, on="category", how="left")
        result["share"] = result["count"] / result["total"].where(result["total"].ne(0), 1)
        result = result[["category", "risk_level", "share"]]

    base_grid = pd.MultiIndex.from_product(
        [categories, RISK_LEVEL_ORDER],
        names=["category", "risk_level"],
    ).to_frame(index=False)
    result = base_grid.merge(result, on=["category", "risk_level"], how="left")
    result["share"] = result["share"].fillna(0.0)
    share_totals = result.groupby("category")["share"].transform("sum")
    result["share"] = result["share"] / share_totals.where(share_totals.ne(0), 1)
    result["comparison_group"] = comparison_group
    result["category_label"] = result["category"].map(ESG_TOPIC_GROUP_LABELS).fillna(result["category"])
    risk_order = {level: index for index, level in enumerate(RISK_LEVEL_ORDER)}
    result["risk_order"] = result["risk_level"].map(risk_order)
    return result[["comparison_group", "category", "category_label", "risk_level", "risk_order", "share"]]


def build_peer_risk_comparison_data(report_rows: pd.DataFrame, peer_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_rows = ensure_greenwashing_risk_columns(report_rows)
    peer_rows = ensure_greenwashing_risk_columns(peer_rows)
    overall = pd.concat(
        [
            build_risk_share_rows(report_rows, "本報告"),
            build_risk_share_rows(peer_rows, "同業平均", average_by_company=True),
        ],
        ignore_index=True,
    )
    by_category = pd.concat(
        [
            build_risk_share_rows(report_rows, "本報告", category_column="esg_category"),
            build_risk_share_rows(peer_rows, "同業平均", category_column="esg_category", average_by_company=True),
        ],
        ignore_index=True,
    )
    return overall, by_category


def risk_share_value(data: pd.DataFrame, comparison_group: str, risk_level: str, category: str = "整體") -> float:
    matched = data[
        data["comparison_group"].eq(comparison_group)
        & data["risk_level"].eq(risk_level)
        & data["category"].eq(category)
    ]
    return float(matched["share"].iloc[0]) if not matched.empty else 0.0


def render_peer_delta_summary(overall_data: pd.DataFrame) -> None:
    summary_cols = st.columns(3)
    for col, risk_level, label in zip(
        summary_cols,
        ["High", "Medium", "Low"],
        ["高風險差異", "中風險差異", "低風險差異"],
    ):
        report_share = risk_share_value(overall_data, "本報告", risk_level)
        peer_share = risk_share_value(overall_data, "同業平均", risk_level)
        delta = report_share - peer_share
        direction = "高" if delta > 0 else "低" if delta < 0 else "相同"
        delta_color = "#d84a3a" if delta > 0 and risk_level == "High" else "#22c55e" if delta < 0 and risk_level == "High" else "var(--text-color, CanvasText)"
        delta_text = "相比同業相同" if delta == 0 else f"相比同業{direction} {abs(delta):.0%}"
        col.markdown(
            f"""
            <div style="padding: 0.25rem 0 0.55rem 0;">
              <div style="font-size: 1.16rem; font-weight: 800; color: var(--text-color, CanvasText);">{label}</div>
              <div style="font-size: 2.45rem; line-height: 1.15; margin-top: 0.35rem; color: var(--text-color, CanvasText); font-weight: 750;">{report_share:.0%}</div>
              <div style="display: inline-block; margin-top: 0.45rem; padding: 0.28rem 0.7rem; border-radius: 999px; background: color-mix(in srgb, {delta_color} 12%, var(--secondary-background-color, Canvas) 88%); color: {delta_color}; font-size: 1.05rem; font-weight: 800;">
                {delta_text}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overall_risk_profile(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("目前沒有足夠資料可繪製同業比較。")
        return

    chart_data = data.copy()
    comparison_sort = ["本報告", "同業平均"]
    comparison_order = {label: index for index, label in enumerate(comparison_sort)}
    chart_data["comparison_order"] = chart_data["comparison_group"].map(comparison_order)
    chart_data["risk_level_label"] = chart_data["risk_level"].map(localize_risk_level)
    chart_data = chart_data.sort_values(["comparison_order", "risk_order"])
    theme_colors = chart_theme_colors()

    color_scale = alt.Scale(
        domain=RISK_LEVEL_LABEL_ORDER,
        range=[RISK_LEVEL_COLORS[level] for level in RISK_LEVEL_ORDER],
    )

    def make_profile_bar(comparison_group: str, show_x_axis: bool) -> alt.Chart:
        axis = alt.Axis(format="%", labelFontSize=18, titleFontSize=20) if show_x_axis else None
        legend = (
            alt.Legend(orient="bottom", labelFontSize=18, titleFontSize=19, symbolSize=210)
            if show_x_axis
            else None
        )
        return (
            alt.Chart(chart_data[chart_data["comparison_group"].eq(comparison_group)])
            .mark_bar(size=32)
            .encode(
                x=alt.X(
                    "share:Q",
                    title="風險等級分布" if show_x_axis else None,
                    stack="zero",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=axis,
                ),
                y=alt.Y(
                    "comparison_group:N",
                    title=None,
                    sort=[comparison_group],
                    axis=alt.Axis(labelFontSize=22, labelFontWeight="bold", labelPadding=14),
                ),
                color=alt.Color(
                    "risk_level_label:N",
                    title="風險等級",
                    scale=color_scale,
                    legend=legend,
                ),
                order=alt.Order("risk_order:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip("comparison_group:N", title="比較對象"),
                    alt.Tooltip("risk_level_label:N", title="風險等級"),
                    alt.Tooltip("share:Q", title="比例", format=".0%"),
                ],
            )
            .properties(height=74)
        )

    chart = (
        alt.vconcat(
            make_profile_bar("本報告", show_x_axis=False),
            make_profile_bar("同業平均", show_x_axis=True),
            spacing=8,
        )
        .properties(padding={"bottom": 70, "left": 12, "right": 12, "top": 14})
        .resolve_scale(x="shared", color="shared")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=theme_colors["text"],
            titleColor=theme_colors["muted"],
            domainColor=theme_colors["border"],
            tickColor=theme_colors["border"],
        )
        .configure_legend(
            labelColor=theme_colors["text"],
            titleColor=theme_colors["muted"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def render_category_risk_profile(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("目前沒有足夠資料可繪製 E / S / G 分組比較。")
        return

    chart_data = data.copy()
    comparison_sort = ["本報告", "同業平均"]
    comparison_order = {label: index for index, label in enumerate(comparison_sort)}
    chart_data["comparison_order"] = chart_data["comparison_group"].map(comparison_order)
    chart_data["risk_level_label"] = chart_data["risk_level"].map(localize_risk_level)
    chart_data = chart_data.sort_values(["category", "comparison_order", "risk_order"])
    theme_colors = chart_theme_colors()

    available_categories = [category for category in PEER_CATEGORY_LABELS if category in set(chart_data["category"])]
    if not available_categories:
        available_categories = chart_data["category"].drop_duplicates().tolist()

    category_width = 96
    comparison_width = 116
    bar_width = 560
    group_height = 92
    color_scale = alt.Scale(
        domain=RISK_LEVEL_LABEL_ORDER,
        range=[RISK_LEVEL_COLORS[level] for level in RISK_LEVEL_ORDER],
    )

    def make_category_group(category: str, show_x_axis: bool, show_legend: bool) -> alt.HConcatChart:
        group_data = chart_data[chart_data["category"].eq(category)].copy()
        category_label = str(group_data["category_label"].iloc[0]) if not group_data.empty else ESG_TOPIC_GROUP_LABELS.get(category, category)
        comparison_table = pd.DataFrame({"comparison_group": comparison_sort})

        category_cell = (
            alt.Chart(pd.DataFrame({"category_label": [category_label]}))
            .mark_rect(fill=theme_colors["panel"], stroke=theme_colors["border"], strokeWidth=1.2)
            .properties(width=category_width, height=group_height)
        )
        category_text = (
            alt.Chart(pd.DataFrame({"category_label": [category_label]}))
            .mark_text(align="center", baseline="middle", fontSize=19, fontWeight="bold", color=theme_colors["text"])
            .encode(x=alt.value(category_width / 2), y=alt.value(group_height / 2), text="category_label:N")
            .properties(width=category_width, height=group_height)
        )
        comparison_cells = (
            alt.Chart(comparison_table)
            .mark_rect(fill=theme_colors["panel_alt"], stroke=theme_colors["border"], strokeWidth=1.2)
            .encode(
                y=alt.Y("comparison_group:N", title=None, sort=comparison_sort, axis=None),
                x=alt.value(0),
                x2=alt.value(comparison_width),
            )
            .properties(width=comparison_width, height=group_height)
        )
        comparison_text = (
            alt.Chart(comparison_table)
            .mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color=theme_colors["muted"])
            .encode(
                x=alt.value(comparison_width / 2),
                y=alt.Y("comparison_group:N", title=None, sort=comparison_sort, axis=None),
                text="comparison_group:N",
            )
            .properties(width=comparison_width, height=group_height)
        )
        bars = (
            alt.Chart(group_data)
            .mark_bar(size=28)
            .encode(
                x=alt.X(
                    "share:Q",
                    title="風險等級分布" if show_x_axis else None,
                    stack="zero",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", labelFontSize=18, titleFontSize=20) if show_x_axis else None,
                ),
                y=alt.Y("comparison_group:N", title=None, sort=comparison_sort, axis=None),
                color=alt.Color(
                    "risk_level_label:N",
                    title="風險等級",
                    scale=color_scale,
                    legend=alt.Legend(orient="bottom", labelFontSize=18, titleFontSize=19, symbolSize=210) if show_legend else None,
                ),
                order=alt.Order("risk_order:Q", sort="ascending"),
                tooltip=[
                    alt.Tooltip("comparison_group:N", title="比較對象"),
                    alt.Tooltip("category_label:N", title="類別"),
                    alt.Tooltip("risk_level_label:N", title="風險等級"),
                    alt.Tooltip("share:Q", title="比例", format=".0%"),
                ],
            )
            .properties(width=bar_width, height=group_height)
        )

        return alt.hconcat(
            category_cell + category_text,
            comparison_cells + comparison_text,
            bars,
            spacing=0,
        )

    category_charts = [
        make_category_group(
            category,
            show_x_axis=index == len(available_categories) - 1,
            show_legend=index == len(available_categories) - 1,
        )
        for index, category in enumerate(available_categories)
    ]

    chart = (
        alt.vconcat(*category_charts, spacing=18)
        .properties(padding={"bottom": 58, "left": 4, "right": 4, "top": 8})
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=theme_colors["text"],
            titleColor=theme_colors["muted"],
            domainColor=theme_colors["border"],
            tickColor=theme_colors["border"],
        )
        .configure_legend(
            labelColor=theme_colors["text"],
            titleColor=theme_colors["muted"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def render_peer_risk_comparison(report_rows: pd.DataFrame, issue_rows: pd.DataFrame) -> None:
    if issue_rows.empty:
        return

    peer_rows = get_peer_rows_for_issue(issue_rows.iloc[0])
    if peer_rows.empty:
        return

    render_section_heading(
        "同業比較",
        "用風險等級比例比較本報告與同業平均，快速檢視高風險揭露是否相對集中；同業平均以每家公司先各自計算分布後再平均。",
        level=2,
    )
    peer_rows = ensure_greenwashing_risk_columns(peer_rows)
    st.caption(f"同業比較基準：{peer_rows['company'].nunique()} 家公司")

    overall_data, category_data = build_peer_risk_comparison_data(report_rows, peer_rows)
    comparison_tabs = st.tabs(["整體風險輪廓", "E / S / G 分組"])
    with comparison_tabs[0]:
        render_peer_delta_summary(overall_data)
        render_overall_risk_profile(overall_data)
    with comparison_tabs[1]:
        render_category_risk_profile(category_data)


def build_issue_summary_display(issue_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    columns = [
        "file_name",
        "esg_category",
        "topic",
        "evidence_count",
    ]
    display_source = issue_df[columns].reset_index(drop=True).copy()

    localized = localize_dataframe(display_source)
    file_column = COLUMN_LABELS["file_name"]
    esg_column = COLUMN_LABELS["esg_category"]

    repeated_file = localized[file_column].eq(localized[file_column].shift())
    repeated_esg = repeated_file & localized[esg_column].eq(localized[esg_column].shift())
    localized.loc[repeated_file, file_column] = ""
    localized.loc[repeated_esg, esg_column] = ""

    return localized.style.format(
        {
            COLUMN_LABELS["evidence_count"]: "{:.0f}",
        }
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
        return RISK_LEVEL_COLORS["High"], "高優先"
    if status == "需追蹤":
        return RISK_LEVEL_COLORS["Medium"], "有一定風險"
    return RISK_LEVEL_COLORS["Low"], "完成"


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


# Issue detail UI
def render_ai_analysis(issue: pd.Series) -> None:
    trust = float(issue["overall_trust_score"])
    promise_rate = float(issue.get("promise_rate", 0.0))
    clear_evidence_rate = float(issue.get("clear_evidence_rate", 0.0))
    evidence_count = int(issue.get("evidence_count", 0) or 0)

    if trust < TRUST_LOW_THRESHOLD:
        verdict = "綜合判斷為高優先複核。此議題雖被辨識出相關揭露，但承諾、證據品質或時程訊號之間的一致性不足。"
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
        f"清楚證據比例為 `{clear_evidence_rate:.1f}%`。"
    )
    st.write(action)


# Topic selector and task distribution
RISK_LEVEL_COLORS = {
    "High": "#d84a3a",
    "Medium": "#d9952f",
    "Low": "#3f9b63",
    "Neutral": "#94a3b8",
    "無資料": "#f1f5f9",
}


PROMISE_STATUS_ORDER = ["Yes", "No"]
PROMISE_STATUS_LABELS = {
    "Yes": "有承諾",
    "No": "無明確承諾",
}


EVIDENCE_STATUS_ORDER = ["Yes", "No", "N/A"]
EVIDENCE_STATUS_LABELS = {
    "Yes": "有證據",
    "No": "無證據",
    "N/A": "未標示",
}
EVIDENCE_STATUS_GROUPS = {
    "Yes": "Yes",
    "No": "No",
    "N/A": "N/A",
}


EVIDENCE_QUALITY_ORDER = ["high", "medium", "low", "unknown"]
EVIDENCE_QUALITY_LABELS = {
    "high": "高清晰度",
    "medium": "中清晰度",
    "low": "低清晰度",
    "unknown": "未標示",
}
EVIDENCE_QUALITY_GROUPS = {
    "Specific": "high",
    "Clear": "high",
    "Vague": "medium",
    "Not Clear": "low",
    "Misleading": "low",
    "N/A": "unknown",
}


def risk_matrix_row_specs() -> list[tuple[str, str]]:
    return [("Yes", quality) for quality in EVIDENCE_QUALITY_ORDER] + [("No", "unknown"), ("N/A", "unknown")]


def build_issue_risk_matrix_data(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    if evidence_rows.empty:
        return pd.DataFrame()

    rows = evidence_rows.copy().reset_index(drop=True)
    rows["promise_status"] = rows["promise_status"].where(rows["promise_status"].isin(PROMISE_STATUS_ORDER), "No")
    rows["evidence_status"] = rows["evidence_status"].map(EVIDENCE_STATUS_GROUPS).fillna("N/A")
    rows["evidence_quality"] = rows["evidence_quality"].map(EVIDENCE_QUALITY_GROUPS).fillna("unknown")
    rows.loc[rows["evidence_status"].isin(["No", "N/A"]), "evidence_quality"] = "unknown"
    rows["risk_level"] = rows["risk_level"].where(rows["risk_level"].isin(RISK_LEVEL_ORDER), "Neutral")

    grouped = rows.groupby(
        ["promise_status", "evidence_status", "evidence_quality", "risk_level"]
    ).size().reset_index(name="count")
    if grouped.empty:
        return pd.DataFrame()

    base_grid = pd.DataFrame(
        [
            {
                "promise_status": promise_status,
                "evidence_status": evidence_status,
                "evidence_quality": evidence_quality,
            }
            for promise_status in PROMISE_STATUS_ORDER
            for evidence_status, evidence_quality in risk_matrix_row_specs()
        ]
    )

    total_counts = (
        grouped.groupby(["promise_status", "evidence_status", "evidence_quality"], as_index=False)
        .agg(total_count=("count", "sum"))
    )
    matrix = base_grid.merge(total_counts, on=["promise_status", "evidence_status", "evidence_quality"], how="left")
    matrix["total_count"] = matrix["total_count"].fillna(0).astype(int)

    risk_rank = {level: index for index, level in enumerate(RISK_LEVEL_ORDER)}
    dominant = grouped.copy()
    dominant["risk_rank"] = dominant["risk_level"].map(risk_rank).fillna(len(risk_rank))
    dominant = dominant.sort_values(
        ["promise_status", "evidence_status", "evidence_quality", "count", "risk_rank"],
        ascending=[True, True, True, False, True],
    ).drop_duplicates(["promise_status", "evidence_status", "evidence_quality"])

    matrix = matrix.merge(
        dominant[["promise_status", "evidence_status", "evidence_quality", "risk_level"]],
        on=["promise_status", "evidence_status", "evidence_quality"],
        how="left",
    )
    matrix["risk_level"] = matrix["risk_level"].fillna("Neutral")
    matrix["display_risk_level"] = matrix["risk_level"]
    matrix.loc[matrix["total_count"].eq(0), "display_risk_level"] = "無資料"
    matrix["risk_level_label"] = matrix["display_risk_level"].map(
        lambda value: "無資料" if str(value) == "無資料" else localize_risk_level(value)
    )
    matrix["display_risk_level_label"] = matrix["display_risk_level"].map(
        lambda value: "無資料" if str(value) == "無資料" else localize_risk_level(value)
    )
    matrix["promise_label"] = matrix["promise_status"].map(PROMISE_STATUS_LABELS)
    matrix["evidence_status_label"] = matrix["evidence_status"].map(EVIDENCE_STATUS_LABELS)
    matrix["evidence_label"] = matrix["evidence_quality"].map(EVIDENCE_QUALITY_LABELS)
    matrix["evidence_combined_label"] = matrix["evidence_status_label"] + " / " + matrix["evidence_label"]
    matrix["cell_label"] = matrix["total_count"].map(lambda value: f"{value} 句" if value else "")
    row_order = {
        f"{EVIDENCE_STATUS_LABELS[status]} / {EVIDENCE_QUALITY_LABELS[quality]}": order
        for order, (status, quality) in enumerate(risk_matrix_row_specs())
    }
    matrix["row_order"] = matrix["evidence_combined_label"].map(row_order)

    risk_breakdown = grouped.pivot_table(
        index=["promise_status", "evidence_status", "evidence_quality"],
        columns="risk_level",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for level in RISK_LEVEL_ORDER:
        if level not in risk_breakdown.columns:
            risk_breakdown[level] = 0
    matrix = matrix.merge(risk_breakdown, on=["promise_status", "evidence_status", "evidence_quality"], how="left")
    for level in RISK_LEVEL_ORDER:
        matrix[level] = matrix[level].fillna(0).astype(int)
    matrix["risk_breakdown"] = matrix.apply(
        lambda row: " / ".join(
            f"{localize_risk_level(level)}: {int(row[level])}" for level in RISK_LEVEL_ORDER if int(row[level]) > 0
        )
        or "無句子",
        axis=1,
    )
    return matrix


def render_issue_risk_matrix(evidence_rows: pd.DataFrame) -> None:
    matrix_df = build_issue_risk_matrix_data(evidence_rows)
    if matrix_df.empty:
        st.info("此議題沒有足夠資料可繪製風險矩陣。")
        return

    x_sort = [PROMISE_STATUS_LABELS[key] for key in PROMISE_STATUS_ORDER]
    y_sort = [
        f"{EVIDENCE_STATUS_LABELS[status]} / {EVIDENCE_QUALITY_LABELS[quality]}"
        for status, quality in risk_matrix_row_specs()
    ]
    matrix_height = 288
    row_count = len(risk_matrix_row_specs())
    theme_colors = chart_theme_colors()
    status_header_df = pd.DataFrame(
        [
            {
                "evidence_status_label": EVIDENCE_STATUS_LABELS[status],
                "row_start": 0 if status == "Yes" else 4 if status == "No" else 5,
                "row_end": 4 if status == "Yes" else 5 if status == "No" else 6,
                "row_mid": 2.0 if status == "Yes" else 4.5 if status == "No" else 5.5,
            }
            for status in EVIDENCE_STATUS_ORDER
        ]
    )
    quality_header_df = pd.DataFrame(
        [
            {
                "evidence_label": EVIDENCE_QUALITY_LABELS[quality],
                "row_start": index,
                "row_end": index + 1,
                "row_mid": index + 0.5,
            }
            for index, (_, quality) in enumerate(risk_matrix_row_specs())
        ]
    )
    header_y_scale = alt.Scale(domain=[0, row_count], reverse=True)

    base_chart = alt.Chart(matrix_df).encode(
        x=alt.X(
            "promise_label:N",
            title="承諾狀態",
            sort=x_sort,
            axis=alt.Axis(labelAngle=0, labelLimit=180, labelPadding=10, labelFontSize=21, titleFontSize=23),
        ),
        y=alt.Y(
            "evidence_combined_label:N",
            title=None,
            sort=y_sort,
            axis=alt.Axis(labels=False, ticks=False, domain=False),
        ),
        tooltip=[
            alt.Tooltip("promise_label:N", title="承諾狀態"),
            alt.Tooltip("evidence_status_label:N", title="證據狀態"),
            alt.Tooltip("evidence_label:N", title="證據品質"),
            alt.Tooltip("risk_level_label:N", title="主要風險等級"),
            alt.Tooltip("total_count:Q", title="句數", format=".0f"),
            alt.Tooltip("risk_breakdown:N", title="風險分布"),
        ],
    )

    heatmap = (
        base_chart
        .mark_rect(stroke=theme_colors["cell_stroke"], strokeWidth=1.6)
        .encode(
            color=alt.Color(
                "display_risk_level_label:N",
                title="主要風險等級",
                scale=alt.Scale(
                    domain=RISK_LEVEL_LABEL_ORDER + ["無資料"],
                    range=[
                        theme_colors["panel_alt"] if key == "無資料" else RISK_LEVEL_COLORS[key]
                        for key in RISK_LEVEL_ORDER + ["無資料"]
                    ],
                ),
            ),
        )
    )
    heatmap_cell_dividers = (
        base_chart
        .mark_rect(fillOpacity=0, stroke=theme_colors["cell_divider"], strokeWidth=2.4)
    )
    labels = (
        base_chart
        .transform_filter("datum.total_count > 0")
        .mark_text(color=theme_colors["text"], fontSize=24, fontWeight="bold")
        .encode(
            text="cell_label:N",
        )
    )

    status_header = (
        alt.Chart(status_header_df)
        .mark_rect(fill=theme_colors["panel"], stroke=theme_colors["border"], strokeWidth=1.2)
        .encode(
            y=alt.Y("row_start:Q", scale=header_y_scale, axis=None),
            y2="row_end:Q",
        )
        .properties(width=88, height=matrix_height)
    )
    status_text = (
        alt.Chart(status_header_df)
        .mark_text(align="center", baseline="middle", fontSize=19, fontWeight="bold", color=theme_colors["text"])
        .encode(
            x=alt.value(44),
            y=alt.Y("row_mid:Q", scale=header_y_scale, axis=None),
            text="evidence_status_label:N",
        )
        .properties(width=88, height=matrix_height)
    )
    status_title = (
        alt.Chart(pd.DataFrame({"label": ["證據狀態"]}))
        .mark_rect(fill=theme_colors["panel"], stroke=theme_colors["border"], strokeWidth=1.2)
        .properties(width=88, height=32)
    )
    status_title_text = (
        alt.Chart(pd.DataFrame({"label": ["證據狀態"]}))
        .mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color=theme_colors["muted"])
        .encode(x=alt.value(44), y=alt.value(16), text="label:N")
        .properties(width=88, height=32)
    )
    quality_header = (
        alt.Chart(quality_header_df)
        .mark_rect(fill=theme_colors["panel_alt"], stroke=theme_colors["border"], strokeWidth=1.2)
        .encode(
            y=alt.Y("row_start:Q", scale=header_y_scale, axis=None),
            y2="row_end:Q",
        )
        .properties(width=126, height=matrix_height)
    )
    quality_text = (
        alt.Chart(quality_header_df)
        .mark_text(align="left", baseline="middle", fontSize=19, color=theme_colors["muted"])
        .encode(
            x=alt.value(12),
            y=alt.Y("row_mid:Q", scale=header_y_scale, axis=None),
            text="evidence_label:N",
        )
        .properties(width=126, height=matrix_height)
    )
    quality_title = (
        alt.Chart(pd.DataFrame({"label": ["證據品質"]}))
        .mark_rect(fill=theme_colors["panel"], stroke=theme_colors["border"], strokeWidth=1.2)
        .properties(width=126, height=32)
    )
    quality_title_text = (
        alt.Chart(pd.DataFrame({"label": ["證據品質"]}))
        .mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color=theme_colors["muted"])
        .encode(x=alt.value(63), y=alt.value(16), text="label:N")
        .properties(width=126, height=32)
    )
    heatmap_panel = (heatmap + heatmap_cell_dividers + labels).properties(
        width=360,
        height=matrix_height,
    )
    heatmap_title = (
        alt.Chart(pd.DataFrame({"label": ["承諾狀態"]}))
        .mark_text(align="center", baseline="middle", fontSize=18, fontWeight="bold", color=theme_colors["subtle"])
        .encode(x=alt.value(180), y=alt.value(16), text="label:N")
        .properties(width=360, height=32)
    )
    chart = alt.hconcat(
        alt.vconcat(status_title + status_title_text, status_header + status_text, spacing=0),
        alt.vconcat(quality_title + quality_title_text, quality_header + quality_text, spacing=0),
        alt.vconcat(heatmap_title, heatmap_panel, spacing=0),
        spacing=0,
    ).resolve_scale(
        y="independent"
    ).configure_view(
        stroke=None
    ).configure_axis(
        labelFontSize=20,
        titleFontSize=22,
        labelColor=theme_colors["text"],
        titleColor=theme_colors["muted"],
        domainColor=theme_colors["border"],
        tickColor=theme_colors["border"],
        grid=False,
    ).configure_legend(
        labelFontSize=18,
        titleFontSize=19,
        symbolSize=210,
        labelColor=theme_colors["text"],
        titleColor=theme_colors["muted"],
    )
    st.altair_chart(chart, use_container_width=True)


def render_topic_selector(issue_df: pd.DataFrame) -> tuple[str, str] | None:
    available_topics = {
        (str(row["esg_category"]), str(row["topic"]))
        for _, row in issue_df.iterrows()
    }
    if not available_topics:
        return None

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
                    button_label = f"✓ {topic}" if is_selected else topic
                    if st.button(
                        button_label,
                        key=f"topic_selector_{category}_{topic}",
                        disabled=not exists,
                        use_container_width=True,
                    ):
                        st.session_state["selected_esg_topic"] = (category, topic)
                        st.rerun()

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

    risk_matrix_col, assessment_col = st.columns([1.08, 0.92])
    with risk_matrix_col:
        st.markdown("**風險矩陣**")
        st.caption("以承諾狀態與「證據狀態 / 證據品質」交叉統計此議題的句子分布；顏色代表該格子的主要風險等級，文字代表句數。")
        render_issue_risk_matrix(evidence_rows)

    with assessment_col:
        st.markdown("**綜合評估**")
        render_ai_analysis(issue)

    task_chart_tab, audit_timeline_tab = st.tabs(
        [
            "品質監督",
            "稽核與時程",
        ]
    )

    with task_chart_tab:
        st.caption("用四個分布圖檢視此議題相關句子的承諾、證據狀態、證據品質與驗證時程。")
        render_task_pie_charts(evidence_rows)

    with audit_timeline_tab:
        st.caption("依據模型判定結果自動整理需複核、補件或追蹤的稽核待辦。")
        render_audit_actions(issue, evidence_rows)


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
        st.info("請先上傳一份或多份 PDF，系統會擷取 ESG 相關句子並評估承諾、證據與風險等級。")
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

result_df = normalize_task_columns(result_df)
result_df = ensure_greenwashing_risk_columns(result_df)
st.session_state["analysis_results"] = result_df

required_columns = {
    "overall_trust_score",
    "risk_level",
    "risk_score",
    "risk_reason",
    "esg_category",
    "topic",
    "sentence",
    "sentence_id",
}
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
    "議題摘要",
    "列出此報告命中的 ESG 議題、所屬類別與相關句數。",
    level=2,
)
st.dataframe(
    build_issue_summary_display(issue_df),
    use_container_width=True,
    hide_index=True,
)

render_risk_statistics_and_results(result_df)

render_peer_risk_comparison(result_df, issue_df)

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
    ].sort_values("file_name", ascending=True)

    topic_title = f"{ESG_TOPIC_GROUP_LABELS.get(selected_category, selected_category)} / {selected_topic_name}"
    topic_description = TOPIC_DESCRIPTIONS.get(selected_topic_name, "")
    if topic_description:
        topic_title = f"{topic_title} | {topic_description}"
    st.markdown(f'<div class="esg-topic-title">{topic_title}</div>', unsafe_allow_html=True)

    for index, issue in selected_issues.reset_index(drop=True).iterrows():
        if len(selected_issues) > 1:
            with st.expander(
                f"{issue['file_name']}",
                expanded=index == 0,
            ):
                render_issue_detail(issue, result_df)
        else:
            render_issue_detail(issue, result_df)
