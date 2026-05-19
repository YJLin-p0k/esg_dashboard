from __future__ import annotations

from io import BytesIO, StringIO

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.hybrid_model import ESG_CATEGORIES, HybridESGAnalyzer
from esg_dashboard.pdf_utils import extract_pdf_text
from esg_dashboard.text_utils import split_chinese_sentences


TOPIC_KEYWORDS = {
    "Environment": {
        "碳排與減碳": ["carbon", "emission", "emissions", "ghg", "net zero", "碳", "排放", "淨零", "溫室氣體"],
        "能源轉型": ["energy", "renewable", "solar", "wind", "再生能源", "能源", "太陽能", "風力"],
        "水資源與廢棄物": ["water", "waste", "recycle", "水資源", "廢棄物", "回收"],
        "氣候與環境風險": ["climate", "biodiversity", "pollution", "氣候", "生物多樣性", "污染"],
    },
    "Social": {
        "員工健康與安全": ["health", "safety", "職安", "安全", "健康"],
        "人權與勞動條件": ["human rights", "labor", "人權", "勞工"],
        "多元與共融": ["diversity", "inclusion", "多元", "共融"],
        "供應鏈責任": ["supplier", "supply chain", "供應商", "供應鏈"],
        "社區影響": ["community", "社區", "公益"],
    },
    "Governance": {
        "董事會與公司治理": ["board", "director", "董事", "治理"],
        "倫理法遵與反貪腐": ["ethics", "compliance", "anti-corruption", "倫理", "法遵", "反貪腐"],
        "稽核與風險管理": ["audit", "risk management", "稽核", "風險管理"],
        "資料隱私與資安": ["privacy", "security", "資安", "隱私"],
    },
}

CATEGORY_LABELS = {
    "Environment": "環境",
    "Social": "社會",
    "Governance": "治理",
    "Other": "其他",
}

VALUE_LABELS = {
    "Yes": "有",
    "No": "無",
    "N/A": "不適用",
    "Clear": "清楚",
    "Not Clear": "不清楚",
    "Misleading": "可能誤導",
    "already": "已完成或可驗證",
    "within_2_years": "2 年內",
    "between_2_and_5_years": "2 到 5 年",
    "longer_than_5_years": "超過 5 年",
}

COLUMN_LABELS = {
    "file_name": "檔案",
    "sentence": "原文句子",
    "esg_category": "ESG 類別",
    "topic": "主題",
    "overall_trust_score": "信任分數",
    "greenwashing_risk": "漂綠風險",
    "avg_confidence": "最低模型信心",
    "confidence": "模型信心",
    "evidence_count": "相關句數",
    "sentence_id": "句子編號",
    "promise_rate": "承諾比例",
    "clear_evidence_rate": "清楚證據比例",
    "representative_sentence": "代表句",
    "risk_reason": "風險原因",
    "promise_status": "是否有承諾",
    "verification_timeline": "驗證時程",
    "evidence_status": "是否有證據",
    "evidence_quality": "證據品質",
    "signal": "風險訊號",
    "score": "分數",
    "benchmark": "比較對象",
    "trust_score": "信任分數",
    "status": "狀態",
    "audit_item": "待辦事項",
    "owner": "負責單位",
    "timeline": "時程",
    "count": "句數",
}


if get_script_run_ctx() is None:
    print("This is a Streamlit app. Start it with: python -m streamlit run app.py")
    raise SystemExit(0)


st.set_page_config(page_title="ESG 信任儀表板", page_icon="📊", layout="wide")


@st.cache_resource
def load_analyzer() -> HybridESGAnalyzer:
    return HybridESGAnalyzer()


def build_results(uploaded_files) -> pd.DataFrame:
    analyzer = load_analyzer()
    rows: list[dict[str, object]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        text = extract_pdf_text(BytesIO(file_bytes))
        sentences = split_chinese_sentences(text)
        predictions = analyzer.predict(sentences)

        for sentence_id, (sentence, prediction) in enumerate(zip(sentences, predictions), start=1):
            if prediction.esg_category not in ESG_CATEGORIES or prediction.esg_category == "Other":
                continue

            rows.append(
                {
                    "file_name": uploaded_file.name,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "esg_category": prediction.esg_category,
                    "topic": detect_topic(sentence, prediction.esg_category),
                    "overall_trust_score": prediction.overall_trust_score,
                    "greenwashing_risk": prediction.greenwashing_risk,
                    "confidence": prediction.confidence,
                    "promise_status": prediction.promise_status,
                    "verification_timeline": prediction.verification_timeline,
                    "evidence_status": prediction.evidence_status,
                    "evidence_quality": prediction.evidence_quality,
                    "risk_reason": prediction.risk_reason,
                }
            )

    return pd.DataFrame(rows)


def detect_topic(sentence: str, category: str) -> str:
    topic_scores: dict[str, int] = {}
    lowered = sentence.lower()
    for topic, keywords in TOPIC_KEYWORDS.get(category, {}).items():
        topic_scores[topic] = sum(1 for keyword in keywords if keyword.lower() in lowered)

    best_topic, best_score = max(topic_scores.items(), key=lambda item: item[1], default=(category, 0))
    return best_topic if best_score > 0 else f"{CATEGORY_LABELS.get(category, category)}一般議題"


def localize_value(value: object) -> object:
    return VALUE_LABELS.get(value, CATEGORY_LABELS.get(value, value))


def localize_risk_reason(reason: object) -> object:
    if not isinstance(reason, str):
        return reason

    translated = reason
    replacements = {
        "commitment without strong supporting evidence": "有承諾，但缺少有力證據支持",
        "evidence quality is Not Clear": "證據品質不清楚",
        "evidence quality is Misleading": "證據可能誤導",
        "verification horizon is longer than five years": "驗證時程超過五年",
        "evidence and timeline are reasonably aligned": "證據與時程大致一致",
    }
    for source, target in replacements.items():
        translated = translated.replace(source, target)
    return translated.replace("; ", "；")


def localize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    for column in localized.columns:
        if column == "risk_reason":
            localized[column] = localized[column].map(localize_risk_reason)
        elif column in {"esg_category", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"}:
            localized[column] = localized[column].map(localize_value)
    return localized.rename(columns=COLUMN_LABELS)


def first_by_lowest_trust(values: pd.Series) -> object:
    return values.iloc[0] if not values.empty else ""


def build_issue_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = result_df.sort_values(
        ["file_name", "esg_category", "topic", "overall_trust_score", "greenwashing_risk"],
        ascending=[True, True, True, True, False],
    )

    return (
        sorted_df.groupby(["file_name", "esg_category", "topic"], as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "min"),
            greenwashing_risk=("greenwashing_risk", "max"),
            avg_confidence=("confidence", "min"),
            evidence_count=("sentence", "count"),
            promise_rate=("promise_status", lambda values: round((values == "Yes").mean() * 100, 1)),
            clear_evidence_rate=("evidence_quality", lambda values: round((values == "Clear").mean() * 100, 1)),
            representative_sentence=("sentence", "first"),
            risk_reason=("risk_reason", first_by_lowest_trust),
        )
        .sort_values(["overall_trust_score", "greenwashing_risk"], ascending=[True, False])
    )


def build_radar_data(issue: pd.Series) -> pd.DataFrame:
    trust_risk = 100 - float(issue["overall_trust_score"])
    evidence_gap = 100 - float(issue["clear_evidence_rate"])
    promise_pressure = float(issue["promise_rate"])
    confidence_gap = max(0, 100 - float(issue["avg_confidence"]) * 100)

    return pd.DataFrame(
        [
            {"signal": "信任缺口", "score": round(trust_risk, 2)},
            {"signal": "證據缺口", "score": round(evidence_gap, 2)},
            {"signal": "承諾壓力", "score": round(promise_pressure, 2)},
            {"signal": "模型不確定性", "score": round(confidence_gap, 2)},
        ]
    )


def build_peer_benchmark(issue: pd.Series) -> pd.DataFrame:
    trust = float(issue["overall_trust_score"])
    category = str(issue["esg_category"])
    median = {"Environment": 72, "Social": 76, "Governance": 79}.get(category, 74)
    leader = min(95, median + 13)

    return pd.DataFrame(
        [
            {"benchmark": "目前議題", "trust_score": trust},
            {"benchmark": "同業中位數", "trust_score": median},
            {"benchmark": "同業領先者", "trust_score": leader},
        ]
    )


def build_audit_feed(issue: pd.Series, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    items: list[dict[str, str]] = []
    low_quality = evidence_rows[evidence_rows["evidence_quality"].isin(["Not Clear", "Misleading"])]
    no_evidence = evidence_rows[evidence_rows["evidence_status"].eq("No")]
    long_timeline = evidence_rows[evidence_rows["verification_timeline"].eq("longer_than_5_years")]

    if not low_quality.empty:
        items.append({"status": "需複核", "audit_item": "確認品質不足或可能誤導的證據。", "owner": "ESG 稽核"})
    if not no_evidence.empty:
        items.append({"status": "需補件", "audit_item": "請公司補充量化證據與來源文件。", "owner": "揭露團隊"})
    if not long_timeline.empty:
        items.append({"status": "需追蹤", "audit_item": "用中期里程碑追蹤長期承諾。", "owner": "永續專案辦公室"})
    if not items:
        items.append({"status": "通過", "audit_item": "證據、時程與承諾訊號一致。", "owner": "ESG 稽核"})

    return pd.DataFrame(items)


def build_milestone_timeline(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    counts = evidence_rows["verification_timeline"].value_counts().to_dict()
    labels = [
        ("already", "已完成或可驗證"),
        ("within_2_years", "近期檢查點"),
        ("between_2_and_5_years", "中期里程碑"),
        ("longer_than_5_years", "長期承諾"),
        ("N/A", "未說明時程"),
    ]
    return pd.DataFrame(
        [{"timeline": label, "count": int(counts.get(key, 0))} for key, label in labels]
    )


def render_ai_analysis(issue: pd.Series) -> None:
    trust = float(issue["overall_trust_score"])
    risk = float(issue["greenwashing_risk"])

    if trust < 55:
        verdict = "高優先複核：這個議題的證據訊號偏弱，且漂綠風險偏高。"
    elif trust < 72:
        verdict = "建議追蹤：揭露內容有部分可用訊號，但證據或時程仍不夠有說服力。"
    else:
        verdict = "短期疑慮較低：目前揭露內容與該 ESG 議題大致一致。"

    st.write(verdict)
    st.write(f"漂綠風險為 `{risk:.1f}`，此議題群組中最低信任分數為 `{trust:.1f}`。")
    st.write(f"主要原因：{localize_risk_reason(issue['risk_reason'])}")
    st.caption(str(issue["representative_sentence"]))


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    localize_dataframe(df).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


st.title("ESG 混合信任儀表板")
st.caption("上傳 ESG 或永續報告 PDF，系統會先判斷句子的 E/S/G 語意，再細分議題並彙總評估。")

uploaded_files = st.file_uploader(
    "上傳 ESG / 永續報告 PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("請上傳一份或多份 PDF 報告，系統會分類 ESG 議題並評估承諾、證據、時程與信任訊號。")
    st.stop()

with st.spinner("正在擷取 PDF 文字並分析 ESG 訊號..."):
    result_df = build_results(uploaded_files)

required_columns = {"overall_trust_score", "esg_category", "topic", "sentence", "sentence_id"}
if result_df.empty or not required_columns.issubset(result_df.columns):
    st.warning("未偵測到 E / S / G 相關句子。請確認 PDF 可選取文字，或檢查 OCR 品質。")
    st.stop()

issue_df = build_issue_summary(result_df)

metric_cols = st.columns(4)
metric_cols[0].metric("議題數", f"{len(issue_df):,}")
metric_cols[1].metric("相關句數", f"{len(result_df):,}")
metric_cols[2].metric("最低信任分數", f"{issue_df['overall_trust_score'].min():.1f}")
metric_cols[3].metric("最高漂綠風險", f"{issue_df['greenwashing_risk'].max():.1f}")

st.download_button(
    "下載議題摘要 CSV",
    data=to_csv_download(issue_df),
    file_name="esg_hybrid_trust_results.csv",
    mime="text/csv",
)

st.subheader("議題摘要")
st.dataframe(
    localize_dataframe(issue_df[
        [
            "file_name",
            "esg_category",
            "topic",
            "overall_trust_score",
            "greenwashing_risk",
            "promise_rate",
            "clear_evidence_rate",
            "evidence_count",
            "representative_sentence",
        ]
    ]),
    use_container_width=True,
    hide_index=True,
)

for index, issue in issue_df.reset_index(drop=True).iterrows():
    title = (
        f"{index + 1}. [{localize_value(issue['esg_category'])}] {issue['topic']} "
        f"- 信任分數 {issue['overall_trust_score']:.1f}"
    )
    evidence_rows = result_df[
        (result_df["file_name"] == issue["file_name"])
        & (result_df["esg_category"] == issue["esg_category"])
        & (result_df["topic"] == issue["topic"])
    ].sort_values("overall_trust_score", ascending=True)

    with st.expander(title, expanded=index == 0):
        metric_row = st.columns(4)
        metric_row[0].metric("保守信任分數", f"{issue['overall_trust_score']:.1f}")
        metric_row[1].metric("最高漂綠風險", f"{issue['greenwashing_risk']:.1f}")
        metric_row[2].metric("承諾比例", f"{issue['promise_rate']:.1f}%")
        metric_row[3].metric("清楚證據比例", f"{issue['clear_evidence_rate']:.1f}%")

        radar_tab, peer_tab, audit_tab, ai_tab, timeline_tab, related_tab = st.tabs(
            [
                "漂綠雷達",
                "同業比較",
                "稽核待辦",
                "AI 分析",
                "里程碑時程",
                "相關文句",
            ]
        )

        with radar_tab:
            radar_chart = (
                alt.Chart(build_radar_data(issue))
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="分數"),
                    y=alt.Y("signal:N", sort="-x", title=None),
                    color=alt.Color("score:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
                    tooltip=[
                        alt.Tooltip("signal:N", title="風險訊號"),
                        alt.Tooltip("score:Q", title="分數"),
                    ],
                )
                .properties(height=240)
            )
            st.altair_chart(radar_chart, use_container_width=True)

        with peer_tab:
            peer_chart = (
                alt.Chart(build_peer_benchmark(issue))
                .mark_bar()
                .encode(
                    x=alt.X("benchmark:N", title="比較對象"),
                    y=alt.Y("trust_score:Q", title="信任分數", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("benchmark:N", title="比較對象"),
                        alt.Tooltip("trust_score:Q", title="信任分數"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(peer_chart, use_container_width=True)

        with audit_tab:
            st.dataframe(localize_dataframe(build_audit_feed(issue, evidence_rows)), use_container_width=True, hide_index=True)

        with ai_tab:
            render_ai_analysis(issue)

        with timeline_tab:
            st.dataframe(localize_dataframe(build_milestone_timeline(evidence_rows)), use_container_width=True, hide_index=True)

        with related_tab:
            st.dataframe(
                localize_dataframe(evidence_rows[
                    [
                        "sentence_id",
                        "overall_trust_score",
                        "greenwashing_risk",
                        "promise_status",
                        "verification_timeline",
                        "evidence_status",
                        "evidence_quality",
                        "risk_reason",
                        "sentence",
                    ]
                ]),
                use_container_width=True,
                hide_index=True,
            )
