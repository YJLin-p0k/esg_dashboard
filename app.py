from __future__ import annotations

import base64
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
        "Carbon and emissions": ["carbon", "emission", "emissions", "ghg", "net zero", "碳", "排放", "淨零", "溫室氣體"],
        "Energy transition": ["energy", "renewable", "solar", "wind", "再生能源", "能源", "太陽能", "風力"],
        "Water and waste": ["water", "waste", "recycle", "水資源", "廢棄物", "回收"],
        "Climate risk": ["climate", "biodiversity", "pollution", "氣候", "生物多樣性", "污染"],
    },
    "Social": {
        "Employee health and safety": ["health", "safety", "職安", "安全", "健康"],
        "Human rights and labor": ["human rights", "labor", "人權", "勞工"],
        "Diversity and inclusion": ["diversity", "inclusion", "多元", "共融"],
        "Supply chain responsibility": ["supplier", "supply chain", "供應商", "供應鏈"],
        "Community impact": ["community", "社區", "公益"],
    },
    "Governance": {
        "Board governance": ["board", "director", "董事", "治理"],
        "Ethics and compliance": ["ethics", "compliance", "anti-corruption", "倫理", "法遵", "反貪腐"],
        "Audit and risk": ["audit", "risk management", "稽核", "風險管理"],
        "Data privacy and security": ["privacy", "security", "資安", "隱私"],
    },
}


if get_script_run_ctx() is None:
    print("This is a Streamlit app. Start it with: python -m streamlit run app.py")
    raise SystemExit(0)


st.set_page_config(page_title="ESG Hybrid Trust Dashboard", page_icon="📊", layout="wide")


@st.cache_resource
def load_analyzer() -> HybridESGAnalyzer:
    return HybridESGAnalyzer()


def build_results(uploaded_files) -> tuple[pd.DataFrame, dict[str, bytes]]:
    analyzer = load_analyzer()
    rows: list[dict[str, object]] = []
    pdf_bytes_by_file: dict[str, bytes] = {}

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        pdf_bytes_by_file[uploaded_file.name] = file_bytes
        text = extract_pdf_text(BytesIO(file_bytes))
        sentences = split_chinese_sentences(text)
        predictions = analyzer.predict(sentences)

        for sentence, prediction in zip(sentences, predictions):
            if prediction.esg_category not in ESG_CATEGORIES or prediction.esg_category == "Other":
                continue

            rows.append(
                {
                    "file_name": uploaded_file.name,
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

    return pd.DataFrame(rows), pdf_bytes_by_file


def detect_topic(sentence: str, category: str) -> str:
    topic_scores: dict[str, int] = {}
    lowered = sentence.lower()
    for topic, keywords in TOPIC_KEYWORDS.get(category, {}).items():
        topic_scores[topic] = sum(1 for keyword in keywords if keyword.lower() in lowered)

    best_topic, best_score = max(topic_scores.items(), key=lambda item: item[1], default=(category, 0))
    return best_topic if best_score > 0 else f"{category} general issue"


def build_issue_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    return (
        result_df.groupby(["file_name", "esg_category", "topic"], as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "min"),
            greenwashing_risk=("greenwashing_risk", "max"),
            avg_confidence=("confidence", "mean"),
            evidence_count=("sentence", "count"),
            promise_rate=("promise_status", lambda values: round((values == "Yes").mean() * 100, 1)),
            clear_evidence_rate=("evidence_quality", lambda values: round((values == "Clear").mean() * 100, 1)),
            representative_sentence=("sentence", "first"),
            risk_reason=("risk_reason", "first"),
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
            {"signal": "Trust gap", "score": round(trust_risk, 2)},
            {"signal": "Evidence gap", "score": round(evidence_gap, 2)},
            {"signal": "Promise pressure", "score": round(promise_pressure, 2)},
            {"signal": "Model uncertainty", "score": round(confidence_gap, 2)},
        ]
    )


def build_peer_benchmark(issue: pd.Series) -> pd.DataFrame:
    trust = float(issue["overall_trust_score"])
    category = str(issue["esg_category"])
    median = {"Environment": 72, "Social": 76, "Governance": 79}.get(category, 74)
    leader = min(95, median + 13)

    return pd.DataFrame(
        [
            {"benchmark": "Selected issue", "trust_score": trust},
            {"benchmark": "Peer median", "trust_score": median},
            {"benchmark": "Peer leader", "trust_score": leader},
        ]
    )


def build_audit_feed(issue: pd.Series, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    items: list[dict[str, str]] = []
    low_quality = evidence_rows[evidence_rows["evidence_quality"].isin(["Not Clear", "Misleading"])]
    no_evidence = evidence_rows[evidence_rows["evidence_status"].eq("No")]
    long_timeline = evidence_rows[evidence_rows["verification_timeline"].eq("longer_than_5_years")]

    if not low_quality.empty:
        items.append({"status": "Review", "audit_item": "Validate low-quality or potentially misleading evidence.", "owner": "ESG Audit"})
    if not no_evidence.empty:
        items.append({"status": "Request", "audit_item": "Ask the company for quantitative evidence and source documents.", "owner": "Disclosure Team"})
    if not long_timeline.empty:
        items.append({"status": "Monitor", "audit_item": "Track long-horizon commitments against interim milestones.", "owner": "Sustainability PMO"})
    if not items:
        items.append({"status": "Pass", "audit_item": "Evidence, timeline, and commitment signals are aligned.", "owner": "ESG Audit"})

    return pd.DataFrame(items)


def build_milestone_timeline(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    counts = evidence_rows["verification_timeline"].value_counts().to_dict()
    labels = [
        ("already", "Completed or already verifiable"),
        ("within_2_years", "Near-term checkpoint"),
        ("between_2_and_5_years", "Mid-term milestone"),
        ("longer_than_5_years", "Long-term commitment"),
        ("N/A", "No explicit timeline"),
    ]
    return pd.DataFrame(
        [{"timeline": label, "count": int(counts.get(key, 0))} for key, label in labels]
    )


def render_pdf_viewer(file_name: str, pdf_bytes_by_file: dict[str, bytes]) -> None:
    pdf_bytes = pdf_bytes_by_file.get(file_name)
    if not pdf_bytes:
        st.caption("No PDF bytes available for preview.")
        return

    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="560"></iframe>',
        unsafe_allow_html=True,
    )


def render_ai_analysis(issue: pd.Series) -> None:
    trust = float(issue["overall_trust_score"])
    risk = float(issue["greenwashing_risk"])

    if trust < 55:
        verdict = "High review priority: the issue combines weak evidence signals with elevated greenwashing risk."
    elif trust < 72:
        verdict = "Needs follow-up: the disclosure has some usable signals, but the evidence or timeline is not fully convincing."
    else:
        verdict = "Lower immediate concern: the available disclosure is relatively consistent with the stated ESG issue."

    st.write(verdict)
    st.write(f"Greenwashing risk is `{risk:.1f}` and the weakest trust score in this issue group is `{trust:.1f}`.")
    st.write(f"Main reason: {issue['risk_reason']}")
    st.caption(str(issue["representative_sentence"]))


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


st.title("ESG Hybrid Trust Dashboard")
st.caption("PDF disclosure analysis using the Hybrid v4 task structure from the notebook.")

uploaded_files = st.file_uploader(
    "Upload ESG / sustainability PDF reports",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDF reports to classify ESG issues and evaluate commitment, evidence, timeline, and trust signals.")
    st.stop()

with st.spinner("Extracting PDF text and running Hybrid ESG analysis..."):
    result_df, pdf_bytes_by_file = build_results(uploaded_files)

required_columns = {"overall_trust_score", "esg_category", "topic", "sentence"}
if result_df.empty or not required_columns.issubset(result_df.columns):
    st.warning("No E / S / G issue sentences were detected. Try a text-based PDF or check OCR quality.")
    st.stop()

issue_df = build_issue_summary(result_df)

metric_cols = st.columns(4)
metric_cols[0].metric("ESG Issues", f"{len(issue_df):,}")
metric_cols[1].metric("Evidence Sentences", f"{len(result_df):,}")
metric_cols[2].metric("Lowest Trust", f"{issue_df['overall_trust_score'].min():.1f}")
metric_cols[3].metric("Highest Risk", f"{issue_df['greenwashing_risk'].max():.1f}")

st.download_button(
    "Download issue summary CSV",
    data=to_csv_download(issue_df),
    file_name="esg_hybrid_trust_results.csv",
    mime="text/csv",
)

st.subheader("Issue Summary")
st.dataframe(
    issue_df[
        [
            "file_name",
            "esg_category",
            "topic",
            "overall_trust_score",
            "greenwashing_risk",
            "promise_rate",
            "clear_evidence_rate",
            "evidence_count",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

for index, issue in issue_df.reset_index(drop=True).iterrows():
    title = (
        f"{index + 1}. [{issue['esg_category']}] {issue['topic']} "
        f"- Trust {issue['overall_trust_score']:.1f}"
    )
    evidence_rows = result_df[
        (result_df["file_name"] == issue["file_name"])
        & (result_df["esg_category"] == issue["esg_category"])
        & (result_df["topic"] == issue["topic"])
    ].sort_values("overall_trust_score", ascending=True)

    with st.expander(title, expanded=index == 0):
        metric_row = st.columns(4)
        metric_row[0].metric("Overall Trust", f"{issue['overall_trust_score']:.1f}")
        metric_row[1].metric("Greenwashing Risk", f"{issue['greenwashing_risk']:.1f}")
        metric_row[2].metric("Promise Rate", f"{issue['promise_rate']:.1f}%")
        metric_row[3].metric("Clear Evidence", f"{issue['clear_evidence_rate']:.1f}%")

        radar_tab, peer_tab, audit_tab, pdf_tab, ai_tab, timeline_tab = st.tabs(
            [
                "Greenwashing Radar",
                "Peer Benchmarking",
                "Active Audit Feed",
                "PDF Viewer",
                "AI Analysis Panel",
                "Milestone Timeline",
            ]
        )

        with radar_tab:
            radar_chart = (
                alt.Chart(build_radar_data(issue))
                .mark_bar()
                .encode(
                    x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Risk signal"),
                    y=alt.Y("signal:N", sort="-x", title=None),
                    color=alt.Color("score:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
                    tooltip=["signal", "score"],
                )
                .properties(height=240)
            )
            st.altair_chart(radar_chart, use_container_width=True)

        with peer_tab:
            st.bar_chart(build_peer_benchmark(issue), x="benchmark", y="trust_score")

        with audit_tab:
            st.dataframe(build_audit_feed(issue, evidence_rows), use_container_width=True, hide_index=True)

        with pdf_tab:
            render_pdf_viewer(str(issue["file_name"]), pdf_bytes_by_file)

        with ai_tab:
            render_ai_analysis(issue)

        with timeline_tab:
            st.dataframe(build_milestone_timeline(evidence_rows), use_container_width=True, hide_index=True)

        st.markdown("**Evidence Sentences**")
        st.dataframe(
            evidence_rows[
                [
                    "overall_trust_score",
                    "greenwashing_risk",
                    "promise_status",
                    "verification_timeline",
                    "evidence_status",
                    "evidence_quality",
                    "risk_reason",
                    "sentence",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
