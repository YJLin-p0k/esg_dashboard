from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO, StringIO
import math

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.hybrid_model import ESG_CATEGORIES, HybridESGAnalyzer
from esg_dashboard.pdf_utils import extract_pdf_text
from esg_dashboard.text_utils import split_chinese_sentence_units


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
    "paragraph_id": "段落編號",
    "paragraph_context": "相近段落",
    "sentence": "原文句子",
    "matched_sentences": "相關句子",
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
        sentence_units = split_chinese_sentence_units(text)
        sentences = [unit.sentence for unit in sentence_units]
        predictions = analyzer.predict(sentences)

        for sentence_id, (unit, prediction) in enumerate(zip(sentence_units, predictions), start=1):
            sentence = unit.sentence
            if prediction.esg_category not in ESG_CATEGORIES or prediction.esg_category == "Other":
                continue

            rows.append(
                {
                    "file_name": uploaded_file.name,
                    "paragraph_id": unit.paragraph_id,
                    "paragraph_text": unit.paragraph_text,
                    "paragraph_context": unit.paragraph_context,
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


def severity_from_trust(score: float) -> tuple[str, str]:
    if score < 55:
        return "高風險", "#d84a3a"
    if score < 72:
        return "需追蹤", "#f2b84b"
    return "穩健", "#2f9e62"


def severity_from_risk(score: float) -> tuple[str, str]:
    if score >= 65:
        return "高風險", "#d84a3a"
    if score >= 35:
        return "需追蹤", "#f2b84b"
    return "低風險", "#2f9e62"


def render_trust_gauge(score: float) -> None:
    severity, color = severity_from_trust(score)
    angle = 180 - max(0, min(100, score)) * 1.8
    st.markdown(
        f"""
        <div style="max-width: 420px; margin: 0 auto 0.75rem auto;">
          <div style="
              position: relative;
              width: 100%;
              aspect-ratio: 2 / 1;
              overflow: hidden;
              border-radius: 420px 420px 0 0;
              background: conic-gradient(from 270deg at 50% 100%,
                #d84a3a 0deg 54deg,
                #f2b84b 54deg 126deg,
                #2f9e62 126deg 180deg,
                transparent 180deg 360deg);
              box-shadow: inset 0 0 0 1px rgba(20, 31, 43, 0.12);
            ">
            <div style="
                position: absolute;
                left: 11%;
                right: 11%;
                bottom: 0;
                height: 78%;
                border-radius: 320px 320px 0 0;
                background: white;
              "></div>
            <div style="
                position: absolute;
                left: 50%;
                bottom: 0;
                width: 42%;
                height: 4px;
                background: #17202a;
                transform-origin: 0% 50%;
                transform: rotate({angle:.1f}deg);
                border-radius: 4px;
              "></div>
            <div style="
                position: absolute;
                left: calc(50% - 10px);
                bottom: -10px;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #17202a;
              "></div>
          </div>
          <div style="display: flex; justify-content: space-between; color: #667085; font-size: 0.8rem;">
            <span>0</span><span>50</span><span>100</span>
          </div>
          <div style="text-align: center; margin-top: 0.1rem;">
            <div style="font-size: 2rem; font-weight: 700; line-height: 1;">{score:.1f}</div>
            <div style="color: {color}; font-weight: 700;">{severity}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_greenwashing_pie_data(issue: pd.Series) -> pd.DataFrame:
    risk = float(issue["greenwashing_risk"])
    severity, color = severity_from_risk(risk)
    return pd.DataFrame(
        [
            {"segment": f"漂綠風險：{severity}", "value": risk, "color": color},
            {"segment": "其餘信任空間", "value": max(0, 100 - risk), "color": "#e7ebef"},
        ]
    )


def build_risk_signal_data(issue: pd.Series) -> pd.DataFrame:
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


def build_peer_radar_data(issue: pd.Series, result_df: pd.DataFrame) -> pd.DataFrame:
    file_name = issue["file_name"]
    report_rows = result_df[result_df["file_name"].eq(file_name)]
    peer_scores = {
        "Environment 信任": 72,
        "Social 信任": 76,
        "Governance 信任": 79,
        "Environment 證據": 66,
        "Social 證據": 70,
        "Governance 證據": 74,
    }
    category_labels = {
        "Environment": "Environment",
        "Social": "Social",
        "Governance": "Governance",
    }

    report_scores: dict[str, float] = {}
    for category, label in category_labels.items():
        category_rows = report_rows[report_rows["esg_category"].eq(category)]
        if category_rows.empty:
            report_scores[f"{label} 信任"] = 0
            report_scores[f"{label} 證據"] = 0
            continue

        report_scores[f"{label} 信任"] = round(float(category_rows["overall_trust_score"].mean()), 2)
        report_scores[f"{label} 證據"] = round(float(category_rows["evidence_quality"].eq("Clear").mean() * 100), 2)

    axes = list(peer_scores.keys())
    rows: list[dict[str, object]] = []
    for series_name, scores in {"同業平均": peer_scores, "本報告": report_scores}.items():
        for order, axis in enumerate(axes):
            angle = (math.pi / 2) - (2 * math.pi * order / len(axes))
            value = float(scores.get(axis, 0))
            rows.append(
                {
                    "series": series_name,
                    "axis": axis,
                    "order": order,
                    "score": value,
                    "x": math.cos(angle) * value,
                    "y": math.sin(angle) * value,
                }
            )
        first = rows[-len(axes)].copy()
        first["order"] = len(axes)
        rows.append(first)

    return pd.DataFrame(rows)


def build_peer_radar_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for score in [25, 50, 75, 100]:
        for order in range(7):
            angle = (math.pi / 2) - (2 * math.pi * (order % 6) / 6)
            rows.append({"score": score, "order": order, "x": math.cos(angle) * score, "y": math.sin(angle) * score})
    return pd.DataFrame(rows)


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


def build_related_paragraphs(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    if evidence_rows.empty:
        return pd.DataFrame()

    paragraph_rows: list[dict[str, object]] = []
    grouped = evidence_rows.sort_values(["paragraph_id", "sentence_id"]).groupby(
        ["file_name", "paragraph_id"],
        as_index=False,
        sort=True,
    )

    for _, paragraph_df in grouped:
        sentence_ids = paragraph_df["sentence_id"].astype(int).tolist()
        matched_sentences = " / ".join(paragraph_df["sentence"].astype(str).tolist())
        reason_values = paragraph_df["risk_reason"].dropna().astype(str).unique().tolist()
        paragraph_rows.append(
            {
                "file_name": paragraph_df["file_name"].iloc[0],
                "paragraph_id": int(paragraph_df["paragraph_id"].iloc[0]),
                "sentence_id": f"{min(sentence_ids)}-{max(sentence_ids)}" if len(sentence_ids) > 1 else str(sentence_ids[0]),
                "overall_trust_score": round(float(paragraph_df["overall_trust_score"].min()), 2),
                "greenwashing_risk": round(float(paragraph_df["greenwashing_risk"].max()), 2),
                "promise_status": "Yes" if paragraph_df["promise_status"].eq("Yes").any() else "No",
                "verification_timeline": paragraph_df["verification_timeline"].mode().iat[0],
                "evidence_status": "Yes" if paragraph_df["evidence_status"].eq("Yes").any() else paragraph_df["evidence_status"].mode().iat[0],
                "evidence_quality": paragraph_df["evidence_quality"].mode().iat[0],
                "risk_reason": "；".join(reason_values),
                "matched_sentences": matched_sentences,
                "paragraph_context": paragraph_df["paragraph_context"].iloc[0],
            }
        )

    return pd.DataFrame(paragraph_rows).sort_values(
        ["overall_trust_score", "greenwashing_risk"],
        ascending=[True, False],
    )


def build_audit_gantt_data(issue: pd.Series, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    today = date.today()
    tasks: list[dict[str, object]] = []

    audit_rows = build_audit_feed(issue, evidence_rows)
    for offset, audit_row in audit_rows.iterrows():
        status = str(audit_row["status"])
        severity = "低風險" if status == "通過" else ("高風險" if status in {"需複核", "需補件"} else "需追蹤")
        start = today + timedelta(days=offset * 7)
        tasks.append(
            {
                "task": str(audit_row["audit_item"]),
                "group": "稽核待辦",
                "owner": str(audit_row["owner"]),
                "status": status,
                "severity": severity,
                "start": start,
                "end": start + timedelta(days=14),
            }
        )

    timeline_map = {
        "already": ("已完成或可驗證", today - timedelta(days=45), today - timedelta(days=15), "低風險"),
        "within_2_years": ("近期檢查點", today + timedelta(days=15), today + timedelta(days=120), "需追蹤"),
        "between_2_and_5_years": ("中期里程碑", today + timedelta(days=120), today + timedelta(days=365), "需追蹤"),
        "longer_than_5_years": ("長期承諾", today + timedelta(days=365), today + timedelta(days=730), "高風險"),
        "N/A": ("未說明時程", today, today + timedelta(days=30), "高風險"),
    }
    counts = evidence_rows["verification_timeline"].value_counts().to_dict()
    for key, (label, start, end, severity) in timeline_map.items():
        count = int(counts.get(key, 0))
        if count == 0:
            continue
        tasks.append(
            {
                "task": f"{label}（{count} 句）",
                "group": "里程碑時程",
                "owner": "ESG 專案",
                "status": label,
                "severity": severity,
                "start": start,
                "end": end,
            }
        )

    return pd.DataFrame(tasks)


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


def uploaded_file_signature(uploaded_files) -> tuple[tuple[str, int], ...]:
    return tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files)


def render_upload_confirmation(uploaded_files) -> tuple[tuple[str, int], ...]:
    signature = uploaded_file_signature(uploaded_files)
    total_size_mb = sum(size for _, size in signature) / (1024 * 1024)

    st.write(f"已選擇 `{len(uploaded_files)}` 份 PDF，總大小 `{total_size_mb:.2f} MB`。")
    st.dataframe(
        pd.DataFrame(
            [
                {"檔案": file_name, "大小 MB": round(size / (1024 * 1024), 2)}
                for file_name, size in signature
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    confirmed_signature = st.session_state.get("confirmed_upload_signature")
    if confirmed_signature != signature:
        st.info("請確認上傳檔案無誤後，再開始分析。")
        if st.button("確認並開始分析", type="primary"):
            st.session_state["confirmed_upload_signature"] = signature
            st.rerun()
        st.stop()

    st.success("PDF 已確認，開始進行分析。")
    return signature


st.title("ESG 混合信任儀表板")
st.caption("上傳 ESG 或永續報告 PDF，系統會先判斷句子的 E/S/G 語意，再細分議題並彙總評估。")

uploaded_files = st.file_uploader(
    "上傳 ESG / 永續報告 PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.session_state.pop("confirmed_upload_signature", None)
    st.info("請上傳一份或多份 PDF 報告，系統會分類 ESG 議題並評估承諾、證據、時程與信任訊號。")
    st.stop()

render_upload_confirmation(uploaded_files)

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
        render_trust_gauge(float(issue["overall_trust_score"]))

        metric_row = st.columns(4)
        metric_row[0].metric("保守信任分數", f"{issue['overall_trust_score']:.1f}")
        metric_row[1].metric("最高漂綠風險", f"{issue['greenwashing_risk']:.1f}")
        metric_row[2].metric("承諾比例", f"{issue['promise_rate']:.1f}%")
        metric_row[3].metric("清楚證據比例", f"{issue['clear_evidence_rate']:.1f}%")

        greenwash_tab, peer_tab, audit_timeline_tab, ai_tab, related_tab = st.tabs(
            [
                "漂綠圓餅",
                "同業比較",
                "稽核與時程",
                "AI 分析",
                "相關文句",
            ]
        )

        with greenwash_tab:
            pie_data = build_greenwashing_pie_data(issue)
            risk_label, risk_color = severity_from_risk(float(issue["greenwashing_risk"]))
            pie_chart = (
                alt.Chart(pie_data)
                .mark_arc(innerRadius=70, outerRadius=125)
                .encode(
                    theta=alt.Theta("value:Q", stack=True),
                    color=alt.Color("segment:N", scale=alt.Scale(domain=pie_data["segment"].tolist(), range=pie_data["color"].tolist()), legend=None),
                    tooltip=[
                        alt.Tooltip("segment:N", title="項目"),
                        alt.Tooltip("value:Q", title="占比", format=".1f"),
                    ],
                )
                .properties(height=285)
            )
            center_text = (
                alt.Chart(pd.DataFrame([{"text": f"{issue['greenwashing_risk']:.1f}", "subtext": risk_label}]))
                .mark_text(fontSize=34, fontWeight="bold", color=risk_color, dy=-8)
                .encode(text="text:N")
            )
            center_subtext = (
                alt.Chart(pd.DataFrame([{"subtext": risk_label}]))
                .mark_text(fontSize=15, fontWeight="bold", color=risk_color, dy=28)
                .encode(text="subtext:N")
            )
            st.altair_chart(pie_chart + center_text + center_subtext, use_container_width=True)
            st.dataframe(build_risk_signal_data(issue), use_container_width=True, hide_index=True)

        with peer_tab:
            peer_data = build_peer_radar_data(issue, result_df)
            grid_chart = (
                alt.Chart(build_peer_radar_grid())
                .mark_line(color="#d8dee6", strokeWidth=1)
                .encode(
                    x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-110, 110])),
                    y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-110, 110])),
                    detail="score:N",
                    order="order:Q",
                )
            )
            axis_label_data = peer_data[peer_data["series"].eq("同業平均") & peer_data["order"].lt(6)].copy()
            axis_label_data["label_x"] = axis_label_data["x"] * 1.16
            axis_label_data["label_y"] = axis_label_data["y"] * 1.16
            label_chart = (
                alt.Chart(axis_label_data)
                .mark_text(fontSize=12, color="#475467")
                .encode(
                    x=alt.X("label_x:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                    y=alt.Y("label_y:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                    text="axis:N",
                )
            )
            peer_line_chart = (
                alt.Chart(peer_data)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                    y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                    color=alt.Color("series:N", scale=alt.Scale(range=["#7a869a", "#2563eb"]), title="比較對象"),
                    detail="series:N",
                    order="order:Q",
                    tooltip=[
                        alt.Tooltip("series:N", title="比較對象"),
                        alt.Tooltip("axis:N", title="向度"),
                        alt.Tooltip("score:Q", title="分數", format=".1f"),
                    ],
                )
                .properties(height=380)
            )
            st.altair_chart(grid_chart + peer_line_chart + label_chart, use_container_width=True)

        with audit_timeline_tab:
            gantt_data = build_audit_gantt_data(issue, evidence_rows)
            if gantt_data.empty:
                st.info("目前沒有可呈現的稽核待辦或里程碑。")
            else:
                gantt_chart = (
                    alt.Chart(gantt_data)
                    .mark_bar(size=18)
                    .encode(
                        x=alt.X("start:T", title="開始"),
                        x2="end:T",
                        y=alt.Y("task:N", sort="-x", title=None),
                        color=alt.Color(
                            "severity:N",
                            scale=alt.Scale(
                                domain=["高風險", "需追蹤", "低風險"],
                                range=["#d84a3a", "#f2b84b", "#2f9e62"],
                            ),
                            title="嚴重程度",
                        ),
                        row=alt.Row("group:N", title=None, header=alt.Header(labelAngle=0, labelFontWeight="bold")),
                        tooltip=[
                            alt.Tooltip("group:N", title="類型"),
                            alt.Tooltip("task:N", title="項目"),
                            alt.Tooltip("owner:N", title="負責"),
                            alt.Tooltip("status:N", title="狀態"),
                            alt.Tooltip("start:T", title="開始", format="%Y-%m-%d"),
                            alt.Tooltip("end:T", title="結束", format="%Y-%m-%d"),
                        ],
                    )
                    .properties(height=180)
                    .resolve_scale(y="independent")
                )
                st.altair_chart(gantt_chart, use_container_width=True)
                st.dataframe(gantt_data[["group", "task", "owner", "status", "severity", "start", "end"]], use_container_width=True, hide_index=True)

        with ai_tab:
            render_ai_analysis(issue)

        with related_tab:
            related_paragraphs = build_related_paragraphs(evidence_rows)
            st.dataframe(
                localize_dataframe(related_paragraphs[
                    [
                        "paragraph_id",
                        "sentence_id",
                        "overall_trust_score",
                        "greenwashing_risk",
                        "promise_status",
                        "verification_timeline",
                        "evidence_status",
                        "evidence_quality",
                        "risk_reason",
                        "matched_sentences",
                        "paragraph_context",
                    ]
                ]),
                use_container_width=True,
                hide_index=True,
            )
