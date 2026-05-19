from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO
import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.hybrid_model import ESG_CATEGORIES, HybridESGAnalyzer, OFFICIAL_WEIGHTS
from esg_dashboard.pdf_utils import extract_pdf_text
from esg_dashboard.text_utils import split_chinese_sentence_units


APP_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = APP_DIR / "data" / "vpesg_4k_train_1000.json"


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

ESG_TYPE_TO_CATEGORY = {
    "E": "Environment",
    "S": "Social",
    "G": "Governance",
}

ESG_CATEGORY_ORDER = {
    "Environment": 0,
    "Social": 1,
    "Governance": 2,
}

COMPANY_ALIASES = {
    "accton": ["accton", "智邦", "2345"],
    "acl": ["acl", "國巨", "2327"],
    "alchip": ["alchip", "世芯", "3661"],
    "aseh": ["aseh", "日月光", "3711"],
    "avc": ["avc", "東元", "1504"],
    "cathay": ["cathay", "國泰", "2882"],
    "chailease": ["chailease", "中租", "5871"],
    "cht": ["cht", "中華電信", "2412"],
    "csc": ["csc", "中鋼", "2002"],
    "ctbc": ["ctbc", "中信", "2891"],
    "delta": ["delta", "台達", "2308"],
    "emc": ["emc", "長榮海運", "2603"],
    "emc2": ["emc2", "長榮", "2603"],
    "esfh": ["esfh", "玉山", "2884"],
    "fet": ["fet", "遠傳", "4904"],
    "ffhc": ["ffhc", "第一金", "2892"],
    "fpc": ["fpc", "台塑", "1301"],
    "fpcc": ["fpcc", "台塑化", "6505"],
    "fubon": ["fubon", "富邦", "2881"],
    "hnfhc": ["hnfhc", "華南金", "2880"],
    "honhai": ["honhai", "鴻海", "2317"],
    "hotaimotor": ["hotaimotor", "和泰", "2207"],
    "kgi": ["kgi", "凱基", "2883"],
    "largan": ["largan", "大立光", "3008"],
    "ltc": ["ltc", "聯強", "2347"],
    "mediatek": ["mediatek", "聯發科", "2454"],
    "mega": ["mega", "兆豐", "2886"],
    "novatek": ["novatek", "聯詠", "3034"],
    "npc": ["npc", "南亞", "1303"],
    "pcsc": ["pcsc", "統一超", "2912"],
    "pec": ["pec", "台光", "2383"],
    "pegatron": ["pegatron", "和碩", "4938"],
    "qci": ["qci", "廣達", "2382"],
    "rt": ["rt", "潤泰", "2915"],
    "scsb": ["scsb", "上海商銀", "5876"],
    "taishin": ["taishin", "台新", "2887"],
    "tcc": ["tcc", "台泥", "1101"],
    "tcfhc": ["tcfhc", "台中銀", "2812"],
    "tsmc": ["tsmc", "台積", "2330"],
    "twm": ["twm", "台灣大", "3045"],
    "umc": ["umc", "聯電", "2303"],
    "unipresident": ["unipresident", "統一企業", "1216"],
    "wanhai": ["wanhai", "萬海", "2615"],
    "wistron": ["wistron", "緯創", "3231"],
    "wiwynn": ["wiwynn", "緯穎", "6669"],
    "yageo": ["yageo", "國巨", "2327"],
    "yfy": ["yfy", "永豐餘", "1907"],
    "ymtc": ["ymtc", "陽明", "2609"],
    "yuanta": ["yuanta", "元大", "2885"],
}

PEER_GROUPS = {
    "semiconductor": {"tsmc", "umc", "mediatek", "novatek", "alchip", "aseh"},
    "electronics": {"accton", "delta", "honhai", "pegatron", "qci", "wistron", "wiwynn", "pec", "largan", "yageo", "acl", "ltc"},
    "finance": {"cathay", "ctbc", "esfh", "ffhc", "fubon", "hnfhc", "kgi", "mega", "scsb", "taishin", "tcfhc", "yuanta", "chailease"},
    "telecom": {"cht", "fet", "twm"},
    "transport": {"emc", "emc2", "wanhai", "ymtc"},
    "materials": {"tcc", "fpc", "fpcc", "npc", "csc", "yfy"},
    "consumer": {"avc", "hotaimotor", "pcsc", "rt", "unipresident"},
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


st.set_page_config(page_title="ESG Sentinal 綠色哨兵", page_icon="📊", layout="wide")


@st.cache_resource
def load_analyzer() -> HybridESGAnalyzer:
    return HybridESGAnalyzer()


@st.cache_data
def load_training_peer_rows() -> pd.DataFrame:
    rows = pd.read_json(TRAINING_DATA_PATH)
    rows["company"] = rows["company"].astype(str).str.lower()
    rows["esg_category"] = rows["esg_type"].map(ESG_TYPE_TO_CATEGORY)
    rows["evidence_quality"] = rows["evidence_quality"].replace("", "N/A").fillna("N/A")
    rows["evidence_status"] = rows["evidence_status"].replace("", "N/A").fillna("N/A")
    rows["verification_timeline"] = rows["verification_timeline"].replace("", "N/A").fillna("N/A")
    rows["promise_status"] = rows["promise_status"].replace("", "No").fillna("No")
    rows["overall_trust_score"] = rows.apply(compute_reference_trust_score, axis=1)
    return rows


def compute_reference_trust_score(row: pd.Series) -> float:
    promise_score = 1.0 if row.get("promise_status") == "Yes" else 0.55
    evidence_score = {"Yes": 1.0, "No": 0.25, "N/A": 0.55}.get(row.get("evidence_status"), 0.55)
    quality_score = {"Clear": 1.0, "Not Clear": 0.35, "Misleading": 0.0, "N/A": 0.55}.get(row.get("evidence_quality"), 0.55)
    timeline_score = {
        "already": 1.0,
        "within_2_years": 0.85,
        "between_2_and_5_years": 0.65,
        "longer_than_5_years": 0.35,
        "N/A": 0.55,
    }.get(row.get("verification_timeline"), 0.55)

    weighted = (
        promise_score * OFFICIAL_WEIGHTS["promise_status"]
        + evidence_score * OFFICIAL_WEIGHTS["evidence_status"]
        + quality_score * OFFICIAL_WEIGHTS["evidence_quality"]
        + timeline_score * OFFICIAL_WEIGHTS["verification_timeline"]
    )
    return round(max(0, min(100, weighted * 100)), 2)


def detect_company(file_name: str, text: str) -> str:
    haystack = f"{file_name}\n{text[:5000]}".lower()
    for company, aliases in COMPANY_ALIASES.items():
        if any(alias.lower() in haystack for alias in aliases):
            return company
    return "unknown"


def get_peer_group(company: str) -> set[str]:
    normalized_company = str(company).lower()
    for peer_companies in PEER_GROUPS.values():
        if normalized_company in peer_companies:
            return peer_companies
    return set()


def build_results(uploaded_files) -> pd.DataFrame:
    analyzer = load_analyzer()
    rows: list[dict[str, object]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        text = extract_pdf_text(BytesIO(file_bytes))
        company = detect_company(uploaded_file.name, text)
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
                    "company": company,
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
        ["file_name", "company", "esg_category", "overall_trust_score", "greenwashing_risk", "topic"],
        ascending=[True, True, True, True, False, True],
    )

    summary = (
        sorted_df.groupby(["file_name", "company", "esg_category", "topic"], as_index=False)
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
    )
    summary["_esg_order"] = summary["esg_category"].map(ESG_CATEGORY_ORDER).fillna(99)
    return (
        summary.sort_values(
            ["file_name", "_esg_order", "overall_trust_score", "greenwashing_risk", "topic"],
            ascending=[True, True, True, False, True],
        )
        .drop(columns="_esg_order")
        .reset_index(drop=True)
    )


def build_issue_summary_display(issue_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    columns = [
        "file_name",
        "esg_category",
        "topic",
        "overall_trust_score",
        "promise_rate",
        "clear_evidence_rate",
        "evidence_count",
        "representative_sentence",
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
        if bool(is_file_min.iloc[row.name]):
            styles[row.index.get_loc(trust_column)] = "color: #d84a3a; font-weight: 800;"
        return styles

    return localized.style.apply(highlight_file_min, axis=1).format(
        {
            COLUMN_LABELS["overall_trust_score"]: "{:.1f}",
            COLUMN_LABELS["promise_rate"]: "{:.1f}%",
            COLUMN_LABELS["clear_evidence_rate"]: "{:.1f}%",
            COLUMN_LABELS["evidence_count"]: "{:.0f}",
        }
    )


def severity_from_trust(score: float) -> tuple[str, str]:
    if score < 35:
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
    angle = 180 + max(0, min(100, score)) * 1.8
    st.markdown(
        f"""
        <div style="max-width: 460px; margin: 0 auto 0.75rem auto;">
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
                #d84a3a 0deg 48deg,
                #f2b84b 60deg 114deg,
                #2f9e62 132deg 180deg,
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
                background: currentColor;
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
                background: currentColor;
                box-shadow: 0 0 0 5px color-mix(in srgb, Canvas 82%, transparent), 0 2px 6px rgba(20, 31, 43, 0.35);
                z-index: 6;
              "></div>
            <div style="position: absolute; left: 0; bottom: -1.15rem; color: inherit; font-size: 0.82rem; font-weight: 700;">0</div>
            <div style="position: absolute; left: 50%; top: -1.25rem; transform: translateX(-50%); color: inherit; font-size: 0.82rem; font-weight: 700;">50</div>
            <div style="position: absolute; right: 0; bottom: -1.15rem; color: inherit; font-size: 0.82rem; font-weight: 700;">100</div>
          </div>
          <div style="text-align: center; margin-top: 1.35rem;">
            <div style="font-size: 2rem; font-weight: 700; line-height: 1;">{score:.1f}</div>
            <div style="color: {color}; font-weight: 700;">{severity}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_greenwashing_pie_data(evidence_rows: pd.DataFrame) -> pd.DataFrame:
    severity_order = [("高風險", "#d84a3a"), ("需追蹤", "#f2b84b"), ("低風險", "#2f9e62")]
    severity_counts = {severity: 0 for severity, _ in severity_order}

    for risk in evidence_rows["greenwashing_risk"].astype(float):
        severity, _ = severity_from_risk(risk)
        severity_counts[severity] += 1

    total = max(1, sum(severity_counts.values()))
    return pd.DataFrame(
        [
            {
                "segment": severity,
                "count": severity_counts[severity],
                "value": round(severity_counts[severity] / total * 100, 1),
                "color": color,
            }
            for severity, color in severity_order
            if severity_counts[severity] > 0
        ]
    )


def render_greenwashing_risk_explanation() -> None:
    st.markdown(
        """
        <div style="color: inherit; background: transparent; padding: 0 0 0.35rem 0; margin-bottom: 0.75rem;">
          <div style="font-size: 0.95rem; line-height: 1.55;">
            漂綠風險代表相關文句中，永續承諾、佐證資料、驗證時程與證據清楚程度之間可能不一致的程度。
            分數越高，表示該文句越需要進一步查核是否有承諾過度、證據不足或時程不明確的疑慮。
          </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.35rem 0 1rem 0; color: inherit;">
          <div style="display: inline-flex; align-items: center; gap: 0.4rem;">
            <span style="width: 0.85rem; height: 0.85rem; border-radius: 50%; background: #d84a3a; display: inline-block;"></span>
            <span style="font-size: 0.9rem; color: inherit; font-weight: 600;">高風險：65 分以上</span>
          </div>
          <div style="display: inline-flex; align-items: center; gap: 0.4rem;">
            <span style="width: 0.85rem; height: 0.85rem; border-radius: 50%; background: #f2b84b; display: inline-block;"></span>
            <span style="font-size: 0.9rem; color: inherit; font-weight: 600;">需追蹤：35 至 64.9 分</span>
          </div>
          <div style="display: inline-flex; align-items: center; gap: 0.4rem;">
            <span style="width: 0.85rem; height: 0.85rem; border-radius: 50%; background: #2f9e62; display: inline-block;"></span>
            <span style="font-size: 0.9rem; color: inherit; font-weight: 600;">低風險：低於 35 分</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
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


def build_peer_radar_data(issue: pd.Series, result_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    file_name = issue["file_name"]
    company = str(issue.get("company", "unknown")).lower()
    report_rows = result_df[result_df["file_name"].eq(file_name)]
    category_labels = {
        "Environment": "Environment",
        "Social": "Social",
        "Governance": "Governance",
    }
    training_rows = load_training_peer_rows()
    peer_group = get_peer_group(company)
    if peer_group:
        peer_rows = training_rows[training_rows["company"].isin(peer_group) & ~training_rows["company"].eq(company)]
    else:
        peer_rows = training_rows

    if peer_rows.empty:
        peer_rows = training_rows

    baseline_note = f"同業平均來源：與本研究資料中同業的公司({peer_rows['company'].nunique()}家)。"

    peer_scores: dict[str, float] = {}
    for category, label in category_labels.items():
        category_rows = peer_rows[peer_rows["esg_category"].eq(category)]
        if category_rows.empty:
            peer_scores[f"{label} 信任"] = 0
            peer_scores[f"{label} 證據"] = 0
            continue

        peer_scores[f"{label} 信任"] = round(float(category_rows["overall_trust_score"].mean()), 2)
        peer_scores[f"{label} 證據"] = round(float(category_rows["evidence_quality"].eq("Clear").mean() * 100), 2)

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

    return pd.DataFrame(rows), baseline_note


def build_peer_radar_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for score in [25, 50, 75, 100]:
        for order in range(7):
            angle = (math.pi / 2) - (2 * math.pi * (order % 6) / 6)
            rows.append({"score": score, "order": order, "x": math.cos(angle) * score, "y": math.sin(angle) * score})
    return pd.DataFrame(rows)


def build_peer_radar_axis_labels(axes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order, axis in enumerate(axes):
        angle = (math.pi / 2) - (2 * math.pi * order / len(axes))
        rows.append(
            {
                "axis": axis,
                "label_x": math.cos(angle) * 122,
                "label_y": math.sin(angle) * 122,
            }
        )
    return pd.DataFrame(rows)


def render_peer_comparison(issue: pd.Series, result_df: pd.DataFrame) -> None:
    peer_data, baseline_note = build_peer_radar_data(issue, result_df)
    radar_width = 520
    radar_height = 420
    chart_domain = [-160, 160]
    st.markdown(
        f"""
        <div style="color: inherit; background: transparent; padding: 0 0 0.35rem 0; margin-bottom: 0.6rem; font-size: 0.9rem;">
          {baseline_note}
        </div>
        """,
        unsafe_allow_html=True,
    )
    grid_chart = (
        alt.Chart(build_peer_radar_grid())
        .mark_line(color="#cbd5e1", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            detail="score:N",
            order="order:Q",
        )
        .properties(width=radar_width, height=radar_height)
    )
    axis_label_data = build_peer_radar_axis_labels(peer_data[peer_data["order"].lt(6)]["axis"].drop_duplicates().tolist())
    label_chart = (
        alt.Chart(axis_label_data)
        .mark_text(fontSize=12, fontWeight="bold", color="currentColor")
        .encode(
            x=alt.X("label_x:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            y=alt.Y("label_y:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            text="axis:N",
        )
        .properties(width=radar_width, height=radar_height)
    )
    peer_line_chart = (
        alt.Chart(peer_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            color=alt.Color("series:N", scale=alt.Scale(range=["#475467", "#2563eb"]), title="比較對象"),
            detail="series:N",
            order="order:Q",
            tooltip=[
                alt.Tooltip("series:N", title="比較對象"),
                alt.Tooltip("axis:N", title="向度"),
                alt.Tooltip("score:Q", title="分數", format=".1f"),
            ],
        )
        .properties(width=radar_width, height=radar_height)
    )
    chart = (
        grid_chart + peer_line_chart + label_chart
    ).configure_view(
        stroke=None
    )

    _, radar_col, _ = st.columns([1, 2, 1])
    with radar_col:
        st.altair_chart(chart, use_container_width=False)


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


def quarter_start(value: date) -> date:
    quarter_month = ((value.month - 1) // 3) * 3 + 1
    return date(value.year, quarter_month, 1)


def add_quarters(value: date, quarters: int) -> date:
    month_index = value.year * 12 + value.month - 1 + quarters * 3
    return date(month_index // 12, month_index % 12 + 1, 1)


def quarter_label(value: date) -> str:
    return f"{value.year} Q{((value.month - 1) // 3) + 1}"


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
    current_quarter = quarter_start(date.today())
    tasks: list[dict[str, object]] = []

    audit_rows = build_audit_feed(issue, evidence_rows)
    for offset, audit_row in audit_rows.iterrows():
        status = str(audit_row["status"])
        severity = "低風險" if status == "通過" else ("高風險" if status in {"需複核", "需補件"} else "需追蹤")
        start = add_quarters(current_quarter, offset)
        end = add_quarters(start, 1)
        tasks.append(
            {
                "task": str(audit_row["audit_item"]),
                "group": "稽核待辦",
                "owner": str(audit_row["owner"]),
                "status": status,
                "severity": severity,
                "start": start,
                "end": end,
                "start_quarter": quarter_label(start),
                "end_quarter": quarter_label(end),
            }
        )

    timeline_map = {
        "already": ("已完成或可驗證", add_quarters(current_quarter, -1), current_quarter, "低風險"),
        "within_2_years": ("近期檢查點", current_quarter, add_quarters(current_quarter, 4), "需追蹤"),
        "between_2_and_5_years": ("中期里程碑", add_quarters(current_quarter, 4), add_quarters(current_quarter, 8), "需追蹤"),
        "longer_than_5_years": ("長期承諾", add_quarters(current_quarter, 8), add_quarters(current_quarter, 12), "高風險"),
        "N/A": ("未說明時程", current_quarter, add_quarters(current_quarter, 1), "高風險"),
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
                "start_quarter": quarter_label(start),
                "end_quarter": quarter_label(end),
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


st.title("ESG Sentinal 承諾驗證")
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
    build_issue_summary_display(issue_df),
    use_container_width=True,
    hide_index=True,
)

st.subheader("同業比較")
for file_name, file_issues in issue_df.groupby("file_name", sort=False):
    st.markdown(f"**目前報告：{file_name}**")
    render_peer_comparison(file_issues.iloc[0], result_df)

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

        metric_row = st.columns(3)
        metric_row[0].metric("保守信任分數", f"{issue['overall_trust_score']:.1f}")
        metric_row[1].metric("承諾比例", f"{issue['promise_rate']:.1f}%")
        metric_row[2].metric("清楚證據比例", f"{issue['clear_evidence_rate']:.1f}%")

        greenwash_tab, audit_timeline_tab, ai_tab, related_tab = st.tabs(
            [
                "漂綠風險",
                "稽核與時程",
                "AI 分析",
                "相關文句",
            ]
        )

        with greenwash_tab:
            render_greenwashing_risk_explanation()
            pie_data = build_greenwashing_pie_data(evidence_rows)
            average_risk = float(evidence_rows["greenwashing_risk"].mean())
            risk_label, risk_color = severity_from_risk(average_risk)
            pie_chart = (
                alt.Chart(pie_data)
                .mark_arc(innerRadius=70, outerRadius=125)
                .encode(
                    theta=alt.Theta("value:Q", stack=True),
                    color=alt.Color(
                        "segment:N",
                        scale=alt.Scale(domain=pie_data["segment"].tolist(), range=pie_data["color"].tolist()),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("segment:N", title="層級"),
                        alt.Tooltip("count:Q", title="文句數"),
                        alt.Tooltip("value:Q", title="比例", format=".1f"),
                    ],
                )
                .properties(height=285)
            )
            center_title = (
                alt.Chart(pd.DataFrame([{"text": "平均漂綠風險"}]))
                .mark_text(fontSize=14, fontWeight="bold", color="currentColor", dy=-38)
                .encode(text="text:N")
            )
            center_score = (
                alt.Chart(pd.DataFrame([{"text": f"{average_risk:.1f}"}]))
                .mark_text(fontSize=34, fontWeight="bold", color=risk_color, dy=-5)
                .encode(text="text:N")
            )
            center_subtext = (
                alt.Chart(pd.DataFrame([{"subtext": risk_label}]))
                .mark_text(fontSize=15, fontWeight="bold", color=risk_color, dy=31)
                .encode(text="subtext:N")
            )
            pie_col, sentence_count_col = st.columns([3, 0.72], gap="small")
            with pie_col:
                greenwashing_chart = (
                    pie_chart + center_title + center_score + center_subtext
                ).configure_view(stroke=None)
                st.altair_chart(greenwashing_chart, use_container_width=True)
            with sentence_count_col:
                st.markdown(
                    f"""
                    <div style="color: inherit; margin-top: 5.2rem;">
                      <div style="font-size: 0.9rem; font-weight: 700;">相關文句數量</div>
                      <div style="font-size: 2rem; font-weight: 800; line-height: 1.1;">{len(evidence_rows):,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with audit_timeline_tab:
            gantt_data = build_audit_gantt_data(issue, evidence_rows)
            if gantt_data.empty:
                st.info("目前沒有可呈現的稽核待辦或里程碑。")
            else:
                gantt_chart = (
                    alt.Chart(gantt_data)
                    .mark_bar(size=18)
                    .encode(
                        x=alt.X(
                            "start:T",
                            title="季度",
                            axis=alt.Axis(format="%Y Q%q", tickCount={"interval": "month", "step": 3}, labelAngle=0),
                        ),
                        x2="end:T",
                        y=alt.Y("task:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
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
                            alt.Tooltip("status:N", title="狀態"),
                            alt.Tooltip("start_quarter:N", title="開始季度"),
                            alt.Tooltip("end_quarter:N", title="結束季度"),
                        ],
                    )
                    .properties(width=860, height=180)
                    .resolve_scale(y="independent")
                )
                gantt_col, _ = st.columns([7, 1])
                with gantt_col:
                    st.altair_chart(gantt_chart, use_container_width=False)
                st.dataframe(
                    gantt_data[["group", "task", "status", "severity", "start_quarter", "end_quarter"]],
                    use_container_width=True,
                    hide_index=True,
                )

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
