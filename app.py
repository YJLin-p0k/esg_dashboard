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

# Fixed ESG issue taxonomy. The dashboard only keeps sentences that match one of
# these 16 topics, so it no longer invents or extracts similar ad-hoc themes.
TOPIC_KEYWORDS = {
    "Environment": {
        "溫室氣體排放": [
            "greenhouse gas", "ghg", "scope 1", "scope 2", "scope 3", "carbon", "emission", "emissions",
            "co2e", "net zero", "sbti", "溫室氣體", "範疇一", "範疇二", "範疇三", "碳排", "碳排放",
            "碳排放強度", "減碳", "淨零", "碳中和", "科學基礎減碳",
        ],
        "能源管理": [
            "energy", "renewable", "solar", "wind", "green power", "electricity", "kwh", "mwh", "gwh",
            "能源", "總能源", "用電", "耗電", "節能", "能效", "能源效率", "綠電", "再生能源", "太陽能", "風力",
        ],
        "水資源管理": [
            "water", "wastewater", "cod", "水資源", "取水", "耗水", "用水", "缺水", "廢水", "水質",
            "化學需氧量", "回收水", "循環用水",
        ],
        "廢棄物與污染控制": [
            "waste", "hazardous waste", "recycle", "recycling", "pollution", "sox", "nox", "pm",
            "廢棄物", "有害廢棄物", "無害廢棄物", "回收", "再利用", "掩埋", "污染", "空氣污染",
            "硫氧化物", "氮氧化物", "懸浮微粒",
        ],
        "產品生態設計": [
            "eco-design", "circular economy", "biodiversity", "fsc", "packaging", "recyclable",
            "生態設計", "循環經濟", "生物多樣性", "生態敏感", "永續採購", "包材", "包裝", "減量",
            "可回收", "原物料",
        ],
    },
    "Social": {
        "員工薪酬與福利": [
            "compensation", "salary", "wage", "benefit", "turnover", "parental leave", "retention",
            "薪酬", "薪資", "工資", "福利", "離職率", "留任率", "育嬰留停", "復職率", "基本工資",
        ],
        "多元與包容（DEI）": [
            "diversity", "equity", "inclusion", "dei", "gender pay gap", "female", "women",
            "多元", "公平", "包容", "女性員工", "女性主管", "性別差距", "身心障礙", "原住民", "少數族群",
        ],
        "職業安全衛生": [
            "occupational safety", "health and safety", "iso 45001", "injury", "fr", "sr",
            "職業安全", "職業衛生", "職安", "工安", "安全衛生", "職災", "失能傷害", "健康檢查",
        ],
        "客戶權益與產品安全": [
            "product safety", "recall", "customer satisfaction", "marketing", "advertising",
            "產品安全", "產品責任", "召回", "客戶權益", "客戶滿意", "消費者", "行銷", "廣告", "違規受罰",
        ],
        "資安與隱私保護": [
            "cybersecurity", "information security", "privacy", "data breach", "iso 27001", "personal data",
            "資安", "資訊安全", "隱私", "個資", "資料外洩", "用戶資料", "資料安全",
        ],
        "人權與社區參與": [
            "human rights", "community", "volunteer", "forced labor", "child labor",
            "人權", "童工", "強迫勞動", "社區", "公益", "志工", "在地", "供應鏈人權",
        ],
    },
    "Governance": {
        "董事會結構": [
            "board", "director", "independent director", "chairman", "ceo",
            "董事會", "董事", "獨立董事", "董事長", "總經理", "職責分離", "外部評估", "專業多元",
        ],
        "資訊透明度": [
            "disclosure", "transparency", "shareholder meeting", "electronic voting", "annual report",
            "sustainability report", "揭露", "透明", "股東會", "電子投票", "逐案表決", "財報", "年報",
            "永續報告", "英文版",
        ],
        "商業道德與誠信": [
            "ethics", "integrity", "anti-corruption", "anti-bribery", "whistleblower", "antitrust",
            "fair competition", "誠信", "商業道德", "反貪腐", "反賄賂", "舉報", "投訴", "反壟斷",
            "公平競爭", "訴訟",
        ],
        "風險控管能力": [
            "risk management", "risk control", "material risk", "climate risk", "cyber risk", "bcp",
            "business continuity", "風險管理", "風險控管", "重大性風險", "氣候風險", "資安風險",
            "鑑別", "因應流程", "營運持續",
        ],
        "供應鏈永續治理": [
            "supplier", "supply chain", "supplier audit", "local procurement",
            "供應鏈", "供應商", "高風險供應商", "esg 評鑑", "實地稽核", "在地採購", "關鍵零組件",
            "供應鏈治理",
        ],
    },
}

FIXED_ESG_TOPICS = {
    topic
    for category_topics in TOPIC_KEYWORDS.values()
    for topic in category_topics
}

ESG_TOPIC_GROUP_LABELS = {
    "Environment": "E 環境",
    "Social": "S 社會",
    "Governance": "G 治理",
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
    "avg_confidence": "最低模型信心",
    "confidence": "模型信心",
    "evidence_count": "相關句數",
    "sentence_id": "句子編號",
    "promise_rate": "承諾比例",
    "clear_evidence_rate": "清楚證據比例",
    "representative_sentence": "代表句",
    "promise_status": "是否有承諾",
    "verification_timeline": "驗證時程",
    "evidence_status": "是否有證據",
    "evidence_quality": "證據品質",
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
            topic = detect_topic(sentence, prediction.esg_category)
            if topic is None:
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
                    "topic": topic,
                    "overall_trust_score": prediction.overall_trust_score,
                    "confidence": prediction.confidence,
                    "promise_status": prediction.promise_status,
                    "verification_timeline": prediction.verification_timeline,
                    "evidence_status": prediction.evidence_status,
                    "evidence_quality": prediction.evidence_quality,
                }
            )

    return pd.DataFrame(rows)


def detect_topic(sentence: str, category: str) -> str | None:
    topic_scores: dict[str, int] = {}
    lowered = sentence.lower()
    for topic, keywords in TOPIC_KEYWORDS.get(category, {}).items():
        topic_scores[topic] = sum(1 for keyword in keywords if keyword.lower() in lowered)

    best_topic, best_score = max(topic_scores.items(), key=lambda item: item[1], default=("", 0))
    return best_topic if best_score > 0 and best_topic in FIXED_ESG_TOPICS else None


def localize_value(value: object) -> object:
    return VALUE_LABELS.get(value, CATEGORY_LABELS.get(value, value))


def localize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    for column in localized.columns:
        if column in {"esg_category", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"}:
            localized[column] = localized[column].map(localize_value)
    return localized.rename(columns=COLUMN_LABELS)


def first_by_lowest_trust(values: pd.Series) -> object:
    return values.iloc[0] if not values.empty else ""


def build_issue_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = result_df.sort_values(
        ["file_name", "company", "esg_category", "overall_trust_score", "topic"],
        ascending=[True, True, True, True, True],
    )

    summary = (
        sorted_df.groupby(["file_name", "company", "esg_category", "topic"], as_index=False)
        .agg(
            overall_trust_score=("overall_trust_score", "mean"),
            avg_confidence=("confidence", "min"),
            evidence_count=("sentence", "count"),
            promise_rate=("promise_status", lambda values: round((values == "Yes").mean() * 100, 1)),
            clear_evidence_rate=("evidence_quality", lambda values: round((values == "Clear").mean() * 100, 1)),
            representative_sentence=("sentence", "first"),
        )
    )
    summary["_esg_order"] = summary["esg_category"].map(ESG_CATEGORY_ORDER).fillna(99)
    return (
        summary.sort_values(
            ["file_name", "_esg_order", "overall_trust_score", "topic"],
            ascending=[True, True, True, True],
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
        if bool(is_file_min.iloc[row.name]):
            styles[row.index.get_loc(trust_column)] = "color: #d84a3a; font-weight: 800;"
        return styles

    return localized.style.apply(highlight_file_min, axis=1).format(
        {
            COLUMN_LABELS["overall_trust_score"]: "{:.1f}",
            COLUMN_LABELS["evidence_count"]: "{:.0f}",
        }
    )


def format_score_metric(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f}"


def avg_trust_score(rows: pd.DataFrame) -> float | None:
    if rows.empty or "overall_trust_score" not in rows:
        return None
    return float(rows["overall_trust_score"].mean())


def severity_from_trust(score: float) -> tuple[str, str]:
    if score < 35:
        return "低信任", "#d84a3a"
    if score < 72:
        return "需追蹤", "#f2b84b"
    return "穩健", "#2f9e62"


def trust_dot(score: float) -> tuple[str, str]:
    _, color = severity_from_trust(score)
    return color, f"信任分數 {score:.1f}"


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


def build_peer_radar_data(issue: pd.Series, result_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    file_name = issue["file_name"]
    company = str(issue.get("company", "unknown")).lower()
    report_rows = result_df[result_df["file_name"].eq(file_name)]
    category_labels = {
        "Environment": "E 信任分數",
        "Social": "S 信任分數",
        "Governance": "G 信任分數",
    }
    training_rows = load_training_peer_rows()
    peer_group = get_peer_group(company)
    if peer_group:
        peer_rows = training_rows[training_rows["company"].isin(peer_group) & ~training_rows["company"].eq(company)]
    else:
        peer_rows = training_rows

    if peer_rows.empty:
        peer_rows = training_rows

    baseline_note = f"同業比較基準：{peer_rows['company'].nunique()} 家公司"

    peer_scores: dict[str, float] = {}
    for category, label in category_labels.items():
        category_rows = peer_rows[peer_rows["esg_category"].eq(category)]
        peer_scores[label] = 0 if category_rows.empty else round(float(category_rows["overall_trust_score"].mean()), 2)

    report_scores: dict[str, float] = {}
    for category, label in category_labels.items():
        category_rows = report_rows[report_rows["esg_category"].eq(category)]
        report_scores[label] = 0 if category_rows.empty else round(float(category_rows["overall_trust_score"].mean()), 2)

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
        rows.append(
            {
                "axis": axis,
                "label_x": math.cos(angle) * 122,
                "label_y": math.sin(angle) * 122,
            }
        )
    return rows


def build_peer_radar_grid(axis_count: int = 3) -> list[dict[str, object]]:
    return build_radar_grid(axis_count=axis_count)


def build_peer_radar_axis_labels(axes: list[str]) -> list[dict[str, object]]:
    return build_radar_axis_labels(axes)


def render_peer_comparison(issue: pd.Series, result_df: pd.DataFrame) -> None:
    peer_data, baseline_note = build_peer_radar_data(issue, result_df)
    axes = peer_data[peer_data["order"].lt(3)]["axis"].drop_duplicates().tolist()
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
        alt.Chart(pd.DataFrame(build_peer_radar_grid(axis_count=len(axes))))
        .mark_line(color="#cbd5e1", strokeWidth=1)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=chart_domain), sort=None),
            detail="score:N",
            order="order:Q",
        )
        .properties(width=radar_width, height=radar_height)
    )
    axis_label_rows = build_peer_radar_axis_labels(axes)
    axis_label_data = pd.DataFrame(axis_label_rows)
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
        paragraph_rows.append(
            {
                "file_name": paragraph_df["file_name"].iloc[0],
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


def build_audit_gantt_data(issue: pd.Series, evidence_rows: pd.DataFrame) -> pd.DataFrame:
    current_quarter = quarter_start(date.today())
    tasks: list[dict[str, object]] = []

    audit_rows = build_audit_feed(issue, evidence_rows)
    for offset, audit_row in audit_rows.iterrows():
        status = str(audit_row["status"])
        severity = "穩健" if status == "通過" else ("低信任" if status in {"需複核", "需補件"} else "需追蹤")
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
        "already": ("已完成或可驗證", add_quarters(current_quarter, -1), current_quarter, "穩健"),
        "within_2_years": ("近期檢查點", current_quarter, add_quarters(current_quarter, 4), "需追蹤"),
        "between_2_and_5_years": ("中期里程碑", add_quarters(current_quarter, 4), add_quarters(current_quarter, 8), "需追蹤"),
        "longer_than_5_years": ("長期承諾", add_quarters(current_quarter, 8), add_quarters(current_quarter, 12), "低信任"),
        "N/A": ("未說明時程", current_quarter, add_quarters(current_quarter, 1), "低信任"),
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

    if trust < 55:
        verdict = "高優先複核：這個議題的信任分數偏低，建議優先檢查承諾、佐證與驗證時程。"
    elif trust < 72:
        verdict = "建議追蹤：信任分數中等，仍需要補強證據清楚度或時程說明。"
    else:
        verdict = "目前信任分數較穩定，相關承諾、證據與時程大致一致。"

    st.write(verdict)
    st.write(f"此議題群組中的最低信任分數為 `{trust:.1f}`。")
    st.caption(str(issue["representative_sentence"]))

def render_topic_selector(issue_df: pd.DataFrame) -> tuple[str, str] | None:
    available_topics = {
        (str(row["esg_category"]), str(row["topic"]))
        for _, row in issue_df.iterrows()
    }
    topic_scores = (
        issue_df.groupby(["esg_category", "topic"], as_index=False)["overall_trust_score"]
        .mean()
        .set_index(["esg_category", "topic"])["overall_trust_score"]
        .to_dict()
    )
    selected = st.session_state.get("selected_esg_topic")
    if selected not in available_topics:
        # Explicit selection order: try all Environment topics first (in defined order),
        # then Social, then Governance. Pick the first topic that has any matches.
        def _pick_default(available: set[tuple[str, str]]):
            for category in ("Environment", "Social", "Governance"):
                for topic in TOPIC_KEYWORDS.get(category, {}).keys():
                    if (category, topic) in available:
                        return (category, topic)
            return None

        selected = _pick_default(available_topics)
        st.session_state["selected_esg_topic"] = selected

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
                    row_cols = st.columns([20, 1])
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
                        if exists:
                            dot_color, dot_title = trust_dot(float(topic_scores.get((category, topic), 0.0)))
                            st.markdown(
                                f'<div title="{dot_title}" style="text-align:right; padding-top: 0.45rem; line-height: 1;">'
                                f'<span style="display:inline-block; color:{dot_color}; font-size:0.82rem; line-height:1;">●</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown("&nbsp;", unsafe_allow_html=True)

    return st.session_state.get("selected_esg_topic")


def render_issue_detail(issue: pd.Series, result_df: pd.DataFrame) -> None:
    evidence_rows = result_df[
        (result_df["file_name"] == issue["file_name"])
        & (result_df["esg_category"] == issue["esg_category"])
        & (result_df["topic"] == issue["topic"])
    ].sort_values("overall_trust_score", ascending=True)

    render_trust_gauge(float(issue["overall_trust_score"]))

    metric_row = st.columns(3)
    metric_row[0].metric("保守信任分數", f"{issue['overall_trust_score']:.1f}")
    metric_row[1].metric("承諾比例", f"{issue['promise_rate']:.1f}%")
    metric_row[2].metric("清楚證據比例", f"{issue['clear_evidence_rate']:.1f}%")

    audit_timeline_tab, ai_tab, related_tab = st.tabs(
        [
            "稽核與時程",
            "AI 評語",
            "相關段落",
        ]
    )

    with audit_timeline_tab:
        gantt_data = build_audit_gantt_data(issue, evidence_rows)
        if gantt_data.empty:
            st.info("此議題目前沒有可整理的稽核或時程資訊。")
        else:
            gantt_chart = (
                alt.Chart(gantt_data)
                .mark_bar(size=18)
                .encode(
                    x=alt.X(
                        "start:T",
                        title="開始",
                        axis=alt.Axis(format="%Y Q%q", tickCount={"interval": "month", "step": 3}, labelAngle=0),
                    ),
                    x2="end:T",
                    y=alt.Y("task:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                    color=alt.Color(
                        "severity:N",
                        scale=alt.Scale(
                            domain=["低信任", "需追蹤", "穩健"],
                            range=["#d84a3a", "#f2b84b", "#2f9e62"],
                        ),
                        title="狀態",
                    ),
                    row=alt.Row("group:N", title=None, header=alt.Header(labelAngle=0, labelFontWeight="bold")),
                    tooltip=[
                        alt.Tooltip("group:N", title="類別"),
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
            localize_dataframe(
                related_paragraphs[
                    [
                        "paragraph_id",
                        "sentence_id",
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


def to_csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    localize_dataframe(df).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def uploaded_file_signature(uploaded_files) -> tuple[tuple[str, int], ...]:
    return tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files)


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def render_upload_confirmation(uploaded_files, show_file_table: bool = True) -> tuple[tuple[str, int], ...]:
    signature = uploaded_file_signature(uploaded_files)
    total_size = format_file_size(sum(size for _, size in signature))

    st.write(f"已選擇 `{len(uploaded_files)}` 份 PDF，總大小 `{total_size}`。")
    if show_file_table:
        render_uploaded_file_table(signature)

    confirmed_signature = st.session_state.get("confirmed_upload_signature")
    if confirmed_signature != signature:
        st.info("請確認上傳檔案無誤後，再開始分析。")
        if st.button("確認並開始分析", type="primary"):
            st.session_state["confirmed_upload_signature"] = signature
            st.rerun()
        st.stop()

    st.success("PDF 已確認，開始進行分析。")
    return signature


def render_uploaded_file_table(signature: tuple[tuple[str, int], ...]) -> None:
    st.dataframe(
        pd.DataFrame(
            [
                {"檔案": file_name, "大小": format_file_size(size)}
                for file_name, size in signature
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


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
    upload_signature = upload_signature if 'upload_signature' in locals() else render_upload_confirmation(uploaded_files, show_file_table=False)
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

result_df = result_df[result_df["file_name"].eq(selected_file_name)].copy()
issue_df = all_issue_df[all_issue_df["file_name"].eq(selected_file_name)].copy()

st.markdown("<hr>", unsafe_allow_html=True)

metric_cols = st.columns(4)
metric_cols[0].metric("整體信任分數", format_score_metric(avg_trust_score(issue_df)))
metric_cols[1].metric("E 信任分數", format_score_metric(avg_trust_score(issue_df[issue_df["esg_category"].eq("Environment")])))
metric_cols[2].metric("S 信任分數", format_score_metric(avg_trust_score(issue_df[issue_df["esg_category"].eq("Social")])))
metric_cols[3].metric("G 信任分數", format_score_metric(avg_trust_score(issue_df[issue_df["esg_category"].eq("Governance")])))

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
render_peer_comparison(issue_df.iloc[0], result_df)

st.subheader("16 項 ESG 議題")
selected_topic = render_topic_selector(issue_df)

if selected_topic is None:
    st.info("此報告書沒有命中固定 16 項 ESG 議題。")
else:
    selected_category, selected_topic_name = selected_topic
    selected_issues = issue_df[
        issue_df["esg_category"].eq(selected_category)
        & issue_df["topic"].eq(selected_topic_name)
    ].sort_values(["overall_trust_score", "file_name"], ascending=[True, True])

    st.markdown(
        f"**{ESG_TOPIC_GROUP_LABELS.get(selected_category, selected_category)} / {selected_topic_name}**"
    )

    for index, issue in selected_issues.reset_index(drop=True).iterrows():
        if len(selected_issues) > 1:
            with st.expander(
                f"{issue['file_name']} - 信任分數 {issue['overall_trust_score']:.1f}",
                expanded=index == 0,
            ):
                render_issue_detail(issue, result_df)
        else:
            render_issue_detail(issue, result_df)
