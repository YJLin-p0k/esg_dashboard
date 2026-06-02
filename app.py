from __future__ import annotations

from html import escape
import importlib
from io import BytesIO, StringIO
import math
from pathlib import Path
import re
from typing import Literal, cast

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from esg_dashboard.hybrid_model import ESG_CATEGORIES, HybridESGAnalyzer, OFFICIAL_WEIGHTS
from esg_dashboard import pdf_utils
from esg_dashboard.text_utils import split_chinese_sentence_units


def process_pdf_chunks(pdf_source):
    """Load the PDF processor lazily so Streamlit cannot keep a stale symbol."""
    if not hasattr(pdf_utils, "process_pdf"):
        importlib.reload(pdf_utils)
    return pdf_utils.process_pdf(pdf_source)


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

TRUST_LOW_THRESHOLD = 35
TRUST_STABLE_THRESHOLD = 70
TRUST_COLOR_LOW = "#d84a3a"
TRUST_COLOR_MEDIUM = "#f2b84b"
TRUST_COLOR_HIGH = "#2f9e62"
TRUST_GAUGE_BLEND_DEGREES = 18
CHART_LABEL_FONT_SIZE = 15
CHART_TITLE_FONT_SIZE = 17
CHART_LEGEND_FONT_SIZE = 14
CHART_LEGEND_SYMBOL_SIZE = 130
TASK_PIE_LEGEND_FONT_SIZE = 22
TASK_PIE_LEGEND_SYMBOL_SIZE = 330
RADAR_CHART_SIZE = 600
RADAR_AXIS_LABEL_FONT_SIZE_MIN = 17
RADAR_AXIS_LABEL_FONT_SIZE_MAX = 24
RADAR_AXIS_LABEL_FONT_SIZE_FALLBACK = 22
RADAR_AXIS_LABEL_RADIUS = 134
RADAR_AXIS_LABEL_LINE_HEIGHT = 26
RADAR_CHART_PADDING = {"left": 128, "right": 128, "top": 92, "bottom": 132}

# Fixed ESG issue taxonomy. The dashboard only keeps sentences that match one of
# these topics, so it no longer invents or extracts similar ad-hoc themes.
TOPIC_KEYWORDS = {
    "Environment": {
        "氣候變遷": [
            "climate change", "global warming", "extreme weather", "greenhouse gas", "ghg", "scope 1",
            "scope 2", "scope 3", "carbon", "emission", "emissions", "co2e", "net zero", "sbti",
            "氣候變遷", "全球暖化", "極端氣候", "溫室氣體", "範疇一", "範疇二", "範疇三", "碳排",
            "碳排放", "碳排放強度", "減碳", "淨零", "碳中和", "科學基礎減碳", "氣候風險",
            "脫碳", "碳管理",
        ],
        "天然資源": [
            "natural resource", "energy", "water", "wastewater", "land use", "raw material",
            "biodiversity", "forest", "fsc", "electricity", "kwh", "mwh", "gwh", "天然資源",
            "能源", "總能源", "用電", "耗電", "節能", "能效", "能源效率", "水資源", "取水",
            "耗水", "用水", "缺水", "廢水", "水質", "回收水", "循環用水", "土地", "土地使用",
            "原物料", "生物多樣性", "森林", "保育", "棲地",
        ],
        "污染濫用": [
            "waste", "hazardous waste", "recycle", "recycling", "pollution", "pollutant", "sox", "nox",
            "pm", "cod", "resource waste", "廢棄物", "有害廢棄物", "無害廢棄物", "回收", "再利用",
            "掩埋", "污染", "污染防治", "空氣污染", "水污染", "土壤污染", "硫氧化物", "氮氧化物",
            "懸浮微粒", "化學需氧量", "資源浪費", "濫用", "減量",
        ],
        "環境機會": [
            "environmental opportunity", "green technology", "clean technology", "renewable", "solar",
            "wind", "green power", "circular economy", "eco-design", "green product", "sustainable product",
            "環境機會", "綠色科技", "綠色技術", "潔淨技術", "再生能源", "太陽能", "風力", "綠電",
            "循環經濟", "生態設計", "環保產品", "綠色產品", "永續產品", "永續採購", "包材",
            "包裝", "可回收", "綠色投資", "綠能",
        ],
    },
    "Social": {
        "人權": [
            "human rights", "forced labor", "child labor", "modern slavery", "discrimination",
            "harassment", "freedom of association", "人權", "基本人權", "童工", "強迫勞動", "歧視",
            "騷擾", "剝削", "結社自由", "供應鏈人權", "人權風險", "弱勢族群",
        ],
        "勞工": [
            "labor", "employee", "workforce", "compensation", "salary", "wage", "benefit", "turnover",
            "parental leave", "retention", "diversity", "equity", "inclusion", "dei", "occupational safety",
            "health and safety", "iso 45001", "injury", "fr", "sr", "勞工", "員工", "薪酬", "薪資",
            "工資", "福利", "離職率", "留任率", "育嬰留停", "復職率", "基本工資", "多元",
            "公平", "包容", "女性員工", "女性主管", "性別差距", "身心障礙", "原住民", "職業安全",
            "職業衛生", "職安", "工安", "安全衛生", "職災", "失能傷害", "健康檢查", "勞動權益",
            "職場安全",
        ],
        "股東": [
            "shareholder", "investor", "minority shareholder", "shareholder rights", "shareholder meeting",
            "electronic voting", "dividend", "股東", "投資人", "股東權益", "少數股東", "股東會",
            "電子投票", "逐案表決", "股利", "公平對待", "投資人關係",
        ],
        "社會機會": [
            "social opportunity", "community", "volunteer", "philanthropy", "charity", "education support",
            "community development", "social impact", "public welfare", "社會機會", "社會公益",
            "公益", "慈善", "志工", "志願服務", "社區", "在地", "教育支持", "助學", "社區發展",
            "社會影響", "社會參與", "社會責任",
        ],
    },
    "Governance": {
        "公司治理": [
            "corporate governance", "board", "director", "independent director", "chairman", "ceo",
            "disclosure", "transparency", "annual report", "sustainability report", "risk management",
            "risk control", "material risk", "cyber risk", "bcp", "business continuity", "compliance",
            "internal control", "audit", "公司治理", "董事會", "董事", "獨立董事", "董事長",
            "總經理", "職責分離", "外部評估", "專業多元", "揭露", "透明", "財報", "年報",
            "永續報告", "英文版", "風險管理", "風險控管", "重大性風險", "資安風險", "鑑別",
            "因應流程", "營運持續", "合規", "法遵", "內控", "內部控制", "稽核",
        ],
        "公司行為": [
            "corporate behavior", "ethics", "integrity", "anti-corruption", "anti-bribery",
            "whistleblower", "antitrust", "fair competition", "supplier", "supply chain", "supplier audit",
            "local procurement", "公司行為", "商業道德", "誠信", "誠信經營", "反貪腐", "反賄賂",
            "舉報", "檢舉", "投訴", "反壟斷", "公平競爭", "訴訟", "法規遵循", "法令遵循",
            "供應鏈", "供應商", "高風險供應商", "esg 評鑑", "實地稽核", "在地採購",
            "關鍵零組件", "供應鏈治理",
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

TOPIC_DESCRIPTIONS = {
    "氣候變遷": "減碳、淨零與氣候風險",
    "天然資源": "能源、水與生物多樣性",
    "污染濫用": "廢棄物、污染與減量",
    "環境機會": "綠能、循環與綠色產品",
    "人權": "人權保障與反歧視",
    "勞工": "薪酬福利與職安",
    "股東": "股東權益與投資人溝通",
    "社會機會": "公益、社區與教育支持",
    "公司治理": "董事會、風控與透明揭露",
    "公司行為": "誠信、法遵與供應鏈",
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

PEER_CATEGORY_LABELS = {
    "Environment": "Environment 環境信任分數",
    "Social": "Social 社會信任分數",
    "Governance": "Governance 治理信任分數",
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
    "page": "頁數",
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

TASK_PIE_CHARTS = [
    ("promise_status", "承諾判定分布", "是否出現明確承諾或目標"),
    ("verification_timeline", "驗證時程分布", "承諾時程或完成狀態"),
    ("evidence_status", "證據狀態分布", "是否提供佐證資料"),
    ("evidence_quality", "證據品質分布", "佐證是否清楚可信"),
]

TASK_PIE_ORDER = {
    "promise_status": ["有", "無", "未判定"],
    "evidence_status": ["有", "無", "未需佐證"],
    "evidence_quality": ["清楚", "不清楚", "可能誤導", "無證據可評"],
    "verification_timeline": ["已完成或可驗證", "2 年內", "2 到 5 年", "超過 5 年", "未說明時程"],
}

TASK_PIE_LABELS = {
    "promise_status": {"N/A": "未判定"},
    "evidence_status": {"N/A": "未需佐證"},
    "evidence_quality": {"N/A": "無證據可評"},
    "verification_timeline": {"N/A": "未說明時程"},
}

TASK_PIE_COLORS = {
    "有": TRUST_COLOR_HIGH,
    "無": TRUST_COLOR_LOW,
    "未判定": "#94a3b8",
    "未需佐證": "#94a3b8",
    "無證據可評": "#94a3b8",
    "清楚": TRUST_COLOR_HIGH,
    "不清楚": TRUST_COLOR_MEDIUM,
    "可能誤導": TRUST_COLOR_LOW,
    "已完成或可驗證": TRUST_COLOR_HIGH,
    "2 年內": "#3b82f6",
    "2 到 5 年": TRUST_COLOR_MEDIUM,
    "超過 5 年": TRUST_COLOR_LOW,
    "未說明時程": "#94a3b8",
}


if get_script_run_ctx() is None:
    print("This is a Streamlit app. Start it with: python -m streamlit run app.py")
    raise SystemExit(0)


# Streamlit setup
st.set_page_config(page_title="ESG Sentinal 綠色哨兵", page_icon="📊", layout="wide")


def inject_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, .stApp {
            font-size: clamp(17px, 0.8vw + 13px, 22px);
        }

        .stApp {
            background: var(--background-color, Canvas);
            color: var(--text-color, CanvasText);
        }

        div[data-testid="stAppViewContainer"] .block-container {
            max-width: min(100%, 1660px);
            padding-top: clamp(2.2rem, 2.5vw, 3rem);
            padding-bottom: 3rem;
            padding-left: clamp(1rem, 2vw, 2.4rem);
            padding-right: clamp(1rem, 2vw, 2.4rem);
        }

        h1 {
            font-size: clamp(2rem, 2vw + 1.35rem, 3.25rem);
            line-height: 1.18;
        }

        h2 {
            font-size: clamp(1.55rem, 1.25vw + 1.1rem, 2.35rem);
            line-height: 1.25;
        }

        h3 {
            font-size: clamp(1.25rem, 0.85vw + 1rem, 1.8rem);
            line-height: 1.3;
        }

        div[data-testid="stMetricValue"] {
            font-size: clamp(1.9rem, 1.45vw + 1rem, 2.9rem);
            line-height: 1.05;
        }

        div[data-testid="stMetricLabel"] {
            font-size: clamp(1rem, 0.5vw + 0.82rem, 1.18rem);
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stMarkdownContainer"] span,
        div[data-testid="stMarkdownContainer"] label,
        div[data-testid="stMarkdownContainer"] div {
            font-size: inherit;
            line-height: 1.65;
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
        }

        button,
        input,
        select,
        textarea {
            font-size: inherit;
        }

        button,
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            min-height: 2.65rem;
        }

        div[data-testid="stFileUploader"] {
            font-size: clamp(0.92rem, 0.28vw + 0.84rem, 1rem);
        }

        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {
            font-size: inherit;
            line-height: 1.35;
        }

        div[data-testid="stFileUploader"] section {
            padding: 0.6rem 0.75rem;
            min-height: 4.2rem;
        }

        div[data-testid="stFileUploader"] button {
            min-height: 2.15rem;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
            font-size: 0.92rem;
        }

        div[data-testid="stTabs"] button {
            font-size: clamp(1rem, 0.45vw + 0.84rem, 1.16rem);
            min-height: 2.75rem;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] div {
            font-size: clamp(1rem, 0.42vw + 0.86rem, 1.14rem);
        }

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            min-height: 2.35rem;
            line-height: 1.45;
        }

        .stSelectbox,
        .stMultiSelect,
        .stFileUploader,
        .stTextInput,
        .stNumberInput,
        .stTextArea {
            font-size: inherit;
        }

        .vg-tooltip {
            font-size: 0.95rem !important;
            line-height: 1.45 !important;
        }

        .esg-topic-title {
            font-size: clamp(1.55rem, 0.9vw + 1.2rem, 2.25rem);
            line-height: 1.32;
            font-weight: 850;
            margin: 0.4rem 0 1rem 0;
        }

        .esg-section-heading {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.35rem 0 0.65rem 0;
        }

        .esg-section-heading h2,
        .esg-section-heading h3,
        .esg-section-heading h4 {
            margin: 0;
            line-height: 1.25;
            letter-spacing: 0;
        }

        .esg-section-help {
            position: relative;
            width: 1.45rem;
            height: 1.45rem;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--primary-color, #2563eb) 42%, transparent);
            background: var(--secondary-background-color, Canvas);
            color: var(--text-color, CanvasText);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 800;
            cursor: help;
            user-select: none;
        }

        .esg-section-tooltip {
            position: absolute;
            top: calc(100% + 0.5rem);
            left: 50%;
            width: min(28rem, calc(100vw - 2rem));
            border-radius: 8px;
            border: 1px solid color-mix(in srgb, var(--text-color, CanvasText) 18%, transparent);
            background: var(--secondary-background-color, Canvas);
            color: var(--text-color, CanvasText);
            box-shadow: 0 18px 38px color-mix(in srgb, var(--text-color, CanvasText) 22%, transparent);
            padding: 0.85rem 0.95rem;
            opacity: 0;
            transform: translate(-50%, -0.25rem);
            pointer-events: none;
            transition: opacity 0.18s ease, transform 0.18s ease;
            z-index: 50;
            text-align: left;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .esg-section-help:hover .esg-section-tooltip,
        .esg-section-help:focus .esg-section-tooltip {
            opacity: 1;
            transform: translate(-50%, 0);
        }

        .esg-section-tooltip p {
            margin: 0;
        }

        .esg-section-tooltip ul {
            margin: 0.55rem 0 0 1.1rem;
            padding: 0;
        }

        .esg-section-tooltip li {
            margin: 0.24rem 0;
            padding-left: 0.15rem;
        }

        @media (max-width: 760px) {
            div[data-testid="stHorizontalBlock"] {
                gap: 0.9rem;
            }

            .esg-section-tooltip {
                left: auto;
                right: -0.25rem;
                transform: translate(0, -0.25rem);
            }

            .esg-section-help:hover .esg-section-tooltip,
            .esg-section-help:focus .esg-section-tooltip {
                transform: translate(0, 0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_tooltip_text(help_text: str) -> str:
    parts = [part.strip() for part in help_text.split("；") if part.strip()]
    if not parts:
        return ""
    lead = f"<p>{escape(parts[0])}</p>"
    if len(parts) == 1:
        return lead
    bullets = "".join(f"<li>{escape(part)}</li>" for part in parts[1:])
    return f"{lead}<ul>{bullets}</ul>"


def render_section_heading(title: str, help_text: str, level: int = 3) -> None:
    heading_tag = "h2" if level <= 2 else "h4" if level >= 4 else "h3"
    tooltip_html = format_tooltip_text(help_text)
    st.markdown(
        f"""
        <div class="esg-section-heading">
          <{heading_tag}>{escape(title)}</{heading_tag}>
          <div class="esg-section-help" tabindex="0" aria-label="{escape(title)}說明">
            ?
            <div class="esg-section-tooltip" role="tooltip">{tooltip_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_responsive_styles()


# Data loading and analysis
@st.cache_resource
def load_analyzer() -> HybridESGAnalyzer:
    return HybridESGAnalyzer()


@st.cache_data
def load_training_peer_rows() -> pd.DataFrame:
    rows = pd.read_json(path_or_buf=str(TRAINING_DATA_PATH))
    rows["company"] = rows["company"].astype(str).str.lower()
    rows["esg_category"] = rows["esg_type"].map(ESG_TYPE_TO_CATEGORY)
    rows["evidence_quality"] = rows["evidence_quality"].replace("", "N/A").fillna("N/A")
    rows["evidence_status"] = rows["evidence_status"].replace("", "N/A").fillna("N/A")
    rows["verification_timeline"] = rows["verification_timeline"].replace("", "N/A").fillna("N/A")
    rows["promise_status"] = rows["promise_status"].replace("", "No").fillna("No")
    rows["overall_trust_score"] = rows.apply(compute_reference_trust_score, axis=1)
    rows["sentence"] = rows["data"].astype(str)
    rows["topic"] = rows.apply(lambda row: detect_topic(str(row["sentence"]), str(row["esg_category"])), axis=1)
    rows = rows[rows["topic"].notna()].copy()
    return rows


def compute_reference_trust_score(row: pd.Series) -> float:
    promise_status = str(row.get("promise_status", "No"))
    evidence_status = str(row.get("evidence_status", "N/A"))
    evidence_quality = str(row.get("evidence_quality", "N/A"))
    verification_timeline = str(row.get("verification_timeline", "N/A"))

    promise_score = 1.0 if promise_status == "Yes" else 0.55
    evidence_score = {"Yes": 1.0, "No": 0.25, "N/A": 0.55}.get(evidence_status, 0.55)
    quality_score = {"Clear": 1.0, "Not Clear": 0.35, "Misleading": 0.0, "N/A": 0.55}.get(evidence_quality, 0.55)
    timeline_score = {
        "already": 1.0,
        "within_2_years": 0.85,
        "between_2_and_5_years": 0.65,
        "longer_than_5_years": 0.35,
        "N/A": 0.55,
    }.get(verification_timeline, 0.55)

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
        chunk_df = process_pdf_chunks(BytesIO(file_bytes))
        text = "\n\n".join(chunk_df["chunk_text"].astype(str).tolist()) if not chunk_df.empty else ""
        company = detect_company(uploaded_file.name, text)
        chunk_units: list[tuple[pd.Series, object]] = []
        for _, chunk in chunk_df.iterrows():
            for unit in split_chinese_sentence_units(str(chunk["chunk_text"])):
                chunk_units.append((chunk, unit))

        sentences = [str(getattr(unit, "sentence", "")) for _, unit in chunk_units]
        predictions = analyzer.predict(sentences)

        for sentence_id, ((chunk, unit), prediction) in enumerate(zip(chunk_units, predictions), start=1):
            sentence = str(getattr(unit, "sentence", ""))
            paragraph_id = int(getattr(unit, "paragraph_id", 0) or 0)
            paragraph_text = str(getattr(unit, "paragraph_text", ""))
            paragraph_context = str(getattr(unit, "paragraph_context", ""))
            if prediction.esg_category not in ESG_CATEGORIES or prediction.esg_category == "Other":
                continue
            topic = detect_topic(sentence, prediction.esg_category)
            if topic is None:
                continue

            rows.append(
                {
                    "file_name": uploaded_file.name,
                    "company": company,
                    "page": int(chunk.get("page", 0) or 0),
                    "paragraph_id": int(chunk.get("chunk_id", paragraph_id) or paragraph_id),
                    "paragraph_text": paragraph_text,
                    "paragraph_context": str(chunk.get("context_text", paragraph_context)),
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


def format_score_metric(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f}"


def calculate_overall_trust_score(rows: pd.DataFrame) -> float | None:
    if rows.empty or "overall_trust_score" not in rows:
        return None
    return round(float(rows["overall_trust_score"].mean()), 2)


# Trust score display
def severity_from_trust(score: float) -> tuple[str, str]:
    if score < TRUST_LOW_THRESHOLD:
        return "低信任", TRUST_COLOR_LOW
    if score < TRUST_STABLE_THRESHOLD:
        return "需追蹤", TRUST_COLOR_MEDIUM
    return "穩健", TRUST_COLOR_HIGH


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


def peer_score_comment(score: float) -> str:
    if score < TRUST_LOW_THRESHOLD:
        return "信任分數偏低，建議優先檢查證據品質與時程揭露。"
    if score < TRUST_STABLE_THRESHOLD:
        return "信任分數中等，揭露基礎尚可，但仍需要追蹤證據完整性。"
    return "信任分數較穩定，承諾、證據與時程訊號大致一致。"


def calculate_esg_trust_scores(rows: pd.DataFrame) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for category in PEER_CATEGORY_LABELS:
        category_rows = rows[rows["esg_category"].eq(category)]
        scores[category] = None if category_rows.empty else round(float(category_rows["overall_trust_score"].mean()), 2)
    return scores


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
