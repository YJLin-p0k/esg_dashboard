from __future__ import annotations

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
RADAR_CHART_SIZE = 720
RADAR_LEGEND_FONT_SIZE = 20
RADAR_LEGEND_TITLE_FONT_SIZE = 21
RADAR_LEGEND_SYMBOL_SIZE = 260
RADAR_AXIS_LABEL_FONT_SIZE_MIN = 17
RADAR_AXIS_LABEL_FONT_SIZE_MAX = 24
RADAR_AXIS_LABEL_FONT_SIZE_FALLBACK = 22
RADAR_AXIS_LABEL_RADIUS = 150
RADAR_AXIS_LABEL_LINE_HEIGHT = 26
RADAR_CHART_PADDING = {"left": 150, "right": 150, "top": 110, "bottom": 170}

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

NORMALIZED_TOPIC_KEYWORDS = {
    category: {
        topic: tuple(keyword.lower() for keyword in keywords)
        for topic, keywords in topics.items()
    }
    for category, topics in TOPIC_KEYWORDS.items()
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
    ("evidence_status", "證據狀態分布", "是否提供佐證資料"),
    ("evidence_quality", "證據品質分布", "佐證是否清楚可信"),
    ("verification_timeline", "驗證時程分布", "承諾時程或完成狀態"),
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
