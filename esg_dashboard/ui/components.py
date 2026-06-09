from __future__ import annotations

from html import escape

import streamlit as st


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

        .stApp[data-theme="dark"] .vega-embed svg text,
        .stApp[data-theme="dark"] div[data-testid="stVegaLiteChart"] svg text {
            fill: #f8fafc !important;
        }

        .stApp[data-theme="dark"] .vega-embed svg .role-axis-title text,
        .stApp[data-theme="dark"] .vega-embed svg .role-legend-title text,
        .stApp[data-theme="dark"] div[data-testid="stVegaLiteChart"] svg .role-axis-title text,
        .stApp[data-theme="dark"] div[data-testid="stVegaLiteChart"] svg .role-legend-title text {
            fill: #e2e8f0 !important;
        }

        .stApp[data-theme="dark"] .vega-embed svg .role-axis path,
        .stApp[data-theme="dark"] .vega-embed svg .role-axis line,
        .stApp[data-theme="dark"] div[data-testid="stVegaLiteChart"] svg .role-axis path,
        .stApp[data-theme="dark"] div[data-testid="stVegaLiteChart"] svg .role-axis line {
            stroke: #64748b !important;
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

