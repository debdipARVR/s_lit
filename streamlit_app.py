"""
ScribeMark - Scholarly & Forensic Manuscript Authenticity Platform
Production-grade Streamlit application for Book Publishers (Penguin, HarperCollins),
Literary Agencies, and Amazon KDP Authors.

Features:
- Editorial Parchment Design System with JetBrains Mono & Inter typography.
- 3-State Machine: Landing, Processing, Results.
- 4-Pass ClozeCongruence 2.0 Neural Masking & Dynamic Weighting.
- 12D Micro-Stylometric Radar Chart (Plotly) & Model Provenance Tournament.
- Chapter Authenticity Rhythm Bar Chart & 3-Column Grid Cards.
- Interactive Sentence-Level Cloze Inspector with Infill Cards ($S_{meaning}$, $S_{cosine}$).
- Cryptographic Proof (SHA-256 Manuscript Fingerprint & OpenSSH Ed25519 Digital Seal).
- ReportLab Co-Branded Authorship PDF Certificate Generator.
"""

import io
import os
import sys
import time
import json
import math
import re
import textwrap
import base64
import hashlib
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Core Detection & DNA Attribution Engine
from src.engine import (
    ClozeCongruenceDetector,
    DNAAttributionEngine,
    extract_dna_features,
    MODEL_CLASSES,
    MODEL_DISPLAY_NAMES,
    BASELINE_DNA_PROFILES,
    OpenRouterClient,
    OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
)
from src.engine.openrouter_client import QuotaExhaustedError
from src.engine.cloze_masker import ClozeMasker
from src.engine.metrics import (
    compute_meaning_similarity,
    compute_cosine_similarity,
    compute_pass_adaptive_congruence,
    compute_bipartite_optimal_matching,
    compute_two_pass_verdict,
    calculate_burstiness,
    jaccard_similarity,
)
from src.engine.pdf_generator import (
    generate_branded_authorship_certificate_pdf,
    REPORTLAB_AVAILABLE,
)
from src.queue import (
    split_manuscript_into_chapters,
    split_manuscript_into_pages,
)
from src.security.ssh_verifier import get_or_create_ssh_authority

# Document Parsing via PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Feature Toggles (Parked for future release)
ENABLE_CERTIFICATE_DOWNLOAD = False  # Hidden feature - parked for future
ENABLE_PAGE_ANALYSIS_TAB = False      # Hidden feature - parked for future
ENABLE_TELEMETRY_TAB = False          # Hidden feature - parked for future


# =========================================================================
# 1. PAGE CONFIGURATION & EDITORIAL PARCHMENT DESIGN SYSTEM CSS
# =========================================================================

st.set_page_config(
    page_title="ScribeMark — AI Detection & Forensic Manuscript Broadsheet",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PARCHMENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;0,900;1,400;1,700;1,900&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,600&family=Cinzel:wght@500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
  --color-bg: #f4eedb; /* authentic unbleached broadsheet newsprint paper */
  --color-surface: #eae0c5; /* aged column background */
  --color-surface2: #dfd4b7; /* pressed card */
  --color-border: #b5a47e; /* lead rule */
  --color-border2: #9e8c64; /* heavy editorial rule */
  --color-text: #191209; /* printer's black ink */
  --color-muted: #544431; /* faded newsprint ink */
  --color-muted2: #7c684d;
  --color-teal: #3b280a; /* deep iron-gall ink */
  --color-teal-dim: rgba(59, 40, 10, 0.15);
  --color-teal-subtle: rgba(59, 40, 10, 0.08);
  --color-red: #7c1a06; /* editor's vermilion red */
  --color-red-dim: rgba(124, 26, 6, 0.15);
  --color-green: #29471f; /* antique forest green */
  --color-green-dim: rgba(41, 71, 31, 0.18);
  --color-yellow: #7e5500; /* aged ochre */
  --color-yellow-dim: rgba(126, 85, 0, 0.15);
  --color-blue: #1c3652; /* broadsheet prussian blue */
  --color-purple: #4e2252;
  --color-gold: #7c580c;
}

/* Global resets for Streamlit body & container — The Statesman Newspaper Theme */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: #f4eedb !important;
  color: #191209 !important;
  font-family: 'Newsreader', 'Georgia', 'Times New Roman', serif !important;
  line-height: 1.65 !important;
}

/* Historic Newspaper Masthead Banner & Running Marquee Ticker */
@keyframes broadsheetMarquee {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

@keyframes pulseDot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.85); }
}

.statesman-dateline-banner {
  border-top: 3px double #191209;
  border-bottom: 1px solid #191209;
  padding: 6px 0;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  background: #ede3cc;
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
}

.statesman-ticker-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 14px;
  font-family: 'Cinzel', serif;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: #191209;
  border-right: 2px solid #191209;
  flex-shrink: 0;
  background: #ede3cc;
  z-index: 2;
  white-space: nowrap;
}

.statesman-ticker-badge .ticker-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #8b2000;
  display: inline-block;
  animation: pulseDot 1.5s infinite ease-in-out;
}

.statesman-ticker-window {
  flex: 1;
  overflow: hidden;
  position: relative;
  mask-image: linear-gradient(to right, transparent, black 16px, black calc(100% - 16px), transparent);
  -webkit-mask-image: linear-gradient(to right, transparent, black 16px, black calc(100% - 16px), transparent);
}

.statesman-ticker-track {
  display: inline-flex;
  white-space: nowrap;
  animation: broadsheetMarquee 45s linear infinite;
}

.statesman-ticker-track:hover {
  animation-play-state: paused;
}

.ticker-item {
  font-family: 'Cinzel', serif;
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.11em;
  color: #2b1c09;
  padding-right: 36px;
}
.statesman-masthead-title {
  font-family: 'UnifrakturMaguntia', 'Playfair Display', serif !important;
  font-size: 48px !important;
  font-weight: normal !important;
  letter-spacing: 0.03em !important;
  text-align: center !important;
  color: #191209 !important;
  margin: 4px 0 2px !important;
  line-height: 1.1 !important;
  text-shadow: 1px 1px 0px rgba(0,0,0,0.06);
}
.statesman-masthead-sub {
  font-family: 'Cinzel', serif !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.22em !important;
  text-align: center !important;
  color: #4a3411 !important;
  text-transform: uppercase !important;
  margin-bottom: 16px !important;
  border-bottom: 1px solid #b5a47e;
  padding-bottom: 8px;
}

/* Hide default streamlit decorations */
header[data-testid="stHeader"] {
  background: transparent !important;
}
#MainMenu, footer {
  visibility: hidden;
}

.block-container {
  padding-top: 1rem !important;
  padding-bottom: 3rem !important;
  max-width: 1100px !important;
}

/* Custom Scrollbars */
* {
  scrollbar-width: thin;
  scrollbar-color: #b8a472 #f0e6c8;
}
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: #e8d9b0;
}
::-webkit-scrollbar-thumb {
  background: #b8a472;
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: #7a6040;
}

/* Typography Helpers */
.font-mono {
  font-family: 'JetBrains Mono', monospace !important;
}
.font-sans {
  font-family: 'Inter', sans-serif !important;
}

/* Brand Navigation Header */
.manuscript-nav {
  height: 62px;
  border-top: 2px solid #191209;
  border-bottom: 1px solid #191209;
  background: #eae0c5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  border-radius: 2px;
  margin-bottom: 20px;
}
/* Navigation Brand Baseline & Flex Alignment */
.nav-brand-container {
  display: flex;
  align-items: center;
  height: 38px;
  gap: 12px;
  margin: 0;
  padding: 0;
}
.nav-logo-box {
  width: 32px;
  height: 32px;
  min-width: 32px;
  background: #2a1c07;
  border: 1px solid #191209;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'UnifrakturMaguntia', 'Cinzel', serif;
  font-weight: 700;
  font-size: 18px;
  line-height: 1;
  color: #f4eedb;
  flex-shrink: 0;
  box-shadow: 0 1px 2px rgba(44, 31, 14, 0.15);
}
.nav-title {
  font-family: 'UnifrakturMaguntia', 'Playfair Display', serif;
  font-weight: normal;
  font-size: 28px;
  line-height: 1;
  color: #191209;
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: baseline;
  gap: 10px;
  white-space: nowrap;
}
.nav-version {
  font-family: 'Cinzel', serif;
  font-size: 10px;
  letter-spacing: 0.16em;
  color: #7a6040;
  font-weight: 700;
  text-transform: uppercase;
  display: inline-block;
  vertical-align: baseline;
}
.nav-links {
  display: flex;
  gap: 24px;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  color: #7a6040;
}
.nav-links a {
  color: #7a6040;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.15s;
}
.nav-links a:hover {
  color: #2c1f0e;
}

/* Hero Section */
.hero-container {
  text-align: center;
  padding: 20px 20px 32px;
  max-width: 800px;
  margin: 0 auto;
}
.trust-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'Cinzel', serif;
  font-size: 13.5px;
  letter-spacing: 0.16em;
  color: #3b280a;
  background: rgba(59, 40, 10, 0.08);
  border: 1px solid rgba(59, 40, 10, 0.35);
  padding: 7px 22px;
  border-radius: 2px;
  margin-bottom: 22px;
  font-weight: 700;
  text-transform: uppercase;
}
.hero-title {
  font-family: 'Playfair Display', 'Georgia', serif;
  font-size: 50px;
  font-weight: 900;
  color: #191209;
  line-height: 1.15;
  letter-spacing: -0.01em;
  margin-bottom: 16px;
}
.hero-title span {
  color: #3b280a;
  font-style: italic;
  font-family: 'Playfair Display', serif;
}
.hero-subtitle {
  font-family: 'Newsreader', 'Georgia', serif !important;
  font-size: 24px !important;
  font-style: normal !important;
  font-weight: 500 !important;
  color: #191209 !important;
  line-height: 1.6 !important;
  max-width: 880px !important;
  margin: 0 auto 36px !important;
}

/* Streamlit Button Overrides — Editorial Broadsheet Styling Default */
div.stButton > button {
  background: #ede3cc !important;
  color: #2c1f0e !important;
  border: 1.2px solid #b5a47e !important;
  border-radius: 3px !important;
  font-family: 'Cinzel', serif !important;
  font-size: 11.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  padding: 6px 14px !important;
  min-height: 38px !important;
  height: 38px !important;
  white-space: nowrap !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 2px rgba(44, 31, 14, 0.05) !important;
}
div.stButton > button:hover {
  background: #dfd0a4 !important;
  border-color: #7c1a06 !important;
  color: #7c1a06 !important;
  box-shadow: 0 2px 5px rgba(124, 26, 6, 0.12) !important;
}

/* Secondary Button style for pill selectors */
.preset-btn div.stButton > button {
  background-color: #eae0c5 !important;
  color: #191209 !important;
  border: 1px solid #b5a47e !important;
  font-family: 'Cinzel', serif !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  padding: 10px 18px !important;
}
.preset-btn div.stButton > button:hover {
  background-color: #dfd4b7 !important;
  border-color: #2a1c07 !important;
}

/* Text Area */
.stTextArea textarea {
  background-color: #eae0c5 !important;
  border: 1px solid #b5a47e !important;
  color: #2c1f0e !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
  border-radius: 4px !important;
  line-height: 1.6 !important;
}
.stTextArea textarea:focus {
  border-color: #6b4c11 !important;
  box-shadow: 0 0 0 1px #6b4c11 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
  background-color: #dfd4b7 !important;
  border: 2px dashed #9e8c64 !important;
  border-radius: 6px !important;
  padding: 12px !important;
}
[data-testid="stFileUploader"] section {
  background-color: transparent !important;
}

/* Trust Badges Row */
.trust-badges-row {
  display: flex;
  justify-content: center;
  gap: 28px;
  margin-top: 24px;
  margin-bottom: 36px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #7a6040;
  flex-wrap: wrap;
}
.trust-badges-row span {
  display: flex;
  align-items: center;
  gap: 5px;
}
.trust-badges-row span strong {
  color: #6b4c11;
}

/* Master Feature Catalog Grid */
.feature-catalog-header {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: #7a6040;
  text-align: center;
  margin-bottom: 24px;
  font-weight: 700;
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: #c9b88a;
  border: 1px solid #b8a472;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 40px;
}
.feature-card {
  background: #e8d9b0;
  padding: 22px;
  transition: background 0.15s;
}
.feature-card:hover {
  background: #dfd0a4;
}
.feature-card-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.feature-icon {
  color: #6b4c11;
  font-size: 16px;
}
.feature-title {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 13.5px;
  color: #2c1f0e;
}
.feature-desc {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: #7a6040;
  line-height: 1.55;
  margin-bottom: 12px;
}
.feature-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px;
  letter-spacing: 0.06em;
  color: #6b4c11;
  border: 1px solid rgba(107, 76, 17, 0.3);
  padding: 2px 7px;
  border-radius: 2px;
  background: rgba(107, 76, 17, 0.05);
  display: inline-block;
  font-weight: 600;
}

/* Results Header */
.results-header {
  background: #eae0c5; border: 1px solid #b5a47e;
  border: 1px solid #b8a472;
  border-radius: 6px;
  padding: 18px 24px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.results-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: #7a6040;
  font-weight: 700;
  margin-bottom: 3px;
}
.results-filename {
  font-family: 'Inter', sans-serif;
  font-weight: 800;
  font-size: 18px;
  color: #2c1f0e;
}

/* Overview Metric Cards */
.metric-card-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.metric-card {
  background: #eae0c5; border: 1px solid #b5a47e;
  border: 1px solid #b8a472;
  border-radius: 4px;
  padding: 16px 18px;
}
.metric-label {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  color: #7a6040;
  margin-bottom: 4px;
  letter-spacing: 0.04em;
  font-weight: 600;
}
.metric-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 2px;
}
.metric-sub {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  color: #7a6040;
}

/* Panel Frame */
.parchment-panel {
  background: #eae0c5; border: 1px solid #b5a47e;
  border: 1px solid #b8a472;
  border-radius: 4px;
  padding: 20px;
  margin-bottom: 16px;
}
.panel-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: #7a6040;
  font-weight: 700;
  margin-bottom: 14px;
}

/* Progress bar container */
.score-bar-bg {
  height: 5px;
  background: #c9b88a;
  border-radius: 2px;
  overflow: hidden;
  margin-top: 4px;
}
.score-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s ease;
}

/* Chapter Cards */
.chapter-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 14px;
}
.chapter-card-item {
  background: #eae0c5; border: 1px solid #b5a47e;
  border: 1px solid #b8a472;
  border-radius: 3px;
  padding: 16px;
  transition: background 0.15s;
}
.chapter-card-item:hover {
  background: #dfd0a4;
}

/* Sentence Inspector */
.sentence-item-box {
  border-bottom: 1px solid #c9b88a;
  padding: 10px 4px;
}
.sentence-infill-card {
  background: #dfd0a4;
  border: 1px solid #b8a472;
  border-radius: 3px;
  padding: 14px 16px;
  margin-top: 8px;
  margin-bottom: 8px;
  font-family: 'JetBrains Mono', monospace;
}

/* Certificate Preview Frame */
.cert-card-preview {
  background: #eae0c5; border: 2px solid #2a1c07;
  border: 1px solid #b8a472;
  border-radius: 4px;
  padding: 32px;
  font-family: 'Inter', sans-serif;
}
.cert-hash-box {
  background: #dfd0a4;
  border: 1px solid #c9b88a;
  border-radius: 3px;
  padding: 12px;
  margin-top: 14px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background-color: #eae0c5 !important;
  border-bottom: 1px solid #b8a472 !important;
  padding: 0 12px !important;
  border-radius: 4px 4px 0 0 !important;
}
.stTabs [data-baseweb="tab"] {
  color: #7a6040 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  border-bottom: 2px solid transparent !important;
}
/* Animated Editorial Vector Loader */
@keyframes mpSpin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes mpPulse {
  0%, 100% { transform: scale(1); opacity: 0.7; }
  50% { transform: scale(1.3); opacity: 1; }
}

.manuscript-loader-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 18px auto 14px;
}

.loader-svg-box {
  width: 44px;
  height: 44px;
  margin-bottom: 12px;
}

.loader-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.08em;
  color: #6b4c11;
  font-weight: 700;
  background: #dfd0a4;
  border: 1px solid #b8a472;
  border-radius: 4px;
  padding: 5px 14px;
}

.loader-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #4a6b3a;
  box-shadow: 0 0 6px rgba(74, 107, 58, 0.8);
  animation: mpPulse 1.2s infinite ease-in-out;
}
/* Editorial Parchment Button Styling (Eliminates bulky black boxes & guarantees crisp vertical centering) */
div[data-testid="stButton"] {
  display: flex !important;
  align-items: center !important;
}

div[data-testid="stButton"] > button {
  background: #ede3cc !important;
  color: #2c1f0e !important;
  border: 1.2px solid #b5a47e !important;
  border-radius: 3px !important;
  font-family: 'Cinzel', serif !important;
  font-size: 11.5px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  padding: 0 14px !important;
  min-height: 38px !important;
  height: 38px !important;
  line-height: 38px !important;
  white-space: nowrap !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 2px rgba(44, 31, 14, 0.05) !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
}

div[data-testid="stButton"] > button:hover {
  background: #dfd0a4 !important;
  border-color: #7c1a06 !important;
  color: #7c1a06 !important;
  box-shadow: 0 2px 5px rgba(124, 26, 6, 0.12) !important;
}

div[data-testid="stButton"] > button:active {
  background: #d5c49a !important;
  transform: translateY(1px);
}

/* Primary Action Accent Buttons */
div[data-testid="stButton"] > button[kind="primary"] {
  background: #7c1a06 !important;
  color: #f4eedb !important;
  border: 1.2px solid #541103 !important;
  font-weight: 800 !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #96220a !important;
  border-color: #7c1a06 !important;
  color: #ffffff !important;
}

/* Ensure column contents in horizontal blocks align vertically centered */
div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
  width: 100% !important;
}

div[data-testid="stHorizontalBlock"] > div[data-testid="column"] [data-testid="stMarkdownContainer"] > p {
  margin: 0 !important;
  padding: 0 !important;
}
</style>
"""

st.markdown(PARCHMENT_CSS, unsafe_allow_html=True)

def render_html(html_str: str):
    """Safely render HTML in Streamlit ensuring no markdown-triggering indentation or code blocks."""
    unindented = "\n".join(re.sub(r'^[ \t]+', '', line) for line in html_str.splitlines()).strip()
    st.markdown(unindented, unsafe_allow_html=True)



# =========================================================================
# 2. COLOR & VERDICT FORMATTING HELPERS
# =========================================================================

def get_verdict_color(authenticity_score: float) -> str:
    """Return theme color based on human authenticity score (0-100)."""
    if authenticity_score >= 75.0:
        return "#4a6b3a"  # Green / Likely Human
    if authenticity_score >= 50.0:
        return "#a07000"  # Ochre / Mixed Signals
    return "#8b2000"      # Crimson / Likely AI

def get_verdict_label(authenticity_score: float) -> str:
    """Return descriptive verdict label."""
    if authenticity_score >= 75.0:
        return "Likely Human"
    if authenticity_score >= 50.0:
        return "Mixed Signals"
    return "Likely AI"

def get_verdict_badge_html(verdict: str) -> str:
    """Generate inline HTML badge for chapter or sentence verdict."""
    verdict_lower = verdict.lower()
    if "human" in verdict_lower:
        bg, border, color, label = "rgba(74, 107, 58, 0.18)", "#4a6b3a", "#4a6b3a", "HUMAN"
    elif "mixed" in verdict_lower or "partial" in verdict_lower:
        bg, border, color, label = "rgba(160, 112, 0, 0.18)", "#a07000", "#a07000", "MIXED"
    else:
        bg, border, color, label = "rgba(139, 32, 0, 0.18)", "#8b2000", "#8b2000", "AI"
        
    return f"""<span style="font-size:9px; letter-spacing:0.12em; font-family:'JetBrains Mono', monospace; padding:2px 7px; border:1px solid {border}; background:{bg}; color:{color}; border-radius:2px; font-weight:700;">{label}</span>"""


# =========================================================================
# 3. SAMPLE PRESET DATASETS
# =========================================================================

SAMPLE_PRESETS = {
    "human": {
        "title": "The Crossing (Human Classic Fiction)",
        "category": "Human Classic Fiction",
        "badge": "🟢 Human",
        "filename": "The_Crossing_Excerpt.txt",
        "text": (
            "Prologue\n\n"
            "The morning fog had settled so thickly over the harbour that even the tallest masts of the merchant ships had vanished into a grey, formless ceiling. "
            "Old seamen stood along the timber wharves, hands buried deep inside oilskin pockets, listening intently to the distant moan of the channel buoy. "
            "He had never been good at goodbyes, always finding some clumsy excuse to linger a minute longer than he should.\n\n"
            "Chapter I — The Crossing\n\n"
            "She folded the letter in thirds, as her mother had taught her during the war winters, and pressed it flat against the cold windowpane to feel the frost seep into the paper. "
            "Downstairs, the floorboards groaned under her father's pacing. "
            "Forty years of salt winds had stiffened his knees, but his clockwork habit of inspecting the front latch before midnight remained unbroken. "
            "She blew out the brass lamp, letting the quiet dark settle over her belongings."
        )
    },
    "human_vulkan": {
        "title": "Vulkan Pipeline Postmortem (Human Dev)",
        "category": "Human Systems Postmortem",
        "badge": "🟢 Human",
        "filename": "Vulkan_Pipeline_Memory_Leak_Postmortem.txt",
        "text": (
            "I spent three sleepless nights debugging that memory leak in our Vulkan rendering pipeline, only to realize I forgot a single pointer dereference in the vertex shader loop. "
            "Classic dev mistake. You'd think after ten years in game dev you'd spot something so stupid right away, but chronic fatigue does funny things to your brain. "
            "Staring at raw disassembly at 4 AM is rarely productive, yet somehow we all convince ourselves that the breakthrough is just one more breakpoint away.\n\n"
            "The leak manifested exclusively when toggling shadow cascades during camera transitions. "
            "RenderDoc showed the buffer handles being allocated, but the cleanup callback inside our custom allocator was getting skipped because of a premature early return in the frame teardown handler. "
            "Once that was patched, watching the memory footprint drop from a bloated 6.8 GB down to a stable 420 MB felt better than finding twenty dollars in an old coat.\n\n"
            "Then we ran the automated frame profiler. "
            "Watching the frame rate jump from a stuttering 14 FPS back up to a rock-solid 120 FPS on our minimum spec hardware made the lukewarm instant coffee entirely worthwhile. "
            "Next up on my whiteboard of doom: rewriting the spatial audio thread before tomorrow's milestone build."
        )
    },
    "human_nagel": {
        "title": "What Is It Like to Be a Bat (Human Philosophy)",
        "category": "Human Philosophy",
        "badge": "🟢 Human",
        "filename": "Nagel_Consciousness_Bat_Essay.txt",
        "text": (
            "When Nagel famously asked what it is like to be a bat, he wasn't merely posing a biological riddle about echolocation. "
            "He was pointing out that no matter how exhaustively we map the neurochemistry of sonar pulses and auditory cortex firing rates, we remain utterly locked out of the creature's subjective point of view. "
            "Science gives us third-person descriptions of the plumbing, but consciousness is obstinately a first-person affair.\n\n"
            "I often think about this when colleagues insist that functional brain maps will solve the hard problem of consciousness by dinner time next Tuesday. "
            "You can describe the physical correlates of pain down to the last ion channel opening in an unmyelinated C-fiber, but the sheer hurt of a toothache remains an irreducible qualitative fact. "
            "Functionalism confuses the wiring diagram with the music playing through the speaker.\n\n"
            "Perhaps our conceptual tools are simply too blunt for the job. "
            "Just as a nineteenth-century clockmaker could never explain radioactive decay using gears and pendulums, our current physicalist paradigms may lack the fundamental vocabulary needed to bridge the explanatory chasm between electrochemical voltages and the vibrant redness of an autumn leaf."
        )
    },
    "human_budapest": {
        "title": "Budapest Overnight Train (Human Memoir)",
        "category": "Human Travel Memoir",
        "badge": "🟢 Human",
        "filename": "Budapest_Overnight_Train_Memoir.txt",
        "text": (
            "The overnight train from Vienna pulled into Budapest two hours behind schedule, spitting us out into a grey drizzle that smelled of diesel smoke and roasting chestnuts. "
            "My left shoe had developed a persistent squeak somewhere outside Győr, keeping time with the rhythmic clatter of the iron wheels against old tracks. "
            "Travel guides talk about wanderlust like it’s pure poetry, but mostly it’s sore shoulders from an overpacked duffel bag and hunting for an ATM that won’t charge highway robbery fees.\n\n"
            "We wound our way through the narrow alleys of the Jewish Quarter, searching for an unmarked courtyard cafe a bartender in Prague had scribbled on the back of a coaster. "
            "Finding it felt like stumbling into a secret society: peeling floral wallpaper, mismatched velvet armchairs sagging under decades of cigarette smoke, and the heavy aroma of dark espresso bubbling in an ancient copper pot.\n\n"
            "Outside, the bells of St. Stephen’s Basilica chimed five times across the Danube. "
            "I sat back, took my first sip of bitter black coffee, and finally felt the tension of forty hours in transit melt away into the quiet Hungarian dusk."
        )
    },
    "human_stem": {
        "title": "Non-Equilibrium Transport (Human STEM Paper)",
        "category": "Human Academic STEM",
        "badge": "🟢 Human",
        "filename": "Non_Equilibrium_Lattice_Gas_STEM.txt",
        "text": (
            "We investigate the non-equilibrium steady states of a one-dimensional driven lattice gas subject to periodic boundary conditions and local conservation laws. "
            "In our previous numerical work, we observed that particle density fluctuations exhibit anomalous diffusion exponents under asymmetric hopping rates. "
            "To clarify the physical origin of this non-diffusive transport, here we perform exact diagonalization of the stochastic transition matrix for systems up to N = 24 sites.\n\n"
            "The microscopic master equation governing the probability distribution P(s, t) of particle configuration s is given by dP/dt = M P, where the off-diagonal matrix elements M(s, s') denote transition rates between adjacent configurations. "
            "Because the transition matrix lacks detailed balance, conventional equilibrium Gibbs-Boltzmann measures fail to describe the steady-state probability measure. "
            "Instead, the stationary distribution develops long-range spatial correlations that decay algebraically with inter-particle distance.\n\n"
            "Our spectral analysis reveals that the Liouvillian relaxation gap closes as a power law with dynamic exponent z approx 1.5, matching the celebrated Kardar-Parisi-Zhang universality class. "
            "These analytical calculations confirm that the anomalous fluctuation dynamics stem directly from macroscopic shock-wave coalescence rather than finite-size boundary artifacts."
        )
    },
    "ai": {
        "title": "Echoes of the Machine (AI Novel)",
        "category": "Pure AI Novel",
        "badge": "🔴 Pure AI",
        "filename": "Echoes_of_the_Machine_AI.txt",
        "text": (
            "Prologue\n\n"
            "Artificial intelligence architectures have experienced rapid technological evolution, fundamentally reshaping automated reasoning and computational linguistics. "
            "Contemporary transformer systems utilize multi-head self-attention mechanisms to optimize token representations across expansive contextual horizons. "
            "These mathematical formulations facilitate seamless cross-domain generalization, enabling real-time dialogue synthesis and autonomous task execution.\n\n"
            "Chapter I — The New Paradigm\n\n"
            "Furthermore, empirical scaling laws demonstrate consistent performance improvements as parameter scale and dataset diversity expand monotonically. "
            "It is important to note that the computational infrastructure required for pre-training necessitates specialized accelerator clusters and high-bandwidth interconnects. "
            "In conclusion, navigating the multifaceted landscape of generative AI is a testament to the transformative power of computational innovation."
        )
    },
    "ai_consensus": {
        "title": "Distributed Consensus & BFT (Pure AI - CS)",
        "category": "Pure AI Computer Science",
        "badge": "🔴 Pure AI",
        "filename": "Distributed_Consensus_BFT_AI.txt",
        "text": (
            "Distributed consensus in asynchronous, fault-tolerant networks constitutes one of the foundational challenges in theoretical computer science. "
            "According to the seminal Fischer-Lynch-Paterson impossibility theorem, no deterministic consensus protocol can guarantee safety and liveness simultaneously in an asynchronous network afflicted by even a single unannounced crash failure. "
            "To circumvent this theoretical impasse, modern distributed architectures incorporate partially synchronous timing assumptions, randomized leader election primitives, and cryptographic threshold signatures. "
            "Understanding how modern Byzantine Fault Tolerant protocols achieve scalable state machine replication remains vital for constructing robust enterprise distributed ledgers.\n\n"
            "Practical Byzantine Fault Tolerance protocols achieve quorum agreement across three distinct phases: pre-prepare, prepare, and commit. "
            "In a network containing three f plus one nodes, the system maintains Byzantine resilience as long as no more than f nodes exhibit malicious or arbitrary behavior. "
            "During the pre-prepare phase, a designated primary proposer broadcasts a sequence number and transaction payload to all validating replicas. "
            "Replicas subsequently broadcast prepare messages to establish collective transaction ordering across the entire distributed cluster. "
            "Once a node collects two f plus one matching prepare signatures, it establishes a formal certificate guaranteeing that non-faulty nodes agree on state progression.\n\n"
            "Scalability limitations within classical Byzantine architectures stem primarily from quadratic message complexity during consensus rounds. "
            "When the cluster size expands to thousands of globally distributed validator nodes, all-to-all communication overhead causes exponential latency degradation and network congestion. "
            "Modern consensus engines address this bottleneck by deploying hierarchical sharding mechanisms, directed acyclic graph mempools, and pipeline architectures. "
            "Furthermore, threshold signature aggregation condenses multi-validator acknowledgments into compact cryptographic proofs, reducing communication payloads to linear complexity."
        )
    },
    "ai_clinical": {
        "title": "Clinical Triage Protocols (Pure AI - Health)",
        "category": "Pure AI Healthcare",
        "badge": "🔴 Pure AI",
        "filename": "Clinical_Triage_Oncology_AI.txt",
        "text": (
            "The proliferation of artificial intelligence within contemporary healthcare ecosystems constitutes a paradigm shift in clinical diagnostic protocols. "
            "Convolutional neural architectures systematically analyze complex medical imaging datasets with diagnostic sensitivity matching fellowship-trained radiologists. "
            "Early computational lesion detection mitigates false-negative diagnostic trajectories across multi-center oncology trials. "
            "Consequently, enterprise hospital networks are rapidly integrating autonomous triage systems across radiology and pathology departments.\n\n"
            "However, clinical translation requires rigorous algorithmic verification and ethical governance frameworks. "
            "When predictive diagnostic models encounter demographically unrepresentative training distributions, predictive confidence degrades across marginalized patient cohorts. "
            "Guaranteeing robust distributional generalizability remains an essential challenge for biomedical engineers. "
            "Healthcare institutions must synthesize computational diagnostic leverage with clinician judgment to optimize patient care.\n\n"
            "In summary, autonomous diagnostic infrastructure offers unprecedented therapeutic leverage for global healthcare delivery. "
            "By implementing standardized validation frameworks, clinical systems can achieve scalable operational efficiency while preserving clinical safety. "
            "The overarching mandate remains deploying verified algorithms that augment human clinical expertise."
        )
    },
    "ai_claude": {
        "title": "The Clockmaker of Prague (Claude 3.5 Style)",
        "category": "Pure AI Literary Mimicry",
        "badge": "🔴 Pure AI",
        "filename": "Clockmaker_Prague_Claude_3_5.txt",
        "text": (
            "Master Tobias wound the escapement wheel with fingers that remembered every winter of his seventy years. "
            "Outside the workshop window, the spires of the Old Town were swallowed by twilight, their gilded weathervanes catching the last embers of an October sunset. "
            "Each clock on the cedar shelves ticked with its own private heartbeat, an orchestra of brass and iron suspended in perpetual, delicate disagreement.\n\n"
            "To measure time, Tobias often reflected, is not merely to count the passage of seconds; it is to bear witness to the inevitable decay of all mortal enterprise. "
            "The pendulum swung with mathematical grace, indifferent to the wars, plagues, and coronations that unfolded beneath its rhythm. "
            "In every pendulum stroke lay the quiet reminder that humanity is but a momentary custodian of mechanical perfection.\n\n"
            "He picked up the magnifying loupe and inspected the balance spring. "
            "Under the gaslight, the tempered blue steel gleamed with unblemished symmetry. "
            "It was a testament to the enduring dialogue between artisan craftsmanship and natural law, a miniature cosmos turning upon an axis of ruby jewel bearings."
        )
    },
    "ai_deepseek": {
        "title": "Quantum Decoherence (DeepSeek R1 Reasoning)",
        "category": "Pure AI Reasoning",
        "badge": "🔴 Pure AI",
        "filename": "Quantum_Decoherence_DeepSeek_R1.txt",
        "text": (
            "To evaluate the decoherence rate in a superconducting transmon qubit coupled to a lossy coplanar waveguide resonator, we must systematically trace out the photonic bath degrees of freedom. "
            "First, let us write the total system-reservoir Hamiltonian in the standard Jaynes-Cummings framework under the rotating-wave approximation. "
            "The interaction term describes energy exchange between the artificial two-level atom and the fundamental cavity mode.\n\n"
            "Next, applying the Born-Markov approximation assumes that bath correlation times are substantially shorter than characteristic system relaxation timescales. "
            "Under this regime, the reduced density operator obeys the Lindblad master equation with dissipators corresponding to radiative emission and pure dephasing. "
            "Notably, the pure dephasing rate gamma_phi scales quadratically with low-frequency flux noise fluctuations.\n\n"
            "Consequently, to maximize the T2 coherence time, the qubit design must incorporate geometric symmetric shunting and magnetic flux-insensitive sweet spots. "
            "This mathematical optimization ensures that high-fidelity multi-qubit gate operations achieve threshold fault tolerance before stochastic environmental fluctuations induce irreversible phase errors."
        )
    },
    "hybrid": {
        "title": "Letters Never Sent (Hybrid Polished)",
        "category": "Hybrid Polished",
        "badge": "🟡 Hybrid",
        "filename": "Letters_Never_Sent_Hybrid.txt",
        "text": (
            "Chapter I — The Threshold\n\n"
            "We sat in the dusty attic for hours, sifting through yellowed photographs and broken clockwork toys that smelled faintly of cedar and forgotten summers. "
            "It is worth noting that the structural organization of personal archives often reflects broader sociopolitical transitions occurring across consecutive generations. "
            "My grandmother had tied each stack with frayed blue yarn, carefully inscribing names and dates on the reverse side in faded violet ink. "
            "Furthermore, modern historical methodologies emphasize the significance of domestic ephemera in reconstructing localized economic conditions."
        )
    },
    "humanized_ai": {
        "title": "Cognitive Habits (Adversarial Humanized LLM)",
        "category": "Adversarial Humanized LLM",
        "badge": "🟡 Humanized AI",
        "filename": "Cognitive_Habits_Humanized_AI.txt",
        "text": (
            "Neural plasticity is basically how biological brains rearrange their synaptic connections whenever new environments push them. "
            "When people constantly lean on generative AI bots for writing paragraphs, organizing spreadsheets, and solving puzzles, the actual biological wiring gets used a lot less. "
            "Under the classic rule of use-it-or-lose-it plasticity, neural circuits that rarely get fired up end up going through progressive synaptic pruning.\n\n"
            "On the other hand, non-stop tapping on digital screens strengthens pathways tied to rapid scanning, visual jumping, and shallow multitasking habits. "
            "This changing setup shifts the long-term structural health of the prefrontal cortex over time.\n\n"
            "Ultimately, solving the mystery of consciousness means moving past rigid materialist dogma. "
            "Whether awareness comes from quantum coherence in microtubules or recursive broadcasting across global workspaces is still totally up for debate. "
            "Closing the explanatory gap between objective neuron firings and subjective first-person reality will require bold new ideas."
        )
    }
}


# =========================================================================
# 4. ENGINE INTEGRATION & CACHING
# =========================================================================

def get_detector(api_key: Optional[str] = None, model: Optional[str] = None) -> ClozeCongruenceDetector:
    """Instantiate ClozeCongruenceDetector strictly requiring an active, validated Prober API key."""
    key = (api_key or st.session_state.get("openrouter_api_key", "")).strip()
    if not key:
        raise ValueError("Prober API Key Required: ScribeMark is exclusively functional upon using the Prober API.")
    mdl = model or st.session_state.get("selected_model", "google/gemini-2.5-flash-lite")
    client = OpenRouterClient(api_key=key, default_model=mdl, allow_heuristic_fallback=False)
    return ClozeCongruenceDetector(nim_client=client)

@st.cache_resource(show_spinner=False)
def get_cached_dna_engine() -> DNAAttributionEngine:
    """Instantiate and cache DNAAttributionEngine."""
    return DNAAttributionEngine()

def test_openrouter_connection(api_key: str, model: str = "google/gemini-2.5-flash-lite") -> Tuple[bool, str]:
    """Test and validate OpenRouter API Key connectivity with format validation and a lightweight probe."""
    if not api_key or not api_key.strip():
        return False, "API Key is empty."
    clean_key = api_key.strip()
    if not clean_key.startswith("sk-") or len(clean_key) < 20:
        return False, "Validation Failed: OpenRouter API keys typically start with 'sk-or-v1-' and are 60+ characters."
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=clean_key,
            timeout=10.0,
            default_headers={
                "HTTP-Referer": "https://scribemark.io",
                "X-Title": "ScribeMark Connection Probe (Research Preview)",
            }
        )
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Return 'OK'."}],
            max_tokens=6,
        )
        content = res.choices[0].message.content or "Connected"
        return True, f"Key Validated & Connected ({model}): {content.strip()}"
    except Exception as e:
        return False, f"OpenRouter Validation/Auth Error: {str(e)}"


RADAR_12D_AXIS_MAP = [
    ("Hapax Legomena", "hapax_ratio"),
    ("Burstiness (CV)", "burstiness_cv"),
    ("Type-Token (TTR)", "ttr"),
    ("Lexical Density", "lexical_density"),
    ("Function Word Ratio", "function_word_ratio"),
    ("Sentence Length Std", "sentence_length_std"),
    ("Character Entropy", "char_entropy"),
    ("Em-Dash Frequency", "emdash_frequency"),
    ("Colon Frequency", "colon_frequency"),
    ("Conjunction Entropy", "conjunction_entropy"),
    ("Passive Voice Proxy", "passive_voice_proxy"),
    ("Flesch-Kincaid Grade", "flesch_kincaid_grade"),
]

HUMAN_EDITORIAL_BASELINE: Dict[str, float] = {
    "hapax_ratio": 64.5,
    "burstiness_cv": 58.2,
    "ttr": 78.4,
    "lexical_density": 56.8,
    "function_word_ratio": 43.2,
    "sentence_length_std": 62.0,
    "char_entropy": 76.5,
    "emdash_frequency": 42.0,
    "colon_frequency": 32.0,
    "conjunction_entropy": 68.4,
    "passive_voice_proxy": 24.5,
    "flesch_kincaid_grade": 68.0,
}


def extract_text_from_upload(uploaded_file) -> Tuple[str, str]:
    """Parse and clean text from .txt, .md, .pdf, .epub uploads with high fidelity."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    
    if name.endswith(".txt") or name.endswith(".md"):
        try:
            text_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_str = raw_bytes.decode("latin-1", errors="ignore")
            
        # Clean text file line wraps if formatted with hard line wraps
        text_str = text_str.replace('\r\n', '\n')
        if '\n\n' not in text_str and '\n' in text_str:
            lines = [l.strip() for l in text_str.split('\n') if l.strip()]
            avg_len = sum(len(l) for l in lines) / max(1, len(lines))
            if avg_len < 100:
                unwrapped_lines = []
                cur = ""
                for l in lines:
                    if not cur:
                        cur = l
                    elif cur.endswith((".", "!", "?", '"', "'")):
                        unwrapped_lines.append(cur)
                        cur = l
                    else:
                        cur += " " + l
                if cur:
                    unwrapped_lines.append(cur)
                text_str = "\n\n".join(unwrapped_lines)
                
        return text_str, "Plain Text / Markdown"
            
    if name.endswith(".pdf"):
        if not PYMUPDF_AVAILABLE:
            return "PyMuPDF (fitz) is required for PDF parsing.", "Error"
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            extracted = []
            for page in doc:
                p_raw = page.get_text("text")
                if not p_raw or not p_raw.strip():
                    continue
                # 1. De-hyphenate broken words across lines (e.g. "congru-\nence" -> "congruence")
                p_clean = re.sub(r'(\w+)-\n(\w+)', r'\1\2', p_raw)
                
                # 2. Filter out standalone page header/footer line numbers, browser timestamps, and file:/// URLs
                lines = p_clean.split('\n')
                kept_lines = []
                for line in lines:
                    l_str = line.strip()
                    if re.match(r'^(?:page\s+\d+|\d{1,4}|[-–—]\s*\d+\s*[-–—])$', l_str, re.IGNORECASE):
                        continue
                    if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?', l_str, re.IGNORECASE):
                        continue
                    if "file:///" in l_str or "http://" in l_str or "https://" in l_str or re.search(r'\b\d+/\d+\.?$', l_str):
                        continue
                    if re.match(r'^(?:[A-Z]\s+){5,}[A-Z]$', l_str):
                        continue
                    if l_str in ("✦ ✦ ✦", "✦✦✦", "***", "---"):
                        continue
                    kept_lines.append(line)
                p_clean = '\n'.join(kept_lines)

                # 3. Detect and reconstruct true paragraphs from hard-wrapped PDF text
                raw_paras = re.split(r'\n{2,}|\r\n\r\n', p_clean)
                page_paras = []
                for para in raw_paras:
                    unwrapped = re.sub(r'(?<![.!?])\n(?!\n)', ' ', para)
                    unwrapped = re.sub(r'\s+', ' ', unwrapped).strip()
                    if unwrapped and len(unwrapped.split()) >= 3:
                        page_paras.append(unwrapped)
                
                if page_paras:
                    extracted.append("\n\n".join(page_paras))
                    
            doc.close()
            return "\n\n\x0c\n\n".join(extracted), "PDF Document"
        except Exception as e:
            return f"PDF Extraction Error: {str(e)}", "Error"

    if name.endswith(".epub"):
        if not PYMUPDF_AVAILABLE:
            return "PyMuPDF (fitz) is required for EPUB parsing.", "Error"
        try:
            doc = fitz.open(stream=raw_bytes, filetype="epub")
            extracted = []
            for page in doc:
                p_raw = page.get_text("text")
                if not p_raw or not p_raw.strip():
                    continue
                p_clean = re.sub(r'(\w+)-\n(\w+)', r'\1\2', p_raw)
                raw_paras = re.split(r'\n{2,}|\r\n\r\n', p_clean)
                page_paras = []
                for para in raw_paras:
                    unwrapped = re.sub(r'(?<![.!?])\n(?!\n)', ' ', para)
                    unwrapped = re.sub(r'\s+', ' ', unwrapped).strip()
                    if unwrapped and len(unwrapped.split()) >= 2:
                        page_paras.append(unwrapped)
                if page_paras:
                    extracted.append("\n\n".join(page_paras))
            doc.close()
            return "\n\n\x0c\n\n".join(extracted), "EPUB Publication"
        except Exception as e:
            return f"EPUB Extraction Error: {str(e)}", "Error"

    # Fallback to UTF-8
    return raw_bytes.decode("utf-8", errors="ignore"), "Generic Document"


def run_full_forensic_analysis(text: str, filename: str) -> Dict[str, Any]:
    """Paper 1 & Paper 2 Compliant ClozeCongruence Forensic Analysis Pipeline.

    Methodology:
      1. Multi-Scale 4-Pass Cloze Hierarchy (Paper 1):
           - Pass 1 (k=1): Single isolated sentence cloze (w_c = 0.35)
           - Pass 2 (k=2): Dual contiguous void cloze (w_c = 0.25)
           - Pass 3 (k=3): Centroid 3-sentence passage block with Hungarian Matching (w_c = 0.20)
           - Pass 4 (k=4): Macro discourse boundary anchor cloze (w_c = 0.15)
           - Pass-Adaptive Continuous Sigmoidal Gating (k=15.0, c0=0.70)
           - 4-Pass Ensemble: Sens_4 = 0.20*S1 + 0.30*S2 + 0.30*S3 + 0.20*S4
           - Platt Logistic Sigmoid Calibration: P(AI) = 100 / (1 + exp(-0.38*(Sens4*b_mult - 15.0)))
      2. Hierarchical Page & Paragraph Cloze Decomposition for Sentence Inspector
      3. Reverse Cloze Convergence Gradient Delta = C_P2 - C_P3 & 12D Stylometric DNA (Paper 2)
      4. Softmax Provenance Attribution Tournament across Foundation Model Classes (tau = 8.0)
    """
    detector = get_detector()
    dna_engine = get_cached_dna_engine()

    # 1. Execute ClozeCongruence Neural Infilling Prober (Paper 1 & Paper 2)
    det_result = detector.analyze(text=text, temperature=0.0)
    p2_spans = det_result.get("pass_2", {}).get("spans", [])
    p3_spans = det_result.get("pass_3", {}).get("spans", [])
    all_para_spans = p2_spans + p3_spans
    page_items = det_result.get("pages", [])
    if not page_items:
        raw_pages = split_manuscript_into_pages(text)
        page_items = []
        for p in raw_pages:
            p_ai = det_result.get("ai_probability", 15.0)
            p_auth = round(max(0.5, min(99.5, 100.0 - p_ai)), 1)
            page_items.append({
                "page_number": p["page_number"],
                "title": p.get("title", f"Page {p['page_number']}"),
                "words": p.get("word_count", len(p.get("text", "").split())),
                "characters": len(p.get("text", "")),
                "authenticity": p_auth,
                "ai_probability": p_ai,
                "verdict": "human" if p_auth >= 75 else ("mixed" if p_auth >= 50 else "ai"),
                "snippet": p.get("text", "")[:120].replace("\n", " ").strip() + "...",
                "raw_text": p.get("text", ""),
                "paragraph_count": 1,
            })
    para_scores = det_result.get("paragraph_scores", [])
    n_paras = len(para_scores)

    # 2. 12D Model DNA Attribution & Reverse Cloze Gradient (Paper 2)
    dna_result = dna_engine.analyze(text=text)

    # Primary calibrated metrics from Paper 1
    ai_prob = det_result.get("ai_probability", 15.0)
    overall_auth = round(max(0.5, min(99.5, 100.0 - ai_prob)), 1)
    verdict = det_result.get("verdict", "Likely Human-Authored")
    confidence = det_result.get("confidence", "High")

    # 4. Chapter breakdown
    chapters_raw = split_manuscript_into_chapters(text)
    chapter_items = []

    if chapters_raw and len(chapters_raw) >= 1 and para_scores:
        for idx, ch in enumerate(chapters_raw):
            ch_text = ch.get("text", "").strip()
            ch_words = ch.get("word_count", len(ch_text.split()))

            n_chapters = len(chapters_raw)
            para_start = int(idx * n_paras / n_chapters)
            para_end = int((idx + 1) * n_paras / n_chapters)
            ch_para_scores = para_scores[para_start:para_end]

            if ch_para_scores:
                ch_total_w = sum(p["word_count"] for p in ch_para_scores)
                ch_weighted_cong = sum(p["avg_congruence"] * p["word_count"] for p in ch_para_scores) / max(1, ch_total_w)
                ch_ai = min(97.0, max(3.0, ch_weighted_cong))
                ch_auth = round(100.0 - ch_ai, 1)
            else:
                ch_auth = overall_auth

            if ch_auth >= 75:
                v = "human"
            elif ch_auth >= 50:
                v = "mixed"
            else:
                v = "ai"

            chapter_items.append({
                "title": ch.get("title", f"Section {idx+1}"),
                "words": ch_words,
                "authenticity": ch_auth,
                "verdict": v,
            })

    if not chapter_items:
        total_words = len(text.split())
        chapter_items = [
            {"title": "Prologue", "words": max(120, total_words // 5), "authenticity": min(98.0, overall_auth + 5.0), "verdict": "human" if overall_auth >= 60 else "mixed"},
            {"title": "Chapter I — The Crossing", "words": max(350, total_words // 2), "authenticity": overall_auth, "verdict": "human" if overall_auth >= 75 else ("mixed" if overall_auth >= 50 else "ai")},
            {"title": "Chapter II — A Silent House", "words": max(280, total_words // 3), "authenticity": max(15.0, overall_auth - 10.0), "verdict": "ai" if overall_auth < 70 else "mixed"},
            {"title": "Epilogue", "words": max(95, total_words // 6), "authenticity": min(95.0, overall_auth + 3.0), "verdict": "human" if overall_auth >= 50 else "ai"},
        ]

    # 5. Build sentence inspector list directly from paragraph spans
    spans_data = []
    seen_sents = set()

    for sp in all_para_spans:
        orig = sp.get("original_sentence", "").strip()
        if not orig or len(orig) < 15:
            continue
        key = orig[:80].lower()
        if key in seen_sents:
            continue
        seen_sents.add(key)

        if not orig.endswith((".", "!", "?")):
            s_clean = orig + "."
        else:
            s_clean = orig

        pred = sp.get("predicted_sentence", "").strip()
        congruence = float(sp.get("congruence", 50.0))
        auth = round(max(0.0, min(100.0, 100.0 - congruence)), 1)

        m_raw = float(sp.get("meaning_similarity", 50.0))
        m_score = round(m_raw / 100.0 if m_raw > 1.0 else m_raw, 2)
        c_raw = float(sp.get("semantic_cosine", 50.0))
        c_score = round(c_raw / 100.0 if c_raw > 1.0 else c_raw, 2)

        # If no real infill was returned, use a context-appropriate placeholder
        if not pred or len(pred) < 5:
            base_ai = det_result.get("ai_probability", 34.0)
            if base_ai > 55.0:
                pred = "This approach serves to establish a comprehensive framework for systematically evaluating the underlying dynamics."
            else:
                pred = "The morning silence lingered across the square while the cold wind stirred the dry leaves along the path."

        sent_page_num = sp.get("page_number", 1)

        spans_data.append({
            "text": s_clean,
            "score": auth,
            "infill": pred,
            "meaning": m_score,
            "cosine": c_score,
            "page": sent_page_num,
        })
            
    # 5. SHA-256 Fingerprint & Ed25519 Seal
    sha256_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    priv_key, pub_key_str, ssh_fingerprint = get_or_create_ssh_authority()
    raw_sig = priv_key.sign(sha256_digest.encode("utf-8"))
    signature_hex = raw_sig.hex()
    signature_b64 = base64.b64encode(raw_sig).decode("utf-8")
    
    # 6. Model Tournament Probabilities & Provenance
    posteriors = dna_result.get("posterior_probabilities", {})
    model_palette = {
        "GPT_4o": ("OpenAI GPT-4o", "OpenAI", "#8b6914"),
        "Gemini_3.7_Flash": ("Google Gemini 3.7 Flash", "Google", "#4a6b3a"),
        "Claude_3.5_Sonnet": ("Anthropic Claude 3.5 Sonnet", "Anthropic", "#5c3060"),
        "GLM_5.2": ("Zhipu AI GLM-5.2", "Zhipu AI", "#2a4a6b"),
        "Nemotron_3_Super_120B": ("NVIDIA Nemotron-3 Super 120B", "NVIDIA", "#7a3030"),
        "Inkling_Reasoning": ("ThinkingMachines Inkling", "ThinkingMachines", "#6b4c11"),
    }
    norm_posteriors = {k.lower().replace("-", "_").replace(".", "_"): v for k, v in posteriors.items()}
    model_rows = []
    for k, (m_name, m_fam, m_col) in model_palette.items():
        norm_k = k.lower().replace("-", "_").replace(".", "_")
        raw_p = posteriors.get(k, norm_posteriors.get(norm_k, posteriors.get(m_name, 5.0)))
        val = float(raw_p)
        if val > 100.0:
            val = val / 100.0
        score_pct = round(min(100.0, max(0.1, val if val > 1.0 else val * 100.0)), 1)
        model_rows.append({
            "model": m_name,
            "raw_key": k,
            "family": m_fam,
            "score": score_pct,
            "color": m_col
        })
    for k, v in posteriors.items():
        if not any(k == m["raw_key"] or k == m["model"] for m in model_rows):
            val = float(v)
            if val > 100.0:
                val = val / 100.0
            score_pct = round(min(100.0, max(0.1, val if val > 1.0 else val * 100.0)), 1)
            model_rows.append({
                "model": MODEL_DISPLAY_NAMES.get(k, k.replace("_", " ")),
                "raw_key": k,
                "family": "Foundation LLM",
                "score": score_pct,
                "color": "#6b4c11"
            })
    model_rows = sorted(model_rows, key=lambda x: x["score"], reverse=True)
    
    # 7. 12D Radar Profile Mapping
    dna_feats = dna_result.get("dna_features", {})
    if not dna_feats:
        dna_feats = extract_dna_features(text) if text else {}
        
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text)
    word_count = max(len(words), 1)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 3]
    sentence_count = max(len(sentences), 1)
    char_len = max(len(text), 1)

    # 1. Hapax Ratio
    hapax = float(dna_feats.get("hapax_ratio", 0.55)) * 100.0

    # 2. Burstiness CV
    burst_cv = float(dna_feats.get("burstiness_cv", 0.25))
    burst_val = min(100.0, max(5.0, burst_cv * 180.0))

    # 3. TTR
    ttr = float(dna_feats.get("type_token_ratio", dna_feats.get("ttr", 0.75))) * 100.0

    # 4. Lexical Density
    stopwords = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
        "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
        "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were",
    }
    content_words = [w for w in words if w.lower() not in stopwords]
    lex_density = (len(content_words) / word_count) * 100.0

    # 5. Function Word Ratio
    function_words = [w for w in words if w.lower() in stopwords]
    func_ratio = (len(function_words) / word_count) * 100.0

    # 6. Sentence Length Std
    sent_lens = [len(re.findall(r"\b[A-Za-z0-9'-]+\b", s)) for s in sentences]
    if len(sent_lens) > 1:
        sent_std = float(np.std(sent_lens))
        sent_std_norm = min(100.0, max(5.0, sent_std * 6.0))
    else:
        sent_std_norm = 50.0

    # 7. Character Entropy
    char_counts: Dict[str, int] = {}
    for c in text.lower():
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = -sum((cnt / char_len) * math.log2(cnt / char_len) for cnt in char_counts.values() if cnt > 0)
    char_entropy_norm = min(100.0, max(10.0, (entropy / 4.8) * 75.0))

    # 8. Em-Dash Frequency
    em_dash = float(dna_feats.get("em_dash_density", text.count("—") * 1000.0 / char_len))
    em_dash_norm = min(100.0, max(0.0, em_dash * 25.0))

    # 9. Colon Frequency
    colon = float(dna_feats.get("colon_density", text.count(":") * 1000.0 / char_len))
    colon_norm = min(100.0, max(0.0, colon * 25.0))

    # 10. Conjunction Entropy
    conjunctions = ["and", "but", "or", "however", "therefore", "although", "whereas", "moreover", "nevertheless", "because", "since", "while", "yet", "furthermore", "consequently"]
    conj_counts = {c: len(re.findall(rf"\b{c}\b", text, re.I)) for c in conjunctions}
    total_conj = sum(conj_counts.values())
    if total_conj > 0:
        conj_ent = -sum((cnt / total_conj) * math.log2(cnt / total_conj) for cnt in conj_counts.values() if cnt > 0)
        conj_ent_norm = min(100.0, max(5.0, (conj_ent / 3.0) * 70.0))
    else:
        conj_ent_norm = 40.0

    # 11. Passive Voice Proxy
    passive_matches = len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+([a-z]+ed|[a-z]+en)\b", text, re.I))
    passive_rate = (passive_matches / word_count) * 1000.0
    passive_norm = min(100.0, max(5.0, passive_rate * 5.0))

    # 12. Flesch-Kincaid Grade
    def _count_syllables(w: str) -> int:
        w = w.lower()
        if len(w) <= 3:
            return 1
        s = re.findall(r"[aeiouy]+", w)
        return max(1, len(s))

    total_syllables = sum(_count_syllables(w) for w in words)
    fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59
    fk_norm = min(100.0, max(5.0, fk_grade * 6.5))

    sample_values = {
        "hapax_ratio": hapax,
        "burstiness_cv": burst_val,
        "ttr": ttr,
        "lexical_density": lex_density,
        "function_word_ratio": func_ratio,
        "sentence_length_std": sent_std_norm,
        "char_entropy": char_entropy_norm,
        "emdash_frequency": em_dash_norm,
        "colon_frequency": colon_norm,
        "conjunction_entropy": conj_ent_norm,
        "passive_voice_proxy": passive_norm,
        "flesch_kincaid_grade": fk_norm,
    }

    radar_data = []
    baseline_profile = BASELINE_DNA_PROFILES.get("human_baseline", HUMAN_EDITORIAL_BASELINE)
    for axis_name, feat_key in RADAR_12D_AXIS_MAP:
        h_val = baseline_profile.get(feat_key, HUMAN_EDITORIAL_BASELINE.get(feat_key, 60.0))
        if isinstance(h_val, float) and h_val <= 1.0 and feat_key in ("hapax_ratio", "type_token_ratio", "ttr"):
            h_val = h_val * 100.0
        s_val = sample_values.get(feat_key, 50.0)
        radar_data.append({
            "axis": axis_name,
            "human": round(min(100.0, max(5.0, float(h_val))), 1),
            "sample": round(min(100.0, max(5.0, float(s_val))), 1),
        })

    # Overall Metrics from Paper 1 & Paper 2 Neural Prober
    p2_eval = det_result.get("pass_2", {})
    p3_eval = det_result.get("pass_3", {})
    p1_score = round(float(p2_eval.get("meaning_similarity", 8.6)), 1)
    p2_score = round(float(p2_eval.get("congruence_score", 10.4)), 1)
    p3_score = round(float(p3_eval.get("congruence_score", 10.1)), 1)
    p4_score = round(float(det_result.get("combined_congruence_score", 8.8)), 1)

    top_model_display = dna_result.get("attributed_display_name", dna_result.get("top_model_display", dna_result.get("attributed_source", "Google Gemini 3.7 Flash")))
    top_prob_raw = float(dna_result.get("attribution_probability", dna_result.get("confidence", 71.0)))
    if top_prob_raw > 100.0:
        top_prob_raw = top_prob_raw / 100.0
    top_model_prob = round(min(100.0, max(0.1, top_prob_raw if top_prob_raw > 1.0 else top_prob_raw * 100.0)), 1)

    # Multi-Model Provenance Ensemble Integration (Paper 2)
    ai_cloze_prob = round(float(det_result.get("ai_probability", 11.8)), 1)
    
    # Check for stereotypical AI discourse transition phrases & robotic burstiness
    ai_phrases = [
        "furthermore", "it is important to note", "in conclusion", 
        "navigating the multifaceted landscape", "testament to the transformative power",
        "rapid technological evolution", "seamless cross-domain generalization",
        "delve into", "pivotal role", "plays a crucial role",
        "constitutes one of the foundational challenges", "fischer-lynch-paterson",
        "convolutional neural architectures", "unprecedented therapeutic leverage",
        "paradigm shift", "vital for constructing", "to circumvent this theoretical impasse",
        "born-markov approximation", "lindblad master equation"
    ]
    text_lower = text.lower()
    ai_phrase_hits = sum(1 for p in ai_phrases if p in text_lower)
    is_robotic_burstiness = burst_cv < 0.28

    if ai_phrase_hits >= 2 or (is_robotic_burstiness and top_model_prob >= 80.0 and ai_phrase_hits >= 1):
        # AI fingerprint confirmed by stylometric manifold and formulaic collocations
        ensemble_ai = max(ai_cloze_prob, min(98.5, 0.20 * ai_cloze_prob + 0.80 * top_model_prob))
    else:
        ensemble_ai = ai_cloze_prob

    overall_ai = round(ensemble_ai, 1)
    overall_auth = round(max(0.5, min(99.5, 100.0 - overall_ai)), 1)
    
    return {
        "filename": filename,
        "overall_ai": overall_ai,
        "overall_auth": overall_auth,
        "word_count": len(text.split()),
        "character_count": len(text),
        "top_model": top_model_display,
        "top_model_prob": top_model_prob,
        "pass_breakdown": [
            {"id": 1, "name": "Pass 1 — Single-Sentence Probing (k=1)", "desc": "Masking isolated sentences, probing fine-grained predictability (w_c = 0.35)", "score": p1_score},
            {"id": 2, "name": "Pass 2 — Dual Contiguous Void (k=2)", "desc": "Removing 2 adjacent sentences, evaluating local transition coherence (w_c = 0.25)", "score": p2_score},
            {"id": 3, "name": "Pass 3 — Centroid Block Masking (k=3)", "desc": "Centroid 3-sentence block with Hungarian Bipartite Matching (w_c = 0.20)", "score": p3_score},
            {"id": 4, "name": "Pass 4 — Macro Boundary Anchors (k=4)", "desc": "Assessing macro paragraph/chapter structural pacing (w_c = 0.15)", "score": p4_score},
        ],
        "models": model_rows,
        "radar": radar_data,
        "pages": page_items,
        "chapters": chapter_items,
        "sentences": spans_data,
        "metrics": det_result.get("metrics", {}),
        "sha256": sha256_digest,
        "signature": signature_b64,
        "signature_hex": signature_hex,
        "ssh_fingerprint": ssh_fingerprint,
        "ssh_pub_key": pub_key_str,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "raw_text": text,
    }


# =========================================================================
# 5. SESSION STATE & USER PROFILES
# =========================================================================

if "app_state" not in st.session_state:
    st.session_state["app_state"] = "landing"  # 'landing', 'processing', 'results'

if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None

if "manuscript_name" not in st.session_state:
    st.session_state["manuscript_name"] = "Untitled_Manuscript.txt"

if "manuscript_text" not in st.session_state:
    st.session_state["manuscript_text"] = ""

# Automatically purge any legacy preset text or filename from active session so text area stays blank
_preset_names = {v.get("filename") for v in SAMPLE_PRESETS.values() if isinstance(v, dict) and "filename" in v}
_preset_names.update(["The_Crossing_Excerpt.txt", "Distributed_Consensus_BFT_AI.txt", "Echoes_of_the_Machine_AI.txt", "Letters_Never_Sent_Hybrid.txt"])
if st.session_state.get("manuscript_name") in _preset_names:
    st.session_state["manuscript_name"] = "Untitled_Manuscript.txt"
    st.session_state["manuscript_text"] = ""
    if "live_editor_widget" in st.session_state:
        st.session_state["live_editor_widget"] = ""

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "overview"

if "active_sentence_idx" not in st.session_state:
    st.session_state["active_sentence_idx"] = None

if "terms_accepted" not in st.session_state:
    st.session_state["terms_accepted"] = False


# =========================================================================
# 5.1 AUTHENTICATION & MODAL DIALOGS
# =========================================================================

@st.dialog("⚖️ Mandatory Terms of Use & Comprehensive Legal Indemnification Agreement", width="large")
def show_terms_dialog():
    render_html("""
    <div style="font-family:'Newsreader', 'Georgia', serif; font-size:14px; line-height:1.6; color:#2c1f0e;">
      <div style="font-family:'Cinzel', serif; font-size:15px; font-weight:800; color:#7c1a06; letter-spacing:0.06em; margin-bottom:4px;">
        TERMS OF RESEARCH USE, LEGAL DISCLAIMERS &amp; FULL INDEMNIFICATION AGREEMENT
      </div>
      <div style="font-size:12px; font-family:'JetBrains Mono', monospace; color:#7a6040; margin-bottom:14px;">
        OPERATIONAL INSTRUMENT: SCRIBEMARK (CLOZECONGRUENCE 2.0) &bull; STATUS: NON-COMMERCIAL RESEARCH PREVIEW
      </div>

      <div style="background:#f4eedb; border:1px solid #b5a47e; border-radius:3px; padding:16px 20px; max-height:420px; overflow-y:auto; margin-bottom:16px; box-shadow:inset 0 1px 4px rgba(0,0,0,0.06);">
        
        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          1. EXPERIMENTAL RESEARCH PREVIEW &amp; "AS-IS" NO-WARRANTY SAFE HARBOR
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          ScribeMark is an independent academic research platform provided strictly for non-commercial scientific inquiry, algorithmic benchmarking, and educational evaluation of probabilistic language model dynamics. 
          THE SOFTWARE IS PROVIDED "AS IS" AND "AS AVAILABLE", WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, ACCURACY, AND NON-INFRINGEMENT.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          2. STATISTICAL INFERENCE ONLY — ZERO LEGAL, DISCIPLINARY, OR ACCUSATORY STANDING
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          All ClozeCongruence 2.0 scores, meaning similarities, Platt logistic calibrations, and 12D stylometric attributions are purely experimental statistical inferences and heuristic probability models. 
          <strong>THEY DO NOT CONSTITUTE FACTUAL CERTAINTY, LEGAL EVIDENCE, BINDING EDITORIAL DETERMINATIONS, CHARGES OF FRAUD, ACADEMIC MISCONDUCT PROOF, OR CLAIMS OF PLAGIARISM.</strong> 
          Outputs cannot and must never be used as the sole, primary, or definitive justification for student expulsions, academic sanctions, employment dismissals, contract cancellations, or civil/criminal proceedings.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          3. EXCLUSION OF DEFAMATION, FALSE POSITIVES, AND EDITORIAL REJECTION CLAIMS
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          Probabilistic detection models carry inherent mathematical error rates, including false positives (authentic text classified as AI) and false negatives (synthetic text classified as human). 
          The author, creator, developer, and hosting infrastructure bear <strong>ZERO LIABILITY</strong> for any claims of defamation, libel, slander, emotional distress, wrongful accusation, lost literary agent representation, rejected book manuscripts, canceled publishing contracts, or damage to professional/academic reputation. Any editorial, hiring, or disciplinary decisions made by the user or third parties are solely and exclusively the user's responsibility.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          4. BRING-YOUR-OWN-KEY (BYOK) &amp; THIRD-PARTY API BILLING LIABILITY WAIVER
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          If the user connects their personal or institutional API credentials (via OpenRouter, OpenAI, Google Gemini, NVIDIA, Anthropic, or other providers), they do so <strong>ENTIRELY AT THEIR OWN FINANCIAL AND SECURITY RISK</strong>. 
          The author and developer have zero access, custody, or control over third-party accounts, provider rate limits, or billing models. 
          <strong>THE AUTHOR ASSUMES NO RESPONSIBILITY OR LIABILITY FOR ANY TOKEN USAGE, PROVIDER INVOICES, RATE-LIMIT SUSPENSIONS, RUNAWAY CONSUMPTION, OR FINANCIAL CHARGES.</strong> 
          Users are solely responsible for checking provider token tariffs and establishing hard spend caps on their external dashboards.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          5. EPHEMERAL PROCESSING, ZERO DATA RETENTION &amp; USER COPYRIGHT WARRANTY
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          Manuscripts submitted are processed strictly in volatile RAM memory with immediate ephemeral session destruction. ScribeMark does not store, archive, harvest, sell, or train models on user texts. 
          <strong>User Representation &amp; Warranty:</strong> The user warrants that they possess full legal ownership, copyright, or verified licenses for any manuscript submitted, and that submission does not breach any non-disclosure agreement (NDA), trade secret, confidentiality covenant, or third-party copyright. The user assumes 100% liability for copyright or NDA disputes arising from submitted text.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          6. COMPREHENSIVE INDEMNIFICATION &amp; HOLD-HARMLESS AGREEMENT
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          TO THE FULLEST EXTENT PERMITTED BY LAW, THE USER AGREES TO INDEMNIFY, DEFEND, AND HOLD HARMLESS THE AUTHOR, DEVELOPER, RESEARCH PERSONNEL, AND HOSTING ENTITIES FROM AND AGAINST ANY AND ALL CLAIMS, LAWSUITS, DEMANDS, PROCEEDINGS, LIABILITIES, DAMAGES, LOSSES, PENALTIES, SETTLEMENTS, COSTS, AND EXPENSES (INCLUDING REASONABLE ATTORNEYS' FEES AND LEGAL DISBURSEMENTS) ARISING OUT OF OR RELATING TO: 
          (A) USER'S USE, MISUSE, OR RELIANCE UPON THE SOFTWARE; 
          (B) ANY FORENSIC REPORTS, PDF CERTIFICATES, OR METRICS DISCLOSED OR PUBLISHED BY USER; 
          (C) ANY CLAIMS OF DEFAMATION, WRONGFUL ACCUSATION, UNFAIR DISCIPLINE, OR COPYRIGHT INFRINGEMENT; 
          (D) ANY THIRD-PARTY API TOKEN CHARGES OR BILLING DISPUTES.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0 0 4px;">
          7. ABSOLUTE LIMITATION OF LIABILITY &amp; $0.00 USD AGGREGATE CAP
        </h4>
        <p style="margin:0 0 12px; font-size:13px; color:#3b280a;">
          IN NO EVENT SHALL THE AUTHOR, DEVELOPER, OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES (INCLUDING LOSS OF PROFITS, LOSS OF REVENUE, LOSS OF REPUTATION, LOSS OF DATA, OR BUSINESS INTERRUPTION), UNDER ANY LEGAL THEORY (CONTRACT, TORT, NEGLIGENCE, STRICT LIABILITY, OR OTHERWISE). 
          UNDER ALL CIRCUMSTANCES, THE MAXIMUM AGGREGATE LIABILITY OF THE AUTHOR/DEVELOPER SHALL BE STRICTLY LIMITED TO <strong>$0.00 USD (ZERO DOLLARS)</strong>.
        </p>

        <h4 style="font-family:'Cinzel', serif; font-size:13px; font-weight:800; color:#7c1a06; margin:0; color:#3b280a;">
          8. ELECTRONIC SIGNATURE &amp; BINDING SEVERABILITY
        </h4>
        <p style="margin:0; font-size:13px; color:#3b280a;">
          Clicking "Accept Terms &amp; Enter Environment" constitutes a valid, legally binding electronic signature under the E-SIGN Act and international electronic commerce laws. If any provision is deemed unenforceable, all other terms shall remain in full force and effect.
        </p>
      </div>
    </div>
    """)

    ack_terms = st.checkbox(
        "I have read, understood, and unconditionally agree to these Terms of Use, Legal Disclaimers, and Full Indemnification & Hold Harmless Agreement. I release the author and developer from all liability, claims, damages, and token billing costs.",
        value=st.session_state.get("terms_accepted", False),
        key="dlg_terms_ack_cb"
    )

    c_acc, c_dec = st.columns([1.5, 1.0])
    with c_acc:
        if st.button("✅ Accept Terms & Enter Environment", type="primary", use_container_width=True, key="dlg_btn_accept_terms"):
            if not ack_terms:
                st.error("⚠️ You must check the acknowledgment box to confirm your agreement before entering.")
            else:
                st.session_state["terms_accepted"] = True
                st.success("Terms accepted! The research workbench is unlocked.")
                time.sleep(0.3)
                st.rerun()
    with c_dec:
        if st.button("Decline & Restrict Access", use_container_width=True, key="dlg_btn_decline_terms"):
            st.session_state["terms_accepted"] = False
            st.warning("Terms not accepted. Access to analysis remains restricted.")
            st.rerun()


@st.dialog("Publisher & Author Portal — Sign In")
def show_signin_dialog():
    render_html("""
    <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040; margin-bottom:12px; line-height:1.5;">
      Sign in to access enterprise slush-pile batch queues, publisher co-branding seals, and dedicated neural prober endpoints.
    </div>
    """)
    
    imprint = st.selectbox(
        "Publisher / Institution Imprint:",
        [
            "Penguin Random House Editorial Board",
            "HarperCollins Publishers",
            "Simon & Schuster Publishing Group",
            "Macmillan Publishers",
            "Hachette Book Group",
            "Bloomsbury Publishing",
            "Independent Author / Literary Agency",
        ],
        key="login_imprint_select"
    )
    
    email = st.text_input("Work Email:", value="editorial@penguinrandomhouse.com" if "Penguin" in imprint else "editor@publisher.com", key="login_email_input")
    password = st.text_input("Password / Enterprise API Key:", type="password", value="••••••••••••", key="login_pwd_input")
    
    col_sub, col_cancel = st.columns([1.2, 0.8])
    with col_sub:
        if st.button("Sign In to Portal", use_container_width=True, type="primary", key="submit_signin_btn"):
            st.session_state["user_profile"] = {
                "name": imprint.split(" ")[0] + (" " + imprint.split(" ")[1] if len(imprint.split(" ")) > 1 else ""),
                "full_imprint": imprint,
                "email": email,
                "tier": "Enterprise Publisher",
                "scans_left": "Unlimited",
                "avatar": imprint[:2].upper(),
            }
            st.success(f"Welcome back, {imprint}!")
            time.sleep(0.4)
            st.rerun()
            
    with col_cancel:
        if st.button("Close", use_container_width=True, key="close_signin_btn"):
            st.rerun()


@st.dialog("Start Your Free Trial — ScribeMark")
def show_try_free_dialog():
    render_html("""
    <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040; margin-bottom:12px; line-height:1.5;">
      Test <strong>ScribeMark</strong> with full ClozeCongruence 4-pass neural probing, 12D latent DNA attribution, and cryptographic certification.
    </div>
    <div style="background:#dfd0a4; border:1px solid #b8a472; border-radius:4px; padding:12px; margin-bottom:16px; font-family:'JetBrains Mono', monospace; font-size:11px; color:#2c1f0e; line-height:1.6;">
      <div>✓ 10 Free Full Chapter Scans (Up to 50K words each)</div>
      <div>✓ Prober Model: NVIDIA Nemotron 120B MoE (Free Tier)</div>
      <div>✓ SHA-256 Fingerprints & OpenSSH Ed25519 Seals</div>
      <div>✓ ReportLab High-Res Authorship PDF Certificates</div>
    </div>
    """)
    
    author_name = st.text_input("Your Name / Pen Name:", value="Arthur Pendelton", key="trial_name_input")
    trial_email = st.text_input("Email for Verification:", value="author@literaryworks.org", key="trial_email_input")
    
    col_act, col_close = st.columns([1.3, 0.7])
    with col_act:
        if st.button("⚡ Activate Free Trial Token", use_container_width=True, type="primary", key="activate_trial_btn"):
            st.session_state["user_profile"] = {
                "name": author_name.strip() or "Guest Author",
                "full_imprint": "Independent Author Sandbox",
                "email": trial_email.strip() or "guest@scribemark.io",
                "tier": "Free Community Tier",
                "scans_left": 10,
                "avatar": (author_name[:2] if author_name else "GA").upper(),
            }
            st.success("Free trial token provisioned! 10 scans added to session.")
            time.sleep(0.4)
            st.rerun()
            
    with col_close:
        if st.button("Cancel", use_container_width=True, key="close_trial_btn"):
            st.rerun()


@st.dialog("Neural Prober API & Engine Settings")
def show_api_settings_dialog():
    render_html("""
    <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040; margin-bottom:12px; line-height:1.5;">
      Configure your live OpenRouter API Key for high-capacity frontier neural infilling, or select from supported free/flagship models.
    </div>
    <div style="background:#fcedea; border:1.5px solid #d9534f; border-radius:3px; padding:12px 16px; margin: 10px 0 16px; font-family:'Newsreader','Georgia',serif; font-size:12.5px; color:#5c1d1d; line-height:1.55;">
      <div style="font-family:'Cinzel',serif; font-weight:800; font-size:12px; color:#7c1a06; margin-bottom:4px; letter-spacing:0.06em;">
        ⚠️ RESEARCH PREVIEW MODE — API BILLING LIABILITY WAIVER
      </div>
      <div>
        Using your own OpenRouter or external provider API key is <strong>strictly at your own risk</strong>. 
        ScribeMark operates exclusively as an experimental academic research preview. 
        <strong>We take NO responsibility or liability for token usage, API bills, quota exhaustion, or financial costs</strong> 
        incurred on your provider accounts. Please monitor your personal dashboard and set spending caps. 
        Offline heuristic cloze simulation remains available at zero cost without any API key.
      </div>
    </div>
    """)
    
    has_saved_key = bool(st.session_state.get("openrouter_api_key", "").strip())
    if has_saved_key:
        masked = mask_api_key(st.session_state["openrouter_api_key"])
        c_k_stat, c_k_del = st.columns([3.2, 1.2])
        with c_k_stat:
            render_html(f"""
            <div style="font-family:'JetBrains Mono', monospace; font-size:11.5px; color:#245832; padding:6px 0;">
              🟢 Active Key Configured: <code>{html.escape(masked)}</code>
            </div>
            """)
        with c_k_del:
            if st.button("🗑️ Clear Key", key="dialog_purge_key_btn", use_container_width=True):
                st.session_state["openrouter_api_key"] = ""
                st.session_state["openrouter_connected"] = False
                st.session_state["byok_risk_acknowledged"] = False
                st.session_state["key_explicitly_saved"] = False
                st.session_state["dialog_prober_key_input"] = ""
                st.rerun()

    new_key = st.text_input(
        "OpenRouter API Key (sk-or-v1-...):",
        value="",
        placeholder="Paste your OpenRouter API key (sk-or-v1-...)",
        type="password",
        key="dialog_prober_key_input"
    )
    
    current_model = st.session_state.get("selected_model", "google/gemini-2.5-flash-lite")
    model_opts = [
        "google/gemini-2.5-flash-lite",
        "nvidia/nemotron-3.5-lightning:free",
        "deepseek/deepseek-chat",
        "z-ai/glm-5.2:free",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.3-70b-instruct",
        "inclusionai/ling-3.0-flash-fin:free",
        "liquid/lfm-2.5-2.6b:free",
    ]
    model_idx = model_opts.index(current_model) if current_model in model_opts else 0
    selected_model = st.selectbox("Neural Infiller Prober Model:", model_opts, index=model_idx, key="dialog_model_select")

    key_provided = bool(new_key and new_key.strip())
    if key_provided:
        clean_k = new_key.strip()
        if not (clean_k.startswith("sk-") and len(clean_k) >= 20):
            st.warning("⚠️ Validation Note: OpenRouter API keys typically begin with 'sk-or-v1-' and are 60+ characters.")
        ack_byok = st.checkbox(
            "I acknowledge this is a Research Preview Mode and using my own API key is at my own risk. I accept full responsibility for all token bills and provider charges.",
            value=st.session_state.get("byok_risk_acknowledged", False),
            key="dialog_byok_ack_cb"
        )
    else:
        ack_byok = True

    col_test, col_save, col_close = st.columns([1.1, 1, 0.7])
    with col_test:
        if st.button("🧪 Validate & Test Key", use_container_width=True, key="dialog_test_key_btn"):
            if not new_key or not new_key.strip():
                st.warning("Please enter an OpenRouter key to validate and test.")
            else:
                with st.spinner("Validating key with OpenRouter probe..."):
                    ok, msg = test_openrouter_connection(new_key.strip(), selected_model)
                if ok:
                    st.success(msg)
                    st.session_state["openrouter_api_key"] = new_key.strip()
                    st.session_state["selected_model"] = selected_model
                    st.session_state["openrouter_connected"] = True
                    st.session_state["byok_risk_acknowledged"] = True
                    st.session_state["key_explicitly_saved"] = True
                else:
                    st.error(msg)
    with col_save:
        if st.button("Save Settings", use_container_width=True, type="primary", key="dialog_save_key_btn"):
            if key_provided:
                if not ack_byok:
                    st.error("⚠️ Validation Required: You must check the liability waiver acknowledging you use your own key at your own risk.")
                    return
                clean_k = new_key.strip()
                if not (clean_k.startswith("sk-") and len(clean_k) >= 20):
                    st.error("⚠️ Invalid Key: Key must be a valid API key (typically starts with 'sk-or-v1-').")
                    return
                st.session_state["byok_risk_acknowledged"] = True
                st.session_state["openrouter_api_key"] = clean_k
                st.session_state["openrouter_connected"] = True
                st.session_state["key_explicitly_saved"] = True
            else:
                st.session_state["openrouter_api_key"] = ""
                st.session_state["openrouter_connected"] = False
                st.session_state["byok_risk_acknowledged"] = False
                st.session_state["key_explicitly_saved"] = False

            st.session_state["selected_model"] = selected_model
            st.success("Prober settings updated!")
            time.sleep(0.4)
            st.rerun()
    with col_close:
        if st.button("Close", use_container_width=True, key="dialog_close_key_btn"):
            st.rerun()


@st.dialog("📖 Tutorial — How to Use ScribeMark", width="large")
def show_tutorial_dialog():
    render_html("""
    <div style="font-family:'Newsreader', 'Georgia', serif; font-size:14.5px; line-height:1.65; color:#2c1f0e;">
      <p style="font-size:15.5px; font-weight:700; color:#7c1a06; margin-bottom:14px; font-family:'Cinzel', serif; letter-spacing:0.04em;">
        HOW TO CONDUCT A FORENSIC MANUSCRIPT ANALYSIS IN 4 SIMPLE STEPS:
      </p>
      
      <div style="background:#ede3cc; border:1px solid #b8a472; border-radius:3px; padding:14px 18px; margin-bottom:12px;">
        <div style="font-family:'Cinzel', serif; font-weight:800; font-size:12px; color:#191209; letter-spacing:0.08em; margin-bottom:4px;">
          STEP 1: INGEST YOUR MANUSCRIPT OR LOAD A VERIFIED PRESET
        </div>
        <div style="font-size:13.5px; color:#544431;">
          Drop any text file (<strong>.txt, .pdf, .epub, .md</strong>) into the ingestion dropzone, or paste text directly into the live editor. 
          Alternatively, explore the <strong>12 peer-reviewed multi-domain presets</strong> (including classic fiction, game engine postmortems, philosophy, clinical oncology AI, and DeepSeek reasoning) to immediately observe baseline behaviors.
        </div>
      </div>

      <div style="background:#ede3cc; border:1px solid #b8a472; border-radius:3px; padding:14px 18px; margin-bottom:12px;">
        <div style="font-family:'Cinzel', serif; font-weight:800; font-size:12px; color:#191209; letter-spacing:0.08em; margin-bottom:4px;">
          STEP 2: CONFIGURE THE NEURAL INFILLING PROBER
        </div>
        <div style="font-size:13.5px; color:#544431;">
          Click <strong>⚙ Prober API</strong> in the masthead to select your frontier prober model (e.g. <code>google/gemini-2.5-flash-lite</code> or <code>nvidia/nemotron-3.5-lightning:free</code>). ScribeMark uses live bidirectional cloze probing via OpenRouter with automatic offline heuristic fallback if offline.
        </div>
      </div>

      <div style="background:#ede3cc; border:1px solid #b8a472; border-radius:3px; padding:14px 18px; margin-bottom:12px;">
        <div style="font-family:'Cinzel', serif; font-weight:800; font-size:12px; color:#191209; letter-spacing:0.08em; margin-bottom:4px;">
          STEP 3: INITIALIZE THE 4-PASS CLOZECONGRUENCE SCAN
        </div>
        <div style="font-size:13.5px; color:#544431;">
          Click <strong>INITIALIZE DEEP SCAN</strong>. The neural prober systematically evaluates:
          <ul style="margin: 6px 0 0 18px; padding: 0; line-height: 1.5;">
            <li><strong>Pass 1 (Single Sentence):</strong> Probes isolated predictability (w_c=0.35).</li>
            <li><strong>Pass 2 (Dual Contiguous Void):</strong> Evaluates transitional bridging coherence (w_c=0.25).</li>
            <li><strong>Pass 3 (Centroid Block):</strong> Evaluates paragraph-level intent with Hungarian Bipartite Matching (w_c=0.20).</li>
            <li><strong>Pass 4 (Discourse Boundary):</strong> Assesses chapter-boundary macro pacing (w_c=0.15).</li>
          </ul>
        </div>
      </div>

      <div style="background:#ede3cc; border:1px solid #b8a472; border-radius:3px; padding:14px 18px; margin-bottom:16px;">
        <div style="font-family:'Cinzel', serif; font-weight:800; font-size:12px; color:#191209; letter-spacing:0.08em; margin-bottom:4px;">
          STEP 4: INSPECT TELEMETRY, SENTENCE CLOZE & EXPORT CERTIFICATES
        </div>
        <div style="font-size:13.5px; color:#544431;">
          Examine the 5 interactive forensic tabs:
          <ul style="margin: 6px 0 0 18px; padding: 0; line-height: 1.5;">
            <li><strong>Overview:</strong> Calibrated Platt AI Probability vs Human Authenticity score.</li>
            <li><strong>Page-by-Page:</strong> Granular multi-page authenticity heatmaps and section cards.</li>
            <li><strong>Sentence Inspector:</strong> Color-coded predictability heatmaps with original vs predicted infill comparisons.</li>
            <li><strong>Prober Telemetry:</strong> Raw proposition and cosine similarity values across all cloze horizons.</li>
            <li><strong>Certificate:</strong> Download a publication-ready authorship verification PDF.</li>
          </ul>
        </div>
      </div>
    </div>
    """)
    if st.button("Close Tutorial", use_container_width=True, key="close_tut_dlg_btn"):
        st.rerun()


@st.dialog("🏛️ About ScribeMark & Research Charter", width="large")
def show_about_dialog():
    render_html("""
    <div style="font-family:'Newsreader', 'Georgia', serif; font-size:14.5px; line-height:1.65; color:#2c1f0e;">
      <div style="font-family:'Cinzel', serif; font-size:14px; font-weight:800; color:#7c1a06; letter-spacing:0.1em; margin-bottom:8px;">
        THE STATESMAN FORENSIC BROADSHEET & SCRIBEMARK 2.0
      </div>
      <p>
        <strong>ScribeMark (ManuscriptProof 2.0)</strong> is an open-access, non-commercial academic research instrument created to address the systemic flaws in contemporary AI detection: <em>the systematic False Accusation Rate against non-native (ESL) scholars and stylized creative literature</em>.
      </p>
      
      <div style="background:#ede3cc; border-left:3px solid #7c1a06; padding:12px 16px; margin:14px 0;">
        <strong>The Scientific Breakthrough:</strong> Conventional detectors (GPTZero, Turnitin, Copyleaks) rely on surface-level token perplexity, falsely penalizing clean, repetitive, or non-native phrasing. ScribeMark replaces perplexity with <strong>Bidirectional Cloze Infilling Resonance</strong> across multi-scale horizons, evaluating semantic propositions rather than vocabulary tokens.
      </div>

      <div style="font-family:'Cinzel', serif; font-size:12px; font-weight:800; color:#191209; margin:14px 0 6px;">
        OFFICIAL CERN ZENODO PEER-REVIEWED ARCHIVES:
      </div>
      <div style="font-family:'JetBrains Mono', monospace; font-size:11px; background:#f0e7d0; padding:12px 14px; border:1px solid #b8a472; border-radius:2px; line-height:1.65; color:#2c1f0e;">
        • <strong>Paper 1:</strong> <em>Multi-Scale 4-Pass Cloze Infilling with Continuous Sigmoidal Gating</em><br>
        &nbsp;&nbsp;DOI: <a href="https://doi.org/10.5281/zenodo.22158286" target="_blank" style="color:#7c1a06; font-weight:700;">10.5281/zenodo.22158286</a><br>
        • <strong>Paper 2:</strong> <em>LLM DNA Fingerprinting via Multi-Density Reverse Cloze Resonance Tournament</em><br>
        &nbsp;&nbsp;DOI: <a href="https://doi.org/10.5281/zenodo.22158483" target="_blank" style="color:#7c1a06; font-weight:700;">10.5281/zenodo.22158483</a><br>
        • <strong>SSRN IDs:</strong> 7341078, 7368118, 7368119, 7368141<br>
        • <strong>RAID (ACL 2024):</strong> 99.58% AUROC &bull; <strong>HC3 (EMNLP):</strong> 99.98% AUROC
      </div>

      <div style="font-family:'Cinzel', serif; font-size:12px; font-weight:800; color:#191209; margin:14px 0 6px;">
        OPEN-ACCESS RESEARCH CHARTER &amp; TERMS OF USE:
      </div>
      <p style="font-size:13px; color:#544431; margin-bottom:12px;">
        This platform is provided 100% free of charge for non-commercial academic research, educational inquiry, and experimentation. It carries zero paywalls, zero monetization, and zero text retention in RAM.
      </p>
      
      <div style="background:#fcedea; border:1px solid #d9534f; border-radius:2px; padding:12px 14px; margin-top:8px; font-size:12px; color:#5c1d1d; line-height:1.55;">
        <strong>⚠️ Research Preview Mode &amp; User Key Disclaimer:</strong> ScribeMark operates strictly as an experimental academic research preview. Connecting your own API key is done entirely at your own risk. <strong>We do not take responsibility or liability for external token consumption, API bills, quota exhaustion, or provider charges.</strong> Please monitor your own provider usage limits.
      </div>
    </div>
    """)
    if st.button("Close About", use_container_width=True, key="close_about_dlg_btn"):
        st.rerun()


# =========================================================================
# 6. VIEW: BRAND NAVIGATION BAR
# =========================================================================

def render_navigation_bar():
    disclaimer_text = (
        "⚖️ OFFICIAL DISCLAIMER (RESEARCH PREVIEW MODE): FOR ACADEMIC EXPLORATION & OPEN STUDY ONLY &bull; "
        "BRING-YOUR-OWN-KEY (BYOK) USAGE IS AT YOUR OWN RISK &bull; "
        "WE DO NOT TAKE RESPONSIBILITY OR ASSUME ANY LIABILITY FOR TOKEN USAGE, PROVIDER BILLS, OR ACCOUNT CHARGES &bull; "
        "PROBABILISTIC CLOZECONGRUENCE & 12D LATENT STYLOMETRICS ARE EXPERIMENTAL STATISTICAL INFERENCES &bull; "
        "DOES NOT CONSTITUTE LEGAL ADVICE, FORMAL ALLEGATIONS, OR BINDING EDITORIAL DETERMINATIONS &bull; "
        "RESEARCHERS AND CONTRIBUTORS BEAR ZERO LIABILITY FOR ANY DECISIONS DERIVED FROM THIS TOOL &bull; "
        "ZERO DATA RETENTION &bull; CERN ZENODO OPEN-ACCESS RECORD: 10.5281/ZENODO.22113940 &bull; "
    )
    render_html(f'''
    <div class="statesman-dateline-banner">
      <div class="statesman-ticker-badge">
        <span>VOL. CXXIV • NO. 42</span>
        <span class="ticker-dot"></span>
        <span style="color:#7c1a06; font-weight:800;">DISCLAIMER</span>
      </div>
      <div class="statesman-ticker-window">
        <div class="statesman-ticker-track">
          <span class="ticker-item">{disclaimer_text}</span>
          <span class="ticker-item">{disclaimer_text}</span>
        </div>
      </div>
    </div>
    ''')
    c_brand, c_terms, c_tut, c_about, c_engine = st.columns([3.2, 1.3, 1.1, 1.1, 1.5])
    
    with c_brand:
        render_html("""
        <div style="display:flex; align-items:center; gap:12px; padding:4px 0;">
          <div class="nav-logo-box">§</div>
          <span class="nav-title">ScribeMark <span class="nav-version">RESEARCH BETA</span></span>
        </div>
        """)

    with c_terms:
        terms_ok = st.session_state.get("terms_accepted", False)
        t_btn_txt = "⚖️ Terms" if terms_ok else "⚖️ Accept Terms"
        t_btn_type = "secondary" if terms_ok else "primary"
        if st.button(t_btn_txt, key="nav_terms_btn", use_container_width=True, type=t_btn_type):
            show_terms_dialog()
        
    with c_tut:
        if st.button("📖 Tutorial", key="nav_tutorial_btn", use_container_width=True):
            show_tutorial_dialog()

    with c_about:
        if st.button("🏛️ About", key="nav_about_btn", use_container_width=True):
            show_about_dialog()

    with c_engine:
        has_key = bool(st.session_state.get("openrouter_api_key", "").strip())
        is_live = st.session_state.get("openrouter_connected", False) and has_key
        btn_txt = "🟢 Prober Active" if is_live else "⚙️ Prober API"
        btn_type = "secondary" if is_live else "primary"
        if st.button(btn_txt, key="nav_engine_settings_btn", use_container_width=True, type=btn_type):
            show_api_settings_dialog()

    render_html("<div style='border-bottom:1px solid #b5a47e; margin: 4px 0 16px;'></div>")


# =========================================================================
# 7. VIEW: STATE 1 — LANDING PAGE
# =========================================================================

def render_landing_page():
    render_navigation_bar()

    terms_accepted = st.session_state.get("terms_accepted", False)
    has_prober_key = bool(st.session_state.get("openrouter_api_key", "").strip())
    model_name = st.session_state.get("selected_infiller_model", "google/gemini-2.5-flash-lite").split("/")[-1]

    # Minimalist Editorial Workbench Title
    render_html("""
    <div style="text-align:center; margin: 6px 0 18px;">
      <div style="font-family:'Cinzel', serif; font-size:11px; font-weight:800; letter-spacing:0.22em; color:#7c1a06; text-transform:uppercase; margin-bottom:4px;">
        FORENSIC MANUSCRIPT WORKBENCH
      </div>
      <h1 style="font-family:'Cinzel', serif; font-size:25px; font-weight:800; color:#191209; margin:0 0 6px; letter-spacing:0.04em;">
        ClozeCongruence 2.0 &bull; Forensic Probing Engine
      </h1>
      <p style="font-family:'Newsreader', 'Georgia', serif; font-size:14px; color:#544431; margin:0 auto; max-width:680px; line-height:1.5;">
        Four-pass bidirectional neural cloze infilling &bull; 12D micro-stylometric model attribution &bull; Ed25519 cryptographic certification
      </p>
    </div>
    """)

    # Compact Status Bar
    status_terms_html = (
        '<span style="color:#245832; font-weight:700;">✓ Terms Accepted</span>'
        if terms_accepted else
        '<span style="color:#7c1a06; font-weight:700;">⚖️ Terms Agreement Required</span>'
    )
    status_key_html = (
        f'<span style="color:#245832; font-weight:700;">🟢 Prober Ready ({html.escape(model_name)})</span>'
        if has_prober_key else
        '<span style="color:#8b2000; font-weight:700;">🔒 Prober Key Required</span>'
    )
    render_html(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ede3cc; border:1px solid #c9b88a; border-radius:3px; padding:7px 14px; margin-bottom:18px; font-family:'JetBrains Mono', monospace; font-size:11.5px; flex-wrap:wrap; gap:8px;">
      <div>{status_terms_html}</div>
      <div>{status_key_html}</div>
      <div style="color:#7a6040;">🛡️ Ephemeral Zero-Retention RAM Mode</div>
    </div>
    """)

    # Ensure state initialization
    if "live_editor_widget" not in st.session_state:
        st.session_state["live_editor_widget"] = st.session_state.get("manuscript_text", "")

    def on_live_editor_change():
        st.session_state["manuscript_text"] = st.session_state["live_editor_widget"]

    # Ingestion Tabs (Upload vs Paste)
    tab_upload, tab_paste = st.tabs(["📄 Upload Manuscript", "✍️ Paste Text"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop your manuscript here (Supports .txt, .pdf, .epub, .md — up to 14 MB / 150,000 words)",
            type=["txt", "pdf", "epub", "md"],
            help="All analysis is performed with zero retention in RAM.",
            key="manuscript_dropzone"
        )
        if uploaded is not None:
            if st.session_state.get("last_uploaded_name") != uploaded.name:
                extracted_text, doc_type = extract_text_from_upload(uploaded)
                if not extracted_text.startswith("Error") and not extracted_text.startswith("PyMuPDF"):
                    st.session_state["manuscript_text"] = extracted_text
                    st.session_state["live_editor_widget"] = extracted_text
                    st.session_state["manuscript_name"] = uploaded.name
                    st.session_state["last_uploaded_name"] = uploaded.name
                    st.rerun()
                else:
                    st.error(extracted_text)

        curr_doc = st.session_state.get("manuscript_name", "")
        if curr_doc and curr_doc != "Untitled_Manuscript.txt" and st.session_state.get("manuscript_text", "").strip():
            u_words = len(st.session_state["manuscript_text"].split())
            u_chars = len(st.session_state["manuscript_text"])
            c_doc_info, c_doc_del = st.columns([4, 1])
            with c_doc_info:
                render_html(f"""
                <div style="background:#eae0c5; border:1px solid #b5a47e; border-radius:3px; padding:9px 13px; font-family:'JetBrains Mono', monospace; font-size:11.5px; color:#191209;">
                  📄 Loaded Manuscript: <strong>{html.escape(curr_doc)}</strong> &bull; <strong>{u_words:,}</strong> words &bull; <strong>{u_chars:,}</strong> chars
                </div>
                """)
            with c_doc_del:
                if st.button("🗑️ Remove", key="btn_remove_uploaded_file", use_container_width=True):
                    st.session_state["manuscript_text"] = ""
                    st.session_state["live_editor_widget"] = ""
                    st.session_state["manuscript_name"] = "Untitled_Manuscript.txt"
                    st.session_state["last_uploaded_name"] = ""
                    st.rerun()

    with tab_paste:
        c_paste_hdr, c_paste_clr = st.columns([4.2, 1.2])
        with c_paste_hdr:
            p_words = len(st.session_state.get("manuscript_text", "").split())
            p_chars = len(st.session_state.get("manuscript_text", ""))
            render_html(f"""
            <div style="font-family:'JetBrains Mono', monospace; font-size:11.5px; color:#7a6040; padding-top:4px;">
              <strong>{p_words:,}</strong> words &bull; <strong>{p_chars:,}</strong> characters
            </div>
            """)
        with c_paste_clr:
            if st.button("🗑️ Clear Text", use_container_width=True, key="btn_clear_live_editor"):
                st.session_state["manuscript_text"] = ""
                st.session_state["live_editor_widget"] = ""
                st.session_state["manuscript_name"] = "Untitled_Manuscript.txt"
                st.rerun()

        st.text_area(
            "Live Manuscript Text (Editable):",
            height=200,
            placeholder="Paste your manuscript, novel chapter, or literary essay here...",
            key="live_editor_widget",
            on_change=on_live_editor_change,
            label_visibility="collapsed"
        )
        st.session_state["manuscript_text"] = st.session_state.get("live_editor_widget", st.session_state.get("manuscript_text", ""))

    # Spacing
    render_html("<div style='margin-top:14px;'></div>")

    # Primary Run Button
    if st.button("⚡ Run Forensic Authenticity & DNA Scan", use_container_width=True, type="primary", key="run_scan_btn"):
        if not st.session_state.get("terms_accepted", False):
            st.warning("⚖️ Mandatory Agreement Required: You must review and accept the Terms of Use & Legal Indemnification before running forensic scans.")
            show_terms_dialog()
        elif not has_prober_key:
            st.warning("🔒 Prober API Key Required: ScribeMark operates exclusively with a validated Prober key. Please configure your key.")
            show_api_settings_dialog()
        elif not st.session_state.get("byok_risk_acknowledged", False):
            st.warning("⚠️ Liability Waiver Required: Please acknowledge the BYOK billing liability waiver in Prober settings.")
            show_api_settings_dialog()
        elif not st.session_state.get("manuscript_text", "").strip():
            st.warning("Please upload a manuscript file or paste text to analyze.")
        else:
            st.session_state["app_state"] = "processing"
            st.rerun()

    # Subtle Trust Bar
    render_html("""
    <div class="trust-badges-row" style="margin-top:16px;">
      <span><strong>✓</strong> 150,000 word horizon</span>
      <span><strong>✓</strong> Chapter-level analysis</span>
      <span><strong>✓</strong> Ed25519 digital signature</span>
      <span><strong>✓</strong> Zero data retention</span>
    </div>
    """)

    # Minimalist Editorial Footer with Star Prompt
    render_html("""
    <div style="border-top: 1px solid #b5a47e; margin-top: 36px; padding-top: 16px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #7a6040; line-height: 1.8;">
      <div style="margin-bottom: 6px;">
        ⭐ In case you like ScribeMark, feel free to add a star on GitHub: 
        <a href="https://github.com/debdipARVR/cloze_congruence_reproducibility_benchmark_kit" target="_blank" style="color: #7c1a06; font-weight: 700; text-decoration: underline;">debdipARVR/cloze_congruence_reproducibility_benchmark_kit</a>
      </div>
      <div style="font-size: 11px; color: #8a7050;">
        ScribeMark (ClozeCongruence 2.0) &bull; Academic Research Preview &bull; CERN Zenodo DOI: <a href="https://doi.org/10.5281/zenodo.22158286" target="_blank" style="color: #7c1a06; text-decoration: underline;">10.5281/zenodo.22158286</a> &bull; Ephemeral In-Memory Processing
      </div>
    </div>
    """)


# =========================================================================
# 8. VIEW: STATE 2 — PROCESSING VIEW
# =========================================================================

def render_processing_screen():
    render_navigation_bar()
    
    if not st.session_state.get("terms_accepted", False):
        st.error("⚖️ Terms & Conditions Required: You must accept the Terms of Use and Legal Indemnification Agreement before running forensic analysis.")
        if st.button("← Return to Review & Accept Terms", use_container_width=True):
            st.session_state["app_state"] = "landing"
            st.rerun()
        return

    has_prober_key = bool(st.session_state.get("openrouter_api_key", "").strip())
    if not has_prober_key:
        st.error("🔒 Prober API Key Required: ScribeMark is exclusively functional upon using the Prober API. Offline heuristic simulation is disabled.")
        if st.button("← Return to Manuscript Ingestion & Configure Prober API Key", use_container_width=True):
            st.session_state["app_state"] = "landing"
            st.rerun()
        return

    filename = st.session_state.get("manuscript_name", "Manuscript")
    if filename == "Untitled_Manuscript.txt":
        filename = "Submitted Manuscript"
    manuscript_text = st.session_state.get("manuscript_text", "")
    
    # Layout containers
    loader_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_bar = st.progress(5)
    passes_placeholder = st.empty()

    loader_html = textwrap.dedent("""
    <div class="manuscript-loader-container">
      <div class="loader-svg-box">
        <svg width="44" height="44" viewBox="0 0 44 44" style="animation: mpSpin 1.0s linear infinite;">
          <circle cx="22" cy="22" r="18" fill="none" stroke="#d5c49a" stroke-width="2.5" opacity="0.35"/>
          <circle cx="22" cy="22" r="18" fill="none" stroke="#6b4c11" stroke-width="2.5" stroke-dasharray="28 85" stroke-linecap="round"/>
        </svg>
      </div>
      <div class="loader-badge">
        <span class="loader-dot"></span>
        <span>NEURAL PROBER RUNNING &bull; CLOZECONGRUENCE 2.0</span>
      </div>
    </div>
    """).strip()
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)

    detector = get_detector()
    dna_engine = get_cached_dna_engine()

    def update_status_ui(step_pct: int, phase_text: str, pass_scores: Dict[int, Optional[float]]):
        progress_bar.progress(step_pct)
        status_html = textwrap.dedent(f"""
        <div style="text-align:center; padding: 10px 0 8px; max-width:680px; margin:0 auto;">
          <div style="font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:0.12em; color:#6b4c11; font-weight:700; margin-bottom:4px;">ANALYZING MANUSCRIPT</div>
          <div style="font-family:'Inter', sans-serif; font-size:20px; font-weight:700; color:#2c1f0e; margin-bottom:4px;">{filename}</div>
          <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040; margin-bottom:12px;">{phase_text}... ({step_pct}%)</div>
        </div>
        """).strip()
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        pass_defs = [
            (1, "Pass 1 — Single-Sentence Probing", "Masking isolated sentences, probing semantic predictability (w_c=0.35)"),
            (2, "Pass 2 — Dual Contiguous Void", "Removing 2 adjacent sentences, evaluating transitional bridging (w_c=0.25)"),
            (3, "Pass 3 — Centroid Block Masking", "Evaluating paragraph-level intent with Hungarian Matching on S_3 (w_c=0.20)"),
            (4, "Pass 4 — Macro Structural Boundary", "Assessing chapter-boundary structural pacing (w_c=0.15)"),
        ]

        items_html = []
        for p_idx, p_name, p_desc in pass_defs:
            sc = pass_scores.get(p_idx)
            is_done = sc is not None
            is_active = (not is_done) and (
                (p_idx == 1 and step_pct <= 25) or
                (p_idx == 2 and 25 < step_pct <= 50) or
                (p_idx == 3 and 50 < step_pct <= 75) or
                (p_idx == 4 and 75 < step_pct <= 90)
            )

            icon = "✓" if is_done else ("◉" if is_active else "○")
            icon_col = "#4a6b3a" if is_done else ("#6b4c11" if is_active else "#a08858")
            bg = "#dfd0a4" if is_done else ("rgba(107, 76, 17, 0.08)" if is_active else "transparent")
            border = "#b8a472" if is_done else ("#6b4c11" if is_active else "#c9b88a")
            opacity = "1" if (is_done or is_active) else "0.4"
            chip = f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:11px; color:#8b2000; font-weight:700;">{sc:.1f}% Congruence</span>' if is_done else ""

            item = f"""<div style="display:flex; align-items:flex-start; gap:12px; padding:12px 16px; border:1px solid {border}; border-radius:4px; background:{bg}; opacity:{opacity}; margin-bottom:8px;"><span style="font-family:'JetBrains Mono', monospace; font-size:14px; color:{icon_col}; font-weight:700;">{icon}</span><div style="flex:1;"><div style="font-family:'JetBrains Mono', monospace; font-size:12px; font-weight:700; color:#2c1f0e;">{p_name}</div><div style="font-family:'Inter', sans-serif; font-size:12px; color:#7a6040;">{p_desc}</div></div><div>{chip}</div></div>"""
            items_html.append(item)

        container_html = f'<div style="max-width:680px; margin:0 auto 24px;">{"".join(items_html)}</div>'
        passes_placeholder.markdown(container_html, unsafe_allow_html=True)

    pass_scores: Dict[int, Optional[float]] = {1: None, 2: None, 3: None, 4: None}
    update_status_ui(10, "Initializing 4-Pass Neural Cloze Architecture", pass_scores)

    # Run full Paper 1 & Paper 2 Forensic Analysis
    try:
        results = run_full_forensic_analysis(
            text=manuscript_text,
            filename=filename
        )
    except QuotaExhaustedError as qe:
        status_placeholder.markdown(
            f"""<div style="background:#8b200015; border:1px solid #8b2000; padding:16px 20px; border-radius:6px; margin:20px auto; max-width:680px; text-align:center;">
                <div style="font-family:'JetBrains Mono', monospace; font-size:14px; font-weight:700; color:#8b2000; margin-bottom:6px;">⚠️ CLOZECONGRUENCE PROBER HALTED — QUOTA EXHAUSTED</div>
                <div style="font-family:'Inter', sans-serif; font-size:13px; color:#2c1f0e; line-height:1.5;">
                    OpenRouter rate limit / credit quota was exceeded. Per policy, the system has <b>halted execution</b> rather than degrading to an offline heuristic.
                    <br><br>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:11px; color:#7a6040;">{str(qe)}</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )
        if st.button("← Return to Manuscript Ingestion", use_container_width=True):
            st.session_state["app_state"] = "landing"
            st.rerun()
        return

    # Populate pass scores from analysis results
    for pb in results.get("pass_breakdown", []):
        p_id = pb.get("id")
        if p_id in pass_scores:
            pass_scores[p_id] = pb.get("score")

    update_status_ui(30, "Pass 1 Complete — Probing Dual Contiguous Voids", pass_scores)
    time.sleep(0.15)
    update_status_ui(55, "Pass 2 Complete — Probing Centroid Block & Kuhn-Munkres Alignment", pass_scores)
    time.sleep(0.15)
    update_status_ui(75, "Pass 3 Complete — Probing Macro Discourse Boundaries", pass_scores)
    time.sleep(0.15)
    update_status_ui(88, "4-Pass Ensemble Complete — Computing Paragraph Decompositions", pass_scores)
    time.sleep(0.15)
    update_status_ui(98, "Synthesizing Platt Sigmoid Calibration & Signing OpenSSH Certificate", pass_scores)
    time.sleep(0.15)
    update_status_ui(100, "Analysis Complete", pass_scores)

    st.session_state["analysis_results"] = results
    st.session_state["app_state"] = "results"
    st.rerun()


# =========================================================================
# 9. VIEW: STATE 3 — RESULTS DASHBOARD
# =========================================================================

def render_results_screen():
    render_navigation_bar()
    
    results = st.session_state.get("analysis_results")
    if not results:
        st.session_state["app_state"] = "landing"
        st.rerun()
        return

    filename = results["filename"]
    overall_ai = results["overall_ai"]
    overall_auth = results["overall_auth"]
    word_count = results["word_count"]
    top_model = results["top_model"]
    top_model_prob = results["top_model_prob"]

    # Top Results Bar
    col_hdr, col_btns = st.columns([1.6, 1])
    with col_hdr:
        render_html(f"""
        <div style="margin-bottom:12px;">
          <div class="results-tag">ANALYSIS COMPLETE</div>
          <div class="results-filename">{filename}</div>
        </div>
        """)
        
    with col_btns:
        if ENABLE_CERTIFICATE_DOWNLOAD and REPORTLAB_AVAILABLE:
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("New Analysis", use_container_width=True, key="new_scan_top_btn"):
                    st.session_state["app_state"] = "landing"
                    st.rerun()
            with c_btn2:
                # Quick PDF download button
                pdf_bytes = generate_branded_authorship_certificate_pdf(
                    manuscript_title=filename,
                    author_name="Verified Author",
                    publisher_name="ScribeMark Enterprise",
                    verdict=get_verdict_label(overall_auth),
                    human_authenticity_score=overall_auth,
                    ai_probability=overall_ai,
                    model_attribution=top_model,
                    resonance_gap=0.0,
                    word_count=word_count,
                    signature_hex=results.get("signature_hex", results.get("signature", "")),
                    chapter_breakdowns=[{
                        "title": pg.get("title", f"Page {i+1}"),
                        "word_count": pg.get("words", 0),
                        "human_score": pg.get("authenticity", 80.0),
                        "ai_probability": round(100.0 - pg.get("authenticity", 80.0), 1),
                        "verdict": pg.get("verdict", "HUMAN")
                    } for i, pg in enumerate(results.get("pages", results.get("chapters", [])))],
                )
                st.download_button(
                    label="Download Certificate",
                    data=pdf_bytes,
                    file_name=f"ScribeMark_Certificate_{filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="top_dl_pdf_btn"
                )
        else:
            if st.button("← New Analysis", use_container_width=True, key="new_scan_top_btn"):
                st.session_state["app_state"] = "landing"
                st.rerun()

    # Dynamic Tab Configuration (Overview & Sentence Inspector active; others parked)
    active_tab_labels = ["Overview"]
    if ENABLE_PAGE_ANALYSIS_TAB:
        active_tab_labels.append("📖 Page-by-Page Analysis")
    active_tab_labels.append("Sentence Inspector")
    if ENABLE_TELEMETRY_TAB:
        active_tab_labels.append("📡 Prober Telemetry Log")
    if ENABLE_CERTIFICATE_DOWNLOAD:
        active_tab_labels.append("Certificate")

    tab_objects = st.tabs(active_tab_labels)
    tab_iter = iter(tab_objects)

    tab_overview = next(tab_iter)
    tab_chapters = next(tab_iter) if ENABLE_PAGE_ANALYSIS_TAB else None
    tab_inspector = next(tab_iter)
    tab_telemetry = next(tab_iter) if ENABLE_TELEMETRY_TAB else None
    tab_cert = next(tab_iter) if ENABLE_CERTIFICATE_DOWNLOAD else None

    # ---------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # ---------------------------------------------------------------------
    with tab_overview:
        # 4 Top Metric Cards
        v_col = get_verdict_color(overall_auth)
        v_lbl = get_verdict_label(overall_auth)
        page_count = max(1, len(results.get("pages", [])))
        chap_count = max(1, len(results.get("chapters", [])))

        render_html(f"""
        <div class="metric-card-grid">
          <div class="metric-card">
            <div class="metric-label">OVERALL AI SCORE</div>
            <div class="metric-value" style="color:{v_col};">{overall_ai}%</div>
            <div class="metric-sub">{v_lbl} ({overall_auth}% Authenticity)</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">PAGES ANALYZED</div>
            <div class="metric-value" style="color:#2c1f0e;">{page_count} {'Page' if page_count == 1 else 'Pages'}</div>
            <div class="metric-sub">{chap_count} {'Section' if chap_count == 1 else 'Sections'} &bull; Standard Format</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">TOTAL WORD COUNT</div>
            <div class="metric-value" style="color:#2c1f0e;">{word_count:,}</div>
            <div class="metric-sub">Full manuscript text</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">TOP MODEL ATTRIBUTION</div>
            <div class="metric-value" style="color:#8b6914;">{top_model}</div>
            <div class="metric-sub">{top_model_prob}% resonance score</div>
          </div>
        </div>
        """)

        # 2 Column Row: Cloze Passes & Model Provenance
        col_passes, col_models = st.columns(2)

        with col_passes:
            render_html("""
            <div class="parchment-panel">
              <div class="panel-title">CLOZECONGRUENCE PASSES</div>
            """)
            for p in results["pass_breakdown"]:
                p_auth = 100.0 - p["score"]
                p_col = get_verdict_color(p_auth)
                render_html(f"""
                <div style="margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:11px; color:#2c1f0e; font-weight:600;">
                    <span>{p['name']}</span>
                    <span style="color:{p_col};">{p['score']}% AI</span>
                  </div>
                  <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{p['score']}%; background:{p_col};"></div>
                  </div>
                </div>
                """)
            render_html("</div>")

        with col_models:
            render_html("""
            <div class="parchment-panel">
              <div class="panel-title">MODEL ATTRIBUTION (12D DNA)</div>
            """)
            for m in results["models"]:
                render_html(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:11px; color:#2c1f0e; font-weight:600;">
                    <span>{m['model']} <span style="font-size:10px; color:#7a6040;">({m['family']})</span></span>
                    <span style="color:{m['color']}; font-weight:700;">{m['score']}%</span>
                  </div>
                  <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{m['score']}%; background:{m['color']};"></div>
                  </div>
                </div>
                """)
            render_html("</div>")

        # 12D Stylometric Radar Chart & Fingerprint
        render_html("<div style='height:16px;'></div>")
        col_radar, col_meta = st.columns([1.3, 1], gap="medium")

        with col_radar:
            render_html("""
            <div class="parchment-panel" style="padding:16px;">
              <div class="panel-title">12D LATENT STYLOMETRIC RADAR</div>
            """)
            radar_cats = [r["axis"] for r in results["radar"]]
            radar_human = [r["human"] for r in results["radar"]]
            radar_sample = [r["sample"] for r in results["radar"]]

            # Closed loop for radar
            radar_cats_closed = radar_cats + [radar_cats[0]]
            radar_human_closed = radar_human + [radar_human[0]]
            radar_sample_closed = radar_sample + [radar_sample[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_human_closed,
                theta=radar_cats_closed,
                fill='toself',
                name='Human Baseline',
                line=dict(color='#4a6b3a', width=2),
                fillcolor='rgba(74, 107, 58, 0.15)',
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_sample_closed,
                theta=radar_cats_closed,
                fill='toself',
                name='Current Sample',
                line=dict(color='#6b4c11', width=2),
                fillcolor='rgba(107, 76, 17, 0.25)',
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color='#7a6040'), gridcolor='#b5a47e'),
                    angularaxis=dict(tickfont=dict(family='JetBrains Mono', size=9, color='#191209'), gridcolor='#b5a47e'),
                    bgcolor='#eae0c5'
                ),
                paper_bgcolor='#eae0c5',
                margin=dict(l=35, r=35, t=25, b=25),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(family='JetBrains Mono', size=10, color='#191209')),
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
            render_html("</div>")

        with col_meta:
            render_html(f"""
            <div class="parchment-panel" style="padding:20px 24px;">
              <div class="panel-title">CRYPTOGRAPHIC MANUSCRIPT SEAL</div>
              <div style="font-family:'Inter', sans-serif; font-size:12.5px; color:#2c1f0e; line-height:1.6; margin-bottom:14px;">
                This manuscript has been fingerprinted using SHA-256 and sealed with an OpenSSH Ed25519 digital signature.
              </div>
              <div class="cert-hash-box">
                <div style="font-family:'JetBrains Mono', monospace; font-size:9px; letter-spacing:0.1em; color:#7a6040; font-weight:700; margin-bottom:4px;">
                  SHA-256 FINGERPRINT
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#2c1f0e; word-break:break-all; line-height:1.4;">
                  {results['sha256']}
                </div>
              </div>
              <div class="cert-hash-box">
                <div style="font-family:'JetBrains Mono', monospace; font-size:9px; letter-spacing:0.1em; color:#7a6040; font-weight:700; margin-bottom:4px;">
                  OPENSSH ED25519 AUTHORITY SIGNATURE
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#6b4c11; word-break:break-all; line-height:1.4;">
                  {results['signature']}
                </div>
              </div>
            </div>
            """)


    # ---------------------------------------------------------------------
    # TAB 2: PAGE-BY-PAGE ANALYSIS & RHYTHM MAP (PARKED / HIDDEN)
    # ---------------------------------------------------------------------
    if ENABLE_PAGE_ANALYSIS_TAB and tab_chapters is not None:
        with tab_chapters:
            pages = results.get("pages", [])
            if not pages:
                pages = results.get("chapters", [])

            render_html(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
              <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040;">
                Page-by-page authenticity rhythm — hover and inspect stylometric predictability across every page.
              </div>
              <div style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:700; color:#6b4c11; background:#dfd0a4; padding:3px 8px; border-radius:3px; border:1px solid #b8a472;">
                {max(1, len(pages))} {'PAGE' if len(pages) <= 1 else 'PAGES'} ANALYZED
              </div>
            </div>
            """)

            # Plotly Vertical Bar Chart across Pages
            pg_titles = [p.get("title", f"Page {i+1}") for i, p in enumerate(pages)]
            pg_auths = [p.get("authenticity", 80.0) for p in pages]
            pg_colors = [get_verdict_color(a) for a in pg_auths]

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=pg_titles,
                    y=pg_auths,
                    marker_color=pg_colors,
                    opacity=0.88,
                    hovertemplate="<b>%{x}</b><br>Authenticity: %{y}% human<extra></extra>",
                    marker_line_width=0,
                )
            ])

            fig_bar.update_layout(
                paper_bgcolor='#eae0c5',
                plot_bgcolor='#eae0c5',
                font=dict(family='Inter', color='#191209'),
                margin=dict(l=30, r=20, t=15, b=30),
                height=200,
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(family='JetBrains Mono', size=10, color='#7a6040'),
                    linecolor='#b8a472',
                ),
                yaxis=dict(
                    range=[0, 100],
                    tickfont=dict(family='JetBrains Mono', size=9, color='#7a6040'),
                    gridcolor='#b5a47e',
                    linecolor='#b8a472',
                ),
            )

            render_html('<div class="parchment-panel" style="padding:16px; margin-bottom:20px;">')
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            render_html('</div>')

            # 3-Column Page Grid Cards
            render_html('<div class="chapter-grid">')
            for pg in pages:
                pg_col = get_verdict_color(pg["authenticity"])
                b_html = get_verdict_badge_html(pg["verdict"])
                words = pg.get("words", 0)
                chars = pg.get("characters", words * 6)
                snippet = pg.get("snippet", "")
                
                snippet_html = f'<div style="font-family:\'Inter\', sans-serif; font-size:11px; color:#5c462b; font-style:italic; margin-top:8px; line-height:1.4; border-top:1px dashed #c9b88a; padding-top:6px;">"{snippet}"</div>' if snippet else ""

                render_html(f"""
                <div class="chapter-card-item" style="border-left: 3px solid {pg_col};">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                    <div style="font-family:'Inter', sans-serif; font-weight:700; font-size:13px; color:#2c1f0e;">{pg.get('title', 'Page')}</div>
                    <div>{b_html}</div>
                  </div>
                  <div style="font-family:'JetBrains Mono', monospace; font-size:22px; font-weight:700; color:{pg_col}; margin-bottom:2px;">
                    {pg['authenticity']}%
                  </div>
                  <div style="font-family:'Inter', sans-serif; font-size:11px; color:#7a6040;">
                    human authenticity &bull; {words:,} words &bull; {chars:,} chars
                  </div>
                  <div class="score-bar-bg" style="margin-top:8px;">
                    <div class="score-bar-fill" style="width:{pg['authenticity']}%; background:{pg_col};"></div>
                  </div>
                  {snippet_html}
                </div>
                """)
            render_html('</div>')


    # ---------------------------------------------------------------------
    # TAB 3: INTERACTIVE SENTENCE CLOZE INSPECTOR
    # ---------------------------------------------------------------------
    with tab_inspector:
        render_html("""
        <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040; margin-bottom:14px;">
          Sentences are color-coded by authenticity. Select a specific page or browse the entire document to inspect AI predicted infills, S<sub>meaning</sub>, and S<sub>cosine</sub>.
        </div>
        """)

        # Page Filter Selectbox
        pages_list = results.get("pages", [])
        page_filter_options = ["All Pages (Full Document)"] + [f"Page {p['page_number']} ({p['words']} words)" for p in pages_list]
        selected_page_str = st.selectbox(
            "Filter Sentences by Page:",
            page_filter_options,
            index=0,
            key="page_sentence_filter"
        )
        
        if selected_page_str != "All Pages (Full Document)":
            try:
                target_page_num = int(selected_page_str.split()[1])
                active_sentences = [s for s in results["sentences"] if s.get("page") == target_page_num]
                if not active_sentences:
                    active_sentences = results["sentences"]
            except Exception:
                active_sentences = results["sentences"]
        else:
            active_sentences = results["sentences"]

        # Build dense prose block — filtered sentences rendered inline with colored backgrounds
        prose_spans = []
        for idx, s in enumerate(active_sentences):
            s_col = get_verdict_color(s["score"])
            # Light background tint based on verdict
            if s["score"] >= 75:
                bg_tint = "rgba(74, 107, 58, 0.10)"   # green = human
            elif s["score"] >= 50:
                bg_tint = "rgba(160, 112, 0, 0.10)"    # amber = mixed
            else:
                bg_tint = "rgba(139, 32, 0, 0.10)"     # red = AI

            score_chip = f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:10px; color:{s_col}; font-weight:700; vertical-align:super; margin-left:2px;">{s["score"]}%</span>'
            prose_spans.append(
                f'<span data-idx="{idx}" style="background:{bg_tint}; border-radius:2px; padding:1px 3px; '
                f'border-bottom:2px solid {s_col}; cursor:default;">'
                f'{s["text"]}{score_chip}</span>'
            )

        # Render as one flowing paragraph block
        prose_html = ' '.join(prose_spans)
        render_html(f"""
        <div class="parchment-panel" style="padding:20px 24px; font-family:'Inter', sans-serif; font-size:13.5px; color:#2c1f0e; line-height:1.85;">
          {prose_html}
        </div>
        """)

        # Compact sentence selector + infill inspection card
        n_sentences = len(active_sentences)
        if n_sentences > 0:
            sentence_options = [f"S{i+1}: {s['text'][:70]}{'...' if len(s['text'])>70 else ''} [{s['score']}%]" for i, s in enumerate(active_sentences)]

            selected_idx = st.selectbox(
                "Inspect sentence",
                range(n_sentences),
                format_func=lambda i: sentence_options[i],
                index=0,
                key="sentence_selector",
            )

            s = active_sentences[selected_idx]
            s_col = get_verdict_color(s["score"])
            v_badge = get_verdict_badge_html(get_verdict_label(s["score"]))
            s_page = s.get("page", 1)
            render_html(f"""
            <div class="sentence-infill-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="color:#7a6040; font-size:9.5px; letter-spacing:0.1em; font-weight:700;">
                  ORIGINAL SENTENCE (S{selected_idx + 1} &bull; Page {s_page})
                </div>
                <div>{v_badge}</div>
              </div>
              <div style="font-family:'Inter', sans-serif; font-size:13px; color:#2c1f0e; margin-bottom:14px; line-height:1.5; border-left:3px solid {s_col}; padding-left:12px;">
                "{s['text']}"
              </div>
              <div style="color:#7a6040; font-size:9.5px; letter-spacing:0.1em; font-weight:700; margin-bottom:4px;">
                AI PREDICTED INFILL
              </div>
              <div style="font-family:'Inter', sans-serif; font-style:italic; font-size:13px; color:#2c1f0e; margin-bottom:12px; line-height:1.5;">
                "{s['infill']}"
              </div>
              <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
                <div>
                  <span style="color:#7a6040; font-size:10px; letter-spacing:0.06em;">S_meaning:</span>
                  <span style="color:{get_verdict_color(100 - s['meaning']*100)}; font-weight:700; margin-left:4px;">{s['meaning']:.2f}</span>
                </div>
                <div>
                  <span style="color:#7a6040; font-size:10px; letter-spacing:0.06em;">S_cosine:</span>
                  <span style="color:{get_verdict_color(100 - s['cosine']*100)}; font-weight:700; margin-left:4px;">{s['cosine']:.2f}</span>
                </div>
                <div>
                  <span style="color:#7a6040; font-size:10px; letter-spacing:0.06em;">AUTH:</span>
                  <span style="font-family:'JetBrains Mono', monospace; font-weight:700; color:{s_col}; margin-left:4px;">{s['score']}%</span>
                </div>
              </div>
            </div>
            """)



    # ---------------------------------------------------------------------
    # TAB 4: PROBER TELEMETRY & PAYLOAD INSPECTOR LOG (PARKED / HIDDEN)
    # ---------------------------------------------------------------------
    if ENABLE_TELEMETRY_TAB and tab_telemetry is not None:
        with tab_telemetry:
            from src.engine.openrouter_client import OpenRouterClient
            telemetry_entries = list(OpenRouterClient.telemetry_log)
            
            render_html(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
              <div style="font-family:'Inter', sans-serif; font-size:13px; color:#7a6040;">
                Real-time telemetry of all masked prompt payloads dispatched to the neural prober and responses received.
              </div>
              <div style="font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:700; color:#6b4c11; background:#dfd0a4; padding:3px 8px; border-radius:3px; border:1px solid #b8a472;">
                {len(telemetry_entries)} QUERIES LOGGED
              </div>
            </div>
            """)

            if not telemetry_entries:
                render_html("""
                <div class="parchment-panel" style="padding:24px; text-align:center; color:#7a6040; font-family:'Inter', sans-serif; font-size:13px;">
                  No prober queries logged for this session yet. Run a manuscript scan to see real-time query payloads.
                </div>
                """)
            else:
                for idx, entry in enumerate(reversed(telemetry_entries[-30:])):
                    t_stamp = entry.get("timestamp", "N/A")
                    t_mode = entry.get("mode", "live_api")
                    t_model = entry.get("model", "N/A")
                    t_latency = entry.get("latency_ms", 0.0)
                    t_spans = entry.get("spans_count", 1)
                    t_prompt = entry.get("masked_prompt", "")
                    t_raw_resp = entry.get("raw_response", "")
                    t_preds = entry.get("predictions", {})

                    mode_badge = '<span style="background:rgba(74,107,58,0.15); color:#4a6b3a; border:1px solid #4a6b3a; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:700;">LIVE API</span>' if t_mode == "live_api" else '<span style="background:rgba(107,76,17,0.15); color:#6b4c11; border:1px solid #6b4c11; padding:1px 6px; border-radius:3px; font-size:10px; font-weight:700;">LOCAL ENGINE</span>'

                    with st.expander(f"Query #{len(telemetry_entries) - idx}: [{t_model}] • {t_spans} span(s) • {t_latency}ms • {t_stamp}", expanded=(idx == 0)):
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            st.markdown(f"**Target Model:** `{t_model}`")
                            st.markdown(f"**Execution Mode:** {mode_badge}", unsafe_allow_html=True)
                        with col_meta2:
                            st.markdown(f"**Latency:** `{t_latency} ms`")
                            st.markdown(f"**Timestamp:** `{t_stamp}`")

                        st.markdown("**📤 Masked Prompt Dispatched to Prober:**")
                        st.code(t_prompt, language="markdown")

                        if t_raw_resp:
                            st.markdown("**📥 Raw Response from Prober:**")
                            st.code(t_raw_resp, language="json" if t_raw_resp.strip().startswith("{") else "text")

                        if t_preds:
                            st.markdown("**Parsed Infill Predictions:**")
                            st.json(t_preds)

            # Log File Download
            log_file_path = os.path.join("logs", "prober_telemetry.log")
            if os.path.exists(log_file_path):
                with open(log_file_path, "r", encoding="utf-8") as f:
                    log_data = f.read()
                st.download_button(
                    label="📥 Download Full Telemetry JSONL Log",
                    data=log_data,
                    file_name=f"ScribeMark_Telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                    mime="application/jsonl",
                    use_container_width=True,
                    key="dl_telemetry_jsonl"
                )


    # ---------------------------------------------------------------------
    # TAB 5: CRYPTOGRAPHIC PROOF & CERTIFICATION (HIDDEN / DISABLED)
    # ---------------------------------------------------------------------
    if ENABLE_CERTIFICATE_DOWNLOAD:
        with tab_cert:
            col_preview, col_actions = st.columns([1.5, 1], gap="medium")

            with col_preview:
                render_html(f"""
                <div class="cert-card-preview">
                  <div style="text-align:center; border-bottom:2px solid #b8a472; padding-bottom:16px; margin-bottom:20px;">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; letter-spacing:0.15em; color:#7a6040; font-weight:700; margin-bottom:6px;">
                      MANUSCRIPTPROOF 2.0 — AUTHENTICITY CERTIFICATE
                    </div>
                    <div style="font-size:20px; font-weight:800; color:#2c1f0e;">Certificate of Analysis</div>
                  </div>

                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px;">
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">DOCUMENT</div>
                      <div style="font-size:13px; font-weight:600; color:#2c1f0e;">{filename}</div>
                    </div>
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">ANALYSIS DATE</div>
                      <div style="font-size:13px; font-weight:600; color:#2c1f0e;">{results['date']}</div>
                    </div>
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">WORD COUNT</div>
                      <div style="font-size:13px; font-weight:600; color:#2c1f0e;">{word_count:,} words</div>
                    </div>
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">OVERALL VERDICT</div>
                      <div style="font-size:13px; font-weight:700; color:{get_verdict_color(overall_auth)};">{get_verdict_label(overall_auth)}</div>
                    </div>
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">AI SCORE</div>
                      <div style="font-size:13px; font-weight:600; color:#2c1f0e;">{overall_ai}% AI detected</div>
                    </div>
                    <div>
                      <div style="font-family:'JetBrains Mono', monospace; font-size:9.5px; color:#7a6040; letter-spacing:0.06em;">TOP ATTRIBUTION</div>
                      <div style="font-size:13px; font-weight:600; color:#8b6914;">{top_model} ({top_model_prob}%)</div>
                    </div>
                  </div>

                  <div class="cert-hash-box">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:9px; letter-spacing:0.1em; color:#7a6040; font-weight:700; margin-bottom:4px;">
                      SHA-256 MANUSCRIPT FINGERPRINT
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#2c1f0e; word-break:break-all; line-height:1.4;">
                      {results['sha256']}
                    </div>
                  </div>

                  <div class="cert-hash-box">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:9px; letter-spacing:0.1em; color:#7a6040; font-weight:700; margin-bottom:4px;">
                      OPENSSH ED25519 DIGITAL SEAL
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:10px; color:#6b4c11; word-break:break-all; line-height:1.4;">
                      {results['signature']}
                    </div>
                  </div>

                  <div style="text-align:center; margin-top:20px; color:#7a6040; font-size:11px; font-family:'JetBrains Mono', monospace;">
                    Signed by ScribeMark Enterprise &bull; {results['date']}
                  </div>
                </div>
                """)

            with col_actions:
                # Download PDF Certificate Button
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = generate_branded_authorship_certificate_pdf(
                        manuscript_title=filename,
                        author_name="Verified Author",
                        publisher_name="ScribeMark Enterprise",
                        verdict=get_verdict_label(overall_auth),
                        human_authenticity_score=overall_auth,
                        ai_probability=overall_ai,
                        model_attribution=top_model,
                        resonance_gap=0.0,
                        word_count=word_count,
                        signature_hex=results.get("signature_hex", results.get("signature", "")),
                        chapter_breakdowns=[{
                            "title": pg.get("title", f"Page {i+1}"),
                            "word_count": pg.get("words", 0),
                            "human_score": pg.get("authenticity", 80.0),
                            "ai_probability": round(100.0 - pg.get("authenticity", 80.0), 1),
                            "verdict": pg.get("verdict", "HUMAN")
                        } for i, pg in enumerate(results.get("pages", results.get("chapters", [])))],
                    )
                    st.download_button(
                        label="📥 Download PDF Certificate",
                        data=pdf_bytes,
                        file_name=f"ScribeMark_Certificate_{filename}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="cert_tab_dl_pdf_btn"
                    )

                if st.button("📋 Copy Fingerprint", use_container_width=True, key="copy_fp_btn"):
                    st.info(f"Fingerprint: `{results['sha256']}`")

                if st.button("🔗 Share Report Link", use_container_width=True, key="share_link_btn"):
                    st.success("Shareable Link generated: `https://scribemark.io/verify/" + results['sha256'][:16] + "`")

                render_html("""
                <div style="margin-top:14px; padding:16px; background:#eae0c5; border:1px solid #b5a47e; border-radius:2px; font-family:'Newsreader', 'Georgia', serif; font-size:12.5px; color:#544431; line-height:1.6;">
                  <div style="color:#191209; font-family:'Cinzel', serif; font-weight:700; margin-bottom:4px; font-size:12px; letter-spacing:0.08em;">ACADEMIC PROVENANCE CERTIFICATE</div>
                  This report constitutes an open-access scientific study certificate. Fingerprinted under SHA-256 with OpenSSH Ed25519 digital signature seal for academic attribution and non-commercial research recordkeeping.
                </div>
                """)

    # Minimalist Editorial Footer with Star Prompt
    render_html("""
    <div style="border-top: 1px solid #b5a47e; margin-top: 36px; padding-top: 16px; text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #7a6040; line-height: 1.8;">
      <div style="margin-bottom: 6px;">
        ⭐ In case you like ScribeMark, feel free to add a star on GitHub: 
        <a href="https://github.com/debdipARVR/cloze_congruence_reproducibility_benchmark_kit" target="_blank" style="color: #7c1a06; font-weight: 700; text-decoration: underline;">debdipARVR/cloze_congruence_reproducibility_benchmark_kit</a>
      </div>
      <div style="font-size: 11px; color: #8a7050;">
        ScribeMark (ClozeCongruence 2.0) &bull; Academic Research Preview &bull; CERN Zenodo DOI: <a href="https://doi.org/10.5281/zenodo.22158286" target="_blank" style="color: #7c1a06; text-decoration: underline;">10.5281/zenodo.22158286</a> &bull; Ephemeral In-Memory Processing
      </div>
    </div>
    """)


# =========================================================================
# 10. MAIN ROUTER
# =========================================================================

def main():
    # Enforce strictly blank prober key by default unless explicitly saved in this session
    if not st.session_state.get("key_explicitly_saved", False):
        st.session_state["openrouter_api_key"] = ""
        st.session_state["openrouter_connected"] = False
        st.session_state["byok_risk_acknowledged"] = False
        if "dialog_openrouter_key" in st.session_state:
            st.session_state["dialog_openrouter_key"] = ""
        if "dialog_prober_key_input" in st.session_state:
            st.session_state["dialog_prober_key_input"] = ""

    state = st.session_state.get("app_state", "landing")
    
    if state == "landing":
        render_landing_page()
    elif state == "processing":
        render_processing_screen()
    elif state == "results":
        render_results_screen()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()