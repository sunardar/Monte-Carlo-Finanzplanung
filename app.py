"""
Monte Carlo Finanzplanung Schweiz — Streamlit App
Masterarbeit  — Stochastische Finanzplanung für private Haushalte
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))

@st.cache_resource(show_spinner="Modell wird geladen …")
def lade_engine():
    from monte_carlo_engine import (
        HaushaltParams, PersonParams, berechne_ahv_rente,
        simuliere_szenario, berechne_kennzahlen,
        SZENARIEN, SIM_BIS_ALTER, get_steuerfuss, _df_steuer,
    )
    try:
        _col = _df_steuer.columns[1]
        gemeinden = sorted([str(n).strip() for n in _df_steuer[_col].dropna().tolist() if str(n).strip()])
    except Exception:
        gemeinden = ["Adliswil","Horgen","Küsnacht","Schlieren","Thalwil","Urdorf","Winterthur","Zürich"]
    return (HaushaltParams, PersonParams, berechne_ahv_rente,
            simuliere_szenario, berechne_kennzahlen,
            SZENARIEN, SIM_BIS_ALTER, get_steuerfuss, gemeinden)

(HaushaltParams, PersonParams, berechne_ahv_rente,
 simuliere_szenario, berechne_kennzahlen,
 SZENARIEN, SIM_BIS_ALTER, get_steuerfuss, GEMEINDEN) = lade_engine()

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MC Finanzplanung",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, p, div, span, input, button, label, select, textarea { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f0f4f8; }
.block-container { padding: 1.5rem 2rem; max-width: 1280px; }
.app-header {
    background: linear-gradient(135deg, #0f2744 0%, #1a4a8a 100%);
    color: white; padding: 1.75rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.app-header h1 { font-size: 1.6rem; font-weight: 600; margin: 0 0 0.2rem; }
.app-header p { font-size: 0.85rem; opacity: 0.7; margin: 0; }
.kz-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
.kz-card { flex: 1; min-width: 140px; background: #f7faff; border: 1px solid #dde8f7; border-radius: 8px; padding: 0.9rem 1.1rem; }
.kz-label { font-size: 0.68rem; color: #6b88b5; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
.kz-val { font-size: 1.15rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; color: #0f2744; }
.kz-val.pos { color: #1a6b3a; } .kz-val.neg { color: #b91c1c; } .kz-val.neu { color: #0f2744; }
.kz-diff { font-size: 0.7rem; margin-top: 0.15rem; }
.kz-diff.pos { color: #1a6b3a; } .kz-diff.neg { color: #b91c1c; }
.section-card { background: white; border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; border: 1px solid #dde4ee; box-shadow: 0 1px 4px rgba(15,39,68,0.06); }
.section-title { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #4a6fa5; margin-bottom: 0.9rem; padding-bottom: 0.5rem; border-bottom: 1px solid #eef2f8; }
.stButton > button { background: linear-gradient(135deg, #0f2744, #1a4a8a) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.65rem 2rem !important; font-weight: 500 !important; font-size: 0.9rem !important; width: 100% !important; }
.result-header { background: linear-gradient(135deg, #0f2744, #1a4a8a); color: white; padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; }
.result-header h2 { margin: 0; font-size: 1.3rem; font-weight: 600; }
.result-header p { margin: 0.25rem 0 0; font-size: 0.8rem; opacity: 0.7; }
hr { border-color: #dde4ee; margin: 1rem 0; }
.stTabs [data-baseweb="tab-list"] { background: #eef2f8; padding: 0.3rem; border-radius: 8px; gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; font-weight: 500; font-size: 0.8rem; color: #4a6fa5; padding: 0.35rem 0.9rem; }
.stTabs [aria-selected="true"] { background: white; color: #0f2744; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ── Konfession Mapping ────────────────────────────────────────────────────────
KONFESSION_ANZEIGE = ["Ohne Konfession", "Reformiert", "Katholisch", "Christkatholisch", "Andere"]
KONFESSION_CODE    = {"Ohne Konfession": "ohne", "Reformiert": "ref", "Katholisch": "kath", "Christkatholisch": "chr", "Andere": "ohne"}

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {"page": "input", "resultate": None, "haushalt": None, "n_sim": 10_000}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def fmt_chf(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if abs(v) >= 1_000_000: return f"CHF {v/1e6:.2f} Mio."
    return f"CHF {v:,.0f}".replace(",", "'")

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v*100:.1f} %"

def fmt_alter(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v:.0f} Jahre"

def kz_klasse(v, positiv_gut=True):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "neu"
    if positiv_gut: return "pos" if v >= 0 else "neg"
    return "neg" if v > 0.05 else "pos"

def kz_html(label, val_str, klasse="neu", diff_str=None, diff_klasse="neu"):
    diff_html = f'<div class="kz-diff {diff_klasse}">{diff_str}</div>' if diff_str else ""
    return f'<div class="kz-card"><div class="kz-label">{label}</div><div class="kz-val {klasse}">{val_str}</div>{diff_html}</div>'

# ── Plotly Chart ──────────────────────────────────────────────────────────────
def erstelle_chart(res_basis, res_szenario, szenario_name, alter_start, median_tod):
    vp_b = res_basis["verm_pfade"]; vp_s = res_szenario["verm_pfade"]
    n_jahre = res_szenario["n_jahre"]
    alter = list(range(alter_start, alter_start + n_jahre + 1))

    def pct(vp, p): return np.nanpercentile(vp, p, axis=0) / 1e6

    b_med = pct(vp_b, 50); b_p25 = pct(vp_b, 25); b_p75 = pct(vp_b, 75)
    s_med = pct(vp_s, 50); s_p10 = pct(vp_s, 10); s_p25 = pct(vp_s, 25)
    s_p75 = pct(vp_s, 75); s_p90 = pct(vp_s, 90)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=alter+alter[::-1], y=list(b_p75)+list(b_p25[::-1]),
        fill='toself', fillcolor='rgba(148,163,184,0.12)', line=dict(color='rgba(0,0,0,0)'),
        name='Basis P25–P75', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=alter+alter[::-1], y=list(s_p90)+list(s_p10[::-1]),
        fill='toself', fillcolor='rgba(59,130,246,0.08)', line=dict(color='rgba(0,0,0,0)'),
        name='Szenario P10–P90', showlegend=True, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=alter+alter[::-1], y=list(s_p75)+list(s_p25[::-1]),
        fill='toself', fillcolor='rgba(29,78,216,0.14)', line=dict(color='rgba(0,0,0,0)'),
        name='Szenario P25–P75', showlegend=True, hoverinfo='skip'))

    hover_b = [f"<b>Alter {a}</b><br>Basis Median: CHF {v*1e6:,.0f}".replace(",","'") for a, v in zip(alter, b_med)]
    fig.add_trace(go.Scatter(x=alter, y=b_med, mode='lines',
        line=dict(color='#94a3b8', width=1.8, dash='dash'),
        name='Basis (Median)', hovertemplate='%{customdata}<extra></extra>', customdata=hover_b))

    for i in range(len(alter)-1):
        farbe = '#dc2626' if (s_med[i] < 0 or s_med[i+1] < 0) else '#1d4ed8'
        fig.add_trace(go.Scatter(x=[alter[i], alter[i+1]], y=[s_med[i], s_med[i+1]],
            mode='lines', line=dict(color=farbe, width=2.5), showlegend=False, hoverinfo='skip'))

    hover_s = [f"<b>Alter {a}</b><br>{szenario_name}: CHF {v*1e6:,.0f}".replace(",","'") for a, v in zip(alter, s_med)]
    fig.add_trace(go.Scatter(x=alter, y=s_med, mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=10),
        name=f'{szenario_name} (Median)',
        hovertemplate='%{customdata}<extra></extra>', customdata=hover_s))

    fig.add_hline(y=0, line_dash="dot", line_color="#dc2626", line_width=0.8, opacity=0.5)
    if alter_start < 85 < alter_start + n_jahre:
        fig.add_vline(x=85, line_dash="dash", line_color="#64748b", line_width=1, opacity=0.4,
                      annotation_text="Alter 85", annotation_position="top right",
                      annotation_font_size=10, annotation_font_color="#64748b")
    if median_tod and alter_start < median_tod < alter_start + n_jahre:
        fig.add_vline(x=median_tod, line_dash="dot", line_color="#0f2744", line_width=1, opacity=0.5,
                      annotation_text=f"Ø Tod {median_tod:.0f}J", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#0f2744")

    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', font_family='IBM Plex Sans',
        xaxis=dict(title='Alter', gridcolor='#f1f5f9', tickfont=dict(size=10)),
        yaxis=dict(title='Liquides Vermögen (Mio. CHF)', gridcolor='#f1f5f9', tickfont=dict(size=10), tickformat='.1f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=9)),
        margin=dict(l=60, r=20, t=40, b=50), hovermode='x unified', height=420,
    )
    return fig

# ── Kennzahlen ────────────────────────────────────────────────────────────────
def zeige_kennzahlen(kz_s, kz_b, szenario_name):
    def diff(v_s, v_b, is_pct=False, positiv_gut=True):
        if np.isnan(v_s) or np.isnan(v_b): return None, "neu"
        d = v_s - v_b
        if is_pct: d_str = f"{'▲' if d > 0 else '▼'} {abs(d)*100:.1f} PP vs. Basis"
        else: d_str = f"{'▲' if d > 0 else '▼'} CHF {abs(d)/1e3:,.0f}k vs. Basis".replace(",","'")
        d_kl = ("pos" if d > 0 else "neg") if positiv_gut else ("neg" if d > 0 else "pos")
        return d_str, d_kl

    v85_s  = kz_s.get("median_netto_85", float("nan"))
    v85_b  = kz_b.get("median_netto_85", float("nan"))
    ruin_s = kz_s.get("p_ruin_bedingt_85", float("nan"))
    luecke_s = kz_s.get("p_vorsorgeluecke_85", float("nan"))
    tod_s  = kz_s.get("median_tod_alter", float("nan"))
    d85_str, d85_kl = diff(v85_s, v85_b, positiv_gut=True)
    d_ruin_str, d_ruin_kl = diff(ruin_s, kz_b.get("p_ruin_bedingt_85", float("nan")), is_pct=True, positiv_gut=False)

    html = '<div class="kz-row">'
    html += kz_html("Netto-Position Alter 85", fmt_chf(v85_s), kz_klasse(v85_s), d85_str, d85_kl)
    html += kz_html("Wahrsch. Verm. aufgebraucht bis 85", fmt_pct(luecke_s), kz_klasse(luecke_s, False))
    html += kz_html("P(Ruin | lebt bis 85)", fmt_pct(ruin_s), kz_klasse(ruin_s, False), d_ruin_str, d_ruin_kl)
    html += kz_html("Median Vermögen Alter 65", fmt_chf(kz_s.get("p50_65", float("nan"))), "neu")
    html += kz_html("Median Vermögen Alter 75", fmt_chf(kz_s.get("p50_75", float("nan"))), "neu")
    html += kz_html("Median Vermögen Alter 85", fmt_chf(kz_s.get("p50_85", float("nan"))), "neu")
    html += kz_html("Median Todesalter", fmt_alter(tod_s), "neu")
    html += kz_html("P(Tod vor 85)", fmt_pct(kz_s.get("p_tod_bis_85", float("nan"))), "neu")
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SEITE: EINGABE
# ═══════════════════════════════════════════════════════════════════════════════
def seite_eingabe():
    st.markdown("""
    <div class="app-header">
        <h1>Monte Carlo Finanzplanung Schweiz</h1>
        <p>Stochastische Langfristplanung für private Haushalte &mdash; Kanton Zürich</p>
    </div>""", unsafe_allow_html=True)

    with st.form("main_form"):

        # ── Person A ──────────────────────────────────────────────────────────
        with st.expander("Person A", expanded=True):
            c1,c2,c3,c4 = st.columns(4)
            name_a    = c1.text_input("Name", "Person A", key="na")
            geb_a     = c2.number_input("Geburtsjahr", min_value=1900, max_value=2005, value=1978, key="ga")
            gesch_a   = c3.selectbox("Geschlecht", ["Männlich","Weiblich"], key="gsa")
            bildung_a = c4.selectbox("Bildung", ["Grundbildung","Berufsbildung","Tertiär"], index=1, key="ba")

            c1,c2,c3,c4 = st.columns(4)
            eink_a     = c1.number_input("Bruttoeinkommen CHF/Jahr", value=120_000, min_value=0, step=5_000, key="ea")
            ra_a       = c2.number_input("Rentenalter", min_value=58, max_value=70, value=65, key="raa")
            vorbezug_a = c3.number_input("AHV Vorbezug Jahre (0–2)", min_value=0, max_value=2, value=0, key="vba")
            aufschub_a = c4.number_input("AHV Aufschub Jahre (0–5)", min_value=0, max_value=5, value=0, key="aua")

            c1,c2,c3,c4 = st.columns(4)
            ahv_a       = c1.number_input("Erwartete AHV-Rente CHF/Jahr", value=26_000, min_value=0, step=500, key="ahva")
            pk_akt_a    = c2.number_input("PK-Guthaben aktuell CHF", value=250_000, step=10_000, key="pka")
            pk_65_a     = c3.number_input("PK-Guthaben Alter 65 CHF (Vorsorgeausweis)", value=650_000, step=10_000, key="pk65a")
            pk_heirat_a = c4.number_input("PK-Guthaben bei Heirat CHF", value=0, step=5_000, key="pkha")

            c1,c2,c3,c4 = st.columns(4)
            umwandlung_a = c1.number_input("Umwandlungssatz %", min_value=4.0, max_value=8.0, value=6.8, step=0.1, format="%.1f", key="uwa") / 100
            kapbezug_a   = c2.number_input("Anteil Kapitalbezug PK %", min_value=0, max_value=100, value=50, step=10, key="kba") / 100
            s3_a         = c3.number_input("Säule 3a Guthaben CHF", value=50_000, step=5_000, key="s3a")
            s3_einz_a    = c4.number_input("Säule 3a Einzahlung/Jahr CHF (max. 7'258)", min_value=0, max_value=7_258, value=7_258, step=100, key="s3ea")

        # ── Person B ──────────────────────────────────────────────────────────
        with st.expander("Person B (optional)", expanded=False):
            hat_partner = st.checkbox("Partner / Partnerin erfassen", True, key="hp")
            if hat_partner:
                c1,c2,c3,c4 = st.columns(4)
                name_b    = c1.text_input("Name", "Person B", key="nb")
                geb_b     = c2.number_input("Geburtsjahr", min_value=1900, max_value=2005, value=1983, key="gb")
                gesch_b   = c3.selectbox("Geschlecht", ["Weiblich","Männlich"], key="gsb")
                bildung_b = c4.selectbox("Bildung", ["Grundbildung","Berufsbildung","Tertiär"], index=1, key="bb")

                c1,c2,c3,c4 = st.columns(4)
                eink_b     = c1.number_input("Bruttoeinkommen CHF/Jahr", value=90_000, min_value=0, step=5_000, key="eb")
                ra_b       = c2.number_input("Rentenalter", min_value=58, max_value=70, value=65, key="rab")
                vorbezug_b = c3.number_input("AHV Vorbezug Jahre (0–2)", min_value=0, max_value=2, value=0, key="vbb")
                aufschub_b = c4.number_input("AHV Aufschub Jahre (0–5)", min_value=0, max_value=5, value=0, key="aub")

                c1,c2,c3,c4 = st.columns(4)
                ahv_b       = c1.number_input("Erwartete AHV-Rente CHF/Jahr", value=22_000, min_value=0, step=500, key="ahvb")
                pk_akt_b    = c2.number_input("PK-Guthaben aktuell CHF", value=180_000, step=10_000, key="pkb")
                pk_65_b     = c3.number_input("PK-Guthaben Alter 65 CHF", value=450_000, step=10_000, key="pk65b")
                pk_heirat_b = c4.number_input("PK-Guthaben bei Heirat CHF", value=0, step=5_000, key="pkhb")

                c1,c2,c3,c4 = st.columns(4)
                umwandlung_b = c1.number_input("Umwandlungssatz %", min_value=4.0, max_value=8.0, value=6.8, step=0.1, format="%.1f", key="uwb") / 100
                kapbezug_b   = c2.number_input("Anteil Kapitalbezug PK %", min_value=0, max_value=100, value=50, step=10, key="kbb") / 100
                s3_b         = c3.number_input("Säule 3a Guthaben CHF", value=30_000, step=5_000, key="s3b")
                s3_einz_b    = c4.number_input("Säule 3a Einzahlung/Jahr CHF (max. 7'258)", min_value=0, max_value=7_258, value=7_258, step=100, key="s3eb")
            else:
                name_b="Person B"; geb_b=1983; gesch_b="Weiblich"; bildung_b="Berufsbildung"
                eink_b=0; ra_b=65; vorbezug_b=0; aufschub_b=0; ahv_b=0
                pk_akt_b=0; pk_65_b=0; pk_heirat_b=0; umwandlung_b=0.068; kapbezug_b=0.5
                s3_b=0; s3_einz_b=0

        # ── Haushalt & Zivilstand ─────────────────────────────────────────────
        with st.expander("Haushalt & Zivilstand", expanded=True):
            c1,c2,c3,c4 = st.columns(4)
            zivilstand   = c1.selectbox("Zivilstand", ["verheiratet","ledig","geschieden","verwitwet"], key="zs")
            if zivilstand == "verheiratet":
                heiratsjahr = c2.number_input("Heiratsjahr", min_value=1900, max_value=date.today().year, value=2010, key="hj")
            else:
                heiratsjahr = None
                c2.markdown("&nbsp;")
            konfession_a_anzeige = c3.selectbox("Konfession Person A", KONFESSION_ANZEIGE, key="ka")
            konfession_b_anzeige = c4.selectbox("Konfession Person B", KONFESSION_ANZEIGE, key="kb")

            c1,c2,c3,c4 = st.columns(4)
            liquide    = c1.number_input("Liquides Vermögen CHF", value=1_000_000, step=50_000, key="lv")
            ausgaben   = c2.number_input("Jahresausgaben CHF/Jahr", value=80_000, min_value=0, step=5_000, key="jav")
            eigengut_a = c3.number_input("Eigengut Person A CHF", value=0, step=10_000, key="ega")
            eigengut_b = c4.number_input("Eigengut Person B CHF", value=0, step=10_000, key="egb")

            c1,c2 = st.columns(2)
            eiserne_reserve = c1.number_input("Eiserne Reserve CHF", value=50_000, step=5_000, key="er")
            gemeinde = c2.selectbox("Wohngemeinde", GEMEINDEN,
                                    index=GEMEINDEN.index("Urdorf") if "Urdorf" in GEMEINDEN else 0, key="gem")

        # ── Liegenschaft ──────────────────────────────────────────────────────
        with st.expander("Liegenschaft (optional)", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            lieg      = c1.number_input("Verkehrswert CHF", value=900_000, min_value=0, step=50_000, key="lw")
            hypo      = c2.number_input("Hypothek CHF", value=600_000, min_value=0, step=50_000, key="hy")
            hypo_zins = c3.number_input("Hypothekarzins %", min_value=0.1, max_value=10.0, value=2.5, step=0.1, format="%.1f", key="hz") / 100
            jahr_kauf = c4.number_input("Kaufjahr", min_value=1900, max_value=date.today().year, value=2010, key="jk")

            c1,c2 = st.columns(2)
            eigenmietwert    = c1.number_input("Eigenmietwert CHF/Jahr (aus Steuererklärung)", value=22_500, min_value=0, step=500, key="emw")
            ist_ersterwerber = c2.checkbox("Ersterwerber (Steuerausnahme 10 Jahre)", False, key="ee")

        # ── Portfolio & Simulation ─────────────────────────────────────────────
        with st.expander("Portfolio & Simulation", expanded=True):
            c1,c2 = st.columns(2)
            risiko    = c1.select_slider("Portfolio freies Vermögen",
                                         ["Konservativ","Ausgewogen","Wachstum"], "Ausgewogen", key="rs")
            risiko_s3 = c2.select_slider("Portfolio Säule 3a",
                                          ["Konservativ","Ausgewogen","Wachstum"], "Konservativ", key="rs3")

        # ── Einmalzahlungen ───────────────────────────────────────────────────
        with st.expander("Geplante Einmalzahlungen", expanded=False):
            st.markdown("**Einmalausgaben** — leere Zeilen werden ignoriert")
            aus_j = []; aus_b = []
            for i in range(5):
                ca, cb = st.columns(2)
                aus_j.append(ca.number_input(f"Jahr Ausgabe {i+1}", min_value=date.today().year, max_value=2100, value=2030+i*2, key=f"ausj{i}"))
                aus_b.append(cb.number_input(f"Betrag CHF Ausgabe {i+1}", min_value=0, value=0, step=5_000, key=f"ausb{i}"))
            st.markdown("**Einmaleinnahmen** — leere Zeilen werden ignoriert")
            ein_j = []; ein_b = []
            for i in range(5):
                ca, cb = st.columns(2)
                ein_j.append(ca.number_input(f"Jahr Einnahme {i+1}", min_value=date.today().year, max_value=2100, value=2028+i*2, key=f"einj{i}"))
                ein_b.append(cb.number_input(f"Betrag CHF Einnahme {i+1}", min_value=0, value=0, step=5_000, key=f"einb{i}"))

        # ── Submit ────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        cs, cn = st.columns([3, 1])
        with cn:
            n_sim_wahl = st.selectbox("Anzahl Simulationen", [1_000, 10_000],
                                      index=1, format_func=lambda x: f"{x:,}".replace(",","'"), key="nsim")
        with cs:
            st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Simulation starten — alle 8 Szenarien berechnen",
                                              use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        einmalausgaben_list = [(aus_j[i], aus_b[i]) for i in range(5) if aus_b[i] > 0]
        einmaleinnahmen_list = [(ein_j[i], ein_b[i]) for i in range(5) if ein_b[i] > 0]
        bmap = {"Grundbildung":1,"Berufsbildung":2,"Tertiär":3}
        gmap = {"Männlich":"m","Weiblich":"f"}
        rmap = {"Konservativ":1,"Ausgewogen":2,"Wachstum":3}

        konfession_a = KONFESSION_CODE[konfession_a_anzeige]
        konfession_b = KONFESSION_CODE[konfession_b_anzeige]

        person_a = PersonParams(
            name=name_a, geburtsjahr=geb_a, geschlecht=gmap[gesch_a], bildung=bmap[bildung_a],
            einkommen=float(eink_a), rentenalter=ra_a,
            pk_guthaben=float(pk_akt_a), pk_guthaben_65=float(pk_65_a), pk_bei_heirat=float(pk_heirat_a),
            pk_umwandlungssatz=umwandlung_a, pk_bezug_kapital_anteil=kapbezug_a,
            saeule3=float(s3_a), saeule3_einzahlung=float(s3_einz_a),
            ahv_rente_erwartet=float(ahv_a), ahv_vorbezug_jahre=int(vorbezug_a), ahv_aufschub_jahre=int(aufschub_a),
        )

        person_b = None
        if hat_partner:
            person_b = PersonParams(
                name=name_b, geburtsjahr=geb_b, geschlecht=gmap.get(gesch_b,"f"), bildung=bmap[bildung_b],
                einkommen=float(eink_b), rentenalter=ra_b,
                pk_guthaben=float(pk_akt_b), pk_guthaben_65=float(pk_65_b), pk_bei_heirat=float(pk_heirat_b),
                pk_umwandlungssatz=umwandlung_b, pk_bezug_kapital_anteil=kapbezug_b,
                saeule3=float(s3_b), saeule3_einzahlung=float(s3_einz_b),
                ahv_rente_erwartet=float(ahv_b), ahv_vorbezug_jahre=int(vorbezug_b), ahv_aufschub_jahre=int(aufschub_b),
            )

        # Zivilstand fürs Modell: verheiratet oder ledig (für Steuer/AHV)
        zs_model = "verheiratet" if zivilstand == "verheiratet" and hat_partner else "ledig"
        # Initialer civil_status für Arbeitslosigkeits-Logit
        zs_logit_map = {"verheiratet": "married", "ledig": "single", "geschieden": "divorced", "verwitwet": "widowed"}
        zs_logit = zs_logit_map.get(zivilstand, "single")
        marriage_dur = (date.today().year - heiratsjahr) if (zivilstand=="verheiratet" and hat_partner and heiratsjahr) else 0

        haushalt = HaushaltParams(
            person_a=person_a, person_b=person_b,
            zivilstand=zs_model,
            heiratsjahr=heiratsjahr if zivilstand=="verheiratet" and hat_partner else None,
            marriage_duration=max(0, marriage_dur),
            liquides_vermoegen=float(liquide),
            eigengut_a=float(eigengut_a),
            eigengut_b=float(eigengut_b) if hat_partner else 0.0,
            eiserne_reserve=float(eiserne_reserve),
            liegenschaft=float(lieg), hypothek=float(hypo), hypothek_zins=hypo_zins,
            eigenmietwert=float(eigenmietwert), ist_ersterwerber=ist_ersterwerber,
            jahr_kauf=int(jahr_kauf),
            ausgaben=float(ausgaben),
            einmalausgaben=einmalausgaben_list,
            einmaleinnahmen=einmaleinnahmen_list,
            gemeinde=gemeinde, konfession_a=konfession_a, konfession_b=konfession_b,
            risikoaversion=rmap[risiko], risikoaversion_saeule3=rmap[risiko_s3],
            zivilstand_logit_a=zs_logit,
        )

        n_sim = int(n_sim_wahl)
        with st.spinner(f"Simulation läuft — 8 Szenarien × {n_sim:,} Simulationen …".replace(",","'")):
            resultate = {}
            progress = st.progress(0)
            for i, (name, flags) in enumerate(SZENARIEN.items()):
                res = simuliere_szenario(name, flags, haushalt, n_sim=n_sim)
                resultate[name] = {**res, "kennzahlen": berechne_kennzahlen(res)}
                progress.progress((i+1) / len(SZENARIEN))
            progress.empty()

        st.session_state["resultate"] = resultate
        st.session_state["haushalt"]  = haushalt
        st.session_state["n_sim"]     = n_sim
        st.session_state["page"]      = "results"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SEITE: RESULTATE
# ═══════════════════════════════════════════════════════════════════════════════
def seite_resultate():
    resultate   = st.session_state["resultate"]
    haushalt    = st.session_state["haushalt"]
    n_sim       = st.session_state.get("n_sim", 10_000)
    alter_start = haushalt.alter_a()
    res_basis   = resultate["Basis"]
    kz_basis    = res_basis["kennzahlen"]
    name_a = haushalt.person_a.name
    name_b = haushalt.person_b.name if haushalt.has_partner else ""
    titel  = f"{name_a}" + (f" & {name_b}" if name_b else "")

    st.markdown(f"""
    <div class="result-header">
        <h2>Simulationsresultate — {titel}</h2>
        <p>{n_sim:,} Monte-Carlo-Simulationen · Alter {alter_start}–100 · {haushalt.gemeinde}</p>
    </div>""".replace(",","'"), unsafe_allow_html=True)

    if st.button("← Eingaben anpassen"):
        st.session_state["page"] = "input"
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    szen_namen = [n for n in resultate.keys() if n != "Basis"]
    gewaehltes = st.selectbox("Szenario vergleichen mit Basis:", szen_namen, key="szen_choice")

    res_s = resultate[gewaehltes]
    kz_s  = res_s["kennzahlen"]
    median_tod = kz_s.get("median_tod_alter", None)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Kennzahlen — Basis vs. {gewaehltes}</div>', unsafe_allow_html=True)
    zeige_kennzahlen(kz_s, kz_basis, gewaehltes)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Vermögensentwicklung — Basis (grau gestrichelt) vs. {gewaehltes} (blau)</div>', unsafe_allow_html=True)
    fig = erstelle_chart(res_basis, res_s, gewaehltes, alter_start, median_tod)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Vermögen nach Alter — Pessimistisch / Median / Optimistisch</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Basis", gewaehltes])
    for tab, (res, label) in zip(tabs, [(res_basis,"Basis"),(res_s,gewaehltes)]):
        with tab:
            vp = res["verm_pfade"]; n_jahre = res["n_jahre"]
            rows = []
            for cp in [65, 70, 75, 80, 85, 90, 95]:
                if alter_start < cp <= alter_start + n_jahre:
                    d = vp[:, cp - alter_start]; d = d[~np.isnan(d)]
                    if len(d) > 0:
                        rows.append({"Alter": cp,
                                     "P10": fmt_chf(np.percentile(d,10)),
                                     "P25": fmt_chf(np.percentile(d,25)),
                                     "Median": fmt_chf(np.percentile(d,50)),
                                     "P75": fmt_chf(np.percentile(d,75)),
                                     "P90": fmt_chf(np.percentile(d,90))})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Alter"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Übersicht alle 8 Szenarien</div>', unsafe_allow_html=True)
    v85_b = kz_basis.get("median_netto_85", float("nan"))
    rows = []
    for name, res in resultate.items():
        kz = res["kennzahlen"]
        v85 = kz.get("median_netto_85", float("nan"))
        delta = (v85-v85_b) if not (np.isnan(v85) or np.isnan(v85_b)) else float("nan")
        rows.append({
            "Szenario": name,
            "Netto-Position 85": fmt_chf(v85),
            "Δ vs. Basis": f"CHF {delta/1e3:+,.0f}k".replace(",","'") if not np.isnan(delta) else "—",
            "Wahrsch. aufgebraucht bis 85": fmt_pct(kz.get("p_vorsorgeluecke_85", float("nan"))),
            "P(Ruin | lebt 85)": fmt_pct(kz.get("p_ruin_bedingt_85", float("nan"))),
            "P(Tod vor 85)": fmt_pct(kz.get("p_tod_bis_85", float("nan"))),
            "Median Todesalter": fmt_alter(kz.get("median_tod_alter", float("nan"))),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Szenario"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state["page"] == "results" and st.session_state["resultate"]:
    seite_resultate()
else:
    seite_eingabe()