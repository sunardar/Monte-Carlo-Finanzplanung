#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO FINANZPLANUNG SCHWEIZ
# Masterarbeit — Stochastische Finanzplanung für private Haushalte
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO FINANZPLANUNG SCHWEIZ
# Masterarbeit — Stochastische Finanzplanung für private Haushalte
# ═══════════════════════════════════════════════════════════════════════════

import numpy  as np
import pandas as pd
import matplotlib.pyplot    as plt
import matplotlib.ticker    as mtick
from pathlib  import Path
from datetime import date, datetime
from dataclasses import dataclass, field
from typing  import Optional
from joblib  import Parallel, delayed
import multiprocessing
import time

# ── Pfade ─────────────────────────────────────────────────────────────────────
NOTEBOOK_DIR = Path().resolve()
DATA_DIR     = NOTEBOOK_DIR

# ── Simulation ────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_SIM         = 10_000
SIM_BIS_ALTER = 100

# ── AHV ───────────────────────────────────────────────────────────────────────
# Quelle: EAK eak.admin.ch (2026)
AHV_RENTENALTER_M      = 65
AHV_RENTENALTER_F      = 65
AHV_ZAHLUNGEN_PRO_JAHR = 13
AHV_MAX_RENTE_MONAT    = 2_520
AHV_MAX_RENTE          = AHV_MAX_RENTE_MONAT * AHV_ZAHLUNGEN_PRO_JAHR
AHV_MIN_RENTE          = AHV_MAX_RENTE // 2
AHV_PLAFOND_EHEPAAR    = int(AHV_MAX_RENTE_MONAT * 1.5) * AHV_ZAHLUNGEN_PRO_JAHR

AHV_VORBEZUG_ABZUG     = {1: 0.068, 2: 0.136}
AHV_VORBEZUG_MIN_ALTER = 63

AHV_AUFSCHUB_ZUSCHLAG  = {
    1: 0.052, 2: 0.108, 3: 0.171, 4: 0.240, 5: 0.315,
}
AHV_AUFSCHUB_MAX_JAHRE = 5

# Steuern
AHV_IV_EO_SATZ = 0.053    # AHV/IV/EO Arbeitnehmeranteil
ALV_SATZ       = 0.011    # ALV Arbeitnehmeranteil
ALV_MAX_LOHN   = 148_200  # Maximaler versicherter Verdienst ALV/NBUV
NBUV_SATZ      = 0.0106   # NBUV Arbeitnehmeranteil

# ── BVG ───────────────────────────────────────────────────────────────────────
KOORDINATIONSABZUG = 26_460
BVG_ZINS           = 0.0125

def altersgutschrift(age: int) -> float:
    if   age < 25:  return 0.00
    elif age < 35:  return 0.07
    elif age < 45:  return 0.10
    elif age < 55:  return 0.15
    else:           return 0.18

# ── Hypothek ──────────────────────────────────────────────────────────────────
HYPOTHEK_ZINS  = 0.025
AMORT_SCHWELLE = 2 / 3

# ── Liegenschaften ────────────────────────────────────────────────────────────
LIEGEN_MU    = 0.0317   # IAZI Private Real Estate Price Index, Ø 1998–2025
LIEGEN_SIGMA = 0.0275   # IAZI Private Real Estate Price Index, Std. 1998–2025
EIGENMIETWERT_ABSCHAFFUNG_JAHR = 2029

# ── Pflegefall ────────────────────────────────────────────────────────────────
# Zustände
ZUHAUSE      = 0
SPITEX_STATE = 1
HEIM_STATE   = 2

# KVG Patientenbeitrag
KVG_PAT_TAG_CHF = 23.00

# EL-Konstanten — Quelle: ELG SR 831.30, Stand 2024
EL_EINTRITTSSCHWELLE_LEDIG       = 100_000   # Art. 9a ELG
EL_EINTRITTSSCHWELLE_VERHEIRATET = 200_000   # Art. 9a ELG
EL_FREIBETRAG_LEDIG              =  30_000   # Art. 11 Abs. 1 lit. c ELG
EL_FREIBETRAG_VERHEIRATET        =  50_000   # Art. 11 Abs. 1 lit. c ELG
EL_VERMOEGENSVERZEHR_ZUHAUSE     =  1 / 10
EL_VERMOEGENSVERZEHR_HEIM        =  1 / 5
EL_AUSLAGEN                      =  6_800    # ELV Art. 16 Abs. 2

# Spitex-Episodendauer — werden in Cell 3 aus SHP-Rohdaten überschrieben
D_SPITEX_MU   = 0.5518
D_SPITEX_SIG  = 0.2938
D_SPITEX_MEAN = 1.81

# Kostensteigerung — werden in Cell 3 aus KTZH-Daten gesetzt
# NEU: separate Parameter für Heim und Spitex
HEIM_KOSTEN_MU     = None   # wird in Cell 3 gesetzt
HEIM_KOSTEN_SIGMA  = None   # wird in Cell 3 gesetzt
SPITEX_KOSTEN_MU   = None   # wird in Cell 3 gesetzt 
SPITEX_KOSTEN_SIGMA = None  # wird in Cell 3 gesetzt 

# ── Haushaltslogik ────────────────────────────────────────────────────────────
AUSGABEN_REDUKTION_TOD_PARTNER = 0.30
HEIM_AUSGABEN_ANTEIL           = 0.30
P_SPITEX_HEIM = 0.159

# ── Arbeitslosigkeit ──────────────────────────────────────────────────────────
AGE_MEAN_AL = 44.2

PARAMS_AL = {
    "Intercept":     -5.0752,
    "age_c":         -0.0138,
    "age_c_sq":       0.0012,
    "female":         0.0762,
    "civ_single":     0.5748,
    "civ_separated":  0.9538,
    "civ_divorced":   0.6386,
    "civ_widowed":    0.5702,
    "edu_basic":      0.2633,
    "edu_tertiary":  -0.0607,
}

Q_EXIT_BY_AGE = {
    "18–24": 0.8837, "25–29": 0.8630, "30–34": 0.8392, "35–39": 0.7778,
    "40–44": 0.8274, "45–49": 0.8008, "50–54": 0.7991, "55–64": 0.8140,
}
Q_BY_DURATION = {1: 0.8571, 2: 0.7462, 3: 0.6818, 4: 0.6190}
Q_DURATION_DEFAULT = 0.6190

WAGE_SCAR = {0: -0.10, 1: -0.06}

# Konjunkturkomponente — aus Arbeitslosigkeit1.ipynb

BIP_MU    = 1.936    # Ø BIP-Realwachstum 1999–2024 (%)
BIP_STD   = 1.812    # Std BIP-Realwachstum 1999–2024 (%)
BIP_ALPHA = -0.0823  # Logit-Koeffizient bip_dev, geschätzt aus SHP+BIP

# ── Scheidung ─────────────────────────────────────────────────────────────────
AGE_MEAN_DIV = 48.6

PARAMS_DIV = {
    "Intercept":  -4.4451,
    "age_c":      -0.0414,
    "age_c_sq":   -0.0021,
    "female":      0.0070,
    "edu_basic":  -0.1308,
    "unemployed":  1.1401,
}
P_MEAN_LOGIT_DIV = 0.00885

P_DIVORCE_BY_DURATION = {
    "0–4 Jahre":   0.009524,
    "5–9 Jahre":   0.019780,
    "10–14 Jahre": 0.015904,
    "15–19 Jahre": 0.012151,
    "20+ Jahre":   0.001479,
}


# In[2]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1 — STEUERFUNKTIONEN
# Quellen:
#   §35 StG ZH     — Einkommensteuer Kanton Zürich
#   §47 StG ZH     — Vermögenssteuer Kanton Zürich
#   SR 642.11 DBG  — Direkte Bundessteuer (Fedlex, vom Nutzer bereitgestellt)
#   DBG Art. 33    — Doppelverdienerabzug
#   DBG Art. 35    — Verheiratetenabzug
#   DBG Art. 36    — Steuertarife Bund
#   §65 StG ZH     — Kapitalleistungssteuer
#   Volksabstimmung 28.09.2025 — Abschaffung Eigenmietwert per 01.01.2029
#   Verifikation Vermögenssteuer: zh.ch Steuerrechner (vom Nutzer bestätigt)
# ═══════════════════════════════════════════════════════════════════════════

# ── Gemeindesteuerfüsse laden ─────────────────────────────────────────────────
_df_steuer = pd.read_csv(
    DATA_DIR / "Gemeindesteuerfüsse_2026.csv",
    encoding="utf-8-sig",
    sep=";"
)

# Spaltenaliase (Gesamtsteuerfuss = Kanton + Gemeinde bereits zusammen)
_COL_OHNE  = "1. Gesamtsteuerfuss ohne Kirche (in Prozent)"
_COL_REF   = "1. Gesamtsteuerfuss mit ref. Kirche (in Prozent)"
_COL_KATH  = "1. Gesamtsteuerfuss mit kath. Kirche (in Prozent)"
_COL_CHR   = "1. Gesamtsteuerfuss mit christkath. Kirche (in Prozent)"

def get_steuerfuss(gemeinde: str, konfession: str = "ohne") -> float:
    """
    Gibt den Gesamtsteuerfuss zurück (Kanton + Gemeinde + Kirche falls zutreffend).
    Werte bereits als Gesamtsteuerfuss in CSV — kein weiterer Zuschlag nötig.
    konfession: 'ohne' | 'ref' | 'kath' | 'chr'
    Quelle: Steueramt Kanton Zürich, Gemeindesteuerfüsse_2026.csv
    """
    row = _df_steuer[_df_steuer["Gemeinde"].str.strip() == gemeinde.strip()]
    if row.empty:
        raise ValueError(f"Gemeinde '{gemeinde}' nicht gefunden.")
    row = row.iloc[0]

    if konfession == "ref":  col = _COL_REF
    elif konfession == "kath": col = _COL_KATH
    elif konfession == "chr":  col = _COL_CHR
    else:                      col = _COL_OHNE

    val = row[col]
    if pd.isna(val):
        # Fallback auf ohne Kirche wenn Kirchensteuerfuss fehlt
        val = row[_COL_OHNE]
    return float(val)

print(f"✓ {len(_df_steuer)} Gemeinden geladen")
print(f"  Urdorf ohne Kirche:    {get_steuerfuss('Urdorf', 'ohne'):.0f}%")
print(f"  Urdorf ref. Kirche:    {get_steuerfuss('Urdorf', 'ref'):.0f}%")
print(f"  Urdorf kath. Kirche:   {get_steuerfuss('Urdorf', 'kath'):.0f}%")
print(f"  Adliswil ohne Kirche:  {get_steuerfuss('Adliswil', 'ohne'):.0f}%")

def get_gemeinden():
    _lade_steuern()
    col = _df_steuer.columns[1]
    return sorted([str(n).strip() for n in _df_steuer[col].dropna().tolist() if str(n).strip()])

# ══════════════════════════════════════════════════════════════════════════════
# EINKOMMENSTEUER KANTON ZÜRICH
# ══════════════════════════════════════════════════════════════════════════════

def _steuer_kt_grundtarif(einkommen: float) -> float:
    """Einfache Staatssteuer ZH — Grundtarif. Quelle: §35 Abs. 1 StG ZH."""
    e = max(0.0, einkommen)
    if   e <=       0: return 0.0
    elif e <=   7_000: return 0.0
    elif e <=  12_000: return                    (e -   7_000) * 0.020
    elif e <=  16_800: return    100.00         + (e -  12_000) * 0.030
    elif e <=  24_800: return    244.00         + (e -  16_800) * 0.040
    elif e <=  34_500: return    564.00         + (e -  24_800) * 0.050
    elif e <=  45_700: return  1_049.00         + (e -  34_500) * 0.060
    elif e <=  58_800: return  1_721.00         + (e -  45_700) * 0.070
    elif e <=  76_400: return  2_638.00         + (e -  58_800) * 0.080
    elif e <= 110_400: return  4_046.00         + (e -  76_400) * 0.090
    elif e <= 144_100: return  7_106.00         + (e - 110_400) * 0.100
    elif e <= 197_400: return 10_476.00         + (e - 144_100) * 0.110
    elif e <= 266_700: return 16_339.00         + (e - 197_400) * 0.120
    else:              return 24_655.00         + (e - 266_700) * 0.130


def _steuer_kt_verheiratetentarif(einkommen: float) -> float:
    """Einfache Staatssteuer ZH — Verheiratetentarif. Quelle: §35 Abs. 2 StG ZH."""
    e = max(0.0, einkommen)
    if   e <=       0: return 0.0
    elif e <=  14_100: return 0.0
    elif e <=  20_500: return                    (e -  14_100) * 0.020
    elif e <=  28_600: return    128.00         + (e -  20_500) * 0.030
    elif e <=  38_400: return    371.00         + (e -  28_600) * 0.040
    elif e <=  49_600: return    763.00         + (e -  38_400) * 0.050
    elif e <=  64_100: return  1_323.00         + (e -  49_600) * 0.060
    elif e <=  96_300: return  2_193.00         + (e -  64_100) * 0.070
    elif e <= 128_700: return  4_447.00         + (e -  96_300) * 0.080
    elif e <= 177_200: return  7_039.00         + (e - 128_700) * 0.090
    elif e <= 235_100: return 11_404.00         + (e - 177_200) * 0.100
    elif e <= 298_000: return 17_194.00         + (e - 235_100) * 0.110
    elif e <= 370_600: return 24_113.00         + (e - 298_000) * 0.120
    else:              return 32_825.00         + (e - 370_600) * 0.130


# ══════════════════════════════════════════════════════════════════════════════
# DIREKTE BUNDESSTEUER
# Quelle: SR 642.11 DBG Art. 36 (Fedlex, vom Nutzer bereitgestellt)
# ══════════════════════════════════════════════════════════════════════════════

def _bundessteuer_grundtarif(einkommen: float) -> float:
    """Direkte Bundessteuer — Grundtarif (unverheiratet). Quelle: DBG Art. 36 Abs. 1."""
    e = max(0.0, einkommen)
    if   e <=  15_200: return 0.0
    elif e <=  33_200: return                    (e -  15_200) * 0.0077
    elif e <=  43_500: return    138.60         + (e -  33_200) * 0.0088
    elif e <=  58_000: return    229.20         + (e -  43_500) * 0.0264
    elif e <=  76_200: return    612.00         + (e -  58_000) * 0.0297
    elif e <=  82_100: return  1_152.50         + (e -  76_200) * 0.0594
    elif e <= 108_900: return  1_502.95         + (e -  82_100) * 0.0660
    elif e <= 141_500: return  3_271.75         + (e - 108_900) * 0.0880
    elif e <= 185_100: return  6_140.55         + (e - 141_500) * 0.1100
    elif e <= 793_900: return 10_936.55         + (e - 185_100) * 0.1320
    elif e <= 794_000: return 91_298.15         + (e - 793_900) * 0.1320
    else:              return 91_310.00         + (e - 794_000) * 0.1150


def _bundessteuer_verheiratetentarif(einkommen: float) -> float:
    """Direkte Bundessteuer — Verheiratetentarif. Quelle: DBG Art. 36 Abs. 2."""
    e = max(0.0, einkommen)
    if   e <=   29_700: return 0.0
    elif e <=   53_400: return                    (e -  29_700) * 0.0100
    elif e <=   61_300: return    237.00         + (e -  53_400) * 0.0200
    elif e <=   79_100: return    395.00         + (e -  61_300) * 0.0300
    elif e <=   94_900: return    929.00         + (e -  79_100) * 0.0400
    elif e <=  108_700: return  1_561.00         + (e -  94_900) * 0.0500
    elif e <=  120_600: return  2_251.00         + (e - 108_700) * 0.0600
    elif e <=  130_500: return  2_965.00         + (e - 120_600) * 0.0700
    elif e <=  138_400: return  3_658.00         + (e - 130_500) * 0.0800
    elif e <=  144_300: return  4_290.00         + (e - 138_400) * 0.0900
    elif e <=  148_300: return  4_821.00         + (e - 144_300) * 0.1000
    elif e <=  150_400: return  5_221.00         + (e - 148_300) * 0.1100
    elif e <=  152_400: return  5_452.00         + (e - 150_400) * 0.1200
    elif e <=  941_300: return  5_692.00         + (e - 152_400) * 0.1300
    elif e <=  941_400: return 108_249.00        + (e - 941_300) * 0.1300
    else:               return 108_261.00        + (e - 941_400) * 0.1150


# Bundessteuer Abzüge
# Quelle: SR 642.11 DBG Art. 33 Abs. 2 + Art. 35 (vom Nutzer bereitgestellt)
_BD_VERHEIRATETENABZUG  = 2_800    # DBG Art. 35
_BD_DOPPELVERDIENER_MIN = 8_600    # DBG Art. 33 Abs. 2
_BD_DOPPELVERDIENER_MAX = 14_100   # DBG Art. 33 Abs. 2


# ══════════════════════════════════════════════════════════════════════════════
# VERMÖGENSSTEUER KANTON ZÜRICH
# Quelle: §47 StG ZH
# Verifikation: zh.ch Steuerrechner (500k→298‰, 2M ledig→1.375‰, 2M verh.→1.295‰)
# Kein Sozialabzug — verifiziert gegen ZH-Steuerrechner
# ══════════════════════════════════════════════════════════════════════════════

def _verm_kt_grundtarif(vermoegen: float) -> float:
    """Vermögenssteuer ZH — Grundtarif. Quelle: §47 Abs. 1 StG ZH."""
    v = max(0.0, vermoegen)
    if   v <=         0: return 0.0
    elif v <=    81_000: return 0.0
    elif v <=   322_000: return                     (v -    81_000) * 0.00050
    elif v <=   726_000: return    120.50          + (v -   322_000) * 0.00100
    elif v <= 1_371_000: return    524.50          + (v -   726_000) * 0.00150
    elif v <= 2_339_000: return  1_492.00          + (v - 1_371_000) * 0.00200
    elif v <= 3_304_000: return  3_428.00          + (v - 2_339_000) * 0.00250
    else:                return  5_840.50          + (v - 3_304_000) * 0.00300


def _verm_kt_verheiratetentarif(vermoegen: float) -> float:
    """Vermögenssteuer ZH — Verheiratetentarif. Quelle: §47 Abs. 2 StG ZH."""
    v = max(0.0, vermoegen)
    if   v <=         0: return 0.0
    elif v <=   161_000: return 0.0
    elif v <=   403_000: return                     (v -   161_000) * 0.00050
    elif v <=   805_000: return    121.00          + (v -   403_000) * 0.00100
    elif v <= 1_451_000: return    523.00          + (v -   805_000) * 0.00150
    elif v <= 2_418_000: return  1_492.00          + (v - 1_451_000) * 0.00200
    elif v <= 3_385_000: return  3_426.00          + (v - 2_418_000) * 0.00250
    else:                return  5_843.50          + (v - 3_385_000) * 0.00300


# ══════════════════════════════════════════════════════════════════════════════
# ÖFFENTLICHE API
# ══════════════════════════════════════════════════════════════════════════════

def berechne_einkommenssteuer(steuerbares_einkommen: float,
                               steuerfuss: float,
                               is_married: bool = False,
                               zweiteinkommen: float = 0.0) -> float:
    """
    Gesamte Einkommenssteuer = Kantonssteuer × Steuerfuss/100 + Bundessteuer.
    Verheiratet: Verheiratetentarif Kanton + Verheiratetentarif Bund
                 + Verheiratetenabzug CHF 2'800 + Doppelverdienerabzug.
    Quellen: §35 StG ZH, DBG Art. 33/35/36.
    """
    eink = max(0.0, steuerbares_einkommen)

    # ── Kantonssteuer ─────────────────────────────────────────────────────────
    if is_married and zweiteinkommen > 0:
        kt_basis = max(0.0, eink - min(zweiteinkommen, 6_200))
    else:
        kt_basis = eink

    kt = (_steuer_kt_verheiratetentarif(kt_basis) if is_married
          else _steuer_kt_grundtarif(kt_basis)) * steuerfuss / 100

    # ── Bundessteuer ──────────────────────────────────────────────────────────
    if is_married:
        if zweiteinkommen > 0:
            dv_abzug = min(
                _BD_DOPPELVERDIENER_MAX,
                max(_BD_DOPPELVERDIENER_MIN,
                    min(eink, zweiteinkommen) * 0.50)
            )
        else:
            dv_abzug = 0.0
        bd_basis = max(0.0, eink - _BD_VERHEIRATETENABZUG - dv_abzug)
        bd       = _bundessteuer_verheiratetentarif(bd_basis)
    else:
        bd = _bundessteuer_grundtarif(eink)

    return kt + bd   

def berechne_vermoegenssteuer(reinvermoegen: float,
                               steuerfuss: float,
                               is_married: bool = False) -> float:
    """
    Vermögenssteuer Kanton ZH. Kein Sozialabzug.
    Verifiziert: CHF 1M→0.935‰, CHF 2M ledig→1.375‰, CHF 2M verh.→1.295‰.
    Quelle: §47 StG ZH.
    """
    v     = max(0.0, reinvermoegen)
    basis = (_verm_kt_verheiratetentarif(v) if is_married
             else _verm_kt_grundtarif(v))
    return basis * steuerfuss / 100


def berechne_kapitalleistungssteuer(kapital: float,
                                     steuerfuss: float,
                                     is_married: bool = False) -> float:
    if kapital <= 0:
        return 0.0

    # ── Kantonssteuer ZH (§ 37 StG ZH) ───────────────────────────────────
    # Steuersatz = einfache Staatssteuer auf 1/20, mind. 2%
    jahresleistung = kapital / 20
    kt_einfach = (_steuer_kt_verheiratetentarif(jahresleistung) if is_married
                  else _steuer_kt_grundtarif(jahresleistung))
    steuersatz = max(kt_einfach / jahresleistung, 0.02) if jahresleistung > 0 else 0.02
    kt = kapital * steuersatz * steuerfuss / 100

    # ── Bundessteuer (DBG Art. 38) ────────────────────────────────────────
    bd_voll = (_bundessteuer_verheiratetentarif(kapital) if is_married
               else _bundessteuer_grundtarif(kapital))
    bd = bd_voll / 5

    return kt + bd

def berechne_liegenschaft_steuereffekt(liegenschaft: float,
                                        hypothek: float,
                                        hypo_zins_satz: float,
                                        simulationsjahr: int,
                                        eigenmietwert: float,           # ← NEU
                                        ist_ersterwerber: bool = False,
                                        jahre_seit_kauf: int = 99
                                        ) -> tuple:
    """
    Steuerlicher Effekt Eigenheim je nach Regime.
    Bis 2028: Eigenmietwert (Inputparameter, aus Steuererklärung) + Hypo-Zinsen abziehbar.
    Ab 2029:  Kein Eigenmietwert, keine Abzüge (Volksabstimmung 28.09.2025).
              Ausnahme: Ersterwerberabzug ≤ 10 Jahre nach Kauf.
    Gibt zurück: (zusaetzliches_einkommen, abzuege)
    """
    if liegenschaft <= 0:
        return 0.0, 0.0
    hypo_zinsen = hypothek * hypo_zins_satz if hypothek > 0 else 0.0
    if simulationsjahr < EIGENMIETWERT_ABSCHAFFUNG_JAHR:
        return eigenmietwert, hypo_zinsen                               # ← geändert
    if ist_ersterwerber and jahre_seit_kauf <= 10:
        return 0.0, hypo_zinsen
    return 0.0, 0.0


# In[3]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — RENDITE- & INFLATIONSMODELL
# Deployment-Version: Asset-Parameter fest hinterlegt (Bloomberg 2007–2026)
# ═══════════════════════════════════════════════════════════════════════════

STOCHASTIC_ASSETS = ["aktien", "obligationen", "immobilien", "gold"]

# ── Asset-Parameter fest hinterlegt (Bloomberg 01.01.2007–08.05.2026) ────────
ASSET_PARAMS = {
    "aktien": {
        "mu": 0.053298, "mu_ln": 0.051926, "sigma": 0.164659,
        "sigma_arith": 0.164659, "mu_arith": 0.066088, "n": 19, "start": 2007,
    },
    "obligationen": {
        "mu": 0.017430, "mu_ln": 0.017280, "sigma": 0.039428,
        "sigma_arith": 0.039428, "mu_arith": 0.018166, "n": 19, "start": 2007,
    },
    "immobilien": {
        "mu": 0.058934, "mu_ln": 0.057263, "sigma": 0.085415,
        "sigma_arith": 0.085415, "mu_arith": 0.062531, "n": 19, "start": 2007,
    },
    "gold": {
        "mu": 0.073873, "mu_ln": 0.071272, "sigma": 0.152282,
        "sigma_arith": 0.152282, "mu_arith": 0.085372, "n": 19, "start": 2007,
    },
    "cash": {
        "mu": 0.000, "mu_ln": 0.000, "sigma": 0.000,
        "sigma_arith": 0.000, "mu_arith": 0.000, "n": 0, "start": 0,
    },
}

# ── Korrelationsmatrix (aktien, obligationen, immobilien, gold) ───────────────
ASSET_KORRELATION = np.array([
    [1.000000, 0.337772, 0.551721, 0.091724],
    [0.337772, 1.000000, 0.759612, 0.263423],
    [0.551721, 0.759612, 1.000000, 0.569319],
    [0.091724, 0.263423, 0.569319, 1.000000],
])

ASSET_CHOLESKY = np.linalg.cholesky(ASSET_KORRELATION)

# ── Portfolios ────────────────────────────────────────────────────────────────
PORTFOLIOS = {
    1: {"name": "Konservativ",
        "aktien": 0.25, "obligationen": 0.60,
        "immobilien": 0.15, "gold": 0.00, "cash": 0.00},
    2: {"name": "Ausgewogen",
        "aktien": 0.50, "obligationen": 0.35,
        "immobilien": 0.15, "gold": 0.00, "cash": 0.00},
    3: {"name": "Wachstum",
        "aktien": 0.75, "obligationen": 0.15,
        "immobilien": 0.10, "gold": 0.00, "cash": 0.00},
}

print("✓ Asset-Parameter fest hinterlegt (Bloomberg 2007–2026):")
print(f"  {'Asset':<15} {'μ geo p.a.':>12} {'σ ln p.a.':>10}")
print("  " + "-" * 40)
for name, p in ASSET_PARAMS.items():
    if name == "cash": continue
    print(f"  {name:<15} {p['mu']*100:>11.2f}% {p['sigma']*100:>9.2f}%")

print(f"\nPortfolio-Erwartungsrenditen:")
print(f"  {'Portfolio':<15} {'μ (p.a.)':>10} {'σ (p.a.)':>10}")
print("  " + "-" * 38)
for _rv, _p in PORTFOLIOS.items():
    _mu_p  = sum(ASSET_PARAMS[a]["mu"]    * w for a, w in _p.items() if a != "name")
    _sig_p = sum(ASSET_PARAMS[a]["sigma"] * w for a, w in _p.items() if a != "name")
    print(f"  {_p['name']:<15} {_mu_p*100:>9.2f}% {_sig_p*100:>9.2f}%")

# ── Inflation aus BFS CPI ─────────────────────────────────────────────────────
_infl_rates = np.array([
    1.0, 0.6, 0.6, 0.8, 1.2,
    1.1, 0.7, 2.4, -0.5, 0.7,
    0.2, -0.7, -0.2, 0.0, -1.1,
    -0.4, 0.5, 0.9, 0.4, -0.7,
    0.6, 2.8, 2.1, 1.1, 0.2
]) / 100

INFLATION_MU    = float(np.prod(1 + _infl_rates) ** (1 / len(_infl_rates)) - 1)
INFLATION_SIGMA = float(np.std(_infl_rates, ddof=1))

print(f"\nInflation Schweiz (BFS 2001–2025):")
print(f"  μ = {INFLATION_MU * 100:.3f}%")
print(f"  σ = {INFLATION_SIGMA * 100:.3f}%")

# ── Simulationsfunktionen ─────────────────────────────────────────────────────
def simuliere_rendite(risikoaversion: int,
                      n_jahre: int,
                      rng: np.random.Generator) -> np.ndarray:
    p      = PORTFOLIOS[risikoaversion]
    z      = rng.standard_normal((n_jahre, len(STOCHASTIC_ASSETS)))
    z_korr = z @ ASSET_CHOLESKY.T
    renditen = np.zeros(n_jahre)
    for i, asset in enumerate(STOCHASTIC_ASSETS):
        w = p.get(asset, 0.0)
        if w == 0: continue
        params = ASSET_PARAMS[asset]
        r = np.exp(z_korr[:, i] * params["sigma"] + params["mu_ln"]) - 1
        renditen += w * r
    renditen += p.get("cash", 0.0) * ASSET_PARAMS["cash"]["mu"]
    return renditen

def simuliere_inflation(n_jahre: int,
                        rng: np.random.Generator) -> np.ndarray:
    return rng.normal(INFLATION_MU, INFLATION_SIGMA, n_jahre)

def simuliere_liegenschaft_renditen(n_jahre: int,
                                     rng: np.random.Generator) -> np.ndarray:
    return rng.normal(LIEGEN_MU, LIEGEN_SIGMA, n_jahre)

# ── Verifikation ──────────────────────────────────────────────────────────────
_rng_test = np.random.default_rng(42)
print(f"\nSimulations-Verifikation (N=10'000):")
for _rv in [1, 2, 3]:
    _r = simuliere_rendite(_rv, 10_000, _rng_test)
    print(f"  {PORTFOLIOS[_rv]['name']:<13}: Ø={_r.mean()*100:.2f}%  σ={_r.std()*100:.2f}%")



# In[4]:


try:
    import pyreadstat
except ImportError:
    pyreadstat = None
    # ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — STERBETAFELN & PFLEGEFALL-PARAMETER
# Quellen:
#   Sterbetafeln:   BFS Kohortensterbetafeln 2023 (sterbetafeln_kohorten.csv)
#   SOMED:          BFS SOMED 2006–2024 (px-x-1404010100_302.px)
#   STATPOP:        BFS Bevölkerungsstatistik (px-x-0102010000_101_...xlsx)
#   Spitex BFS:     BFS Spitex-Statistik (px-x-1404040000_102.px)
#   SHP:            Swiss Household Panel H$$F17, 565 Episoden
#   KTZH Heim:      KTZH SOMED 2019–2024 (KTZH_00003102_00006632.xlsx)
#   KTZH Spitex:    KTZH Spitex 2019–2024 (KTZH_00003103_00006633.xlsx)
#   EL:             ELG SR 831.30, Stand 2024
# ═══════════════════════════════════════════════════════════════════════════

import pyreadstat
from scipy.stats import norm as _scipy_norm

PATH_STERBETAFELN  = DATA_DIR / "sterbetafeln_kohorten.csv"
PATH_SOMED         = DATA_DIR / "px-x-1404010100_302.px"
PATH_STATPOP       = DATA_DIR / "px-x-0102010000_101_20260414-223248.xlsx"
PATH_SPITEX        = DATA_DIR / "px-x-1404040000_102.px"
PATH_SHP_H         = DATA_DIR / "shplong_h_user.dta"
PATH_SHP_P         = DATA_DIR / "shplong_p_user.dta"
PATH_KTZH_HEIM     = DATA_DIR / "KTZH_00003102_00006632.xlsx"
PATH_KTZH_SPITEX   = DATA_DIR / "KTZH_00003103_00006633.xlsx"   # NEU


# ═══════════════════════════════════════════════════════════════════════════
# 1. STERBETAFELN
# ═══════════════════════════════════════════════════════════════════════════

_df_q     = pd.read_csv(PATH_STERBETAFELN)
_qx_index = _df_q.set_index(["Geschlecht", "Jahrgang", "Alter"])["qx"]

def get_qx(geburtsjahr: int, female: int, alter: int) -> float:
    """
    Jährliche Sterbewahrscheinlichkeit aus BFS Kohortensterbetafeln.
    Fallback: nächster Jahrgang → nächstes Alter → Max qx.
    Quelle: BFS Kohortensterbetafeln 2023.
    """
    geschlecht = "Frau" if female == 1 else "Mann"
    try:
        return float(_qx_index.loc[(geschlecht, geburtsjahr, alter)])
    except KeyError:
        pass
    _subset_alter = _df_q[
        (_df_q["Geschlecht"] == geschlecht) & (_df_q["Alter"] == alter)]
    if not _subset_alter.empty:
        _idx = (_subset_alter["Jahrgang"] - geburtsjahr).abs().idxmin()
        return float(_subset_alter.loc[_idx, "qx"])
    _subset_jg = _df_q[
        (_df_q["Geschlecht"] == geschlecht) & (_df_q["Jahrgang"] == geburtsjahr)]
    if not _subset_jg.empty:
        _idx = (_subset_jg["Alter"] - alter).abs().idxmin()
        return float(_subset_jg.loc[_idx, "qx"])
    return float(_df_q[_df_q["Geschlecht"] == geschlecht]["qx"].max())

print("✓ Sterbetafeln geladen")
print(f"  Beispiel Mann 1978, Alter 65: qx={get_qx(1978, 0, 65):.4%}")
print(f"  Beispiel Frau 1983, Alter 65: qx={get_qx(1983, 1, 65):.4%}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. PX-PARSER
# ═══════════════════════════════════════════════════════════════════════════

def _parse_px(path) -> list:
    with open(path, encoding="iso-8859-15") as f:
        content = f.read()
    start = content.find("DATA=")
    if start == -1:
        raise ValueError(f"Kein DATA= gefunden in {path}")
    data_str = content[start + 5:].strip().rstrip(";")
    data_str = data_str.replace('"', '').replace('\n', ' ').replace('\r', ' ')
    _MISSING = {".", "..", "...", "....", "*", "-", "n/a", "na", "nan"}
    result   = []
    _unbekannte = set()
    for token in data_str.split():
        token = token.strip()
        if token in _MISSING:
            result.append(np.nan)
        else:
            try:
                result.append(float(token))
            except ValueError:
                result.append(np.nan)
                if token not in _unbekannte:
                    print(f"  ⚠️  _parse_px: unbekannter Token '{token}' → nan")
                    _unbekannte.add(token)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. STATPOP — BEVÖLKERUNG NACH EINZELALTER
# Struktur: Zeile 1 = Alter-Header (Spalte 11 = «0 Jahre», ..., Spalte 111 = «100+»)
#           Geschlecht in Spalte 9: «Mann» / «Frau»
#           Jahr in Spalte 1
# ═══════════════════════════════════════════════════════════════════════════

_df_statpop_raw = pd.read_excel(PATH_STATPOP, header=None)

_pop_data = {}   # (jahr, 'Mann'/'Frau') -> {alter: anzahl}
_current_year_statpop = None

for _, _row in _df_statpop_raw.iterrows():
    try:
        _yr = int(float(_row.iloc[1]))
        if 2009 < _yr < 2026:
            _current_year_statpop = _yr
    except (ValueError, TypeError):
        pass
    _sex_str = str(_row.iloc[9]).strip()
    if _current_year_statpop is not None and _sex_str in ("Mann", "Frau"):
        _ages = {}
        for _a in range(101):
            _val = _row.iloc[11 + _a]
            _ages[_a] = float(_val) if not pd.isna(_val) else 0.0
        _pop_data[(_current_year_statpop, _sex_str)] = _ages

_jahre_statpop = sorted({yr for (yr, _) in _pop_data.keys()})
print(f"✓ STATPOP geladen: {len(_pop_data)} Einträge, "
      f"Jahre {_jahre_statpop[0]}–{_jahre_statpop[-1]}")
print(f"  Test 2020 Mann 80-84: {sum(_pop_data.get((2020,'Mann'),{}).get(a,0) for a in range(80,85)):,.0f}")


def get_pop_band(jahr: int, sex: str, alter_von: int, alter_bis: int) -> float:
    """
    Bevölkerungssumme [alter_von, alter_bis) für ein Jahr aus STATPOP.
    sex: 'mann' | 'frau'
    """
    key  = (jahr, "Mann" if sex == "mann" else "Frau")
    ages = _pop_data.get(key, {})
    return sum(ages.get(a, 0.0) for a in range(alter_von, min(alter_bis, 101)))


# ═══════════════════════════════════════════════════════════════════════════
# 4. HEIM-PARAMETER (SOMED)
# Methodik Eintrittsraten:
#   λ_Heim(a,g) = Ø[ Eintritte(a,g,t) / Bevölkerung(a,g,t) ]  t ∈ 2010–2024
# Methodik Aufenthaltsparameter:
#   D(a)     = Ø Bestand / Ø Eintritte
#   p_tod(a) = Ø Todesfälle / Ø Eintritte
# ═══════════════════════════════════════════════════════════════════════════

_somed_j = list(range(2006, 2025))
_ref_j   = list(range(2010, 2025))
_age_lbl = ["50-54", "55-59", "60-64", "65-69", "70-74",
             "75-79", "80-84", "85-89", "90-94", "95+"]
_age_ai  = {"50-54": 11, "55-59": 12, "60-64": 13, "65-69": 14, "70-74": 15,
             "75-79": 16, "80-84": 17, "85-89": 18, "90-94": 19, "95+":   20}
_age_von_bis = {
    "50-54": (50, 55), "55-59": (55, 60), "60-64": (60, 65),
    "65-69": (65, 70), "70-74": (70, 75), "75-79": (75, 80),
    "80-84": (80, 85), "85-89": (85, 90), "90-94": (90, 95),
    "95+":   (95, 101)
}

_LH_FALLBACK = {
    "mann": {
        "50-54": 0.0008, "55-59": 0.0015, "60-64": 0.0025,
        "65-69": 0.0045, "70-74": 0.0100, "75-79": 0.0230,
        "80-84": 0.0550, "85-89": 0.1100, "90-94": 0.1800, "95+": 0.2500,
    },
    "frau": {
        "50-54": 0.0006, "55-59": 0.0012, "60-64": 0.0020,
        "65-69": 0.0040, "70-74": 0.0090, "75-79": 0.0220,
        "80-84": 0.0600, "85-89": 0.1300, "90-94": 0.2100, "95+": 0.3000,
    },
}

try:
    _somed = np.array(_parse_px(PATH_SOMED)).reshape([34, 7, 8, 9, 4, 21, 2, 19])

    _lh = {"mann": {}, "frau": {}}
    for _si, _sk in [(1, "mann"), (2, "frau")]:
        for _age in _age_lbl:
            _ai        = _age_ai[_age]
            _av, _ab   = _age_von_bis[_age]
            _rates     = []
            for _yr in _ref_j:
                _ji = _somed_j.index(_yr)
                _ev = _somed[0, 1, 1, 0, _si, _ai, 0, _ji]
                _pn = get_pop_band(_yr, _sk, _av, _ab)
                if _ev > 0 and not np.isnan(_pn) and _pn > 0:
                    _rates.append(_ev / _pn)
            _lh[_sk][_age] = (float(np.mean(_rates))
                               if _rates
                               else _LH_FALLBACK[_sk][_age])

    _heim_params = {}
    for _age in _age_lbl:
        _ai = _age_ai[_age]
        _abg, _bst, _tod = [], [], []
        for _yr in _ref_j:
            _ji = _somed_j.index(_yr)
            _a  = sum(_somed[0, 1, 0, _di, 0, _ai, 0, _ji]
                      for _di in [1, 2, 3, 4, 5, 8]
                      if _somed[0, 1, 0, _di, 0, _ai, 0, _ji] > 0)
            _b  = _somed[0, 1, 0, 0, 0, _ai, 1, _ji]
            _t  = _somed[0, 1, 0, 5, 0, _ai, 0, _ji]
            if _a > 0 and _b > 0:
                _abg.append(_a); _bst.append(_b); _tod.append(_t)
        _heim_params[_age] = {
            "D":     float(np.mean(_bst)) / float(np.mean(_abg)) if _abg else 2.0,
            "p_tod": float(np.mean(_tod)) / float(np.mean(_abg)) if _abg else 0.7,
        }

    def get_heim_params(age: int) -> dict:
        """D: mittlere Aufenthaltsdauer, p_tod: P(Tod im Heim). Quelle: BFS SOMED."""
        if   age < 55: return _heim_params["50-54"]
        elif age < 60: return _heim_params["55-59"]
        elif age < 65: return _heim_params["60-64"]
        elif age < 70: return _heim_params["65-69"]
        elif age < 75: return _heim_params["70-74"]
        elif age < 80: return _heim_params["75-79"]
        elif age < 85: return _heim_params["80-84"]
        elif age < 90: return _heim_params["85-89"]
        elif age < 95: return _heim_params["90-94"]
        else:          return _heim_params["95+"]

    def get_lambda_heim(age: int, female: int) -> float:
        """Heim-Eintrittswahrscheinlichkeit. Quelle: BFS SOMED + STATPOP."""
        if age < 50: return 0.0
        _sk = "frau" if female == 1 else "mann"
        for _band, _lo, _hi in [
            ("50-54", 50, 55), ("55-59", 55, 60), ("60-64", 60, 65),
            ("65-69", 65, 70), ("70-74", 70, 75), ("75-79", 75, 80),
            ("80-84", 80, 85), ("85-89", 85, 90), ("90-94", 90, 95),
            ("95+",   95, 200)]:
            if _lo <= age < _hi:
                return _lh[_sk][_band]
        return 0.0

    print("✓ SOMED geladen — Heim-Parameter berechnet")
    print(f"  Alter 80-84: D={_heim_params['80-84']['D']:.2f}J  "
          f"p_tod={_heim_params['80-84']['p_tod']:.3f}")
    print(f"  λ_heim Mann 80: {get_lambda_heim(82, 0):.4f}  "
          f"λ_heim Frau 80: {get_lambda_heim(82, 1):.4f}")

except Exception as e:
    print(f"  ⚠️  SOMED: {e} — verwende BFS-Fallback")

    def get_heim_params(age):
        return {"D": 2.0, "p_tod": 0.7}

    def get_lambda_heim(age: int, female: int) -> float:
        if age < 50: return 0.0
        _sk = "frau" if female == 1 else "mann"
        for _band, _lo, _hi in [
            ("50-54", 50, 55), ("55-59", 55, 60), ("60-64", 60, 65),
            ("65-69", 65, 70), ("70-74", 70, 75), ("75-79", 75, 80),
            ("80-84", 80, 85), ("85-89", 85, 90), ("90-94", 90, 95),
            ("95+",   95, 200)]:
            if _lo <= age < _hi:
                return _LH_FALLBACK[_sk][_band]
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. SPITEX-EPISODENLÄNGEN (SHP)
# ═══════════════════════════════════════════════════════════════════════════

try:
    _dfh_ep, _ = pyreadstat.read_dta(PATH_SHP_H, usecols=["idhous", "year", "hf17"])
    _dfp_ep, _ = pyreadstat.read_dta(PATH_SHP_P, usecols=["idpers", "idhous", "year", "age"])
    _dfh_ep["spitex"] = (_dfh_ep["hf17"] == 1).astype(int)
    _dfa = (_dfp_ep.groupby(["idhous", "year"])["age"]
            .max().reset_index().rename(columns={"age": "max_age"}))
    _df50 = (_dfh_ep.merge(_dfa, on=["idhous", "year"], how="left")
             .pipe(lambda d: d[d["max_age"] >= 50])
             .sort_values(["idhous", "year"]).copy())
    _df50["sp_prev"]  = _df50.groupby("idhous")["spitex"].shift(1)
    _df50["ep_start"] = ((_df50["spitex"] == 1) &
                         (_df50["sp_prev"].fillna(0) == 0)).astype(int)
    _df50["ep_id"]    = _df50.groupby("idhous")["ep_start"].cumsum()
    _MIN_YR = int(_df50["year"].min())
    _MAX_YR = int(_df50["year"].max())
    _eps = []
    for (_hh, _ep), _grp in _df50[_df50["spitex"] == 1].groupby(["idhous", "ep_id"]):
        _yrs = sorted(_grp["year"].tolist())
        if not all(_yrs[i + 1] - _yrs[i] == 1 for i in range(len(_yrs) - 1)):
            continue
        _nxt = _df50[(_df50["idhous"] == _hh) & (_df50["year"] == max(_yrs) + 1)]
        _rc  = ((len(_nxt) > 0 and _nxt["spitex"].values[0] == 1) or
                (len(_nxt) == 0 and max(_yrs) == _MAX_YR))
        _eps.append({"n_years": len(_yrs), "lc": min(_yrs) == _MIN_YR, "rc": _rc})
    _edf      = pd.DataFrame(_eps)
    _complete = _edf[~_edf["lc"] & ~_edf["rc"]].copy()
    _log_d        = np.log(_complete["n_years"].values + 0.5)
    D_SPITEX_MU   = float(_log_d.mean())
    D_SPITEX_SIG  = float(_log_d.std(ddof=1))
    D_SPITEX_MEAN = float(np.exp(D_SPITEX_MU + 0.5 * D_SPITEX_SIG**2))
    print(f"✓ SHP Spitex-Episoden geladen: N={len(_complete)}")
    print(f"  D_SPITEX_MU={D_SPITEX_MU:.4f}  "
          f"D_SPITEX_SIG={D_SPITEX_SIG:.4f}  "
          f"E[D]={D_SPITEX_MEAN:.2f}J")
except Exception as e:
    print(f"  ⚠️  SHP Spitex-Episoden: {e} — Platzhalter aktiv")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SPITEX-EINTRITTSRATEN (BFS)
# Methodik: λ_Spitex(a,g) = [Ø Klienten / Ø Bevölkerung] / D_SPITEX_MEAN
# ═══════════════════════════════════════════════════════════════════════════

_spitex_grenzen = {"50-64": (50, 65), "65-79": (65, 80), "80+": (80, 101)}
_sp_jahre       = list(range(2011, 2025))

try:
    _sp_arr = np.array(_parse_px(PATH_SPITEX)).reshape([4, 27, 6, 3, 8, 4, 14])
    _ls     = {}
    for _ai, _band in [(3, "50-64"), (4, "65-79"), (5, "80+")]:
        _ls[_band] = {}
        _av, _ab   = _spitex_grenzen[_band]
        for _gi, _sk in [(1, "mann"), (2, "frau")]:
            _faelle = [_sp_arr[0, 0, _ai, _gi, 1, 1, _ji]
                       for _ji in range(14)
                       if _sp_arr[0, 0, _ai, _gi, 1, 1, _ji] > 0]
            _pop_werte = [get_pop_band(_yr, _sk, _av, _ab) for _yr in _sp_jahre]
            _pop_werte = [v for v in _pop_werte if not np.isnan(v) and v > 0]
            if _faelle and _pop_werte:
                _ls[_band][_sk] = (float(np.mean(_faelle)) /
                                   float(np.mean(_pop_werte))) / D_SPITEX_MEAN
            else:
                _ls[_band][_sk] = 0.0
                print(f"  ⚠️  Spitex-Rate {_sk} {_band}: Fallback 0.0")

    def get_lambda_spitex(age: int, female: int) -> float:
        """Spitex-Eintrittswahrscheinlichkeit. Quelle: BFS Spitex + STATPOP."""
        if age < 50: return 0.0
        _band = "50-64" if age < 65 else ("65-79" if age < 80 else "80+")
        _sk   = "frau" if female == 1 else "mann"
        return _ls[_band][_sk]

    print("✓ BFS Spitex-Eintrittsraten geladen")
except Exception as e:
    print(f"  ⚠️  BFS Spitex: {e}")
    def get_lambda_spitex(age: int, female: int) -> float:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. HEIMKOSTEN NACH GEMEINDE (KTZH SOMED 2024)
# ═══════════════════════════════════════════════════════════════════════════

def _lade_heimkosten_gemeinde(path) -> dict:
    df = pd.read_excel(path, sheet_name="2024")
    gemeinde_avg = (df.groupby("Ort")[["Kosten pro Tag Hotellerie",
                                       "Kosten pro Tag Betreuung"]].mean())
    result = {}
    for ort, row in gemeinde_avg.iterrows():
        kosten_tag   = (row["Kosten pro Tag Hotellerie"] +
                        row["Kosten pro Tag Betreuung"] +
                        KVG_PAT_TAG_CHF)
        result[str(ort)] = float(kosten_tag * 365)
    kt_tag = (df["Kosten pro Tag Hotellerie"].mean() +
              df["Kosten pro Tag Betreuung"].mean() +
              KVG_PAT_TAG_CHF)
    result["__kanton__"] = float(kt_tag * 365)
    return result

try:
    _HEIMKOSTEN_GEMEINDE = _lade_heimkosten_gemeinde(PATH_KTZH_HEIM)
    print(f"✓ Heimkosten nach Gemeinde: {len(_HEIMKOSTEN_GEMEINDE) - 1} Gemeinden")
    print(f"  Kanton ZH Ø: CHF {_HEIMKOSTEN_GEMEINDE['__kanton__']:,.0f}/Jahr")
except Exception as e:
    _HEIMKOSTEN_GEMEINDE = {"__kanton__": 92_484.0}
    print(f"  ⚠️  Heimkosten: {e}")

def get_kosten_heim_jahr(gemeinde: str) -> float:
    """Basisjahr-Heimkosten (2024, nominal) nach Gemeinde. Fallback: Kanton ZH."""
    return _HEIMKOSTEN_GEMEINDE.get(
        gemeinde, _HEIMKOSTEN_GEMEINDE.get("__kanton__", 92_484.0))


# ═══════════════════════════════════════════════════════════════════════════
# 8. HEIM-KOSTENSTEIGERUNG (KTZH SOMED 2019–2024)
# ═══════════════════════════════════════════════════════════════════════════

def _berechne_kostensteigerung_heim(path) -> tuple:
    xl       = pd.ExcelFile(path)
    kant_avg = {}
    for sheet in xl.sheet_names:
        try:
            yr = int(str(sheet).strip())
            df = pd.read_excel(path, sheet_name=sheet)
            cols = {"Kosten pro Tag Hotellerie", "Kosten pro Tag Betreuung"}
            if cols.issubset(df.columns):
                kt = (df["Kosten pro Tag Hotellerie"].mean() +
                      df["Kosten pro Tag Betreuung"].mean())
                if kt > 0:
                    kant_avg[yr] = float(kt)
        except (ValueError, KeyError, Exception):
            continue
    if len(kant_avg) < 2:
        return 0.025, 0.010
    jahre = sorted(kant_avg.keys())
    raten = [(kant_avg[y] - kant_avg[y - 1]) / kant_avg[y - 1]
             for y in jahre if y - 1 in kant_avg]
    return float(np.mean(raten)), float(np.std(raten, ddof=1)) if len(raten) > 1 else 0.010

try:
    HEIM_KOSTEN_MU, HEIM_KOSTEN_SIGMA = _berechne_kostensteigerung_heim(PATH_KTZH_HEIM)
    print(f"\n✓ Heimkostensteigerung: μ={HEIM_KOSTEN_MU:.2%}  σ={HEIM_KOSTEN_SIGMA:.2%}")
except Exception as e:
    HEIM_KOSTEN_MU, HEIM_KOSTEN_SIGMA = 0.025, 0.010
    print(f"  ⚠️  Heimkostensteigerung Fallback 2.5%/1.0%: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. SPITEX-KOSTENSTEIGERUNG (KTZH SPITEX 2019–2024)  — NEU
# Methodik: Gewichtetes Mittel (Vollkosten/Std × Stunden) je Jahr
#           Nur Institutionen mit Leistungspflicht
# ═══════════════════════════════════════════════════════════════════════════

def _berechne_kostensteigerung_spitex(path) -> tuple:
    _COL_KOSTEN  = "Vollkosten pro verr. Stunden Total Pflege"
    _COL_STUNDEN = "Anzahl verr. Stunden Total Pflege"
    _COL_TYP     = "Typ"
    _TYP_LP      = "Spitex-Institution mit Leistungspflicht"

    xl       = pd.ExcelFile(path)
    kant_avg = {}
    for sheet in xl.sheet_names:
        try:
            yr = int(str(sheet).strip())
            df = pd.read_excel(path, sheet_name=sheet)
            df = df[df[_COL_TYP] == _TYP_LP].copy()
            df[_COL_KOSTEN]  = pd.to_numeric(df[_COL_KOSTEN],  errors="coerce")
            df[_COL_STUNDEN] = pd.to_numeric(df[_COL_STUNDEN], errors="coerce")
            df = df[
                df[_COL_KOSTEN].between(10, 500) &
                df[_COL_STUNDEN].notna() &
                (df[_COL_STUNDEN] > 0)
            ]
            if len(df) == 0:
                continue
            gewichtet = ((df[_COL_KOSTEN] * df[_COL_STUNDEN]).sum() /
                          df[_COL_STUNDEN].sum())
            kant_avg[yr] = float(gewichtet)
        except (ValueError, KeyError, Exception):
            continue
    if len(kant_avg) < 2:
        return 0.020, 0.015
    jahre = sorted(kant_avg.keys())
    raten = [(kant_avg[y] - kant_avg[y - 1]) / kant_avg[y - 1]
             for y in jahre if y - 1 in kant_avg]
    return float(np.mean(raten)), float(np.std(raten, ddof=1)) if len(raten) > 1 else 0.015

try:
    SPITEX_KOSTEN_MU, SPITEX_KOSTEN_SIGMA = _berechne_kostensteigerung_spitex(
        PATH_KTZH_SPITEX)
    print(f"✓ Spitex-Kostensteigerung: μ={SPITEX_KOSTEN_MU:.2%}  "
          f"σ={SPITEX_KOSTEN_SIGMA:.2%}")
except Exception as e:
    SPITEX_KOSTEN_MU, SPITEX_KOSTEN_SIGMA = 0.020, 0.015
    print(f"  ⚠️  Spitex-Kostensteigerung Fallback 2.0%/1.5%: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. SPITEX-KOSTEN (EIGENLEISTUNG)
# ═══════════════════════════════════════════════════════════════════════════

def get_spitex_kosten_monatlich(alter: int) -> float:
    """
    Monatliche Eigenleistung Spitex (CHF).
    KVG-Selbstbehalt (~1.5h/Tag × CHF 15.95) + Betreuung/Haushaltshilfe.
    Quelle: KVG Art. 25a Abs. 5, BFS Spitex-Statistik 2023.
    """
    kvg = 720.0
    if alter < 75:
        betreuung = 400.0
    elif alter < 85:
        betreuung = 700.0
    else:
        betreuung = 1_100.0
    return kvg + betreuung


# ═══════════════════════════════════════════════════════════════════════════
# 11. EL-BERECHNUNG (ELG SR 831.30, Stand 2024)
# ═══════════════════════════════════════════════════════════════════════════

def berechne_el_anspruch(rente_ahv:  float,
                          rente_pk:   float,
                          wealth:     float,
                          gemeinde:   str,
                          is_married: bool  = False,
                          im_heim:    bool  = True,
                          kosten_override: float = None) -> dict:
    """EL-Berechnung nach ELG Art. 9–11."""
    eintrittsschwelle = (EL_EINTRITTSSCHWELLE_VERHEIRATET
                         if is_married else EL_EINTRITTSSCHWELLE_LEDIG)
    if wealth > eintrittsschwelle:
        return {
            "el_anspruch": 0.0, "hat_anspruch": False,
            "grund": f"Vermögen > Eintrittsschwelle CHF {eintrittsschwelle:,.0f}",
            "ausgaben_total": 0.0, "einnahmen_total": 0.0,
            "verm_anrechenbar": 0.0,
            "freibetrag": EL_FREIBETRAG_VERHEIRATET if is_married else EL_FREIBETRAG_LEDIG,
            "eintrittsschwelle": eintrittsschwelle, "kosten_heim": 0.0,
        }
    kosten_heim    = (kosten_override if kosten_override is not None
                      else get_kosten_heim_jahr(gemeinde))
    ausgaben_total = kosten_heim + EL_AUSLAGEN
    freibetrag     = (EL_FREIBETRAG_VERHEIRATET if is_married else EL_FREIBETRAG_LEDIG)
    verzehrsatz    = (EL_VERMOEGENSVERZEHR_HEIM if im_heim
                      else EL_VERMOEGENSVERZEHR_ZUHAUSE)
    verm_anrechenbar = max(0.0, wealth - freibetrag) * verzehrsatz
    einnahmen_total  = rente_ahv + rente_pk + verm_anrechenbar
    el_anspruch      = max(0.0, ausgaben_total - einnahmen_total)
    return {
        "el_anspruch": el_anspruch, "hat_anspruch": el_anspruch > 0,
        "grund": "EL berechtigt" if el_anspruch > 0 else "Einnahmen > Ausgaben",
        "ausgaben_total": ausgaben_total, "einnahmen_total": einnahmen_total,
        "rente_ahv": rente_ahv, "rente_pk": rente_pk,
        "verm_anrechenbar": verm_anrechenbar, "freibetrag": freibetrag,
        "eintrittsschwelle": eintrittsschwelle, "kosten_heim": kosten_heim,
    }


def get_el_reduktion(rente_ahv, rente_pk, wealth, gemeinde="Urdorf",
                      is_married=False, im_heim=True, kosten_override=None) -> float:
    return berechne_el_anspruch(rente_ahv, rente_pk, wealth, gemeinde,
                                 is_married, im_heim,
                                 kosten_override=kosten_override)["el_anspruch"]


def get_heim_netto_kosten(rente_ahv, rente_pk, wealth, gemeinde="Urdorf",
                           is_married=False, kosten_override=None) -> float:
    """Netto-Heimkosten nach EL-Abzug. wealth muss Liegenschaftsequity enthalten."""
    el     = get_el_reduktion(rente_ahv, rente_pk, wealth, gemeinde,
                               is_married, im_heim=True,
                               kosten_override=kosten_override)
    brutto = (kosten_override if kosten_override is not None
              else get_kosten_heim_jahr(gemeinde)) + EL_AUSLAGEN
    return max(0.0, brutto - el)


# In[5]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4 — PARAMETER-SCHEMA (ZWEI-PERSONEN-HAUSHALT)
# Generisch für beliebige Personen — Dario/Partner nur Demonstrationsbeispiel
# Streamlit liest diese Parameter direkt aus User-Input
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PersonParams:
    """
    Inputparameter für eine Person.
    Alle monetären Werte in CHF/Jahr (Einkommen, Renten)
    oder CHF (Vermögen, Kapital).
    """
    # ── Persönliche Angaben ───────────────────────────────────────────────
    name:            str
    geburtsjahr:     int
    geschlecht:      str           # 'm' | 'f'
    bildung:         int           # 1=Grundbildung, 2=Berufsbildung, 3=Tertiär

    # ── Erwerbseinkommen ──────────────────────────────────────────────────
    einkommen:       float         # Bruttolohn CHF/Jahr
    rentenalter:     int           # geplantes Pensionierungsalter

    # ── Vorsorge ──────────────────────────────────────────────────────────
    pk_guthaben:        float          # aktuelles PK-Guthaben CHF (Simulationsstartwert)
    pk_guthaben_65:     float  = 0.0   # Projektion Vorsorgeausweis: Guthaben bei Alter 65
                                   # Referenzwert für Basismodell ohne Risiken
    pk_bei_heirat:   float  = 0.0  # PK-Guthaben bei Heirat (für Splitting)
    pk_umwandlungssatz: float = 0.068   # BVG Mindestumwandlungssatz 2024
    pk_bezug_kapital_anteil: float = 0.50  # Anteil PK-Kapital als Einmalbezug
    saeule3:         float  = 0.0  # aktuelles Säule-3a-Guthaben CHF
    saeule3_einzahlung: float = 7_258.0  # max. Einzahlung 2024 CHF/Jahr
    ahv_rente_erwartet: float = 0.0  # erwartete AHV-Rente CHF/Jahr
    ahv_vorbezug_jahre: int   = 0    # 0=normal, 1-2=Vorbezug
    ahv_aufschub_jahre: int   = 0    # 0=normal, 1-5=Aufschub

    # ── Pflegefall ────────────────────────────────────────────────────────
    care_state:      int    = ZUHAUSE
    care_years:      int    = 0
    care_duration:   int    = 0
    care_exit_tod:   bool   = False

    # ── Arbeitslosigkeit ──────────────────────────────────────────────────
    is_unemployed:   bool   = False
    unemp_years:     int    = 0
    wage_scar:       float  = 1.0   # Lohnfaktor nach Narbe (1.0 = keine Narbe)
    pk_aktiv:        bool   = True

    # ── Flags ─────────────────────────────────────────────────────────────
    alive:           bool   = True
    pk_bezogen:      bool   = False
    saeule3_bezogen: bool   = False


@dataclass
class HaushaltParams:
    """
    Inputparameter für den gesamten Haushalt.
    Enthält Person A (obligatorisch) und optional Person B.
    """
    # ── Personen ──────────────────────────────────────────────────────────
    person_a:        PersonParams
    person_b:        Optional[PersonParams] = None  # None = Einzelperson

    # ── Zivilstand & Ehe ──────────────────────────────────────────────────
    zivilstand:      str    = "ledig"      # 'ledig' | 'verheiratet'
    heiratsjahr:     Optional[int] = None
    marriage_duration: int  = 0            # Jahre seit Heirat

    # ── Gemeinsames Vermögen ──────────────────────────────────────────────
    liquides_vermoegen: float = 0.0        # freies liquides Vermögen CHF
    eigengut_a:      float  = 0.0          # Eigengut Person A (ZGB Art. 198)
    eigengut_b:      float  = 0.0          # Eigengut Person B (ZGB Art. 198)
    eiserne_reserve: float  = 0.0          # nicht investierter Puffer CHF

    # ── Liegenschaft ──────────────────────────────────────────────────────
    liegenschaft:    float  = 0.0          # Verkehrswert CHF
    hypothek:        float  = 0.0          # ausstehende Hypothek CHF
    hypothek_zins:   float  = HYPOTHEK_ZINS
    eigenmietwert:   float  = 0.0   
    ist_ersterwerber: bool  = False
    jahr_kauf:       Optional[int] = None  # für Ersterwerberabzug

    # ── Gemeinsame Ausgaben ───────────────────────────────────────────────
    ausgaben:        float  = 0.0          # Haushaltsausgaben CHF/Jahr
    einmalausgaben:  list   = field(default_factory=list)   # [(jahr, betrag)]
    einmaleinnahmen: list   = field(default_factory=list)   # [(jahr, betrag)]

    # ── Steuern ───────────────────────────────────────────────────────────
    gemeinde:        str    = "Zürich"
    konfession_a:    str    = "ohne"       # 'ohne'|'ref'|'kath'|'chr'
    konfession_b:    str    = "ohne"

    # ── Simulation ────────────────────────────────────────────────────────
    risikoaversion:  int    = 2            # 1=Konservativ, 2=Ausgewogen, 3=Wachstum
    risikoaversion_saeule3: int = 1        # Säule 3a meist konservativer

    zivilstand_logit_a: str = "single"     # initialer civil_status für Arbeitslosigkeits-Logit

    # ── Berechnete Felder ─────────────────────────────────────────────────
    @property
    def is_married(self) -> bool:
        return self.zivilstand == "verheiratet" and self.person_b is not None

    @property
    def has_partner(self) -> bool:
        return self.person_b is not None

    @property
    def steuerfuss(self) -> float:
        return get_steuerfuss(self.gemeinde, self.konfession_a)

    def jahresstart(self) -> int:
        """Aktuelles Kalenderjahr."""
        return date.today().year

    def alter_a(self) -> int:
        return self.jahresstart() - self.person_a.geburtsjahr

    def alter_b(self) -> Optional[int]:
        if self.person_b is None:
            return None
        return self.jahresstart() - self.person_b.geburtsjahr


# ── AHV-Rente berechnen ───────────────────────────────────────────────────────

def berechne_ahv_rente(ahv_rente_erwartet: float,
                        vorbezug_jahre: int = 0,
                        aufschub_jahre: int = 0) -> float:
    """
    AHV-Rente nach Vorbezug/Aufschub-Korrekturen.
    Quellen: EAK eak.admin.ch/de/vorbezug + /de/aufschub.
    Plafond Ehepaar wird in jahresschritt() geprüft.
    """
    rente = ahv_rente_erwartet

    if vorbezug_jahre > 0:
        vj    = min(vorbezug_jahre, max(AHV_VORBEZUG_ABZUG.keys()))
        rente = rente * (1 - AHV_VORBEZUG_ABZUG[vj])

    if aufschub_jahre > 0:
        aj    = min(aufschub_jahre, AHV_AUFSCHUB_MAX_JAHRE)
        rente = rente * (1 + AHV_AUFSCHUB_ZUSCHLAG[aj])

    return min(rente, AHV_MAX_RENTE)   # nie über Maximum


# ── Demonstrationsbeispiel (Dario + Partner) ─────────────────────────────────

person_a = PersonParams(
    name             = "Dario",
    geburtsjahr      = 1978,
    geschlecht       = "m",
    bildung          = 2,              # Berufsbildung
    einkommen        = 120_000,
    rentenalter      = 65,
    pk_guthaben      = 280_000,
    pk_guthaben_65   = 650_000,
    pk_bei_heirat    = 0.0,            # wird automatisch geschätzt
    saeule3          = 45_000,
    saeule3_einzahlung = 7_258,
    ahv_rente_erwartet = 26_000,
)

person_b = PersonParams(
    name             = "Partner",
    geburtsjahr      = 1983,
    geschlecht       = "f",
    bildung          = 2,
    einkommen        = 90_000,
    rentenalter      = 65,
    pk_guthaben      = 150_000,
    pk_guthaben_65   = 420_000,
    pk_bei_heirat    = 0.0,
    saeule3          = 30_000,         # Inputparameter
    saeule3_einzahlung = 7_258,
    ahv_rente_erwartet = 22_000,       # Inputparameter
)

haushalt = HaushaltParams(
    person_a            = person_a,
    person_b            = person_b,
    zivilstand          = "verheiratet",
    heiratsjahr         = 2008,
    marriage_duration   = date.today().year - 2008,
    liquides_vermoegen  = 1_000_000,
    eigengut_a          = 200_000,     # Inputparameter
    eigengut_b          = 100_000,     # Inputparameter
    eiserne_reserve     = 50_000,
    liegenschaft        = 900_000,
    hypothek            = 600_000,
    hypothek_zins       = 0.025,
    eigenmietwert       = 22_500,
    ist_ersterwerber    = False,
    jahr_kauf           = 2010,
    ausgaben            = 95_000,
    einmalausgaben      = [],
    einmaleinnahmen     = [],
    gemeinde            = "Urdorf",
    konfession_a        = "ohne",
    konfession_b        = "ohne",
    risikoaversion      = 2,
    risikoaversion_saeule3 = 1,
)

# ── Verifikation ──────────────────────────────────────────────────────────────
print("✓ Cell 4: Parameter-Schema geladen")
print(f"\nDemonstrationsbeispiel:")
print(f"  Haushalt:         {haushalt.person_a.name} + {haushalt.person_b.name}")
print(f"  Zivilstand:       {haushalt.zivilstand} seit {haushalt.heiratsjahr}")
print(f"  Alter A/B:        {haushalt.alter_a()} / {haushalt.alter_b()}")
print(f"  Einkommen A/B:    CHF {haushalt.person_a.einkommen:,.0f} / "
      f"CHF {haushalt.person_b.einkommen:,.0f}")
print(f"  Liquides Verm.:   CHF {haushalt.liquides_vermoegen:,.0f}")
print(f"  Liegenschaft:     CHF {haushalt.liegenschaft:,.0f} "
      f"(Hypo: CHF {haushalt.hypothek:,.0f})")
print(f"  Gemeinde:         {haushalt.gemeinde}  "
      f"Steuerfuss: {haushalt.steuerfuss:.0f}%")
print(f"  is_married:       {haushalt.is_married}")
print(f"  has_partner:      {haushalt.has_partner}")
print(f"\nAHV-Renten:")
ahv_a = berechne_ahv_rente(person_a.ahv_rente_erwartet,
                             person_a.ahv_vorbezug_jahre,
                             person_a.ahv_aufschub_jahre)
ahv_b = berechne_ahv_rente(person_b.ahv_rente_erwartet,
                             person_b.ahv_vorbezug_jahre,
                             person_b.ahv_aufschub_jahre)
ahv_total = ahv_a + ahv_b
ahv_plafond_aktiv = haushalt.is_married and ahv_total > AHV_PLAFOND_EHEPAAR
print(f"  AHV Person A:     CHF {ahv_a:,.0f}")
print(f"  AHV Person B:     CHF {ahv_b:,.0f}")
print(f"  Total:            CHF {ahv_total:,.0f}")
if ahv_plafond_aktiv:
    print(f"  ⚠️  Plafond aktiv: CHF {AHV_PLAFOND_EHEPAAR:,.0f} "
          f"(Reduktion CHF {ahv_total - AHV_PLAFOND_EHEPAAR:,.0f})")
else:
    print(f"  Plafond:          nicht aktiv (Limit CHF {AHV_PLAFOND_EHEPAAR:,.0f})")


# In[6]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5 — CRN-ARCHITEKTUR (COMMON RANDOM NUMBERS)
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
import numpy as np


@dataclass
class BasisSchocks:
    renditen:          np.ndarray
    renditen_saeule3:  np.ndarray
    inflation:         np.ndarray
    liegenschaft:      np.ndarray
    heim_kosten_r:     np.ndarray
    spitex_kosten_r:   np.ndarray
    mortalitaet_a:     np.ndarray
    mortalitaet_b:     np.ndarray
    bip_r:             np.ndarray   # NEU: standardisierte BIP-Abweichung


@dataclass
class RisikoSchocks:
    """
    Risiko-Zufallszahlen für eine Simulation.
    Getrennt von BasisSchocks → kein Einfluss auf Basispfade.
    """
    al_a:     np.ndarray   # shape (n_jahre, 3)
    al_b:     np.ndarray   # shape (n_jahre, 3)
    divorce:  np.ndarray   # shape (n_jahre, 2)
    pflege_a: np.ndarray   # shape (n_jahre, 5)
    pflege_b: np.ndarray   # shape (n_jahre, 5)
    # draws[0]: Eintritts-U       (p_pflege)
    # draws[1]: Typ-U             (p_heim_cond)
    # draws[2]: Heim-Dauer-U      (Exponential inverse CDF bei Eintritt)
    # draws[3]: Tod-U             (p_tod bei Episodenende)
    # draws[4]: Spitex-Dauer-U    (LogNormal inverse CDF bei Eintritt)


def generiere_basis_schocks(n_jahre, sim_seed, haushalt):
    rng = np.random.default_rng(sim_seed)

    renditen         = simuliere_rendite(haushalt.risikoaversion, n_jahre, rng)
    renditen_saeule3 = simuliere_rendite(haushalt.risikoaversion_saeule3, n_jahre, rng)
    inflation        = simuliere_inflation(n_jahre, rng)
    liegenschaft     = simuliere_liegenschaft_renditen(n_jahre, rng)
    heim_kosten_r    = rng.normal(HEIM_KOSTEN_MU,   HEIM_KOSTEN_SIGMA,   n_jahre)
    spitex_kosten_r  = rng.normal(SPITEX_KOSTEN_MU, SPITEX_KOSTEN_SIGMA, n_jahre)
    bip_r            = rng.standard_normal(n_jahre)   # NEU: N(0,1), unabhängig
    mortalitaet_a    = rng.random(n_jahre)
    mortalitaet_b    = rng.random(n_jahre) if haushalt.has_partner else np.zeros(n_jahre)

    return BasisSchocks(
        renditen         = renditen,
        renditen_saeule3 = renditen_saeule3,
        inflation        = inflation,
        liegenschaft     = liegenschaft,
        heim_kosten_r    = heim_kosten_r,
        spitex_kosten_r  = spitex_kosten_r,
        bip_r            = bip_r,
        mortalitaet_a    = mortalitaet_a,
        mortalitaet_b    = mortalitaet_b,
    )


def generiere_risiko_schocks(n_jahre: int,
                              sim_seed: int) -> RisikoSchocks:
    """
    Generiert alle Risiko-Zufallszahlen aus einem SEPARATEN RNG-Stream.
    sim_seed + 10_000_000 → garantiert andere Sequenz als BasisSchocks.
    """
    rng = np.random.default_rng(sim_seed + 10_000_000)   # Stream 2

    return RisikoSchocks(
        al_a     = rng.random((n_jahre, 3)),
        al_b     = rng.random((n_jahre, 3)),
        divorce  = rng.random((n_jahre, 2)),
        pflege_a = rng.random((n_jahre, 5)),
        pflege_b = rng.random((n_jahre, 5)),
    )


# ── Szenario-Definition ───────────────────────────────────────────────────────

SZENARIEN = {
    "Basis":                     {"al": False, "divorce": False, "pflege": False},
    "Basis + Arbeitslosigkeit":  {"al": True,  "divorce": False, "pflege": False},
    "Basis + Scheidung":         {"al": False, "divorce": True,  "pflege": False},
    "Basis + Pflegefall":        {"al": False, "divorce": False, "pflege": True },
    "Basis + AL + Scheidung":    {"al": True,  "divorce": True,  "pflege": False},
    "Basis + AL + Pflegefall":   {"al": True,  "divorce": False, "pflege": True },
    "Basis + Scheidung + Pflege":{"al": False, "divorce": True,  "pflege": True },
    "Basis + Alle":              {"al": True,  "divorce": True,  "pflege": True },
}


def initialisiere_haushalt_state(haushalt: HaushaltParams,
                                  basis: BasisSchocks) -> dict:
    """Erstellt den initialen Simulations-State aus HaushaltParams + BasisSchocks."""
    jahr_aktuell = date.today().year
    alter_a      = jahr_aktuell - haushalt.person_a.geburtsjahr

    ahv_a = berechne_ahv_rente(
        haushalt.person_a.ahv_rente_erwartet,
        haushalt.person_a.ahv_vorbezug_jahre,
        haushalt.person_a.ahv_aufschub_jahre,
    )

    s = {
        # ── Zeitachse ─────────────────────────────────────────────────────
        "jahr":          jahr_aktuell,
        "sim_jahr":      0,
        "bip_r": basis.bip_r,
        "alter_a":       alter_a,

        # ── Vermögen ──────────────────────────────────────────────────────
        "vermoegen":          haushalt.liquides_vermoegen,
        "eigengut_a":         haushalt.eigengut_a,
        "eigengut_b":         haushalt.eigengut_b,
        "eiserne_reserve":    haushalt.eiserne_reserve,
        "liegenschaft":       haushalt.liegenschaft,
        "hypothek":           haushalt.hypothek,
        "hypothek_zins":      haushalt.hypothek_zins,
        "ist_ersterwerber":   haushalt.ist_ersterwerber,
        "jahr_kauf":          haushalt.jahr_kauf or jahr_aktuell,

        # ── Person A ──────────────────────────────────────────────────────
        "einkommen_a":        haushalt.person_a.einkommen,
        "pk_kapital_a":       haushalt.person_a.pk_guthaben,
        "pk_bei_heirat_a":    haushalt.person_a.pk_bei_heirat,
        "pk_rente_a":         0.0,
        "pk_bezogen_a":       False,
        "pk_aktiv_a":         True,
        "pk_kapital_a_vor_bezug": 0.0,
        "saeule3_a":          haushalt.person_a.saeule3,
        "saeule3_bezogen_a":  False,
        "ahv_rente_a":        ahv_a,
        "pensioniert_a":      False,
        "alive_a":            True,
        "is_unemployed_a":    False,
        "unemp_years_a":      0,
        "wage_scar_a":        1.0,
        "civil_status_init_a": haushalt.zivilstand_logit_a,
        "care_state_a":       ZUHAUSE,
        "care_years_a":       0,
        "care_duration_a":    0,

        # ── Person B (falls vorhanden) ────────────────────────────────────
        "has_partner":        haushalt.has_partner,
        "alive_b":            haushalt.has_partner,
        "pensioniert_b":      False,
        "is_unemployed_b":    False,
        "unemp_years_b":      0,
        "wage_scar_b":        1.0,
        "care_state_b":       ZUHAUSE,
        "care_years_b":       0,
        "care_duration_b":    0,

        # ── Haushalt ──────────────────────────────────────────────────────
        "is_married":         haushalt.is_married,
        "marriage_duration":  haushalt.marriage_duration,
        "ausgaben":           haushalt.ausgaben,
        "gemeinde":           haushalt.gemeinde,
        "konfession_a":       haushalt.konfession_a,

        # ── Flags ─────────────────────────────────────────────────────────
        "ruiniert":                False,
        "geschieden":              False,
        "verwitwet_a":             False,
        "liegenschaft_verkauft":   False,
        "liegenschaft_hinweis":    False,
        "hinweis_alter":           None,

        # ── Kostenfaktoren (NEU: beide separat) ───────────────────────────
        "heim_kosten_faktor":   1.0,
        "spitex_kosten_faktor": 1.0,   # NEU

        # ── Basispfade (CRN) ──────────────────────────────────────────────
        "renditen":          basis.renditen,
        "renditen_saeule3":  basis.renditen_saeule3,
        "inflation":         basis.inflation,
        "liegenschaft_r":    basis.liegenschaft,
        "heim_kosten_r":     basis.heim_kosten_r,
        "spitex_kosten_r":   basis.spitex_kosten_r,   # NEU
        "mortalitaet_a":     basis.mortalitaet_a,
        "mortalitaet_b":     basis.mortalitaet_b,
    }

    if haushalt.has_partner:
        alter_b = jahr_aktuell - haushalt.person_b.geburtsjahr
        ahv_b   = berechne_ahv_rente(
            haushalt.person_b.ahv_rente_erwartet,
            haushalt.person_b.ahv_vorbezug_jahre,
            haushalt.person_b.ahv_aufschub_jahre,
        )
        s.update({
            "alter_b":               alter_b,
            "einkommen_b":           haushalt.person_b.einkommen,
            "pk_kapital_b":          haushalt.person_b.pk_guthaben,
            "pk_bei_heirat_b":       haushalt.person_b.pk_bei_heirat,
            "pk_rente_b":            0.0,
            "pk_bezogen_b":          False,
            "pk_aktiv_b":            True,
            "pk_kapital_b_vor_bezug": 0.0,
            "saeule3_b":             haushalt.person_b.saeule3,
            "saeule3_bezogen_b":     False,
            "ahv_rente_b":           ahv_b,
        })

    return s


# ── Verifikation ──────────────────────────────────────────────────────────────

n_jahre_test = SIM_BIS_ALTER - haushalt.alter_a()
_basis_1  = generiere_basis_schocks(n_jahre_test, RANDOM_SEED, haushalt)
_basis_2  = generiere_basis_schocks(n_jahre_test, RANDOM_SEED, haushalt)
_risiko_1 = generiere_risiko_schocks(n_jahre_test, RANDOM_SEED)
_risiko_2 = generiere_risiko_schocks(n_jahre_test, RANDOM_SEED)

assert np.allclose(_basis_1.renditen, _basis_2.renditen), \
    "❌ BasisSchocks nicht reproduzierbar!"
assert np.allclose(_risiko_1.al_a, _risiko_2.al_a), \
    "❌ RisikoSchocks nicht reproduzierbar!"
assert not np.allclose(_basis_1.renditen, _risiko_1.al_a[:, 0]), \
    "❌ Basis- und Risikoschocks nicht getrennt!"
assert len(_basis_1.heim_kosten_r)   == n_jahre_test, "❌ heim_kosten_r Länge!"
assert len(_basis_1.spitex_kosten_r) == n_jahre_test, "❌ spitex_kosten_r Länge!"

_state_test = initialisiere_haushalt_state(haushalt, _basis_1)

print("✓ Cell 5: CRN-Architektur geladen")
print(f"  BasisSchocks reproduzierbar:          ✅")
print(f"  RisikoSchocks reproduzierbar:         ✅")
print(f"  Basis ≠ Risiko (getrennte Streams):   ✅")
print(f"  heim_kosten_r vorhanden:              ✅")
print(f"  spitex_kosten_r vorhanden (NEU):      ✅")
print(f"\nSzenarien ({len(SZENARIEN)}):")
for name, flags in SZENARIEN.items():
    aktiv = [k for k, v in flags.items() if v]
    print(f"  {name:<35} "
          f"{'(' + ', '.join(aktiv) + ')' if aktiv else '(nur Basis)'}")
print(f"\nState initialisiert:")
print(f"  heim_kosten_faktor:   {_state_test['heim_kosten_faktor']:.2f}")
print(f"  spitex_kosten_faktor: {_state_test['spitex_kosten_faktor']:.2f}")


# In[7]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6 — JAHRESSCHRITT
# Cashflow-Reihenfolge:
#   1. Sterblichkeit  2. Risiken  3. Einkommen  4. Ausgaben
#   5. Steuern        6. Cashflow 7. Renditen   8. Vermögen  9. Ruin
#
# Fix 3: Pflege-Tod setzt nur Flag — kein frühes return mehr.
#         Cashflow (Kosten, Renten, Steuern, Rendite) läuft im Todesjahr durch.
# Fix 4: BVG-Cashflow-Abzug = Arbeitnehmeranteil (50%).
#         PK-Gutschrift bleibt voll (AG + AN). Basis: BVG Art. 66.
# ═══════════════════════════════════════════════════════════════════════════


def _civil_status_a(s: dict) -> str:
    if s["is_married"]:       return "married"
    if s.get("geschieden"):   return "divorced"
    if s.get("verwitwet_a"): return "widowed"
    return s.get("civil_status_init_a", "single")


def jahresschritt(s: dict,
                  haushalt: HaushaltParams,
                  risiko_schocks: RisikoSchocks,
                  do_al:      bool = False,
                  do_divorce: bool = False,
                  do_pflege:  bool = False) -> dict:
    s   = dict(s)
    t   = s["sim_jahr"]
    s["sim_jahr"] += 1

    jahr    = s["jahr"]
    alter_a = s["alter_a"]
    alter_b = s.get("alter_b", None)
    sf      = get_steuerfuss(s["gemeinde"], s["konfession_a"])

    ra_a = haushalt.person_a.rentenalter
    ra_b = haushalt.person_b.rentenalter if haushalt.has_partner else 99

    pensioniert_a = alter_a >= ra_a
    pensioniert_b = (alter_b is not None and alter_b >= ra_b
                     and s.get("alive_b", False))
    s["pensioniert_a"] = pensioniert_a
    s["pensioniert_b"] = pensioniert_b

    r_portfolio    = s["renditen"][t]
    r_saeule3      = s["renditen_saeule3"][t]
    inflation      = s["inflation"][t]
    r_liegenschaft = s["liegenschaft_r"][t]

    s["eigengut_a"] = s["eigengut_a"] * (1 + r_portfolio)
    if s.get("has_partner"):
        s["eigengut_b"] = s.get("eigengut_b", 0.0) * (1 + r_portfolio)

    # Kostenfaktoren separat fortschreiben
    s["heim_kosten_faktor"]   = (s.get("heim_kosten_faktor",   1.0)
                                  * (1 + s["heim_kosten_r"][t]))
    s["spitex_kosten_faktor"] = (s.get("spitex_kosten_faktor", 1.0)
                                  * (1 + s["spitex_kosten_r"][t]))
    bip_dev = float(s["bip_r"][t])

    # ══════════════════════════════════════════════════════════════════════
    # 1. STERBLICHKEIT (allgemeine Mortalität — frühes return bleibt hier)
    # ══════════════════════════════════════════════════════════════════════
    if s["alive_a"] and s["care_state_a"] != HEIM_STATE:
        qx_a = get_qx(haushalt.person_a.geburtsjahr,
                      1 if haushalt.person_a.geschlecht == "f" else 0,
                      alter_a)
        if s["mortalitaet_a"][t] < qx_a:
            s["alive_a"] = False

    vorher_alive_b = s.get("alive_b", False)
    if (s.get("has_partner", False) and s.get("alive_b", False)
            and s.get("care_state_b", ZUHAUSE) != HEIM_STATE):
        qx_b = get_qx(haushalt.person_b.geburtsjahr,
                      1 if haushalt.person_b.geschlecht == "f" else 0,
                      alter_b)
        if s["mortalitaet_b"][t] < qx_b:
            s["alive_b"] = False

    if vorher_alive_b and not s.get("alive_b", False):
        s["ausgaben"]   *= (1 - AUSGABEN_REDUKTION_TOD_PARTNER)
        s["is_married"]  = False
        s["verwitwet_a"] = True

    # Allgemeiner Tod Person A: keine Kosten angefallen → frühes return ok
    if not s["alive_a"]:
        s["alter_a"] = alter_a + 1
        s["jahr"]    = jahr + 1
        if alter_b is not None:
            s["alter_b"] = alter_b + 1
        return s

    # ══════════════════════════════════════════════════════════════════════
    # 2. RISIKEN
    # ══════════════════════════════════════════════════════════════════════
    income_factor_a = 1.0
    income_factor_b = 1.0
    zusatz_ausgaben = 0.0

    # ── Arbeitslosigkeit ──────────────────────────────────────────────────
    if do_al and not pensioniert_a:
        al_r_a = _unemployment_step_crn(
    alter         = alter_a,
    female        = 1 if haushalt.person_a.geschlecht == "f" else 0,
    civil_status  = _civil_status_a(s),
    bildung       = haushalt.person_a.bildung,
    is_unemployed = s["is_unemployed_a"],
    unemp_years   = s["unemp_years_a"],
    wage_scar     = s["wage_scar_a"],
    bip_dev       = bip_dev,          # NEU
    draws         = risiko_schocks.al_a[t],
)
        s["is_unemployed_a"] = al_r_a["is_unemployed"]
        s["unemp_years_a"]   = al_r_a["unemp_years"]
        s["wage_scar_a"]     = al_r_a["wage_scar"]
        s["pk_aktiv_a"]      = al_r_a["pk_aktiv"]
        income_factor_a      = al_r_a["income_factor"]

    if do_al and s.get("has_partner") and s.get("alive_b") and not pensioniert_b:
        al_r_b = _unemployment_step_crn(
            alter         = alter_b,
            female        = 1 if haushalt.person_b.geschlecht == "f" else 0,
            civil_status  = "married" if s["is_married"] else "single",
            bildung       = haushalt.person_b.bildung,
            is_unemployed = s["is_unemployed_b"],
            unemp_years   = s["unemp_years_b"],
            wage_scar     = s["wage_scar_b"],
            bip_dev = bip_dev,
            draws         = risiko_schocks.al_b[t],
        )
        s["is_unemployed_b"] = al_r_b["is_unemployed"]
        s["unemp_years_b"]   = al_r_b["unemp_years"]
        s["wage_scar_b"]     = al_r_b["wage_scar"]
        s["pk_aktiv_b"]      = al_r_b["pk_aktiv"]
        income_factor_b      = al_r_b["income_factor"]

    # ── Scheidung ─────────────────────────────────────────────────────────
    if (do_divorce and s["is_married"] and not pensioniert_a
            and not s.get("geschieden", False)):
        div_r = _divorce_step_crn(
            alter_a           = alter_a,
            female_a          = 1 if haushalt.person_a.geschlecht == "f" else 0,
            bildung_a         = haushalt.person_a.bildung,
            is_unemployed_a   = s["is_unemployed_a"],
            marriage_duration = s["marriage_duration"],
            vermoegen         = s["vermoegen"],
            eigengut_a        = s["eigengut_a"],
            eigengut_b        = s.get("eigengut_b", 0.0),
            pk_a              = s["pk_kapital_a"],
            pk_b              = s.get("pk_kapital_b", 0.0),
            pk_bei_heirat_a   = s["pk_bei_heirat_a"],
            pk_bei_heirat_b   = s.get("pk_bei_heirat_b", 0.0),
            draws             = risiko_schocks.divorce[t],
        )
        if div_r["geschieden"]:
            s["vermoegen"]         = div_r["vermoegen_nach"]
            s["pk_kapital_a"]      = div_r["pk_a_neu"]
            s["pk_kapital_b"]      = div_r["pk_b_neu"]
            s["eigengut_a"]        = div_r["eigengut_a_neu"]
            s["eigengut_b"]        = div_r["eigengut_b_neu"]
            s["is_married"]        = False
            s["geschieden"]        = True
            s["marriage_duration"] = 0
            if s["liegenschaft"] > 0:
                erloes = max(0.0, s["liegenschaft"] - s["hypothek"])
                s["vermoegen"]            += erloes / 2
                s["liegenschaft"]          = 0.0
                s["hypothek"]              = 0.0
                s["liegenschaft_verkauft"] = True
        else:
            s["marriage_duration"] = s["marriage_duration"] + 1

    # ── Pflegefall Person A ───────────────────────────────────────────────
    # Fix 3: kein frühes return — nur Flag setzen, Cashflow läuft durch
    pf_kosten_a   = 0.0
    _pflege_tod_a = False

    if do_pflege and alter_a >= 50:
        pf_r_a = _pflegefall_step_crn(
            alter                = alter_a,
            female               = 1 if haushalt.person_a.geschlecht == "f" else 0,
            geburtsjahr          = haushalt.person_a.geburtsjahr,
            care_state           = s["care_state_a"],
            care_years           = s["care_years_a"],
            care_duration        = s["care_duration_a"],
            rente_ahv            = s["ahv_rente_a"],
            rente_pk             = s["pk_rente_a"],
            wealth               = s["vermoegen"],
            gemeinde             = s["gemeinde"],
            is_married           = s["is_married"],
            liegenschaft         = s["liegenschaft"],
            hypothek             = s["hypothek"],
            heim_kosten_faktor   = s["heim_kosten_faktor"],
            spitex_kosten_faktor = s["spitex_kosten_faktor"],
            draws                = risiko_schocks.pflege_a[t],
        )
        s["care_state_a"]    = pf_r_a["care_state"]
        s["care_years_a"]    = pf_r_a["care_years"]
        s["care_duration_a"] = pf_r_a["care_duration"]
        pf_kosten_a          = pf_r_a["pflegekosten_netto_jahr"]

        if pf_r_a["care_state"] == HEIM_STATE and pf_r_a.get("heim_neu"):
            s["ausgaben"] *= (1 - HEIM_AUSGABEN_ANTEIL)

        # Fix 3: Tod als Flag — kein return s hier
        if pf_r_a.get("gestorben"):
            _pflege_tod_a = True
            s["alive_a"]  = False

    # ── Pflegefall Person B ───────────────────────────────────────────────
    pf_kosten_b = 0.0
    if (do_pflege and s.get("has_partner") and s.get("alive_b")
            and alter_b is not None and alter_b >= 50):
        pf_r_b = _pflegefall_step_crn(
            alter                = alter_b,
            female               = 1 if haushalt.person_b.geschlecht == "f" else 0,
            geburtsjahr          = haushalt.person_b.geburtsjahr,
            care_state           = s["care_state_b"],
            care_years           = s["care_years_b"],
            care_duration        = s["care_duration_b"],
            rente_ahv            = s.get("ahv_rente_b", 0.0),
            rente_pk             = s.get("pk_rente_b", 0.0),
            wealth               = s["vermoegen"],
            gemeinde             = s["gemeinde"],
            is_married           = s["is_married"],
            liegenschaft         = s["liegenschaft"],
            hypothek             = s["hypothek"],
            heim_kosten_faktor   = s["heim_kosten_faktor"],
            spitex_kosten_faktor = s["spitex_kosten_faktor"],
            draws                = risiko_schocks.pflege_b[t],
        )
        s["care_state_b"]    = pf_r_b["care_state"]
        s["care_years_b"]    = pf_r_b["care_years"]
        s["care_duration_b"] = pf_r_b["care_duration"]
        pf_kosten_b          = pf_r_b["pflegekosten_netto_jahr"]

        if pf_r_b["care_state"] == HEIM_STATE and pf_r_b.get("heim_neu"):
            s["ausgaben"] *= (1 - HEIM_AUSGABEN_ANTEIL)

        if pf_r_b.get("gestorben"):
            s["alive_b"]     = False
            s["ausgaben"]   *= (1 - AUSGABEN_REDUKTION_TOD_PARTNER)
            s["is_married"]  = False
            s["verwitwet_a"] = True

    zusatz_ausgaben += pf_kosten_a + pf_kosten_b

    # ══════════════════════════════════════════════════════════════════════
    # 3. EINKOMMEN
    # ══════════════════════════════════════════════════════════════════════
    s["ausgaben"] *= (1 + inflation)

    # ── PK-Kapitalbezug bei Pensionierung ─────────────────────────────────
    if pensioniert_a and not s["pk_bezogen_a"]:
        if s.get("szenario") in ["Basis", "Basis + Pflegefall"]:
            pk_basis_a = haushalt.person_a.pk_guthaben_65
        else:
            pk_basis_a = s["pk_kapital_a"]
        s["pk_kapital_a_vor_bezug"] = pk_basis_a
        pk_kap_a  = pk_basis_a * haushalt.person_a.pk_bezug_kapital_anteil
        pk_rent_a = (pk_basis_a
                     * (1 - haushalt.person_a.pk_bezug_kapital_anteil)
                     * haushalt.person_a.pk_umwandlungssatz)
        if pk_kap_a > 0:
            s["vermoegen"] += pk_kap_a - berechne_kapitalleistungssteuer(pk_kap_a, sf)
        s["pk_rente_a"]  = pk_rent_a
        s["pk_kapital_a"] = 0.0
        s["pk_bezogen_a"] = True

    if (pensioniert_b and s.get("has_partner") and s.get("alive_b")
            and not s.get("pk_bezogen_b", False)):
        if s.get("szenario") in ["Basis", "Basis + Pflegefall"]:
            pk_basis_b = haushalt.person_b.pk_guthaben_65
        else:
            pk_basis_b = s.get("pk_kapital_b", 0.0)
        s["pk_kapital_b_vor_bezug"] = pk_basis_b
        pk_kap_b  = pk_basis_b * haushalt.person_b.pk_bezug_kapital_anteil
        pk_rent_b = (pk_basis_b
                     * (1 - haushalt.person_b.pk_bezug_kapital_anteil)
                     * haushalt.person_b.pk_umwandlungssatz)
        if pk_kap_b > 0:
            s["vermoegen"] += pk_kap_b - berechne_kapitalleistungssteuer(pk_kap_b, sf)
        s["pk_rente_b"]   = pk_rent_b
        s["pk_kapital_b"]  = 0.0
        s["pk_bezogen_b"]  = True

    # ── Säule 3a Bezug ────────────────────────────────────────────────────
    if pensioniert_a and not s["saeule3_bezogen_a"] and s["saeule3_a"] > 0:
        steuer_s3_a    = berechne_kapitalleistungssteuer(s["saeule3_a"], sf)
        s["vermoegen"] += max(0.0, s["saeule3_a"] - steuer_s3_a)
        s["saeule3_a"]  = 0.0
        s["saeule3_bezogen_a"] = True

    if (pensioniert_b and s.get("has_partner") and s.get("alive_b")
            and not s.get("saeule3_bezogen_b", False)
            and s.get("saeule3_b", 0.0) > 0):
        steuer_s3_b    = berechne_kapitalleistungssteuer(s.get("saeule3_b", 0.0), sf)
        s["vermoegen"] += max(0.0, s.get("saeule3_b", 0.0) - steuer_s3_b)
        s["saeule3_b"]  = 0.0
        s["saeule3_bezogen_b"] = True

    # ── Bruttoeinkommen ───────────────────────────────────────────────────
    eink_a = (s["einkommen_a"] * income_factor_a
              if not pensioniert_a
              else s["ahv_rente_a"] + s["pk_rente_a"])

    eink_b = 0.0
    if s.get("has_partner") and s.get("alive_b") and s["is_married"]:
        eink_b = (s.get("einkommen_b", 0.0) * income_factor_b
                  if not pensioniert_b
                  else s.get("ahv_rente_b", 0.0) + s.get("pk_rente_b", 0.0))

    # ── AHV-Plafond ───────────────────────────────────────────────────────
    if s["is_married"] and pensioniert_a and pensioniert_b:
        ahv_total = s["ahv_rente_a"] + s.get("ahv_rente_b", 0.0)
        if ahv_total > AHV_PLAFOND_EHEPAAR:
            faktor           = AHV_PLAFOND_EHEPAAR / ahv_total
            s["ahv_rente_a"] *= faktor
            if "ahv_rente_b" in s:
                s["ahv_rente_b"] *= faktor
            eink_a = s["ahv_rente_a"] + s["pk_rente_a"]
            eink_b = s.get("ahv_rente_b", 0.0) + s.get("pk_rente_b", 0.0)

    # ── PK aufbauen — volle Altersgutschrift (AG + AN) der PK gutschreiben
    # Fix 4: Cashflow-Abzug nur Arbeitnehmeranteil (50%), BVG Art. 66
    if not pensioniert_a and not s["pk_bezogen_a"]:
        bvg_a = max(eink_a - KOORDINATIONSABZUG, 0) * altersgutschrift(alter_a)
        if s.get("pk_aktiv_a", True) and bvg_a > 0:
            s["pk_kapital_a"] = s["pk_kapital_a"] * (1 + BVG_ZINS) + bvg_a
        else:
            s["pk_kapital_a"] = s["pk_kapital_a"] 
    else:
        bvg_a = 0.0

    if (s.get("has_partner") and s.get("alive_b") and s["is_married"]
            and not pensioniert_b and not s.get("pk_bezogen_b")):
        bvg_b = (max(s.get("einkommen_b", 0.0) * income_factor_b
                     - KOORDINATIONSABZUG, 0) * altersgutschrift(alter_b))
        if s.get("pk_aktiv_b", True) and bvg_b > 0:
            s["pk_kapital_b"] = s.get("pk_kapital_b", 0.0) * (1 + BVG_ZINS) + bvg_b
        else:
            s["pk_kapital_b"] = s.get("pk_kapital_b", 0.0) 
    else:
        bvg_b = 0.0

    # ── Säule 3a aufbauen ─────────────────────────────────────────────────
    s3_einz_a = 0.0
    if not pensioniert_a and not s["saeule3_bezogen_a"] and income_factor_a > 0.5:
        s3_einz_a      = haushalt.person_a.saeule3_einzahlung
        s["saeule3_a"] = s["saeule3_a"] * (1 + r_saeule3) + s3_einz_a
    elif not s["saeule3_bezogen_a"]:
        s["saeule3_a"] = s["saeule3_a"] * (1 + r_saeule3)

    s3_einz_b = 0.0
    if (s.get("has_partner") and s.get("alive_b") and s["is_married"]
            and not pensioniert_b
            and not s.get("saeule3_bezogen_b", False)
            and income_factor_b > 0.5):
        s3_einz_b      = haushalt.person_b.saeule3_einzahlung
        s["saeule3_b"] = s.get("saeule3_b", 0.0) * (1 + r_saeule3) + s3_einz_b
    elif not s.get("saeule3_bezogen_b", False):
        s["saeule3_b"] = s.get("saeule3_b", 0.0) * (1 + r_saeule3)

    # ══════════════════════════════════════════════════════════════════════
    # 4. AUSGABEN
    # ══════════════════════════════════════════════════════════════════════
    ausgaben_jahr = s["ausgaben"] + zusatz_ausgaben

    for (j, betrag) in haushalt.einmalausgaben:
        if j == jahr: ausgaben_jahr += betrag
    for (j, betrag) in haushalt.einmaleinnahmen:
        if j == jahr: ausgaben_jahr -= betrag

    hypo_zinsen = 0.0
    if s["hypothek"] > 0 and not s["liegenschaft_verkauft"]:
        hypo_zinsen    = s["hypothek"] * s["hypothek_zins"]
        ausgaben_jahr += hypo_zinsen
        if s["liegenschaft"] > 0:
            ltv = s["hypothek"] / s["liegenschaft"]
            if ltv > AMORT_SCHWELLE:
                jahre_bis_rente_a = max(1, ra_a - alter_a)
                jahre_bis_rente_b = max(1, ra_b - alter_b) if alter_b else 999
                horizont          = min(jahre_bis_rente_a, jahre_bis_rente_b, 15)
                ueberschuss       = s["hypothek"] - AMORT_SCHWELLE * s["liegenschaft"]
                amort             = ueberschuss / horizont
                s["hypothek"]     = max(0.0, s["hypothek"] - amort)
                ausgaben_jahr    += amort

    if s["liegenschaft"] > 0 and not s["liegenschaft_verkauft"]:
        s["liegenschaft"] *= (1 + r_liegenschaft)

    # ══════════════════════════════════════════════════════════════════════
    # 5. STEUERN
    # ══════════════════════════════════════════════════════════════════════
    emw, hypo_abzug = berechne_liegenschaft_steuereffekt(
        liegenschaft     = s["liegenschaft"],
        hypothek         = s["hypothek"],
        hypo_zins_satz   = s["hypothek_zins"],
        simulationsjahr  = jahr,
        eigenmietwert    = haushalt.eigenmietwert,
        ist_ersterwerber = s["ist_ersterwerber"],
        jahre_seit_kauf  = jahr - s["jahr_kauf"],
    )

    if not pensioniert_a:
        abzug_verpfl_a = min(3_200, 15 * 220)
        abzug_fahrt_a  = 3_995
        netto_bk_a     = max(0.0, eink_a - bvg_a * 0.5)
        abzug_bk_a     = max(2_000, min(4_000, netto_bk_a * 0.03))
        abzug_vers_a   = 5_800 if s["is_married"] else 3_000
        ahv_abzug_a    = eink_a * AHV_IV_EO_SATZ
        alv_abzug_a    = min(eink_a, ALV_MAX_LOHN) * ALV_SATZ
        nbuv_abzug_a   = min(eink_a, ALV_MAX_LOHN) * NBUV_SATZ
        steuerbares_a  = max(0.0,
            eink_a - bvg_a * 0.5 - s3_einz_a
            - ahv_abzug_a - alv_abzug_a - nbuv_abzug_a
            + emw - hypo_abzug
            - abzug_verpfl_a - abzug_fahrt_a
            - abzug_bk_a - abzug_vers_a)
    else:
        abzug_vers_a  = 5_800 if s["is_married"] else 3_000
        steuerbares_a = max(0.0, eink_a + emw - hypo_abzug - abzug_vers_a)

    steuerbares_b = 0.0
    if s.get("has_partner") and s.get("alive_b") and s["is_married"] and eink_b > 0:
        if not pensioniert_b:
            netto_bk_b   = max(0.0, eink_b - bvg_b * 0.5)
            abzug_bk_b   = max(2_000, min(4_000, netto_bk_b * 0.03))
            ahv_abzug_b  = eink_b * AHV_IV_EO_SATZ
            alv_abzug_b  = min(eink_b, ALV_MAX_LOHN) * ALV_SATZ
            nbuv_abzug_b = min(eink_b, ALV_MAX_LOHN) * NBUV_SATZ
            steuerbares_b = max(0.0,
                eink_b - bvg_b * 0.5 - s3_einz_b
                - ahv_abzug_b - alv_abzug_b - nbuv_abzug_b
                - abzug_bk_b)
        else:
            steuerbares_b = max(0.0, eink_b)

    if s["is_married"] and s.get("alive_b"):
        steuer_eink = berechne_einkommenssteuer(
            steuerbares_a + steuerbares_b, sf,
            is_married=True, zweiteinkommen=steuerbares_b)
    else:
        steuer_eink = berechne_einkommenssteuer(steuerbares_a, sf)

    liegenschafts_equity = max(0.0, s["liegenschaft"] - s["hypothek"])
    reinvermoegen        = s["vermoegen"] + liegenschafts_equity
    steuer_verm          = berechne_vermoegenssteuer(
        reinvermoegen, sf, is_married=s["is_married"])

    # ══════════════════════════════════════════════════════════════════════
    # 6. CASHFLOW
    # Fix 4: nur Arbeitnehmeranteil (bvg / 2) vom Cashflow abziehen
    #        PK-Gutschrift oben bleibt voll (AG + AN). Basis: BVG Art. 66.
    # ══════════════════════════════════════════════════════════════════════
    einnahmen_netto = (eink_a + eink_b
                       - bvg_a / 2 - bvg_b / 2
                       - s3_einz_a - s3_einz_b
                       - steuer_eink)

    cashflow = einnahmen_netto - ausgaben_jahr - steuer_verm

    # ══════════════════════════════════════════════════════════════════════
    # 7. RENDITEN
    # ══════════════════════════════════════════════════════════════════════
    invest_verm    = max(0.0, s["vermoegen"] - s["eiserne_reserve"])
    s["vermoegen"] = (invest_verm * (1 + r_portfolio)
                      + s["eiserne_reserve"]
                      + cashflow)

    # ══════════════════════════════════════════════════════════════════════
    # 8. VERMÖGEN AKTUALISIEREN
    # ══════════════════════════════════════════════════════════════════════
    s["nettovermoegen"] = (s["vermoegen"]
                           + s.get("saeule3_a", 0.0)
                           + s.get("saeule3_b", 0.0)
                           + liegenschafts_equity)

    if (s["vermoegen"] <= s["eiserne_reserve"] * 1.1
            and s["liegenschaft"] > 0
            and not s["liegenschaft_verkauft"]
            and not s.get("liegenschaft_hinweis", False)):
        s["liegenschaft_hinweis"] = True
        s["hinweis_alter"]        = alter_a

    # ══════════════════════════════════════════════════════════════════════
    # 9. RUIN
    # ══════════════════════════════════════════════════════════════════════
    s["ruiniert"] = s["vermoegen"] <= 0

    s["alter_a"] = alter_a + 1
    s["jahr"]    = jahr    + 1
    if alter_b is not None:
        s["alter_b"] = alter_b + 1

    return s


print("✓ Cell 6: jahresschritt() definiert")
print(f"  Fix 3: Pflege-Tod → nur Flag, kein frühes return ✓")
print(f"  Fix 4: BVG-Cashflow-Abzug = AN-Anteil (50%), BVG Art. 66 ✓")


# In[8]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7 — RISIKOMODULE
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. ARBEITSLOSIGKEIT ───────────────────────────────────────────────────
# [_unemployment_step_crn() identisch zu bisheriger Version — unverändert]

def _get_exit_prob_al(alter: int, unemp_years: int) -> float:
    age_band = ("18–24" if alter <= 24 else "25–29" if alter <= 29 else
                "30–34" if alter <= 34 else "35–39" if alter <= 39 else
                "40–44" if alter <= 44 else "45–49" if alter <= 49 else
                "50–54" if alter <= 54 else "55–64")
    q_age = Q_EXIT_BY_AGE.get(age_band, Q_EXIT_BY_AGE["55–64"])
    q_dur = Q_BY_DURATION.get(unemp_years, Q_DURATION_DEFAULT)
    return min(q_age * q_dur, 1.0)


def _unemployment_step_crn(alter, female, civil_status, bildung,
                             is_unemployed, unemp_years, wage_scar,
                             bip_dev: float = 0.0,   # NEU
                             draws: np.ndarray = None) -> dict:

    if alter < 18 or alter > 64:
        return {"is_unemployed": False, "unemp_years": 0,
                "pk_aktiv": True, "income_factor": wage_scar, "wage_scar": wage_scar}

    age_c = alter - AGE_MEAN_AL

    xb = (PARAMS_AL["Intercept"]
          + PARAMS_AL["age_c"]         * age_c
          + PARAMS_AL["age_c_sq"]      * age_c**2
          + PARAMS_AL["female"]        * female
          + PARAMS_AL["civ_single"]    * int(civil_status == "single")
          + PARAMS_AL["civ_separated"] * int(civil_status == "separated")
          + PARAMS_AL["civ_divorced"]  * int(civil_status == "divorced")
          + PARAMS_AL["civ_widowed"]   * int(civil_status == "widowed")
          + PARAMS_AL["edu_basic"]     * int(bildung == 1)
          + PARAMS_AL["edu_tertiary"]  * int(bildung == 3)
          + BIP_ALPHA * bip_dev)       # NEU

    p_eintritt = 1 / (1 + np.exp(-xb))

    # Rest unverändert...
    if not is_unemployed:
        if draws[0] < p_eintritt:
            return {"is_unemployed": True,  "unemp_years": 1,
                    "pk_aktiv": False, "income_factor": 0.70, "wage_scar": wage_scar}
        else:
            return {"is_unemployed": False, "unemp_years": 0,
                    "pk_aktiv": True,  "income_factor": wage_scar, "wage_scar": wage_scar}
    else:
        q          = _get_exit_prob_al(alter, unemp_years)
        neue_jahre = unemp_years + 1
        if draws[0] < q:
            narbe          = 1.0 + WAGE_SCAR[female]
            neue_wage_scar = wage_scar * narbe
            return {"is_unemployed": False, "unemp_years": 0,
                    "pk_aktiv": True, "income_factor": neue_wage_scar,
                    "wage_scar": neue_wage_scar}
        else:
            alv_faktor = 0.70 if neue_jahre <= 2 else 0.0
            return {"is_unemployed": True,  "unemp_years": neue_jahre,
                    "pk_aktiv": False, "income_factor": alv_faktor, "wage_scar": wage_scar}


# ── 2. SCHEIDUNG ──────────────────────────────────────────────────────────
# [_divorce_step_crn() identisch zu bisheriger Version — unverändert]

def _get_divorce_prob(alter_a, female_a, bildung_a, is_unemployed_a,
                       marriage_duration) -> float:
    if   marriage_duration <  5: dur_label = "0–4 Jahre"
    elif marriage_duration < 10: dur_label = "5–9 Jahre"
    elif marriage_duration < 15: dur_label = "10–14 Jahre"
    elif marriage_duration < 20: dur_label = "15–19 Jahre"
    else:                        dur_label = "20+ Jahre"
    p_basis = P_DIVORCE_BY_DURATION[dur_label]
    age_c   = alter_a - AGE_MEAN_DIV
    xb = (PARAMS_DIV["Intercept"]
          + PARAMS_DIV["age_c"]      * age_c
          + PARAMS_DIV["age_c_sq"]   * age_c**2
          + PARAMS_DIV["female"]     * female_a
          + PARAMS_DIV["edu_basic"]  * int(bildung_a == 1)
          + PARAMS_DIV["unemployed"] * int(is_unemployed_a))
    p_ind  = 1 / (1 + np.exp(-xb))
    faktor = p_ind / P_MEAN_LOGIT_DIV
    return min(p_basis * faktor, 1.0)


def _divorce_step_crn(alter_a, female_a, bildung_a, is_unemployed_a,
                       marriage_duration, vermoegen, eigengut_a, eigengut_b,
                       pk_a, pk_b, pk_bei_heirat_a, pk_bei_heirat_b,
                       draws) -> dict:
    p = _get_divorce_prob(alter_a, female_a, bildung_a,
                           is_unemployed_a, marriage_duration)
    if draws[0] >= p:
        return {"geschieden": False}
    errungenschaft = max(0.0, vermoegen - eigengut_a - eigengut_b)
    vermoegen_nach = eigengut_a + errungenschaft / 2
    zuwachs_a = max(0.0, pk_a - pk_bei_heirat_a)
    zuwachs_b = max(0.0, pk_b - pk_bei_heirat_b)
    pk_a_neu  = pk_a - zuwachs_a / 2 + zuwachs_b / 2
    pk_b_neu  = pk_b - zuwachs_b / 2 + zuwachs_a / 2
    return {
        "geschieden":       True,
        "vermoegen_nach":   vermoegen_nach,
        "eigengut_a_neu":   eigengut_a,
        "eigengut_b_neu":   eigengut_b,
        "pk_a_neu":         pk_a_neu,
        "pk_b_neu":         pk_b_neu,
        "errungenschaft":   errungenschaft,
        "vermögensverlust": vermoegen - vermoegen_nach,
    }


def _pflegefall_step_crn(alter:                int,
                          female:               int,
                          geburtsjahr:          int,
                          care_state:           int,
                          care_years:           int,
                          care_duration:        int,
                          rente_ahv:            float,
                          rente_pk:             float,
                          wealth:               float,
                          gemeinde:             str,
                          is_married:           bool,
                          liegenschaft:         float = 0.0,
                          hypothek:             float = 0.0,
                          heim_kosten_faktor:   float = 1.0,
                          spitex_kosten_faktor: float = 1.0,
                          draws:                np.ndarray = None) -> dict:

   
    ps = {
        "care_state":              care_state,
        "care_years":              care_years,
        "care_duration":           care_duration,
        "gestorben":               False,
        "heim_neu":                False,
        "spitex_neu":              False,
        "pflegekosten_netto_jahr": 0.0,
        "el_leistung_jahr":        0.0,
        "el_details":              None,
    }

    # ── HEIM — laufende Episode ───────────────────────────────────────────
    if care_state == HEIM_STATE:
        ps["care_years"] = care_years + 1

        wealth_el      = wealth + max(0.0, liegenschaft - hypothek)
        kosten_aktuell = get_kosten_heim_jahr(gemeinde) * heim_kosten_faktor

        el_r  = berechne_el_anspruch(rente_ahv, rente_pk, wealth_el,
                                      gemeinde, is_married, im_heim=True,
                                      kosten_override=kosten_aktuell)
        netto = get_heim_netto_kosten(rente_ahv, rente_pk, wealth_el,
                                       gemeinde, is_married,
                                       kosten_override=kosten_aktuell)
        ps["pflegekosten_netto_jahr"] = netto
        ps["el_leistung_jahr"]        = el_r["el_anspruch"]
        ps["el_details"]              = el_r

        if ps["care_years"] >= care_duration:
            if draws[3] < get_heim_params(alter)["p_tod"]:
                ps["gestorben"] = True
            else:
                ps["care_state"]    = ZUHAUSE
                ps["care_years"]    = 0
                ps["care_duration"] = 0
        return ps

    # ── SPITEX — laufende Episode ─────────────────────────────────────────
    if care_state == SPITEX_STATE:
        ps["care_years"] = care_years + 1

        spitex_brutto = get_spitex_kosten_monatlich(alter) * 12 * spitex_kosten_faktor
        el_r          = berechne_el_anspruch(rente_ahv, rente_pk, wealth,
                                              gemeinde, is_married, im_heim=False,
                                              kosten_override=spitex_brutto)
        ps["pflegekosten_netto_jahr"] = max(0.0, spitex_brutto - el_r["el_anspruch"])
        ps["el_leistung_jahr"]        = el_r["el_anspruch"]

        if ps["care_years"] >= care_duration:
            # Übergang Spitex → Heim (P_SPITEX_HEIM = 15%)
            # Herleitung: OBSAN (2025) Quasi-Gleichgewicht + SHP-Proxy
            if draws[3] < P_SPITEX_HEIM:
                hp         = get_heim_params(alter)
                dauer_heim = max(1, round(
                    -hp["D"] * np.log(np.clip(1 - draws[2], 1e-9, 1.0))
                ))
                wealth_el      = wealth + max(0.0, liegenschaft - hypothek)
                kosten_aktuell = get_kosten_heim_jahr(gemeinde) * heim_kosten_faktor
                el_r  = berechne_el_anspruch(rente_ahv, rente_pk, wealth_el,
                                              gemeinde, is_married, im_heim=True,
                                              kosten_override=kosten_aktuell)
                netto = get_heim_netto_kosten(rente_ahv, rente_pk, wealth_el,
                                               gemeinde, is_married,
                                               kosten_override=kosten_aktuell)
                ps["care_state"]             = HEIM_STATE
                ps["care_years"]             = 1
                ps["care_duration"]          = dauer_heim
                ps["heim_neu"]               = True
                ps["pflegekosten_netto_jahr"] = netto
                ps["el_leistung_jahr"]        = el_r["el_anspruch"]
                ps["el_details"]             = el_r
            else:
                ps["care_state"]    = ZUHAUSE
                ps["care_years"]    = 0
                ps["care_duration"] = 0
        return ps

    # ── ZUHAUSE — Eintritt prüfen ─────────────────────────────────────────
    if alter >= 50:
        p_heim   = get_lambda_heim(alter, female)
        p_spitex = get_lambda_spitex(alter, female)
        p_pflege = min(1.0, p_heim + p_spitex)

        if draws[0] < p_pflege:
            p_heim_cond = p_heim / p_pflege

            if draws[1] < p_heim_cond:
                # ── Heim-Eintritt: Dauer sofort vorziehen ─────────────────
                hp         = get_heim_params(alter)
                dauer_heim = max(1, round(
                    -hp["D"] * np.log(np.clip(1 - draws[2], 1e-9, 1.0))
                ))

                wealth_el      = wealth + max(0.0, liegenschaft - hypothek)
                kosten_aktuell = get_kosten_heim_jahr(gemeinde) * heim_kosten_faktor
                el_r  = berechne_el_anspruch(rente_ahv, rente_pk, wealth_el,
                                              gemeinde, is_married, im_heim=True,
                                              kosten_override=kosten_aktuell)
                netto = get_heim_netto_kosten(rente_ahv, rente_pk, wealth_el,
                                               gemeinde, is_married,
                                               kosten_override=kosten_aktuell)

                ps["care_state"]             = HEIM_STATE
                ps["care_years"]             = 1
                ps["care_duration"]          = dauer_heim
                ps["heim_neu"]               = True
                ps["pflegekosten_netto_jahr"] = netto
                ps["el_leistung_jahr"]        = el_r["el_anspruch"]
                ps["el_details"]              = el_r

                if ps["care_years"] >= dauer_heim:
                    if draws[3] < hp["p_tod"]:
                        ps["gestorben"]     = True
                    else:
                        ps["care_state"]    = ZUHAUSE
                        ps["care_years"]    = 0
                        ps["care_duration"] = 0

            else:
                # ── Spitex-Eintritt: Dauer sofort vorziehen ───────────────
                z_normal     = _scipy_norm.ppf(np.clip(draws[4], 1e-6, 1 - 1e-6))
                dauer_spitex = max(1, round(
                    np.exp(z_normal * D_SPITEX_SIG + D_SPITEX_MU) - 0.5
                ))

                spitex_brutto = (get_spitex_kosten_monatlich(alter) * 12
                                 * spitex_kosten_faktor)
                el_r          = berechne_el_anspruch(rente_ahv, rente_pk, wealth,
                                                      gemeinde, is_married,
                                                      im_heim=False,
                                                      kosten_override=spitex_brutto)
                ps["care_state"]             = SPITEX_STATE
                ps["care_years"]             = 1
                ps["care_duration"]          = dauer_spitex
                ps["spitex_neu"]             = True
                ps["pflegekosten_netto_jahr"] = max(0.0,
                                                    spitex_brutto - el_r["el_anspruch"])
                ps["el_leistung_jahr"]        = el_r["el_anspruch"]

                if ps["care_years"] >= dauer_spitex:
                    ps["care_state"]    = ZUHAUSE
                    ps["care_years"]    = 0
                    ps["care_duration"] = 0

    return ps



print("\n✓ Cell 7: Risikomodule geladen")


# In[9]:


# ═══════════════════════════════════════════════════════════════════════════
# CELL 8 — SIMULATIONS-LOOP
# Läuft alle 8 Szenarien auf identischen Basispfaden (CRN).
#
# Fix 1: s["szenario"] = name wird nach State-Init gesetzt.
#         Wird in jahresschritt() für PK-Bezugslogik benötigt.
# Fix 2: Vermögen im Todesjahr von Person A wird vor break gespeichert.
#         Verhindert NaN in median_verm_tod und Verzerrung der Pfade.
# ═══════════════════════════════════════════════════════════════════════════

from typing import Dict
from joblib import Parallel, delayed
import time


def simuliere_szenario(name: str,
                        flags: dict,
                        haushalt: HaushaltParams,
                        n_sim: int = N_SIM) -> dict:
    alter_start = haushalt.alter_a()
    n_jahre     = SIM_BIS_ALTER - alter_start

    verm_pfade  = np.full((n_sim, n_jahre + 1), np.nan)
    netto_pfade = np.full((n_sim, n_jahre + 1), np.nan)
    tod_alter_a = np.full(n_sim, np.nan)
    tod_alter_b = np.full(n_sim, np.nan)
    ruin_alter  = np.full(n_sim, np.nan)
    geschieden  = np.zeros(n_sim, dtype=bool)
    al_jahre_a  = np.zeros(n_sim)
    al_jahre_b  = np.zeros(n_sim)

    for sim in range(n_sim):
        sim_seed = RANDOM_SEED + sim
        basis    = generiere_basis_schocks(n_jahre, sim_seed, haushalt)
        risiken  = generiere_risiko_schocks(n_jahre, sim_seed)
        s        = initialisiere_haushalt_state(haushalt, basis)
        s["szenario"] = name    # Fix 1: Szenario-Name im State setzen

        verm_pfade[sim, 0]  = s["vermoegen"]
        netto_pfade[sim, 0] = s.get("nettovermoegen", s["vermoegen"])

        for t in range(n_jahre):
            s = jahresschritt(
                s              = s,
                haushalt       = haushalt,
                risiko_schocks = risiken,
                do_al          = flags["al"],
                do_divorce     = flags["divorce"],
                do_pflege      = flags["pflege"],
            )

            # Tod Person A — Vermögen speichern, dann Simulation beenden
            if not s["alive_a"]:
                tod_alter_a[sim]        = s["alter_a"] - 1
                # Fix 2: Todesjahr-Vermögen sichern (nicht NaN lassen)
                verm_pfade[sim, t + 1]  = s["vermoegen"]
                netto_pfade[sim, t + 1] = s.get("nettovermoegen", s["vermoegen"])
                break

            # Ruin — Flag setzen, weiter simulieren
            if s["vermoegen"] <= 0 and np.isnan(ruin_alter[sim]):
                ruin_alter[sim] = s["alter_a"]

            verm_pfade[sim, t + 1]  = s["vermoegen"]
            netto_pfade[sim, t + 1] = s.get("nettovermoegen", s["vermoegen"])

            # Tod Person B
            if not s.get("alive_b", False) and np.isnan(tod_alter_b[sim]):
                tod_alter_b[sim] = s.get("alter_b", np.nan) - 1

            # Ereignis-Tracking
            if s.get("geschieden") and not geschieden[sim]:
                geschieden[sim] = True
            if s.get("is_unemployed_a"):
                al_jahre_a[sim] += 1
            if s.get("is_unemployed_b"):
                al_jahre_b[sim] += 1

    return {
        "name":        name,
        "flags":       flags,
        "verm_pfade":  verm_pfade,
        "netto_pfade": netto_pfade,
        "tod_alter_a": tod_alter_a,
        "tod_alter_b": tod_alter_b,
        "ruin_alter":  ruin_alter,
        "geschieden":  geschieden,
        "al_jahre_a":  al_jahre_a,
        "al_jahre_b":  al_jahre_b,
        "n_sim":       n_sim,
        "alter_start": alter_start,
        "n_jahre":     n_jahre,
    }


def berechne_kennzahlen(res: dict) -> dict:
    """
    Erweiterte Kennzahlen mit Competing-Risk-Logik.

    Competing Risks: Tod und Ruin konkurrieren.
    Unbedingte Ruinrate unterschätzt ökonomisches Risiko wenn
    Tod als konkurrierendes Ereignis früher eintritt.
    """
    vp          = res["verm_pfade"]
    n_sim       = res["n_sim"]
    n_jahre     = res["n_jahre"]
    alter_start = res["alter_start"]
    tod_a       = res["tod_alter_a"]
    ruin_a      = res["ruin_alter"]

    def idx(alter):
        return min(alter - alter_start, n_jahre)

    # ── Masken ────────────────────────────────────────────────────────────
    alive_85 = np.isnan(tod_a) | (tod_a > 85)
    alive_65 = np.isnan(tod_a) | (tod_a > 65)
    alive_75 = np.isnan(tod_a) | (tod_a > 75)
    ruiniert  = ~np.isnan(ruin_a)
    tot_85    = ~alive_85

    # ── Competing Risks ───────────────────────────────────────────────────
    p_ruin_unbeding_85 = (ruiniert & alive_85).mean()
    p_ruin_bedingt_85  = (ruiniert[alive_85].mean()
                          if alive_85.sum() > 0 else np.nan)
    p_tod_bis_85       = tot_85.mean()
    p_ruin_oder_tod_85 = (ruiniert | tot_85).mean()

    ruin_vor_tod = np.zeros(n_sim, dtype=bool)
    for i in range(n_sim):
        if not np.isnan(ruin_a[i]):
            if np.isnan(tod_a[i]) or ruin_a[i] < tod_a[i]:
                ruin_vor_tod[i] = True
    p_ruin_vor_tod = ruin_vor_tod.mean()

    # ── Medianes Ruinalter ────────────────────────────────────────────────
    ruin_alter_valid  = ruin_a[~np.isnan(ruin_a)]
    median_ruin_alter = (float(np.median(ruin_alter_valid))
                         if len(ruin_alter_valid) > 0 else np.nan)

    # ── Medianes Vermögen bei Tod ─────────────────────────────────────────
    # Fix 2 wirkt hier: Todesjahr-Vermögen ist nicht mehr NaN
    verm_bei_tod = []
    for i in range(n_sim):
        if not np.isnan(tod_a[i]):
            t_idx = int(tod_a[i]) - alter_start
            if 0 <= t_idx < vp.shape[1]:
                v = vp[i, t_idx]
                if not np.isnan(v):
                    verm_bei_tod.append(v)
    median_verm_tod = (float(np.median(verm_bei_tod))
                       if verm_bei_tod else np.nan)

    # ── Vermögen bei Schlüsselaltern (nur Lebende) ────────────────────────
    p50_65 = (np.nanpercentile(vp[alive_65, idx(65)], 50)
              if alive_65.any() else np.nan)
    p25_65 = (np.nanpercentile(vp[alive_65, idx(65)], 25)
              if alive_65.any() else np.nan)
    p75_65 = (np.nanpercentile(vp[alive_65, idx(65)], 75)
              if alive_65.any() else np.nan)
    p50_75 = (np.nanpercentile(vp[alive_75, idx(75)], 50)
              if alive_75.any() else np.nan)
    p50_85_lebend = (np.nanpercentile(vp[alive_85, idx(85)], 50)
                     if alive_85.any() else np.nan)
    p25_85_lebend = (np.nanpercentile(vp[alive_85, idx(85)], 25)
                     if alive_85.any() else np.nan)
    p75_85_lebend = (np.nanpercentile(vp[alive_85, idx(85)], 75)
                     if alive_85.any() else np.nan)

    tod_valid = tod_a[~np.isnan(tod_a)]

    # ── Netto-Position ────────────────────────────────────────────────────
    median_netto_85     = (float(np.nanmedian(vp[alive_85, idx(85)]))
                           if alive_85.any() else np.nan)
    p_vorsorgeluecke_85 = (float(np.mean(vp[alive_85, idx(85)] < 0))
                           if alive_85.any() else np.nan)

    return {
        "p50_65":              p50_65,
        "p25_65":              p25_65,
        "p75_65":              p75_65,
        "p50_75":              p50_75,
        "p50_85":              p50_85_lebend,
        "p25_85":              p25_85_lebend,
        "p75_85":              p75_85_lebend,
        "median_netto_85":     median_netto_85,
        "p_vorsorgeluecke_85": p_vorsorgeluecke_85,
        "p_ruin_unbeding_85":  p_ruin_unbeding_85,
        "p_ruin_bedingt_85":   p_ruin_bedingt_85,
        "p_tod_bis_85":        p_tod_bis_85,
        "p_ruin_oder_tod_85":  p_ruin_oder_tod_85,
        "p_ruin_vor_tod":      p_ruin_vor_tod,
        "median_ruin_alter":   median_ruin_alter,
        "median_tod_alter":    (float(np.median(tod_valid))
                                if len(tod_valid) > 0 else np.nan),
        "median_verm_tod":     median_verm_tod,
        "p_geschieden":        (res["geschieden"].mean()
                                if res["flags"]["divorce"] else np.nan),
        "mean_al_jahre_a":     (res["al_jahre_a"].mean()
                                if res["flags"]["al"] else np.nan),
        "p_ruin_85":           p_ruin_unbeding_85,
        "p_tod_vor_85":        p_tod_bis_85,
    }


def simuliere_alle_szenarien_parallel(haushalt: HaushaltParams,
                                       n_sim: int = N_SIM) -> dict:
    """Alle 8 Szenarien parallel — nutzt alle CPU-Kerne."""
    def _run(name, flags):
        res = simuliere_szenario(name, flags, haushalt, n_sim=n_sim)
        kz  = berechne_kennzahlen(res)
        return name, {**res, "kennzahlen": kz}

    try:
        ergebnisse = Parallel(n_jobs=-1, backend="loky", verbose=1)(
            delayed(_run)(name, flags)
            for name, flags in SZENARIEN.items()
        )
        return dict(ergebnisse)
    except Exception as e:
        print(f"  ⚠️  Parallel fehlgeschlagen ({e}) — sequenziell")
        resultate = {}
        for name, flags in SZENARIEN.items():
            res = simuliere_szenario(name, flags, haushalt, n_sim=n_sim)
            kz  = berechne_kennzahlen(res)
            resultate[name] = {**res, "kennzahlen": kz}
        return resultate


# ═══════════════════════════════════════════════════════════════════════════
# ALLE SZENARIEN SIMULIEREN
# ═══════════════════════════════════════════════════════════════════════════

