"""Dataset-dependent channel mappings for LaBraM inference.

This module provides channel name mappings from various EEG systems to standard
10-20 electrode positions. Mappings are organized by dataset/device to allow
duplicate channel names with different meanings across datasets.

Usage:
    from channel_maps import get_channel_map, list_channel_maps

    # Get merged map for specific datasets
    channel_map = get_channel_map(["default", "ds004460"])

    # List available maps
    print(list_channel_maps())
"""

from typing import Dict, List, Optional


def _build_tuh_eeg_prefix_map() -> Dict[str, str]:
    """Build mapping for TUH-style channel names.

    TUH EDFs commonly use labels like "EEG FP1-REF". The inference pipeline
    strips reference suffixes and hyphenated parts, so these become "EEG FP1".
    This map converts those to standard 10-20 labels.
    """

    # Common TUH referential labels (post-stripping they look like "EEG <CH>").
    base = {
        # Core 10-20
        "EEG FP1": "FP1",
        "EEG FP2": "FP2",
        "EEG F3": "F3",
        "EEG F4": "F4",
        "EEG C3": "C3",
        "EEG C4": "C4",
        "EEG P3": "P3",
        "EEG P4": "P4",
        "EEG O1": "O1",
        "EEG O2": "O2",
        "EEG F7": "F7",
        "EEG F8": "F8",

        # Midline
        "EEG FZ": "FZ",
        "EEG CZ": "CZ",
        "EEG PZ": "PZ",

        # References / auxiliaries often present in TUH
        "EEG A1": "A1",
        "EEG A2": "A2",
        "EEG T1": "T1",
        "EEG T2": "T2",

        # Legacy temporal names seen in TUH: map to modern 10-20
        "EEG T3": "T7",
        "EEG T4": "T8",
        "EEG T5": "P7",
        "EEG T6": "P8",
    }

    # Also accept variants that show up after upstream normalization.
    # (Keeping this conservative; add more as needed.)
    base.update(
        {
            "EEG FPZ": "FPZ",
            "EEG OZ": "OZ",
        }
    )

    return base

# =============================================================================
# Standard 10-20 electrode positions (used for position embeddings in LaBraM)
# These positions correspond to the position embedding indices in the model.
# Index 0 is reserved for the CLS token, so electrode indices start at 1.
# =============================================================================
STANDARD_1020: List[str] = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


# =============================================================================
# Channel Maps by Dataset/Device
# =============================================================================

CHANNEL_MAPS: Dict[str, Dict[str, str]] = {
    # =========================================================================
    # Default mappings (common aliases that don't conflict)
    # =========================================================================
    "default": {
        # Legacy 10-20 naming
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',

        # Common reference electrode aliases
        'MASTL': 'M1',
        'MASTR': 'M2',
        'LMAS': 'M1',
        'RMAS': 'M2',
        'LM': 'M1',
        'RM': 'M2',
        'EARR': 'A2',
        'EARL': 'A1',
    },

    # =========================================================================
    # TUH EEG datasets (e.g., TUAB, TUEV)
    # Channels commonly appear as "EEG FP1-REF" etc.
    # =========================================================================
    "tuab": _build_tuh_eeg_prefix_map(),
    "tuev": _build_tuh_eeg_prefix_map(),

    # =========================================================================
    # Shock dataset (CNT/EDF via shock utils)
    # Usually already uses standard channel names after preprocessing.
    # =========================================================================
    "shock": {},

    # =========================================================================
    # Generic BIDS conversions
    # MNE-BIDS typically yields standard names; keep map for explicit opt-in.
    # =========================================================================
    "bids": {},

    # =========================================================================
    # cEEGrid: Around-the-ear electrode array
    # Reference: Debener et al. "Unobtrusive ambulatory EEG using a smartphone"
    # =========================================================================
    "ceegrid": {
        'L1': 'FT9',   # Left anterior (in front of ear)
        'L2': 'T9',    # Left upper anterior
        'L3': 'T7',    # Left at ear level
        'L4': 'TP9',   # Left upper posterior
        'L4A': 'TP9',
        'L4B': 'TP9',
        'L5': 'TP9',   # Left mid posterior
        'L6': 'M1',    # Left lower posterior
        'L7': 'M1',
        'L8': 'M1',
        'R1': 'FT10',  # Right anterior
        'R2': 'T10',   # Right upper anterior
        'R3': 'T8',    # Right at ear level
        'R4': 'TP10',  # Right upper posterior
        'R4A': 'TP10',
        'R4B': 'TP10',
        'R5': 'TP10',  # Right mid posterior
        'R6': 'M2',    # Right lower posterior
        'R7': 'M2',
        'R8': 'M2',
    },

    # =========================================================================
    # earEEG / In-ear EEG
    # =========================================================================
    "eareeg": {
        'LB': 'T9',    # Left Bottom
        'LT': 'T9',    # Left Top
        'RB': 'T10',   # Right Bottom
        'RT': 'T10',   # Right Top
        'ELE': 'T9',   # Generic ear electrode
        'LE': 'T9',
        'RE': 'T10',
        'LEA': 'T9',
        'LEB': 'T9',
        'REA': 'T10',
        'REB': 'T10',
    },

    # =========================================================================
    # IDUN Guardian earbuds
    # =========================================================================
    "idun": {
        'L_A': 'T9',
        'L_B': 'T9',
        'L_C': 'T9',
        'R_A': 'T10',
        'R_B': 'T10',
        'R_C': 'T10',
    },

    # =========================================================================
    # Muse headband
    # =========================================================================
    "muse": {
        'TP9_MUSE': 'TP9',
        'AF7_MUSE': 'AF7',
        'AF8_MUSE': 'AF8',
        'TP10_MUSE': 'TP10',
        # Also support raw Muse channel names
        'TP9': 'TP9',
        'AF7': 'AF7',
        'AF8': 'AF8',
        'TP10': 'TP10',
    },

    # =========================================================================
    # Emotiv EPOC/EPOC+
    # =========================================================================
    "emotiv": {
        'AF3_EMOTIV': 'AF3',
        'F7_EMOTIV': 'F7',
        'F3_EMOTIV': 'F3',
        'FC5_EMOTIV': 'FC5',
        'T7_EMOTIV': 'T7',
        'P7_EMOTIV': 'P7',
        'O1_EMOTIV': 'O1',
        'O2_EMOTIV': 'O2',
        'P8_EMOTIV': 'P8',
        'T8_EMOTIV': 'T8',
        'FC6_EMOTIV': 'FC6',
        'F4_EMOTIV': 'F4',
        'F8_EMOTIV': 'F8',
        'AF4_EMOTIV': 'AF4',
    },

    # =========================================================================
    # OpenBCI Cyton default channels
    # =========================================================================
    "openbci": {
        'EXG1': 'FP1',
        'EXG2': 'FP2',
        'EXG3': 'C3',
        'EXG4': 'C4',
        'EXG5': 'P7',
        'EXG6': 'P8',
        'EXG7': 'O1',
        'EXG8': 'O2',
    },

    # =========================================================================
    # PSG (Polysomnography) with prefix
    # =========================================================================
    "psg": {
        'PSG_F3': 'F3',
        'PSG_F4': 'F4',
        'PSG_C3': 'C3',
        'PSG_C4': 'C4',
        'PSG_O1': 'O1',
        'PSG_O2': 'O2',
        'PSG_FP1': 'FP1',
        'PSG_FP2': 'FP2',
        'PSG_FZ': 'FZ',
        'PSG_CZ': 'CZ',
        'PSG_PZ': 'PZ',
        'PSG_OZ': 'OZ',
        'PSG_T3': 'T7',
        'PSG_T4': 'T8',
        'PSG_T5': 'P7',
        'PSG_T6': 'P8',
    },

    # =========================================================================
    # Generic headband devices
    # =========================================================================
    "headband": {
        'HB_1': 'AF7',
        'HB_2': 'AF8',
        'HB_3': 'FP1',
        'HB_4': 'FP2',
        'HEADBAND_L': 'AF7',
        'HEADBAND_R': 'AF8',
        'FOREHEAD_L': 'AF7',
        'FOREHEAD_R': 'AF8',
        'FRONT_L': 'AF7',
        'FRONT_R': 'AF8',
    },

    # =========================================================================
    # Dreem headband
    # =========================================================================
    "dreem": {
        'F7_DREEM': 'F7',
        'F8_DREEM': 'F8',
        'FPZ_DREEM': 'FPZ',
        'O1_DREEM': 'O1',
        'O2_DREEM': 'O2',
    },

    # =========================================================================
    # ds004460: EASYCAP Equidistant Cap (Spot Rotation dataset)
    # Auto-generated from 3D electrode coordinates
    # =========================================================================
    "ds004460": {
        # G (Green) group - right anterior/lateral
        'G1': 'FP2',
        'G2': 'FP2',
        'G3': 'FP2',
        'G4': 'AF4',
        'G5': 'FP2',
        'G6': 'FP2',
        'G7': 'AF8',
        'G8': 'AF4',
        'G9': 'AF8',
        'G10': 'F8',
        'G11': 'F8',
        'G12': 'FT8',
        'G14': 'T8',
        'G15': 'T8',
        'G16': 'FT10',
        'G17': 'T8',
        'G18': 'T10',
        'G19': 'TP8',
        'G20': 'TP8',
        'G21': 'P8',
        'G22': 'P8',
        'G23': 'PO8',
        'G24': 'P6',
        'G25': 'PO8',
        'G26': 'PO8',
        'G27': 'M2',
        'G28': 'PO4',
        'G29': 'PO4',
        'G30': 'O2',
        'G31': 'O2',
        'G32': 'M2',
        # R (Red) group - midline/central
        'R1': 'AFZ',
        'R2': 'AFZ',
        'R3': 'AFZ',
        'R4': 'FZ',
        'R5': 'F3',
        'R6': 'FZ',
        'R7': 'FZ',
        'R8': 'FZ',
        'R9': 'FC1',
        'R10': 'F4',
        'R11': 'FZ',
        'R12': 'FCZ',
        'R13': 'CZ',
        'R14': 'FC2',
        'R15': 'FCZ',
        'R16': 'CZ',
        'R17': 'CZ',
        'R18': 'CP2',
        'R19': 'CZ',
        'R20': 'CPZ',
        'R21': 'P2',
        'R22': 'P2',
        'R23': 'C1',
        'R24': 'C1',
        'R25': 'CP1',
        'R26': 'P1',
        'R27': 'PZ',
        'R28': 'CP3',
        'R29': 'P1',
        'R30': 'P1',
        'R31': 'POZ',
        'R32': 'POZ',
        # W (White) group - left hemisphere
        'W1': 'FP1',
        'W2': 'AF7',
        'W3': 'FC5',
        'W4': 'FP1',
        'W5': 'AF3',
        'W6': 'AF3',
        'W7': 'FC5',
        'W8': 'FC5',
        'W9': 'FC5',
        'W10': 'AF3',
        'W11': 'F3',
        'W12': 'F3',
        'W13': 'FC3',
        'W14': 'FC3',
        'W15': 'C3',
        'W16': 'C3',
        'W17': 'C3',
        'W18': 'C3',
        'W19': 'CP3',
        'W20': 'P3',
        'W21': 'PO3',
        'W22': 'C5',
        'W23': 'C5',
        'W24': 'CP5',
        'W25': 'P5',
        'W26': 'P5',
        'W27': 'PO3',
        'W28': 'C5',
        'W29': 'CP5',
        'W30': 'P7',
        'W31': 'P7',
        'W32': 'PO7',
        # Y (Yellow) group - right posterior/central
        'Y1': 'AF4',
        'Y2': 'F4',
        'Y5': 'AF4',
        'Y6': 'F4',
        'Y7': 'FC4',
        'Y19': 'C4',
        'Y20': 'C4',
        'Y22': 'CP6',
        'Y23': 'P6',
        'Y24': 'C4',
        'Y25': 'CP4',
        'Y26': 'P4',
        'Y27': 'P4',
        'Y28': 'C2',
        'Y29': 'C2',
        'Y30': 'CP2',
        'Y31': 'P4',
        'Y32': 'P2',
    },
}


# =============================================================================
# Dataset aliases / per-dataset entries
#
# These keys are convenience aliases so users can select a map by dataset
# identifier (e.g., OpenNeuro accession). Some datasets already use standard
# 10-20 names and therefore do not require any additional alias mapping.
# =============================================================================

# Datasets observed under D:/neuro_datasets (BIDS roots).
# - If mapping isn't needed, keep an empty dict (still allowing explicit opt-in).
# - If a dataset uses a known non-standard naming scheme, alias it to a device map.

CHANNEL_MAPS.setdefault("ds003969", {})   # Standard 10-20 names (case-insensitive)
CHANNEL_MAPS.setdefault("ds004279", {})   # Standard 10-20 names

# ds004408 appears to use non-10-20 channel names (e.g., A1..A19). No built-in map.
CHANNEL_MAPS.setdefault("ds004408", {})

# ds004460 is already defined above (equidistant cap). Add a common folder-name alias.
CHANNEL_MAPS.setdefault("ds004460-download", CHANNEL_MAPS["ds004460"])

# Ear-EEG sleep monitoring dataset uses LB/LT/RB/RT.
CHANNEL_MAPS.setdefault("ds005178", CHANNEL_MAPS["eareeg"])

# Surrey cEEGrid sleep dataset uses L1..L8 / R1..R8.
CHANNEL_MAPS.setdefault("ds005207", CHANNEL_MAPS["ceegrid"])

# HBN EEG appears to use E1..En channel names (likely EGI/NetStation). No built-in map.
CHANNEL_MAPS.setdefault("ds005514", {})

# BOAS headband dataset uses HB_1/HB_2.
CHANNEL_MAPS.setdefault("ds005555", CHANNEL_MAPS["headband"])

CHANNEL_MAPS.setdefault("ds006104", {})   # Standard 10-20 names (mixed case)

# =============================================================================
# High-density cap systems
#
# Some BIDS datasets use manufacturer channel labels (e.g., BioSemi A1..D32,
# EGI E1..E128). These overlap with a few standard labels (notably A1/A2), so
# they should be selected explicitly (or via dataset aliases below).
#
# These maps were generated as a best-effort nearest-neighbor assignment from
# the vendor montage to standard_1005, restricted to LaBraM's STANDARD_1020.
# =============================================================================

CHANNEL_MAPS.setdefault(
    "biosemi128",
    {
        'A1': 'CZ',
        'A10': 'P7',
        'A11': 'P7',
        'A12': 'P9',
        'A13': 'PO9',
        'A14': 'PO9',
        'A15': 'O1',
        'A16': 'PO5',
        'A17': 'PO3',
        'A18': 'P3',
        'A19': 'PZ',
        'A2': 'CZ',
        'A20': 'POZ',
        'A21': 'POZ',
        'A22': 'OZ',
        'A23': 'OZ',
        'A24': 'IZ',
        'A25': 'IZ',
        'A26': 'PO10',
        'A27': 'PO10',
        'A28': 'O2',
        'A29': 'PO6',
        'A3': 'CPZ',
        'A30': 'PO4',
        'A31': 'P4',
        'A32': 'P2',
        'A4': 'CPZ',
        'A5': 'P1',
        'A6': 'CP3',
        'A7': 'CP3',
        'A8': 'P5',
        'A9': 'P5',
        'B1': 'CZ',
        'B10': 'TPP10H',
        'B11': 'TP8',
        'B12': 'TPP8H',
        'B13': 'CP6',
        'B14': 'T8',
        'B15': 'CCP6',
        'B16': 'CCP6',
        'B17': 'CCP4',
        'B18': 'CCP4',
        'B19': 'CCP2',
        'B2': 'CCP2',
        'B20': 'C2',
        'B21': 'C2',
        'B22': 'C4',
        'B23': 'C4',
        'B24': 'C6',
        'B25': 'C6',
        'B26': 'T8',
        'B27': 'FT8',
        'B28': 'FC6',
        'B29': 'FC6',
        'B3': 'CP4',
        'B30': 'FC4',
        'B31': 'FC4',
        'B32': 'FC2',
        'B4': 'CP4',
        'B5': 'P6',
        'B6': 'P6',
        'B7': 'P8',
        'B8': 'P8',
        'B9': 'P10',
        'C1': 'FCZ',
        'C10': 'F4',
        'C11': 'FC2',
        'C12': 'F2',
        'C13': 'F2',
        'C14': 'AF2',
        'C15': 'AF4',
        'C16': 'FP2',
        'C17': 'FPZ',
        'C18': 'FPZ',
        'C19': 'AFZ',
        'C2': 'FC2',
        'C20': 'AFZ',
        'C21': 'FZ',
        'C22': 'FZ',
        'C23': 'FCZ',
        'C24': 'FC1',
        'C25': 'F1',
        'C26': 'F1',
        'C27': 'AF1',
        'C28': 'AF3',
        'C29': 'FP1',
        'C3': 'F2',
        'C30': 'AF7',
        'C31': 'AF5',
        'C32': 'F3',
        'C4': 'F4',
        'C5': 'F6',
        'C6': 'F6',
        'C7': 'F8',
        'C8': 'AF8',
        'C9': 'AF6',
        'D1': 'FCZ',
        'D10': 'FC5',
        'D11': 'FC3',
        'D12': 'FC3',
        'D13': 'FC1',
        'D14': 'C1',
        'D15': 'CZ',
        'D16': 'CCP1',
        'D17': 'CCP1',
        'D18': 'C3',
        'D19': 'C3',
        'D2': 'FC1',
        'D20': 'C3',
        'D21': 'C5',
        'D22': 'C5',
        'D23': 'T7',
        'D24': 'TTP7H',
        'D25': 'TTP7H',
        'D26': 'CCP5',
        'D27': 'CCP3',
        'D28': 'CCP3',
        'D29': 'CP5',
        'D3': 'F1',
        'D30': 'CP5',
        'D31': 'TP7',
        'D32': 'TPP9H',
        'D4': 'F3',
        'D5': 'F5',
        'D6': 'F5',
        'D7': 'F7',
        'D8': 'FT7',
        'D9': 'FC5',
    },
)

CHANNEL_MAPS.setdefault(
    "hydrocel128",
    {
        'E1': 'AF8',
        'E2': 'AF6',
        'E3': 'AF4',
        'E4': 'F2',
        'E5': 'FZ',
        'E6': 'FCZ',
        'E7': 'FCZ',
        'E8': 'FP2',
        'E9': 'FP2',
        'E10': 'AF2',
        'E11': 'AFZ',
        'E12': 'FZ',
        'E13': 'FC1',
        'E14': 'FPZ',
        'E15': 'FPZ',
        'E16': 'AFZ',
        'E17': 'FPZ',
        'E18': 'AF1',
        'E19': 'F1',
        'E20': 'F1',
        'E21': 'FPZ',
        'E22': 'AF3',
        'E23': 'AF3',
        'E24': 'F3',
        'E25': 'FP1',
        'E26': 'AF5',
        'E27': 'F3',
        'E28': 'F3',
        'E29': 'FC3',
        'E30': 'FC1',
        'E31': 'C1',
        'E32': 'AF7',
        'E33': 'F7',
        'E34': 'FC5',
        'E35': 'FC3',
        'E36': 'C3',
        'E37': 'C1',
        'E38': 'F9',
        'E39': 'FT7',
        'E40': 'C5',
        'E41': 'C3',
        'E42': 'CCP3',
        'E43': 'F9',
        'E44': 'FT9',
        'E45': 'T7',
        'E46': 'CCP5',
        'E47': 'CCP5',
        'E48': 'F9',
        'E49': 'FT9',
        'E50': 'TP7',
        'E51': 'CP5',
        'E52': 'CP3',
        'E53': 'CP1',
        'E54': 'CCP1',
        'E55': 'CZ',
        'E56': 'A1',
        'E57': 'TP9',
        'E58': 'P7',
        'E59': 'P5',
        'E60': 'P3',
        'E61': 'P1',
        'E62': 'PZ',
        'E63': 'M1',
        'E64': 'P9',
        'E65': 'PO7',
        'E66': 'PO3',
        'E67': 'P1',
        'E68': 'P9',
        'E69': 'PO9',
        'E70': 'O1',
        'E71': 'PO1',
        'E72': 'POZ',
        'E73': 'PO9',
        'E74': 'IZ',
        'E75': 'OZ',
        'E76': 'PO2',
        'E77': 'P2',
        'E78': 'P2',
        'E79': 'CCP2',
        'E80': 'C2',
        'E81': 'IZ',
        'E82': 'IZ',
        'E83': 'O2',
        'E84': 'PO4',
        'E85': 'P4',
        'E86': 'CP2',
        'E87': 'C2',
        'E88': 'PO10',
        'E89': 'PO10',
        'E90': 'PO8',
        'E91': 'P6',
        'E92': 'CP4',
        'E93': 'CCP4',
        'E94': 'P10',
        'E95': 'P10',
        'E96': 'P8',
        'E97': 'CP6',
        'E98': 'CCP6',
        'E99': 'M2',
        'E100': 'TP10',
        'E101': 'TP8',
        'E102': 'CCP6',
        'E103': 'C4',
        'E104': 'C4',
        'E105': 'FC2',
        'E106': 'FCZ',
        'E107': 'A2',
        'E108': 'T8',
        'E109': 'C6',
        'E110': 'FC4',
        'E111': 'FC4',
        'E112': 'FC2',
        'E113': 'FT10',
        'E114': 'FT10',
        'E115': 'FT8',
        'E116': 'FC6',
        'E117': 'F4',
        'E118': 'F2',
        'E119': 'F10',
        'E120': 'F10',
        'E121': 'F10',
        'E122': 'F8',
        'E123': 'F4',
        'E124': 'F4',
        'E125': 'F10',
        'E126': 'AF10',
        'E127': 'AF9',
        'E128': 'F9',
    },
)

# Wire dataset ids to the appropriate high-density montage maps.
CHANNEL_MAPS["ds004408"] = CHANNEL_MAPS["biosemi128"]
CHANNEL_MAPS["ds005514"] = CHANNEL_MAPS["hydrocel128"]


def list_channel_maps() -> List[str]:
    """Return list of available channel map names."""
    return list(CHANNEL_MAPS.keys())


def get_channel_map(
    map_names: Optional[List[str]] = None,
    include_default: bool = True
) -> Dict[str, str]:
    """Get merged channel mapping from specified maps.

    Args:
        map_names: List of map names to merge. If None, uses common defaults.
        include_default: Whether to always include the "default" map.

    Returns:
        Merged dictionary mapping channel names to standard 10-20 positions.
        Later maps override earlier ones for duplicate keys.

    Example:
        # Get default + cEEGrid + earEEG mappings
        channel_map = get_channel_map(["ceegrid", "eareeg"])

        # Get only ds004460 mappings (no cEEGrid R* conflict)
        channel_map = get_channel_map(["ds004460"], include_default=True)
    """
    if map_names is None:
        # Default: common consumer devices (no conflicting R* channels)
        map_names = [
            "ceegrid",
            "eareeg",
            "idun",
            "muse",
            "emotiv",
            "openbci",
            "psg",
            "headband",
            "dreem",
            # TUH datasets: safe to include by default because keys are prefixed
            # with "EEG " and won't collide with other maps.
            "tuab",
            "tuev",
        ]

    result: Dict[str, str] = {}

    # Always start with default if requested
    if include_default and "default" not in map_names:
        result.update(CHANNEL_MAPS["default"])

    # Merge requested maps in order (later overrides earlier)
    for name in map_names:
        if name in CHANNEL_MAPS:
            result.update(CHANNEL_MAPS[name])
        else:
            raise ValueError(f"Unknown channel map: {name}. "
                           f"Available: {list_channel_maps()}")

    return result


def get_standard_1020() -> List[str]:
    """Return the standard 10-20 electrode position list."""
    return STANDARD_1020.copy()
