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
        map_names = ["ceegrid", "eareeg", "idun", "muse", "emotiv",
                     "openbci", "psg", "headband", "dreem"]

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
