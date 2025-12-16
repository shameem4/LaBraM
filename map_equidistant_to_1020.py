"""Map equidistant EEG electrodes to nearest standard 10-20 positions.

This script reads electrode coordinates from BIDS electrodes.tsv files
and finds the nearest equidistant electrode for each standard 10-20 position.

Usage:
    python map_equidistant_to_1020.py --electrodes path/to/electrodes.tsv
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Standard 10-20 electrode coordinates in MNI space (approximate, mm)
# Based on standard head model, nasion-inion and preauricular points
# Source: Oostenveld & Praamstra (2001), extended 10-20 system
STANDARD_1020_COORDS: Dict[str, Tuple[float, float, float]] = {
    # Midline
    'FPZ': (0, 88, 15),
    'FZ': (0, 60, 95),
    'CZ': (0, 0, 125),
    'PZ': (0, -60, 100),
    'OZ': (0, -100, 20),

    # Frontal pole
    'FP1': (-25, 88, 10),
    'FP2': (25, 88, 10),

    # Frontal
    'F7': (-70, 55, 5),
    'F3': (-45, 55, 80),
    'F4': (45, 55, 80),
    'F8': (70, 55, 5),

    # Anterior frontal
    'AF7': (-55, 75, 10),
    'AF3': (-35, 75, 50),
    'AFZ': (0, 75, 55),
    'AF4': (35, 75, 50),
    'AF8': (55, 75, 10),

    # Fronto-central
    'FC5': (-70, 30, 50),
    'FC3': (-50, 30, 90),
    'FC1': (-25, 30, 105),
    'FCZ': (0, 30, 110),
    'FC2': (25, 30, 105),
    'FC4': (50, 30, 90),
    'FC6': (70, 30, 50),

    # Central
    'T7': (-85, 0, 25),  # T3 in old nomenclature
    'C5': (-75, 0, 60),
    'C3': (-55, 0, 100),
    'C1': (-28, 0, 115),
    'C2': (28, 0, 115),
    'C4': (55, 0, 100),
    'C6': (75, 0, 60),
    'T8': (85, 0, 25),   # T4 in old nomenclature

    # Centro-parietal
    'CP5': (-70, -30, 55),
    'CP3': (-50, -30, 95),
    'CP1': (-25, -30, 110),
    'CPZ': (0, -30, 115),
    'CP2': (25, -30, 110),
    'CP4': (50, -30, 95),
    'CP6': (70, -30, 55),

    # Temporal-parietal
    'TP7': (-80, -35, 20),
    'TP8': (80, -35, 20),
    'TP9': (-85, -45, -10),  # Near mastoid
    'TP10': (85, -45, -10),

    # Parietal
    'P7': (-70, -60, 30),  # T5 in old nomenclature
    'P5': (-60, -60, 60),
    'P3': (-45, -60, 85),
    'P1': (-22, -60, 100),
    'P2': (22, -60, 100),
    'P4': (45, -60, 85),
    'P6': (60, -60, 60),
    'P8': (70, -60, 30),   # T6 in old nomenclature

    # Parieto-occipital
    'PO7': (-55, -80, 25),
    'PO3': (-35, -80, 60),
    'POZ': (0, -85, 65),
    'PO4': (35, -80, 60),
    'PO8': (55, -80, 25),

    # Occipital
    'O1': (-25, -100, 15),
    'O2': (25, -100, 15),

    # Inferior temporal
    'FT7': (-75, 45, 0),
    'FT8': (75, 45, 0),
    'FT9': (-80, 35, -20),
    'FT10': (80, 35, -20),
    'T9': (-85, -10, -15),
    'T10': (85, -10, -15),

    # Mastoids
    'M1': (-80, -55, -30),
    'M2': (80, -55, -30),
}


def load_electrodes_tsv(filepath: Path) -> Dict[str, np.ndarray]:
    """Load electrode coordinates from BIDS electrodes.tsv file."""
    electrodes = {}
    with open(filepath, 'r') as f:
        header = f.readline().strip().split('\t')
        name_idx = header.index('name')
        x_idx = header.index('x')
        y_idx = header.index('y')
        z_idx = header.index('z')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                name = parts[name_idx].upper()
                try:
                    x, y, z = float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])
                    electrodes[name] = np.array([x, y, z])
                except ValueError:
                    continue
    return electrodes


def find_nearest_electrode(
    target_coord: np.ndarray,
    electrodes: Dict[str, np.ndarray],
    exclude_neck: bool = True
) -> Tuple[str, float]:
    """Find the electrode nearest to target coordinates."""
    min_dist = float('inf')
    nearest: str | None = None

    for name, coord in electrodes.items():
        # Skip neck electrodes if requested
        if exclude_neck and name.startswith('N'):
            continue

        dist = float(np.linalg.norm(target_coord - coord))
        if dist < min_dist:
            min_dist = float(dist)
            nearest = name

    if nearest is None:
        raise ValueError("No electrodes available to search (after exclusions)")

    return nearest, float(min_dist)


def find_nearest_1020(eq_coord: np.ndarray) -> Tuple[str, float]:
    """Find the 10-20 position nearest to an equidistant electrode coordinate."""
    min_dist = float('inf')
    nearest: str | None = None

    for std_name, std_coord in STANDARD_1020_COORDS.items():
        std_coord = np.array(std_coord)
        dist = float(np.linalg.norm(eq_coord - std_coord))
        if dist < min_dist:
            min_dist = float(dist)
            nearest = std_name

    if nearest is None:
        raise ValueError("STANDARD_1020_COORDS is empty")

    return nearest, float(min_dist)


def create_mapping(
    electrodes: Dict[str, np.ndarray],
    max_distance: float = 50.0,  # mm
    verbose: bool = True
) -> Dict[str, str]:
    """Create mapping from equidistant electrode to nearest standard 10-20 position.

    Note: The coordinate systems may not be perfectly aligned. This function
    finds the closest 10-20 position for each equidistant electrode based on
    Euclidean distance.

    Returns:
        Dict mapping equidistant electrode names to standard 10-20 position names.
    """
    mapping = {}

    # Find the centroid of the equidistant electrodes (excluding neck)
    scalp_coords = [c for name, c in electrodes.items() if not name.startswith('N')]
    if not scalp_coords:
        print("No scalp electrodes found!")
        return {}

    eq_centroid = np.mean(scalp_coords, axis=0)

    # Find centroid of standard 10-20
    std_centroid = np.mean(list(STANDARD_1020_COORDS.values()), axis=0)

    if verbose:
        print(f"Equidistant centroid: {eq_centroid}")
        print(f"Standard 10-20 centroid: {std_centroid}")

    # For each equidistant electrode, find nearest 10-20 position
    for eq_name, eq_coord in electrodes.items():
        # Skip neck electrodes
        if eq_name.startswith('N'):
            continue

        nearest, dist = find_nearest_1020(eq_coord)

        if nearest and dist < max_distance:
            mapping[eq_name] = nearest
            if verbose:
                print(f"  {eq_name} -> {nearest} (dist: {dist:.1f} mm)")
        else:
            if verbose:
                print(f"  {eq_name} -> NO MATCH (nearest: {nearest}, dist: {dist:.1f} mm)")

    return mapping


def generate_channel_aliases(mapping: Dict[str, str]) -> str:
    """Generate Python code for CHANNEL_ALIASES dictionary.

    Args:
        mapping: Dict mapping equidistant electrode names to 10-20 position names.
    """
    lines = [
        "    # =========================================================================",
        "    # EASYCAP Equidistant Cap (auto-generated mapping)",
        "    # Dataset: ds004460 (Spot Rotation)",
        "    # These mappings are approximate based on 3D coordinate matching",
        "    # =========================================================================",
    ]

    # Sort by equidistant electrode name for consistent output
    for eq_name, std_name in sorted(mapping.items()):
        lines.append(f"    '{eq_name}': '{std_name}',")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Map equidistant electrodes to 10-20")
    parser.add_argument(
        "--electrodes", "-e",
        type=Path,
        required=True,
        help="Path to BIDS electrodes.tsv file"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=30.0,
        help="Maximum distance (mm) for a valid mapping"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for mapping (default: print to stdout)"
    )
    args = parser.parse_args()

    if not args.electrodes.exists():
        print(f"File not found: {args.electrodes}")
        return 1

    print(f"Loading electrodes from: {args.electrodes}")
    electrodes = load_electrodes_tsv(args.electrodes)
    print(f"Found {len(electrodes)} electrodes")

    # Filter out neck electrodes for display
    scalp = {k: v for k, v in electrodes.items() if not k.startswith('N')}
    neck = {k: v for k, v in electrodes.items() if k.startswith('N')}
    print(f"  Scalp: {len(scalp)}, Neck: {len(neck)}")

    print("\nCreating 10-20 mapping...")
    mapping = create_mapping(electrodes, max_distance=args.max_distance)

    print(f"\nSuccessfully mapped {len(mapping)} / {len(STANDARD_1020_COORDS)} standard positions")

    # Generate code
    code = generate_channel_aliases(mapping)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"\nMapping saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("Generated CHANNEL_ALIASES entries:")
        print("=" * 60)
        print(code)

    return 0


if __name__ == "__main__":
    exit(main())
