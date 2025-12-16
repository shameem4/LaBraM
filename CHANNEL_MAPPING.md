# Channel Mapping for Non-Standard EEG Systems

This document describes how `run_labram_inference.py` maps non-standard electrode names to the standard 10-20 positions that LaBraM was trained on.

## Overview

LaBraM uses position embeddings based on the extended 10-20 system. When processing EEG data from non-standard devices (ear EEG, consumer headsets, etc.), we map electrode positions to their **anatomically closest** standard 10-20 equivalent.

**Important caveats:**
- These mappings are approximations based on anatomical proximity
- Signal characteristics may differ due to different tissue paths and reference schemes
- Results from ear-based electrodes should be interpreted with caution
- Multiple non-standard electrodes may map to the same 10-20 position

## Mapping Logic

The channel matching follows this priority:

1. **Direct match**: Standard 10-20 names (FP1, CZ, O2, etc.)
2. **Reference stripping**: Remove reference notation (`C4:A1` → `C4`, `O2-A1` → `O2`)
3. **Alias lookup**: Map via `CHANNEL_ALIASES` dictionary

---

## cEEGrid (Around-the-Ear Array)

The cEEGrid is a flex-PCB electrode array that wraps around the ear from anterior (in front) to posterior (behind). Electrodes are numbered 1-8 from front to back.

**Reference**: Debener et al., "Unobtrusive ambulatory EEG using a smartphone and flexible printed electrodes around the ear" (2015)

```
Anatomical layout (left ear, viewed from side):

        L3 (T7)
       /      \
    L2 (T9)    L4 (TP9)
    |              |
    L1 (FT9)   L5 (TP9)
                   |
               L6-L8 (M1)
      [EAR]
```

| cEEGrid | 10-20 | Anatomical Rationale |
|---------|-------|---------------------|
| L1 / R1 | FT9 / FT10 | Anterior to tragus, preauricular region |
| L2 / R2 | T9 / T10 | Superior-anterior, near temporal bone |
| L3 / R3 | T7 / T8 | At ear level, closest to standard temporal |
| L4, L4A, L4B / R4, R4A, R4B | TP9 / TP10 | Posterior-superior to ear |
| L5 / R5 | TP9 / TP10 | Mid-posterior, temporo-parietal region |
| L6-L8 / R6-R8 | M1 / M2 | Mastoid process, behind/below ear |

---

## In-Ear EEG (earEEG)

In-ear electrodes are positioned in or around the ear canal. All electrodes are anatomically closest to T9/T10 (inferior temporal) due to their proximity to the external auditory meatus.

| earEEG | 10-20 | Notes |
|--------|-------|-------|
| LB | T9 | Left Bottom - lower ear canal |
| LT | T9 | Left Top - upper ear canal |
| RB | T10 | Right Bottom |
| RT | T10 | Right Top |
| ELE | T9 | Generic ear electrode (defaults to left) |
| LE / RE | T9 / T10 | Generic left/right ear |
| LEA, LEB / REA, REB | T9 / T10 | Multi-electrode variants |

---

## IDUN Guardian Earbuds

Consumer in-ear EEG earbuds with 3 electrodes per ear.

| IDUN | 10-20 | Notes |
|------|-------|-------|
| L_A, L_B, L_C | T9 | Left ear electrodes |
| R_A, R_B, R_C | T10 | Right ear electrodes |

---

## Muse Headband

4-channel consumer headband with forehead and behind-ear electrodes.

| Muse | 10-20 | Notes |
|------|-------|-------|
| TP9_MUSE | TP9 | Left behind ear |
| AF7_MUSE | AF7 | Left forehead |
| AF8_MUSE | AF8 | Right forehead |
| TP10_MUSE | TP10 | Right behind ear |

---

## Emotiv EPOC/EPOC+

14-channel consumer headset. Standard names with `_EMOTIV` suffix are mapped directly.

| Emotiv | 10-20 |
|--------|-------|
| AF3_EMOTIV | AF3 |
| F7_EMOTIV | F7 |
| F3_EMOTIV | F3 |
| FC5_EMOTIV | FC5 |
| T7_EMOTIV | T7 |
| P7_EMOTIV | P7 |
| O1_EMOTIV | O1 |
| O2_EMOTIV | O2 |
| P8_EMOTIV | P8 |
| T8_EMOTIV | T8 |
| FC6_EMOTIV | FC6 |
| F4_EMOTIV | F4 |
| F8_EMOTIV | F8 |
| AF4_EMOTIV | AF4 |

---

## OpenBCI Cyton Default Channels

When OpenBCI channels aren't configured with proper names, they use EXG1-8. This mapping assumes a standard clinical montage:

| OpenBCI | 10-20 | Assumed Position |
|---------|-------|-----------------|
| EXG1 | FP1 | Left frontal pole |
| EXG2 | FP2 | Right frontal pole |
| EXG3 | C3 | Left central |
| EXG4 | C4 | Right central |
| EXG5 | P7 | Left parietal |
| EXG6 | P8 | Right parietal |
| EXG7 | O1 | Left occipital |
| EXG8 | O2 | Right occipital |

**Note**: If your OpenBCI setup uses different positions, update channel names in the source data or modify the alias mapping.

---

## Legacy 10-20 Names

Older 10-20 naming conventions are mapped to modern equivalents:

| Legacy | Modern | Region |
|--------|--------|--------|
| T3 | T7 | Left temporal |
| T4 | T8 | Right temporal |
| T5 | P7 | Left posterior temporal |
| T6 | P8 | Right posterior temporal |

---

## Reference Electrode Aliases

Common reference electrode naming variations:

| Alias | 10-20 |
|-------|-------|
| MASTL, LMAS, LM | M1 |
| MASTR, RMAS, RM | M2 |
| EARL | A1 |
| EARR | A2 |

---

## PSG (Polysomnography) Channels

PSG recordings often use reference notation like `C4:A1` (C4 referenced to left mastoid). The script strips the reference suffix to extract the base electrode name:

- `C4:A1` → `C4`
- `O2-A1` → `O2`
- `F3:A2` → `F3`

---

## Adding Custom Mappings

To add mappings for additional electrode systems, edit the `CHANNEL_ALIASES` dictionary in `run_labram_inference.py`:

```python
CHANNEL_ALIASES: dict[str, str] = {
    # ... existing mappings ...

    # Your custom system
    'CUSTOM1': 'FP1',  # Map CUSTOM1 to standard FP1
    'CUSTOM2': 'CZ',   # Map CUSTOM2 to standard CZ
}
```

Guidelines for choosing mappings:
1. Consider the anatomical location of the electrode
2. Choose the nearest standard 10-20 position
3. When in doubt, prefer temporal positions (T7/T8/T9/T10) for ear-adjacent electrodes
4. Document your rationale in comments

---

## Troubleshooting

### "No valid channels found"

This means none of the channel names in your recording match standard 10-20 names or aliases. Run with `--log-level DEBUG` to see which channels are being rejected:

```bash
python run_labram_inference.py --log-level DEBUG ...
```

Then either:
1. Add mappings to `CHANNEL_ALIASES` for your electrode names
2. Rename channels in your source data preprocessing

### Multiple channels mapping to same position

This is expected for systems like earEEG where all electrodes are near the same anatomical location. LaBraM will receive multiple channels with the same position embedding, which may affect results.

### Poor results with ear EEG

Ear-based electrodes capture different neural signals than scalp electrodes due to:
- Different tissue/bone conduction paths
- Greater distance from cortical sources
- Different reference configurations

Consider these mappings as "best effort" approximations rather than true equivalents.
