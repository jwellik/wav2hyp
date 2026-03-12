"""
NonLinLoc velocity model file parsing.

Reads 1D LAYER-format velocity model files and returns layer data compatible
with nllpy's NLLocConfig.layer.layers (list of 7-tuples).
"""

from pathlib import Path
from typing import List, Tuple


def parse_nll_layer_file(filepath: str) -> List[Tuple[float, float, float, float, float, float, float]]:
    """
    Parse a NonLinLoc LAYER-format velocity model file.

    Each line is: LAYER  depth  VpTop  VpGrad  VsTop  VsGrad  rhoTop  rhoGrad
    Comment lines (starting with #) are skipped.

    Parameters
    ----------
    filepath : str
        Path to the velocity model file.

    Returns
    -------
    list of tuples
        Each tuple is (depth, VpTop, VpGrad, VsTop, VsGrad, rhoTop, rhoGrad)
        for use with nllpy NLLocConfig.layer.layers.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a LAYER line does not have exactly 7 numeric fields.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Velocity model file not found: {filepath}")

    layers = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts or parts[0].upper() != "LAYER":
                continue
            if len(parts) != 8:
                raise ValueError(
                    f"{filepath}: line {line_num}: LAYER line must have 8 fields "
                    f"(LAYER + 7 numbers), got {len(parts)}"
                )
            try:
                values = (float(parts[1]), float(parts[2]), float(parts[3]),
                          float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
            except ValueError as e:
                raise ValueError(
                    f"{filepath}: line {line_num}: LAYER values must be numeric: {e}"
                ) from e
            layers.append(values)

    if not layers:
        raise ValueError(f"{filepath}: no LAYER lines found")
    return layers
