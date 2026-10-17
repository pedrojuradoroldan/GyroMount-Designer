"""
GyroMount Designer - predesign model

This module contains the inverse-design logic used by the Streamlit app.
It does not run the Bloch-FE solver. Instead, it uses the empirical scaling
laws extracted from the thesis numerical database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Reference material used in the empirical fits
# -----------------------------------------------------------------------------
E0_PA = 18e6
RHO0_KGM3 = 1210.0


# -----------------------------------------------------------------------------
# Empirical fits from the updated thesis model.
# L is expressed in mm and frequencies in Hz.
# For the reference material:
#   fc * L = Phi(rho*)
#   df * L = Psi(rho*)
# -----------------------------------------------------------------------------
def phi_center(rho_rel: np.ndarray | float) -> np.ndarray | float:
    """Return Phi(rho*) = f_c L for the reference material."""
    return 42987.0 * rho_rel + 9051.6


def psi_width(rho_rel: np.ndarray | float) -> np.ndarray | float:
    """Return Psi(rho*) = Delta f L for the reference material."""
    return -27128.0 * np.asarray(rho_rel) ** 2 + 24867.0 * np.asarray(rho_rel) + 513.5


@dataclass
class PredesignInputs:
    f_low_target_hz: float
    f_high_target_hz: float
    E_pa: float
    rho_kgm3: float
    L_range_mm: Tuple[float, float] = (7.0, 20.0)
    rho_rel_range: Tuple[float, float] = (0.15, 0.42)
    min_thickness_mm: float = 0.4
    require_full_coverage: bool = False
    n_candidates: int = 1500


@dataclass
class PredesignResult:
    candidate_table: pd.DataFrame
    best: pd.Series
    inputs: PredesignInputs
    material_scale: float
    database_file: Optional[Path]
    warnings: list[str]


# -----------------------------------------------------------------------------
# Database loading and interpolation helpers
# -----------------------------------------------------------------------------
def load_database(database_file: Optional[str | Path]) -> Optional[pd.DataFrame]:
    """Load the MATLAB design database if available.

    The MATLAB files created during the thesis use semicolon-separated CSV.
    This function also tolerates comma-separated CSV if needed.
    """
    if database_file is None or str(database_file).strip() == "":
        return None

    path = Path(database_file)
    if not path.exists():
        return None

    # Try semicolon first, then comma.
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    # Normalize likely numeric columns. This protects against decimal-comma imports.
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
            )
            converted = pd.to_numeric(cleaned, errors="ignore")
            df[col] = converted

    return df


def _unique_xy_map(x: np.ndarray, y: np.ndarray, round_decimals: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Sort and average duplicated x-values to build a stable interpolation map."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        raise ValueError("No valid data available for interpolation.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_round = np.round(x, round_decimals)
    unique_vals = np.unique(x_round)

    x_unique = []
    y_unique = []
    for val in unique_vals:
        mask = x_round == val
        x_unique.append(float(np.mean(x[mask])))
        y_unique.append(float(np.mean(y[mask])))

    return np.asarray(x_unique), np.asarray(y_unique)


def threshold_from_database(rho_rel: np.ndarray, db: Optional[pd.DataFrame]) -> Tuple[np.ndarray, str]:
    """Estimate threshold from rho* using the design database.

    If no database is available, a fallback approximation is used only for demo
    purposes. The report/tool should state that database interpolation is preferred.
    """
    rho_rel = np.asarray(rho_rel, dtype=float)

    if db is not None and {"rhoRel_est", "threshold"}.issubset(db.columns):
        x, y = _unique_xy_map(db["rhoRel_est"].to_numpy(), db["threshold"].to_numpy())
        tau = np.interp(rho_rel, x, y, left=np.nan, right=np.nan)

        # Linear extrapolation for very small out-of-range excursions.
        if np.any(np.isnan(tau)) and len(x) >= 2:
            tau_extrap = np.interp(rho_rel, x, y)
            tau = np.where(np.isnan(tau), tau_extrap, tau)
        return tau, "database"

    # Fallback map: deliberately conservative and only approximate.
    # Built around the observed fact that rho* ~ 0.255 corresponds to tau ~ -0.95.
    rho_map = np.array([0.15, 0.20, 0.255, 0.32, 0.40])
    tau_map = np.array([-1.25, -1.08, -0.95, -0.78, -0.58])
    return np.interp(rho_rel, rho_map, tau_map), "fallback approximation"


def t5_from_database(rho_rel: np.ndarray, L_mm: np.ndarray, db: Optional[pd.DataFrame]) -> Tuple[np.ndarray, str]:
    """Estimate t5 thickness in mm.

    We interpolate t5/L as a function of rho*, then scale by the predicted L.
    This is only a predesign estimate, not a substitute for computing thickness
    from the final geometry.
    """
    rho_rel = np.asarray(rho_rel, dtype=float)
    L_mm = np.asarray(L_mm, dtype=float)

    if db is not None and {"rhoRel_est", "L_mm", "t5_mm"}.issubset(db.columns):
        valid = (
            np.isfinite(db["rhoRel_est"].to_numpy(dtype=float))
            & np.isfinite(db["L_mm"].to_numpy(dtype=float))
            & np.isfinite(db["t5_mm"].to_numpy(dtype=float))
            & (db["L_mm"].to_numpy(dtype=float) > 0)
        )
        x = db.loc[valid, "rhoRel_est"].to_numpy(dtype=float)
        y = (db.loc[valid, "t5_mm"].to_numpy(dtype=float) / db.loc[valid, "L_mm"].to_numpy(dtype=float))

        if x.size >= 2:
            xu, yu = _unique_xy_map(x, y)
            t5_norm = np.interp(rho_rel, xu, yu, left=np.nan, right=np.nan)
            if np.any(np.isnan(t5_norm)):
                t5_norm_extrap = np.interp(rho_rel, xu, yu)
                t5_norm = np.where(np.isnan(t5_norm), t5_norm_extrap, t5_norm)
            return t5_norm * L_mm, "database"

    # Fallback: unknown. Returning NaN means the app will not use t5 to filter.
    return np.full_like(rho_rel, np.nan, dtype=float), "not available"


# -----------------------------------------------------------------------------
# Main inverse design function
# -----------------------------------------------------------------------------
def predesign_gyroid(
    inputs: PredesignInputs,
    database_file: Optional[str | Path] = None,
) -> PredesignResult:
    """Generate candidate gyroid designs for a target frequency range."""
    if inputs.f_high_target_hz <= inputs.f_low_target_hz:
        raise ValueError("The upper target frequency must be larger than the lower one.")
    if inputs.L_range_mm[1] <= inputs.L_range_mm[0]:
        raise ValueError("Invalid L range.")
    if inputs.rho_rel_range[1] <= inputs.rho_rel_range[0]:
        raise ValueError("Invalid rho* range.")

    warnings: list[str] = []
    db = load_database(database_file)
    db_path = Path(database_file) if database_file is not None and Path(database_file).exists() else None

    f_low = float(inputs.f_low_target_hz)
    f_high = float(inputs.f_high_target_hz)
    f_center_target = 0.5 * (f_low + f_high)
    target_width = f_high - f_low

    material_scale = np.sqrt(inputs.E_pa / inputs.rho_kgm3) / np.sqrt(E0_PA / RHO0_KGM3)

    rho_grid = np.linspace(inputs.rho_rel_range[0], inputs.rho_rel_range[1], int(inputs.n_candidates))

    # We choose L so that each rho* candidate targets the requested center frequency.
    L_mm = material_scale * phi_center(rho_grid) / f_center_target
    fc_pred = material_scale * phi_center(rho_grid) / L_mm
    width_pred = material_scale * psi_width(rho_grid) / L_mm

    gap_low = fc_pred - 0.5 * width_pred
    gap_high = fc_pred + 0.5 * width_pred
    overlap = np.maximum(0.0, np.minimum(gap_high, f_high) - np.maximum(gap_low, f_low))

    tau_est, tau_source = threshold_from_database(rho_grid, db)
    if tau_source != "database":
        warnings.append("Threshold estimated using fallback approximation. Copy design_database.csv into data/ for better interpolation.")

    t5_est, t5_source = t5_from_database(rho_grid, L_mm, db)
    if t5_source == "not available":
        warnings.append("t5 thickness could not be estimated from database; printability filtering by t5 is disabled.")

    printable = np.ones_like(rho_grid, dtype=bool)
    if np.any(np.isfinite(t5_est)):
        printable = t5_est >= inputs.min_thickness_mm

    valid = (
        (L_mm >= inputs.L_range_mm[0])
        & (L_mm <= inputs.L_range_mm[1])
        & (width_pred > 0)
        & printable
        & np.isfinite(tau_est)
    )
    if inputs.require_full_coverage:
        valid &= overlap >= target_width

    table = pd.DataFrame(
        {
            "rho_rel": rho_grid,
            "threshold_est": tau_est,
            "L_mm": L_mm,
            "fc_pred_Hz": fc_pred,
            "width_pred_Hz": width_pred,
            "gap_low_Hz": gap_low,
            "gap_high_Hz": gap_high,
            "target_overlap_Hz": overlap,
            "t5_est_mm": t5_est,
            "printable": printable,
            "valid": valid,
        }
    )

    if not valid.any():
        raise RuntimeError(
            "No valid candidate found. Try relaxing L range, rho* range, minimum thickness, or full-coverage requirement."
        )

    # Selection criterion: maximize target overlap, then maximize predicted width.
    valid_table = table.loc[valid].copy()
    max_overlap = valid_table["target_overlap_Hz"].max()
    tied = valid_table[np.isclose(valid_table["target_overlap_Hz"], max_overlap)]
    best_idx = tied["width_pred_Hz"].idxmax()
    best = table.loc[best_idx]

    return PredesignResult(
        candidate_table=table,
        best=best,
        inputs=inputs,
        material_scale=float(material_scale),
        database_file=db_path,
        warnings=warnings,
    )
