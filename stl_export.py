"""
GyroMount Designer - STL export

This module generates a gyroid solid using an implicit field and exports it as STL.
The geometry is intended for predesign and visualization, not as a substitute for
process-specific manufacturing compensation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from skimage import measure
import trimesh


ShapeType = Literal["box", "cylinder"]


@dataclass
class STLInfo:
    out_file: Path
    L_mm: float
    threshold: float
    shape: str
    n_vertices: int
    n_faces: int
    dimensions_mm: tuple[float, float, float]
    mesh: trimesh.Trimesh


def _gyroid_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L_mm: float) -> np.ndarray:
    return (
        np.sin(2 * np.pi * X / L_mm) * np.cos(2 * np.pi * Y / L_mm)
        + np.sin(2 * np.pi * Y / L_mm) * np.cos(2 * np.pi * Z / L_mm)
        + np.sin(2 * np.pi * Z / L_mm) * np.cos(2 * np.pi * X / L_mm)
    )


def export_gyroid_stl(
    L_mm: float,
    threshold: float,
    nx: int,
    ny: int,
    nz: int,
    out_file: str | Path,
    shape: ShapeType = "cylinder",
    diameter_mm: Optional[float] = None,
    height_mm: Optional[float] = None,
    resolution_per_cell: int = 35,
    shift_fraction: float = 1 / 8,
    reduction_fraction: float = 1.0,
) -> STLInfo:
    """Generate and export a gyroid STL.

    Parameters
    ----------
    L_mm : float
        Unit-cell size in mm.
    threshold : float
        Gyroid threshold tau. Solid phase is f(x,y,z) < threshold.
    nx, ny, nz : int
        Number of cells in x, y, z used to define the bounding box.
    out_file : str | Path
        Output STL path.
    shape : {'box', 'cylinder'}
        External clipping shape.
    diameter_mm, height_mm : float, optional
        Cylinder dimensions. If omitted, derived from nx, ny, nz and L.
    resolution_per_cell : int
        Number of grid intervals per unit cell, approximately.
    shift_fraction : float
        Domain shift as a fraction of L. This mimics the shifted unit-cell convention.
    reduction_fraction : float
        If < 1, simplify mesh using trimesh's quadratic decimation when available.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if shape not in {"box", "cylinder"}:
        raise ValueError("shape must be 'box' or 'cylinder'.")
    if L_mm <= 0:
        raise ValueError("L_mm must be positive.")
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1.")
    if resolution_per_cell < 12:
        raise ValueError("resolution_per_cell should be at least 12.")

    box_x = nx * L_mm
    box_y = ny * L_mm
    box_z = nz * L_mm

    if shape == "cylinder":
        diameter = float(diameter_mm) if diameter_mm is not None else min(box_x, box_y)
        height = float(height_mm) if height_mm is not None else box_z
        if diameter <= 0 or height <= 0:
            raise ValueError("Cylinder diameter and height must be positive.")
    else:
        diameter = np.nan
        height = box_z

    shift = shift_fraction * L_mm
    margin = 0.25 * L_mm

    x_min = shift - margin
    x_max = shift + box_x + margin
    y_min = shift - margin
    y_max = shift + box_y + margin
    z_min = shift - margin
    z_max = shift + height + margin if shape == "cylinder" else shift + box_z + margin

    n_x = max(24, int(round(nx * resolution_per_cell)) + 2)
    n_y = max(24, int(round(ny * resolution_per_cell)) + 2)
    n_z = max(24, int(round((height / L_mm) * resolution_per_cell)) + 2)

    x = np.linspace(x_min, x_max, n_x, dtype=np.float32)
    y = np.linspace(y_min, y_max, n_y, dtype=np.float32)
    z = np.linspace(z_min, z_max, n_z, dtype=np.float32)

    # indexing='ij' gives array axes x,y,z. marching_cubes will return vertices
    # in these axis coordinates, to which we add the origin offset.
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    F = _gyroid_field(X, Y, Z, L_mm).astype(np.float32)
    gyroid_field = F - np.float32(threshold)  # solid is gyroid_field < 0

    cx = shift + box_x / 2
    cy = shift + box_y / 2

    if shape == "box":
        cz = shift + box_z / 2
        sx = box_x / 2
        sy = box_y / 2
        sz = box_z / 2
        shape_field = np.maximum.reduce(
            [
                np.abs(X - cx) - sx,
                np.abs(Y - cy) - sy,
                np.abs(Z - cz) - sz,
            ]
        ).astype(np.float32)
        dimensions = (box_x, box_y, box_z)
    else:
        cz = shift + height / 2
        radius = diameter / 2
        radial_field = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - radius
        axial_field = np.abs(Z - cz) - height / 2
        shape_field = np.maximum(radial_field, axial_field).astype(np.float32)
        dimensions = (diameter, diameter, height)

    # Boolean intersection: final solid if both fields are negative.
    final_field = np.maximum(gyroid_field, shape_field).astype(np.float32)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    verts, faces, normals, _ = measure.marching_cubes(
        final_field,
        level=0.0,
        spacing=(dx, dy, dz),
        allow_degenerate=False,
    )
    verts[:, 0] += x_min
    verts[:, 1] += y_min
    verts[:, 2] += z_min

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    # Optional simplification. This may require optional dependencies, so fail gracefully.
    if reduction_fraction < 1.0:
        try:
            target_faces = max(100, int(len(mesh.faces) * reduction_fraction))
            mesh = mesh.simplify_quadratic_decimation(target_faces)
        except Exception:
            pass

    # Explicitly force STL export. This avoids errors if the output filename
    # is not recognized correctly by trimesh on some systems.
    mesh.export(str(out_file), file_type="stl")

    return STLInfo(
        out_file=out_file,
        L_mm=float(L_mm),
        threshold=float(threshold),
        shape=shape,
        n_vertices=int(len(mesh.vertices)),
        n_faces=int(len(mesh.faces)),
        dimensions_mm=tuple(float(v) for v in dimensions),
        mesh=mesh,
    )


def mesh_to_plotly(mesh: trimesh.Trimesh, max_faces: int = 15000) -> dict:
    """Return downsampled mesh arrays for Plotly Mesh3d."""
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)

    if len(faces) > max_faces:
        idx = np.linspace(0, len(faces) - 1, max_faces).astype(int)
        faces = faces[idx]

    return {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
    }
