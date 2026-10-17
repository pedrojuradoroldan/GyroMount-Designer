from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gyroid_model import PredesignInputs, predesign_gyroid
from stl_export import export_gyroid_stl, mesh_to_plotly


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
GENERATED_DIR = APP_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
DEFAULT_DB = DATA_DIR / "design_database.csv"


st.set_page_config(
    page_title="GyroMount Designer",
    page_icon="🌀",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .small-note {
        font-size: 0.88rem;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">GyroMount Designer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Proof-of-concept predesign tool for printable gyroid motor-mount metamaterials.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Target frequency range")
    f_low = st.number_input("Lower bound, f_low (Hz)", min_value=1.0, value=1000.0, step=50.0)
    f_high = st.number_input("Upper bound, f_high (Hz)", min_value=1.0, value=1400.0, step=50.0)

    st.header("Material")
    E_mpa = st.number_input("Young's modulus E (MPa)", min_value=0.1, value=18.0, step=1.0)
    rho_kgm3 = st.number_input("Density rho (kg/m³)", min_value=100.0, value=1210.0, step=10.0)

    st.header("Design constraints")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        L_min = st.number_input("L min (mm)", min_value=1.0, value=7.0, step=0.5)
    with col_l2:
        L_max = st.number_input("L max (mm)", min_value=1.0, value=20.0, step=0.5)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        rho_rel_min = st.number_input("rho* min", min_value=0.01, max_value=0.95, value=0.15, step=0.01)
    with col_r2:
        rho_rel_max = st.number_input("rho* max", min_value=0.01, max_value=0.95, value=0.42, step=0.01)

    require_full = st.checkbox("Require full target-band coverage", value=False)

    st.header("Manufacturing")
    process = st.selectbox("Reference process", ["SLA / DLP", "FDM", "SLS / MJF"])
    min_thickness = st.number_input("Minimum printable thickness (mm)", min_value=0.01, value=0.40, step=0.05)

    st.header("STL export")
    shape = st.selectbox("External shape", ["cylinder", "box"])
    nx = st.number_input("Cells nx", min_value=1, value=3, step=1)
    ny = st.number_input("Cells ny", min_value=1, value=3, step=1)
    nz = st.number_input("Cells nz", min_value=1, value=6, step=1)
    diameter = st.number_input("Cylinder diameter (mm)", min_value=1.0, value=30.0, step=1.0)
    height = st.number_input("Cylinder height (mm)", min_value=1.0, value=60.0, step=1.0)
    resolution = st.slider("STL resolution per cell", min_value=15, max_value=60, value=32, step=1)

    st.header("Database")
    uploaded_db = st.file_uploader("Upload design_database.csv", type=["csv"], help="Optional. If absent, the app uses data/design_database.csv if available.")


# Save uploaded database in session temp location
if uploaded_db is not None:
    db_path = GENERATED_DIR / "uploaded_design_database.csv"
    db_path.write_bytes(uploaded_db.getbuffer())
else:
    db_path = DEFAULT_DB if DEFAULT_DB.exists() else None


# -----------------------------------------------------------------------------
# Run predesign
# -----------------------------------------------------------------------------
run = st.button("Run predesign", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "stl_path" not in st.session_state:
    st.session_state.stl_path = None
if "stl_mesh_data" not in st.session_state:
    st.session_state.stl_mesh_data = None

if run:
    try:
        inputs = PredesignInputs(
            f_low_target_hz=f_low,
            f_high_target_hz=f_high,
            E_pa=E_mpa * 1e6,
            rho_kgm3=rho_kgm3,
            L_range_mm=(L_min, L_max),
            rho_rel_range=(rho_rel_min, rho_rel_max),
            min_thickness_mm=min_thickness,
            require_full_coverage=require_full,
            n_candidates=1500,
        )
        result = predesign_gyroid(inputs, database_file=db_path)
        st.session_state.result = result
        st.session_state.stl_path = None
        st.session_state.stl_mesh_data = None
    except Exception as exc:
        st.error(f"Predesign failed: {exc}")

result = st.session_state.result


# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
if result is None:
    st.info("Set the target range and material properties in the sidebar, then click **Run predesign**.")

    with st.expander("What this app does"):
        st.markdown(
            r"""
            This app implements the empirical scaling law developed in the thesis:

            $$f_c \approx \frac{\sqrt{E/\rho}}{L}\,\Phi(\rho^*)$$

            $$\Delta f \approx \frac{\sqrt{E/\rho}}{L}\,\Psi(\rho^*)$$

            It proposes an initial gyroid geometry for a target frequency range and can export an STL file for additive manufacturing.
            The result is a **predesign candidate**, not a substitute for a final Bloch simulation or experimental validation.
            """
        )
else:
    best = result.best
    inputs = result.inputs

    for warning in result.warnings:
        st.warning(warning)

    st.subheader("Recommended design")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cell size L", f"{best['L_mm']:.2f} mm")
    m2.metric("Relative density rho*", f"{best['rho_rel']:.3f}")
    m3.metric("Threshold tau", f"{best['threshold_est']:.3f}")
    m4.metric("Printable", "Yes" if bool(best["printable"]) else "No")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Predicted f_c", f"{best['fc_pred_Hz']:.0f} Hz")
    m6.metric("Predicted width", f"{best['width_pred_Hz']:.0f} Hz")
    m7.metric("Predicted gap", f"{best['gap_low_Hz']:.0f}–{best['gap_high_Hz']:.0f} Hz")
    if np.isfinite(best["t5_est_mm"]):
        m8.metric("Estimated t5", f"{best['t5_est_mm']:.2f} mm")
    else:
        m8.metric("Estimated t5", "n/a")

    st.markdown(
        f"""
        <div class="small-note">
        Material scale relative to the reference model: {result.material_scale:.3f}.<br>
        Database used: {result.database_file if result.database_file is not None else 'none / fallback threshold map'}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Bandgap", "Candidate map", "STL export", "Data"])

    with tab1:
        fig = go.Figure()
        # Target band
        fig.add_shape(
            type="rect",
            x0=inputs.f_low_target_hz,
            x1=inputs.f_high_target_hz,
            y0=0,
            y1=1,
            fillcolor="lightgray",
            opacity=0.5,
            line_width=0,
        )
        # Predicted bandgap
        fig.add_shape(
            type="rect",
            x0=best["gap_low_Hz"],
            x1=best["gap_high_Hz"],
            y0=0.2,
            y1=0.8,
            fillcolor="deepskyblue",
            opacity=0.65,
            line_width=0,
        )
        fig.add_vline(x=best["fc_pred_Hz"], line_width=3, line_color="black")
        fig.add_trace(go.Scatter(x=[inputs.f_low_target_hz, inputs.f_high_target_hz], y=[0.98, 0.98], mode="lines", name="Target range"))
        fig.add_trace(go.Scatter(x=[best["gap_low_Hz"], best["gap_high_Hz"]], y=[0.5, 0.5], mode="lines", name="Predicted bandgap"))
        fig.update_layout(
            title="Target frequency range vs predicted bandgap",
            xaxis_title="Frequency (Hz)",
            yaxis=dict(visible=False, range=[0, 1]),
            height=360,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df = result.candidate_table.copy()
        valid_df = df[df["valid"]]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=valid_df["rho_rel"],
                y=valid_df["width_pred_Hz"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=valid_df["L_mm"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="L (mm)"),
                ),
                name="Valid candidates",
                text=[
                    f"rho*={r:.3f}<br>L={l:.2f} mm<br>tau={t:.3f}<br>width={w:.1f} Hz"
                    for r, l, t, w in zip(valid_df["rho_rel"], valid_df["L_mm"], valid_df["threshold_est"], valid_df["width_pred_Hz"])
                ],
                hoverinfo="text",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[best["rho_rel"]],
                y=[best["width_pred_Hz"]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="Selected design",
            )
        )
        fig2.update_layout(
            title="Candidate designs for the target center frequency",
            xaxis_title="Relative density rho*",
            yaxis_title="Predicted bandgap width (Hz)",
            height=430,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("Generate an STL file using the selected predesign geometry.")
        c1, c2 = st.columns([1, 2])
        with c1:
            generate_stl = st.button("Generate STL", use_container_width=True)
        with c2:
            st.markdown(
                f"Selected geometry: **L = {best['L_mm']:.2f} mm**, **tau = {best['threshold_est']:.3f}**"
            )

        if generate_stl:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # IMPORTANT: sanitize only the base name, not the .stl extension.
                # If we call replace(".", "p") after adding ".stl", the filename
                # becomes "...pstl" and trimesh cannot infer the exporter.
                base_name = f"gyromount_L{best['L_mm']:.2f}_tau{best['threshold_est']:.3f}_{timestamp}"
                base_name = base_name.replace(".", "p").replace("-", "m")
                file_name = base_name + ".stl"
                out_file = GENERATED_DIR / file_name

                with st.spinner("Generating STL. This may take a few seconds..."):
                    info = export_gyroid_stl(
                        L_mm=float(best["L_mm"]),
                        threshold=float(best["threshold_est"]),
                        nx=int(nx),
                        ny=int(ny),
                        nz=int(nz),
                        out_file=out_file,
                        shape=shape,
                        diameter_mm=float(diameter),
                        height_mm=float(height),
                        resolution_per_cell=int(resolution),
                    )
                    st.session_state.stl_path = info.out_file
                    st.session_state.stl_mesh_data = mesh_to_plotly(info.mesh)
                    st.session_state.stl_info = info

                st.success(f"STL generated: {out_file.name}")
            except Exception as exc:
                st.error(f"STL generation failed: {exc}")

        if st.session_state.stl_path is not None and Path(st.session_state.stl_path).exists():
            stl_path = Path(st.session_state.stl_path)
            with open(stl_path, "rb") as f:
                st.download_button(
                    label="Download STL",
                    data=f.read(),
                    file_name=stl_path.name,
                    mime="model/stl",
                    use_container_width=True,
                )

            if st.session_state.stl_mesh_data is not None:
                md = st.session_state.stl_mesh_data
                mesh_fig = go.Figure(
                    data=[
                        go.Mesh3d(
                            x=md["x"],
                            y=md["y"],
                            z=md["z"],
                            i=md["i"],
                            j=md["j"],
                            k=md["k"],
                            color="deepskyblue",
                            opacity=1.0,
                            flatshading=True,
                        )
                    ]
                )
                mesh_fig.update_layout(
                    title="STL preview",
                    scene=dict(aspectmode="data"),
                    height=620,
                    margin=dict(l=0, r=0, b=0, t=40),
                )
                st.plotly_chart(mesh_fig, use_container_width=True)

    with tab4:
        st.markdown("Candidate table generated by the predesign model.")
        st.dataframe(result.candidate_table[result.candidate_table["valid"]].head(200), use_container_width=True)
        csv_bytes = result.candidate_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download candidate table CSV",
            data=csv_bytes,
            file_name="gyromount_candidate_table.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Validity notes"):
        st.markdown(
            r"""
            - The tool uses an empirical model identified from the simulated design region.
            - The scaling law should be used mainly in the regime where the useful gap remains the 8--9 gap.
            - The exported STL is a predesign geometry. Final designs should be validated with the full Bloch-FE model and, eventually, experimental testing.
            - The threshold is estimated from the numerical database when available; without a database, the app uses only a rough fallback map.
            - Manufacturing effects such as overcuring, FDM anisotropy, viscoelasticity, and finite-specimen boundary conditions are not included in this first tool.
            """
        )
