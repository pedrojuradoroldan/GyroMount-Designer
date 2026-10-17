# GyroMount Designer

Proof-of-concept web app for predesigning printable gyroid motor-mount metamaterials.

The app implements the empirical scaling laws developed in the thesis:

\[
f_c \approx \frac{\sqrt{E/\rho}}{L}\Phi(\rho^*)
\]

\[
\Delta f \approx \frac{\sqrt{E/\rho}}{L}\Psi(\rho^*)
\]

It recommends a unit-cell size, relative density, threshold, and predicted bandgap, then exports a gyroid STL file for additive manufacturing.

## Folder structure

```text
gyromount_app/
├── app.py
├── gyroid_model.py
├── stl_export.py
├── requirements.txt
├── data/
│   └── design_database.csv   # optional but recommended
└── generated/
    └── exported STL files
```

## Installation

From the `gyromount_app` folder:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

A browser window should open automatically.

## Database

For best threshold interpolation, copy your MATLAB-generated `design_database.csv` into:

```text
data/design_database.csv
```

The app can also run without the database, but in that case it uses a rough fallback approximation to estimate the threshold from relative density.

## Typical workflow

1. Enter the problematic frequency range.
2. Enter the material properties.
3. Set design and manufacturing constraints.
4. Click **Run predesign**.
5. Inspect the recommended geometry and predicted bandgap.
6. Click **Generate STL**.
7. Download the generated STL file.

## Important limitations

- This is a predesign tool, not a final validation solver.
- The empirical law is intended for the regime where the useful gap remains the 8--9 gap.
- Final designs should be validated with the complete Bloch-FE workflow and, eventually, experimental testing.
- Manufacturing effects such as FDM anisotropy, overcuring, viscoelasticity, and preload are not included in this first version.
