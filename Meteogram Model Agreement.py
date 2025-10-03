# streamlit_app.py
# --- Meteogram Model Agreement Viewer ---
# Run with:  streamlit run streamlit_app.py

import io
import csv
from typing import Dict, List, Tuple
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================
# App Config
# ============================
st.set_page_config(page_title="Meteogram Model Agreement", layout="wide")
st.title("📈 Meteogram Model Agreement")
st.caption("Upload multiple .csv meteograms from different models to compare time series and visualize model agreement.")

# ============================
# Helpers
# ============================
AUTO_TIME_COL_CANDIDATES = [
    "time", "timestamp", "date", "datetime", "valid_time", "valid", "validtime",
    "w. europe daylight time"  # Expedition export header seen in your CSVs
]
AUTO_TWS_COL_CANDIDATES = [
    "tws", "wind", "windspeed", "wind_speed", "ff", "ff10", "ws", "spd", "kt"  # knots column
]
AUTO_TWD_COL_CANDIDATES = [
    "twd", "winddir", "wind_direction", "dd", "dir", "wd", "wind 10m"  # degrees column
]

@st.cache_data(show_spinner=False)
def _sniff_and_read(file: bytes) -> pd.DataFrame:
    """Try to robustly read CSV, handling delimiters and EU decimals."""
    raw = file.decode("utf-8", errors="ignore")
    # Try sniffing delimiter
    try:
        dialect = csv.Sniffer().sniff(raw[:2000])
        delim = dialect.delimiter
    except Exception:
        # Fallback common delimiters
        for delim in [',', ';', '	', '|']:
            if delim in raw.splitlines()[0]:
                break
    # Parse with flexible decimal handling
    try:
        df = pd.read_csv(io.StringIO(raw), delimiter=delim, decimal=',')  # handles comma decimals
    except Exception:
        df = pd.read_csv(io.StringIO(raw), delimiter=delim)  # fallback
    return df

def _auto_pick(colnames: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fuzzy contains
    for name in colnames:
        ln = name.lower().replace(" ", "").replace("-", "_")
        for cand in candidates:
            if cand in ln:
                return name
    return None

def to_datetime_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        raise ValueError("No time column selected.")
    s = df[col]
    # EU-style first, then fallback
    dt = pd.to_datetime(s, errors='coerce', dayfirst=True, utc=False)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors='coerce')
    if dt.isna().any():
        dt = s.apply(lambda x: pd.to_datetime(x, errors='coerce', dayfirst=True))
    return dt

def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return df with added 'time','TWS','TWD' when possible plus a mapping."""
    mapping: Dict[str, str] = {}
    cols = list(df.columns)

    time_col = _auto_pick(cols, AUTO_TIME_COL_CANDIDATES)
    tws_col  = _auto_pick(cols, AUTO_TWS_COL_CANDIDATES)
    twd_col  = _auto_pick(cols, AUTO_TWD_COL_CANDIDATES)

    out = df.copy()
    if time_col:
        out['time'] = to_datetime_series(out, time_col)
        mapping['time'] = time_col
    if tws_col:
        out['TWS'] = pd.to_numeric(out[tws_col], errors='coerce')
        mapping['TWS'] = tws_col
    if twd_col:
        out['TWD'] = pd.to_numeric(out[twd_col], errors='coerce')
        mapping['TWD'] = twd_col

    if 'time' in out:
        out = out.dropna(subset=['time']).sort_values('time')
        out = out.loc[~out['time'].duplicated(keep='first')]
    return out, mapping

def infer_common_freq(times: pd.Series) -> pd.Timedelta:
    if times.empty or len(times) < 3:
        return pd.Timedelta(hours=1)
    diffs = np.diff(times.values.astype('datetime64[ns]'))
    diffs = pd.to_timedelta(diffs)
    med = diffs.median()
    candidates = [pd.Timedelta(minutes=10), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30), pd.Timedelta(hours=1), pd.Timedelta(hours=3)]
    if med <= pd.Timedelta(minutes=12):
        return candidates[0]
    if med <= pd.Timedelta(minutes=22):
        return candidates[1]
    if med <= pd.Timedelta(minutes=45):
        return candidates[2]
    if med <= pd.Timedelta(hours=2):
        return candidates[3]
    return candidates[4]

def build_common_index(models: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    spans = []
    freqs = []
    for _, df in models.items():
        if 'time' not in df:
            continue
        t = df['time']
        if len(t) == 0:
            continue
        freqs.append(infer_common_freq(t))
        spans.append((t.min(), t.max()))
    if not spans:
        return pd.DatetimeIndex([])
    start = min(s for s, _ in spans)
    end   = max(e for _, e in spans)
    step  = min(freqs) if freqs else pd.Timedelta(hours=1)
    return pd.date_range(start, end, freq=step)

def regrid(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty or 'time' not in df:
        return pd.DataFrame(index=index)
    out = df.set_index('time').sort_index()
    keep = [c for c in ['TWS', 'TWD'] if c in out.columns]
    out = out[keep]
    out = out.reindex(index)
    if 'TWS' in out:
        out['TWS'] = out['TWS'].interpolate(method='time', limit_direction='both')
    if 'TWD' in out:
        rad = np.deg2rad(out['TWD'])
        z = np.exp(1j*rad)
        z_real = pd.Series(np.real(z), index=out.index).interpolate(method='time', limit_direction='both')
        z_imag = pd.Series(np.imag(z), index=out.index).interpolate(method='time', limit_direction='both')
        out['TWD'] = (np.angle(z_real + 1j*z_imag) * 180/np.pi) % 360
    return out

def circular_mean_deg(deg_values: np.ndarray) -> float:
    rad = np.deg2rad(deg_values)
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    mean = (np.arctan2(S, C) * 180/np.pi) % 360
    return mean

def circular_resultant_length(deg_values: np.ndarray) -> float:
    rad = np.deg2rad(deg_values)
    C = np.mean(np.cos(rad))
    S = np.mean(np.sin(rad))
    R = np.sqrt(C*C + S*S)  # 0..1 (1 = perfect agreement)
    return float(R)

def agreement_speed(series_list: List[pd.Series]) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    mu = M.mean(axis=1)
    std = M.std(axis=1)
    eps = 1e-6
    score = 1 - (std/(mu.abs()+eps))  # 0..1
    return score.clip(lower=0, upper=1)

def agreement_speed_threshold(series_list: List[pd.Series], band: float = 2.0) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    med = M.median(axis=1)
    within = (M.sub(med, axis=0).abs() <= band)
    return within.sum(axis=1) / within.shape[1]  # 0..1 fraction

def agreement_direction(series_list: List[pd.Series]) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    def row_R(row):
        vals = row.dropna().values.astype(float)
        if len(vals) == 0:
            return np.nan
        return circular_resultant_length(vals)
    return M.apply(row_R, axis=1)  # 0..1

# >>> Smart unwrap (no 600° drift)
def circular_unwrap_deg(y: np.ndarray) -> np.ndarray:
    """Smart unwrap: keep continuity while constraining the curve to stay within
    ±180° of the first value (avoids long drifts to 500–600°)."""
    if y is None or len(y) == 0:
        return y
    arr = np.asarray(y, dtype=float)
    arr = np.mod(arr, 360.0)  # normalize to [0,360)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(arr)))
    base = unwrapped[0]
    for i in range(len(unwrapped)):
        while unwrapped[i] - base > 180:
            unwrapped[i] -= 360
        while unwrapped[i] - base < -180:
            unwrapped[i] += 360
    return unwrapped

# ============================
# Sidebar — Options
# ============================
with st.sidebar:
    st.header("Options")
    speed_unit = st.selectbox("Wind speed unit", ["kt", "m/s"], index=0)
    band_val = st.number_input(
        "Agreement band for speed (±)", min_value=0.0, max_value=20.0,
        value=2.0, step=0.5,
        help="Speed models within this band around the median count as agreeing."
    )
    show_mean = st.checkbox("Show ensemble mean", value=True)
    show_spread = st.checkbox("Shade ±1σ spread (speed)", value=True)
    smooth = st.checkbox("Apply mild smoothing (rolling 3)", value=False)
    st.markdown("---")
    st.caption("If auto-detection fails, override columns and rename models below.")

# ============================
# Upload Section
# ============================
uploaded = st.file_uploader(
    "Upload meteogram CSV files (multiple)", type=["csv"], accept_multiple_files=True
)

if not uploaded:
    st.info("Upload two or more CSV files to begin. Each should include a time column and wind speed (TWS) and/or wind direction (TWD).")
    st.stop()

# Read and standardize each file
models_raw: Dict[str, pd.DataFrame] = {}
mappings: Dict[str, Dict[str, str]] = {}
for f in uploaded:
    name = f.name.rsplit('.', 1)[0]
    try:
        df = _sniff_and_read(f.getvalue())
        std, mapping = standardize_columns(df)
        models_raw[name] = std
        mappings[name] = mapping
    except Exception as e:
        st.error(f"Failed to read {f.name}: {e}")

if not models_raw:
    st.error("No readable CSVs found.")
    st.stop()

# Column overrides per file
with st.expander("Column detection per file (override if needed)", expanded=False):
    override: Dict[str, Dict[str, str]] = {}
    for name, df in models_raw.items():
        st.subheader(name)
        # Re-parse original for choices
        raw_df = None
        for f in uploaded:
            if f.name.rsplit('.',1)[0] == name:
                raw_df = _sniff_and_read(f.getvalue())
                break
        if raw_df is None:
            raw_df = df.copy()
        time_choice = st.selectbox(
            f"Time column — {name}", options=raw_df.columns.tolist(),
            index=(raw_df.columns.tolist().index(mappings.get(name,{}).get('time'))
                   if mappings.get(name,{}).get('time') in raw_df.columns.tolist() else 0),
            key=f"time_{name}"
        )
        tws_choice = st.selectbox(
            f"Wind speed column — {name}", options=["<none>"]+raw_df.columns.tolist(),
            index=((raw_df.columns.tolist().index(mappings.get(name,{}).get('TWS'))+1)
                   if mappings.get(name,{}).get('TWS') in raw_df.columns.tolist() else 0),
            key=f"tws_{name}"
        )
        twd_choice = st.selectbox(
            f"Wind direction column — {name}", options=["<none>"]+raw_df.columns.tolist(),
            index=((raw_df.columns.tolist().index(mappings.get(name,{}).get('TWD'))+1)
                   if mappings.get(name,{}).get('TWD') in raw_df.columns.tolist() else 0),
            key=f"twd_{name}"
        )
        override[name] = {"time": time_choice, "TWS": (None if tws_choice=="<none>" else tws_choice), "TWD": (None if twd_choice=="<none>" else twd_choice)}

    if st.button("Apply column overrides"):
        new_models: Dict[str, pd.DataFrame] = {}
        new_maps: Dict[str, Dict[str, str]] = {}
        for name in models_raw.keys():
            # Re-parse from original
            raw_df = None
            for f in uploaded:
                if f.name.rsplit('.',1)[0] == name:
                    raw_df = _sniff_and_read(f.getvalue())
                    break
            if raw_df is None:
                raw_df = models_raw[name].copy()
            use = pd.DataFrame()
            use['time'] = to_datetime_series(raw_df, override[name]['time'])
            if override[name]['TWS'] is not None:
                use['TWS'] = pd.to_numeric(raw_df[override[name]['TWS']], errors='coerce')
            if override[name]['TWD'] is not None:
                use['TWD'] = pd.to_numeric(raw_df[override[name]['TWD']], errors='coerce')
            use = use.dropna(subset=['time']).sort_values('time')
            new_models[name] = use
            new_maps[name] = {k:v for k,v in override[name].items() if v is not None}
        models_raw = new_models
        mappings = new_maps
        st.success("Overrides applied.")

# Rename models
with st.expander("Rename models (display names)", expanded=False):
    name_inputs = {}
    for name in list(models_raw.keys()):
        name_inputs[name] = st.text_input(f"Display name for '{name}'", value=name, key=f"rename_{name}").strip()

    # Build unique mapping
    taken = set()
    rename_map: Dict[str, str] = {}
    for orig in models_raw.keys():
        desired = name_inputs.get(orig) or orig
        base = desired
        idx = 2
        while desired in taken:
            desired = f"{base} ({idx})"
            idx += 1
        taken.add(desired)
        rename_map[orig] = desired

    if rename_map and any(k != v for k, v in rename_map.items()):
        st.caption("Names will be applied to plots and exports.")

# Apply rename mapping
if 'rename_map' in locals() and rename_map:
    models_raw = {rename_map[k]: v for k, v in models_raw.items()}

# Build common grid and regrid all models
common_index = build_common_index(models_raw)
if len(common_index) == 0:
    st.error("Could not build a common time index. Check that uploaded files contain valid time columns.")
    st.stop()

models: Dict[str, pd.DataFrame] = {name: regrid(df, common_index) for name, df in models_raw.items()}

# ---- Time selection
st.subheader("Time selection")
mode = st.radio("Select time range using:", ["Slider", "Manual inputs"], horizontal=True)

t_min = common_index.min().to_pydatetime()
t_max = common_index.max().to_pydatetime()

if mode == "Slider":
    win = st.slider(
        "Time window",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        step=timedelta(hours=1),  # hourly increments
        format="YYYY-MM-DD HH:mm",
        key="time_window_slider",
    )
    start_t, end_t = pd.to_datetime(win[0]), pd.to_datetime(win[1])
else:
    start_t = st.datetime_input("Start", value=t_min, min_value=t_min, max_value=t_max, key="start_dt")
    end_t   = st.datetime_input("End",   value=t_max, min_value=t_min, max_value=t_max, key="end_dt")
    start_t, end_t = pd.to_datetime(start_t), pd.to_datetime(end_t)

if start_t > end_t:
    st.warning("Start is after end; swapping.")
    start_t, end_t = end_t, start_t

# Apply window to model data
models = {name: d.loc[(d.index >= start_t) & (d.index <= end_t)] for name, d in models.items()}

# Build the time index for plots/exports
selected_index = common_index[(common_index >= start_t) & (common_index <= end_t)]
time_index = selected_index if len(selected_index) > 0 else common_index

# Optional smoothing
if smooth:
    for name, df in models.items():
        if 'TWS' in df:
            df['TWS'] = df['TWS'].rolling(3, min_periods=1, center=True).mean()
        if 'TWD' in df:
            rad = np.deg2rad(df['TWD'])
            z = np.exp(1j*rad)
            z = pd.Series(z).rolling(3, min_periods=1, center=True).mean()
            df['TWD'] = (np.angle(z) * 180/np.pi) % 360

# Units
conv = 0.514444 if st.session_state.get('speed_unit', speed_unit) == "m/s" else 1.0
if speed_unit == "m/s":
    conv = 0.514444
else:
    conv = 1.0

# ============================
# Compute ensemble aggregates
# ============================
frames_speed = [df['TWS']*conv for df in models.values() if 'TWS' in df]
frames_dir   = [df['TWD'] for df in models.values() if 'TWD' in df]

if not frames_speed and not frames_dir:
    st.error("No TWS or TWD found in the uploaded files. Map columns in the overrides panel.")
    st.stop()

mean_speed = pd.concat(frames_speed, axis=1).mean(axis=1) if frames_speed else None
std_speed  = pd.concat(frames_speed, axis=1).std(axis=1) if frames_speed else None

mean_dir = None
if frames_dir:
    M = pd.concat(frames_dir, axis=1)
    mean_dir = M.apply(lambda row: circular_mean_deg(row.dropna().values.astype(float)) if row.notna().any() else np.nan, axis=1)

# Agreements (as PERCENT)
speed_agree_cv = agreement_speed(frames_speed) if frames_speed else None
speed_agree_band = agreement_speed_threshold(frames_speed, band=band_val) if frames_speed else None

dir_agree_R = agreement_direction(frames_dir) if frames_dir else None

speed_agree_cv_pct   = (speed_agree_cv*100.0) if speed_agree_cv is not None else None
speed_agree_band_pct = (speed_agree_band*100.0) if speed_agree_band is not None else None
dir_agree_R_pct      = (dir_agree_R*100.0) if dir_agree_R is not None else None

# ============================
# PLOTS
# ============================
_tab1, _tab2 = st.tabs(["Wind Speed", "Wind Direction"])

with _tab1:
    if not frames_speed:
        st.info("No wind speed columns detected.")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        # Individual models
        for name, df in models.items():
            if 'TWS' not in df:
                continue
            ax1.plot(df.index, df['TWS']*conv, alpha=0.6, linewidth=1.5, label=name)
        # Mean and spread
        if show_mean and mean_speed is not None:
            ax1.plot(mean_speed.index, mean_speed.values, linewidth=2.5, label="Ensemble mean", linestyle='--')
        if show_spread and std_speed is not None and mean_speed is not None:
            ax1.fill_between(mean_speed.index, (mean_speed-std_speed).values, (mean_speed+std_speed).values, alpha=0.15, label="±1σ")
        ax1.set_ylabel(f"Wind speed [{speed_unit}]")
        ax1.set_xlabel("Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend(ncols=3, fontsize=9)
        st.pyplot(fig1, clear_figure=True)

        # Agreement score charts (PERCENT)
        if speed_agree_cv_pct is not None or speed_agree_band_pct is not None:
            fig2, ax2 = plt.subplots(figsize=(12, 2.5))
            if speed_agree_cv_pct is not None:
                ax2.plot(speed_agree_cv_pct.index, speed_agree_cv_pct.values, label="Agreement (1 - σ/μ) %", linewidth=1.8)
            if speed_agree_band_pct is not None:
                ax2.plot(speed_agree_band_pct.index, speed_agree_band_pct.values, label=f"Within ±{band_val:g} {speed_unit} %", linewidth=1.8)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("Agreement [%]")
            ax2.set_xlabel("Time")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

with _tab2:
    if not frames_dir:
        st.info("No wind direction columns detected.")
    else:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for name, df in models.items():
            if 'TWD' not in df:
                continue
            y = circular_unwrap_deg(df['TWD'].values)
            ax3.plot(df.index, y, alpha=0.7, linewidth=1.5, label=name)
        if mean_dir is not None:
            y = circular_unwrap_deg(mean_dir.values)
            ax3.plot(mean_dir.index, y, linewidth=2.5, linestyle='--', label="Circular mean")
        ax3.set_ylabel("Wind direction [°]")
        ax3.set_xlabel("Time")
        ax3.grid(True, alpha=0.3)
        ax3.legend(ncols=3, fontsize=9)
        st.pyplot(fig3, clear_figure=True)

        if dir_agree_R_pct is not None:
            fig4, ax4 = plt.subplots(figsize=(12, 2.5))
            ax4.plot(dir_agree_R_pct.index, dir_agree_R_pct.values, linewidth=1.8, label="Directional agreement (R) %")
            ax4.set_ylim(0, 100)
            ax4.set_ylabel("Agreement [%]")
            ax4.set_xlabel("Time")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            st.pyplot(fig4, clear_figure=True)

# ============================
# Summary Table & Export
# ============================
summary_rows = []
for name, df in models.items():
    row = {"Model": name}
    if 'TWS' in df:
        row.update({
            f"{speed_unit} mean": (df['TWS']*conv).mean(),
            f"{speed_unit} std":  (df['TWS']*conv).std(),
        })
    if 'TWD' in df:
        row.update({
            "dir circular R %": (circular_resultant_length(df['TWD'].dropna().values) * 100.0) if df['TWD'].notna().any() else np.nan
        })
    summary_rows.append(row)

if summary_rows:
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summary_rows).set_index("Model").round(2))

# Export merged dataset
merged = pd.DataFrame(index=(time_index if 'time_index' in globals() else common_index))
for name, df in models.items():
    if 'TWS' in df:
        merged[f"{name}_TWS_{speed_unit}"] = (df['TWS']*conv)
    if 'TWD' in df:
        merged[f"{name}_TWD_deg"] = df['TWD']
if speed_agree_cv_pct is not None:
    merged["agree_speed_cv_pct"] = speed_agree_cv_pct
if speed_agree_band_pct is not None:
    merged["agree_speed_band_pct"] = speed_agree_band_pct
if dir_agree_R_pct is not None:
    merged["agree_dir_R_pct"] = dir_agree_R_pct

csv_bytes = merged.to_csv(index_label="time").encode('utf-8')
st.download_button(
    "Download merged & agreement CSV",
    data=csv_bytes,
    file_name="meteogram_agreement_merged.csv",
    mime="text/csv",
)

st.caption("Agreement metrics shown as percentages: speed — (1 - σ/μ)×100 and fraction within ±band×100; direction — circular resultant length R×100. Higher is better.")
