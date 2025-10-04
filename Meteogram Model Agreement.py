# streamlit_app.py
# Minimal Meteogram Model Agreement (+ direction ±1σ)
# Run: streamlit run streamlit_app.py

import io
import csv
from typing import Dict, List, Tuple
from datetime import timedelta
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Meteogram Model Agreement", layout="wide")
st.title("📈 Meteogram Model Agreement")
st.caption("Upload multiple CSV meteograms, compare time series, and see agreement.")

# -------------------------
# Helpers
# -------------------------
AUTO_TIME_COL_CANDIDATES = [
    "time","timestamp","date","datetime","valid_time","valid","validtime",
    "w. europe daylight time"
]
AUTO_TWS_COL_CANDIDATES = [
    "tws","wind","windspeed","wind_speed","ff","ff10","ws","spd","kt"
]
AUTO_TWD_COL_CANDIDATES = [
    "twd", "winddir", "wind_direction", "dd", "dir", "wd",
    "wind 10m", "wind10m", "wind10m °", "wind10m deg"
]


@st.cache_data(show_spinner=False)
def _sniff_and_read(file: bytes) -> pd.DataFrame:
    # Try UTF-8 first; if it fails, fall back to latin-1
    for encoding in ("utf-8", "latin-1"):
        try:
            raw = file.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # last resort: ignore errors
        raw = file.decode("utf-8", errors="ignore")

    # Delimiter sniff
    try:
        delim = csv.Sniffer().sniff(raw[:2000]).delimiter
    except Exception:
        delim = next((d for d in [",",";","\t","|"] if d in raw.splitlines()[0]), ",")

    # EU decimal comma first, then fallback
    try:
        return pd.read_csv(io.StringIO(raw), delimiter=delim, decimal=",")
    except Exception:
        return pd.read_csv(io.StringIO(raw), delimiter=delim)


def _auto_pick(colnames: List[str], candidates: List[str]) -> str | None:
    # Normalize: lowercase + drop non [a-z0-9]
    def norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    norm_map = {norm(c): c for c in colnames}  # normalized -> original

    # 1) exact normalized match
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]

    # 2) substring normalized match
    cand_keys = [norm(c) for c in candidates]
    for c in colnames:
        nc = norm(c)
        if any(ck in nc for ck in cand_keys):
            return c

    return None


def to_datetime_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None: raise ValueError("No time column selected.")
    s = df[col]
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        dt = s.apply(lambda x: pd.to_datetime(x, errors="coerce", dayfirst=True))
    return dt

def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    mapping: Dict[str,str] = {}
    cols = list(df.columns)
    print(cols)
    time_col = _auto_pick(cols, AUTO_TIME_COL_CANDIDATES)
    tws_col  = _auto_pick(cols, AUTO_TWS_COL_CANDIDATES)
    twd_col  = _auto_pick(cols, AUTO_TWD_COL_CANDIDATES)
    print(twd_col)

    out = df.copy()
    if time_col:
        out["time"] = to_datetime_series(out, time_col)
        mapping["time"] = time_col
    if tws_col:
        out["TWS"] = pd.to_numeric(out[tws_col], errors="coerce")
        mapping["TWS"] = tws_col
    if twd_col:
        out["TWD"] = pd.to_numeric(out[twd_col], errors="coerce")
        mapping["TWD"] = twd_col

    if "time" in out:
        out = out.dropna(subset=["time"]).sort_values("time")
        out = out.loc[~out["time"].duplicated(keep="first")]
    return out, mapping

def infer_common_freq(times: pd.Series) -> pd.Timedelta:
    if times.empty or len(times) < 3:
        return pd.Timedelta(hours=1)
    diffs = pd.to_timedelta(np.diff(times.values.astype("datetime64[ns]")))
    med = diffs.median()
    if med <= pd.Timedelta(minutes=12): return pd.Timedelta(minutes=10)
    if med <= pd.Timedelta(minutes=22): return pd.Timedelta(minutes=15)
    if med <= pd.Timedelta(minutes=45): return pd.Timedelta(minutes=30)
    if med <= pd.Timedelta(hours=2):    return pd.Timedelta(hours=1)
    return pd.Timedelta(hours=3)

def build_common_index(models: Dict[str,pd.DataFrame]) -> pd.DatetimeIndex:
    spans, freqs = [], []
    for _, df in models.items():
        if "time" not in df or df["time"].empty: continue
        t = df["time"]
        freqs.append(infer_common_freq(t))
        spans.append((t.min(), t.max()))
    if not spans: return pd.DatetimeIndex([])
    start = min(s for s,_ in spans); end = max(e for _,e in spans)
    step = min(freqs) if freqs else pd.Timedelta(hours=1)
    return pd.date_range(start, end, freq=step)

def regrid(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty or "time" not in df:
        return pd.DataFrame(index=index)
    out = df.set_index("time").sort_index()
    keep = [c for c in ["TWS","TWD"] if c in out.columns]
    out = out[keep].reindex(index)
    if "TWS" in out:
        out["TWS"] = out["TWS"].interpolate(method="time", limit_direction="both")
    if "TWD" in out:
        rad = np.deg2rad(out["TWD"])
        z = np.exp(1j*rad)
        zr = pd.Series(np.real(z), index=out.index).interpolate(method="time", limit_direction="both")
        zi = pd.Series(np.imag(z), index=out.index).interpolate(method="time", limit_direction="both")
        out["TWD"] = (np.angle(zr + 1j*zi) * 180/np.pi) % 360
    return out

def circular_mean_deg(deg_values: np.ndarray) -> float:
    rad = np.deg2rad(deg_values)
    C = np.mean(np.cos(rad)); S = np.mean(np.sin(rad))
    return (np.arctan2(S, C) * 180/np.pi) % 360

def circular_resultant_length(deg_values: np.ndarray) -> float:
    rad = np.deg2rad(deg_values)
    C = np.mean(np.cos(rad)); S = np.mean(np.sin(rad))
    return float(np.sqrt(C*C + S*S))  # 0..1

def agreement_speed(series_list: List[pd.Series]) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    mu = M.mean(axis=1); std = M.std(axis=1)
    eps = 1e-6
    return (1 - (std/(mu.abs()+eps))).clip(0,1)

def agreement_speed_threshold(series_list: List[pd.Series], band: float=2.0) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    med = M.median(axis=1)
    within = (M.sub(med, axis=0).abs() <= band)
    return within.sum(axis=1) / within.shape[1]

def agreement_direction(series_list: List[pd.Series]) -> pd.Series:
    M = pd.concat(series_list, axis=1)
    def row_R(row):
        vals = row.dropna().values.astype(float)
        if len(vals)==0: return np.nan
        return circular_resultant_length(vals)
    return M.apply(row_R, axis=1)

def circular_unwrap_deg(y: np.ndarray) -> np.ndarray:
    """Smart unwrap: continuous but constrained within ±180° of first value."""
    if y is None or len(y)==0: return y
    arr = np.mod(np.asarray(y, dtype=float), 360.0)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(arr)))
    base = unwrapped[0]
    for i in range(len(unwrapped)):
        while unwrapped[i] - base > 180:  unwrapped[i] -= 360
        while unwrapped[i] - base < -180: unwrapped[i] += 360
    return unwrapped

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Options")
    speed_unit = st.selectbox("Wind speed unit", ["kt","m/s"], index=0)
    band_val = st.number_input("Agreement band for speed (±)", 0.0, 20.0, 2.0, 0.5)
    show_mean = st.checkbox("Show ensemble mean", True)
    show_spread = st.checkbox("Shade ±1σ spread (speed)", True)
    show_dir_sigma = st.checkbox("Shade ±1σ spread (direction)", True)
    smooth = st.checkbox("Apply mild smoothing (rolling 3)", False)
    st.caption("Rename models below if auto-detect is off.")
    wrap_dir_display = st.checkbox("Wrap direction to 0–360° (display only)", True)


# -------------------------
# Upload
# -------------------------
uploaded = st.file_uploader("Upload meteogram CSV files (multiple)", type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("Upload two or more CSV files to begin.")
    st.stop()

models_raw: Dict[str,pd.DataFrame] = {}
for f in uploaded:
    name = f.name.rsplit(".",1)[0]
    try:
        df = _sniff_and_read(f.getvalue())
        std,_ = standardize_columns(df)
        models_raw[name] = std
    except Exception as e:
        st.error(f"Failed to read {f.name}: {e}")

if not models_raw:
    st.error("No readable CSVs found.")
    st.stop()

# Rename models
with st.expander("Rename models (display names)", expanded=False):
    name_inputs = {}
    for name in list(models_raw.keys()):
        name_inputs[name] = st.text_input(f"Display name for '{name}'", value=name, key=f"rename_{name}").strip()

    taken, rename_map = set(), {}
    for orig in models_raw.keys():
        desired = name_inputs.get(orig) or orig
        base, idx = desired, 2
        while desired in taken:
            desired = f"{base} ({idx})"; idx += 1
        taken.add(desired); rename_map[orig] = desired

if 'rename_map' in locals() and rename_map:
    models_raw = {rename_map[k]: v for k,v in models_raw.items()}

# Build common grid + regrid
common_index = build_common_index(models_raw)
if len(common_index)==0:
    st.error("Could not build a common time index.")
    st.stop()

models: Dict[str,pd.DataFrame] = {name: regrid(df, common_index) for name,df in models_raw.items()}

# Time selection (hourly slider)
st.subheader("Time selection")
t_min = common_index.min().to_pydatetime()
t_max = common_index.max().to_pydatetime()
win = st.slider("Time window", min_value=t_min, max_value=t_max,
                value=(t_min,t_max), step=timedelta(hours=1),
                format="YYYY-MM-DD HH:mm")
start_t, end_t = pd.to_datetime(win[0]), pd.to_datetime(win[1])
if start_t > end_t:
    st.warning("Start after end; swapping.")
    start_t, end_t = end_t, start_t

# Apply window
models = {name: d.loc[(d.index>=start_t)&(d.index<=end_t)] for name,d in models.items()}
time_index = common_index[(common_index>=start_t)&(common_index<=end_t)]
if len(time_index)==0:
    time_index = common_index

# Optional smoothing
if smooth:
    for df in models.values():
        if "TWS" in df: df["TWS"] = df["TWS"].rolling(3, min_periods=1, center=True).mean()
        if "TWD" in df:
            rad = np.deg2rad(df["TWD"])
            z = np.exp(1j*rad)
            z = pd.Series(z).rolling(3, min_periods=1, center=True).mean()
            df["TWD"] = (np.angle(z) * 180/np.pi) % 360

# Units
conv = 0.514444 if speed_unit=="m/s" else 1.0

# -------------------------
# Aggregates & agreement
# -------------------------
frames_speed = [df["TWS"]*conv for df in models.values() if "TWS" in df]
frames_dir   = [df["TWD"]       for df in models.values() if "TWD" in df]
if not frames_speed and not frames_dir:
    st.error("No TWS or TWD found.")
    st.stop()

mean_speed = pd.concat(frames_speed, axis=1).mean(axis=1) if frames_speed else None
std_speed  = pd.concat(frames_speed, axis=1).std(axis=1)  if frames_speed else None

mean_dir = None
if frames_dir:
    M = pd.concat(frames_dir, axis=1)
    mean_dir = M.apply(lambda row: circular_mean_deg(row.dropna().values.astype(float))
                       if row.notna().any() else np.nan, axis=1)

speed_agree_cv   = agreement_speed(frames_speed) if frames_speed else None
speed_agree_band = agreement_speed_threshold(frames_speed, band=band_val) if frames_speed else None
dir_agree_R      = agreement_direction(frames_dir) if frames_dir else None

speed_agree_cv_pct   = (speed_agree_cv*100.0)   if speed_agree_cv   is not None else None
speed_agree_band_pct = (speed_agree_band*100.0) if speed_agree_band is not None else None
dir_agree_R_pct      = (dir_agree_R*100.0)      if dir_agree_R      is not None else None

# Circular σ(t) for direction (in degrees)
dir_sigma_deg = None
if dir_agree_R is not None:
    R_clip = dir_agree_R.clip(lower=1e-12, upper=1.0)
    dir_sigma_rad = np.sqrt(-2.0 * np.log(R_clip))
    dir_sigma_deg = np.rad2deg(dir_sigma_rad)  # Series aligned to time

# -------------------------
# PLOTS — per-tab shared x + common titles
# -------------------------
tab_speed, tab_dir = st.tabs(["Wind Speed", "Wind Direction"])

# ---- SPEED TAB ----
with tab_speed:
    if not frames_speed:
        st.info("No wind speed columns detected.")
    else:
        # Figure: Speed (top) + Agreement % (bottom), shared x
        figS, (axS1, axS2) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 8),
            gridspec_kw={"height_ratios": [2.0, 1.0]}
        )

        # Top: wind speed time series
        for name, df in models.items():
            if "TWS" not in df:
                continue
            axS1.plot(df.index, (df["TWS"]*conv).values, alpha=0.7, linewidth=1.5, label=name)

        if show_mean and mean_speed is not None:
            axS1.plot(mean_speed.index, mean_speed.values, linewidth=2.2, linestyle="--", label="Ensemble mean")

        if show_spread and std_speed is not None and mean_speed is not None:
            axS1.fill_between(
                mean_speed.index,
                (mean_speed - std_speed).values,
                (mean_speed + std_speed).values,
                alpha=0.15, label="±1σ"
            )

        axS1.set_ylabel(f"Wind speed [{speed_unit}]")
        axS1.grid(True, alpha=0.3)
        axS1.legend(ncols=3, fontsize=9)

        # Bottom: agreement %
        has_any_speed_agree = (speed_agree_cv_pct is not None) or (speed_agree_band_pct is not None)
        if has_any_speed_agree:
            if speed_agree_cv_pct is not None:
                axS2.plot(
                    speed_agree_cv_pct.index, speed_agree_cv_pct.values,
                    linewidth=1.8, label="Agreement (1−σ/μ) %"
                )
            #if speed_agree_band_pct is not None:
            #    axS2.plot(
            #        speed_agree_band_pct.index, speed_agree_band_pct.values,
            #        linewidth=1.8, label=f"Within ±{band_val:g} {speed_unit} %"
            #    )
            axS2.set_ylim(0, 100)
            axS2.set_ylabel("Agreement [%]")
            axS2.set_xlabel("Time")
            axS2.grid(True, alpha=0.3)
            axS2.legend()
        else:
            axS2.set_axis_off()
            axS2.text(0.5, 0.5, "No agreement metrics available", ha="center", va="center", transform=axS2.transAxes)

        figS.suptitle("Wind Speed — Models & Agreement", y=0.98)
        figS.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(figS, clear_figure=True)

# ---- DIRECTION TAB ----
with tab_dir:
    if not frames_dir:
        st.info("No wind direction columns detected.")
    else:
        # Figure: Direction (top) + Agreement % (bottom), shared x
        figD, (axD1, axD2) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 8),
            gridspec_kw={"height_ratios": [2.0, 1.0]}
        )

        # Top: direction time series (smart unwrapped)
for name, df in models.items():
    if "TWD" not in df:
        continue
    y_unwrapped = circular_unwrap_deg(df["TWD"].values)
    y_plot = (y_unwrapped + 360) % 360 if wrap_dir_display else y_unwrapped
    axD1.plot(df.index, y_plot, alpha=0.7, linewidth=1.5, label=name)

# Circular mean (dashed) + optional ±1σ circular band
if mean_dir is not None:
    mean_unwrapped = pd.Series(circular_unwrap_deg(mean_dir.values), index=mean_dir.index)
    mean_plot = ((mean_unwrapped + 360) % 360) if wrap_dir_display else mean_unwrapped
    axD1.plot(mean_plot.index, mean_plot.values, linewidth=2.2, linestyle="--", label="Circular mean")

    if show_dir_sigma and dir_sigma_deg is not None:
        sig = dir_sigma_deg.reindex(mean_plot.index).interpolate().fillna(method="bfill").fillna(method="ffill")
        upper = mean_unwrapped + sig
        lower = mean_unwrapped - sig
        # wrap the band for display if requested (note: may look odd at 0° crossings)
        if wrap_dir_display:
            upper = (upper + 360) % 360
            lower = (lower + 360) % 360
            axD1.fill_between(mean_plot.index, lower.values, upper.values, alpha=0.15, label="±1σ (circular)")
        else:
            axD1.fill_between(mean_plot.index, lower.values, upper.values, alpha=0.15, label="±1σ (circular)")


        # Bottom: R% agreement
        if dir_agree_R_pct is not None:
            axD2.plot(dir_agree_R_pct.index, dir_agree_R_pct.values, linewidth=1.8, label="Directional agreement (R) %")
            axD2.set_ylim(0, 100)
            axD2.set_ylabel("Agreement [%]")
            axD2.set_xlabel("Time")
            axD2.grid(True, alpha=0.3)
            axD2.legend()
        else:
            axD2.set_axis_off()
            axD2.text(0.5, 0.5, "No directional agreement available", ha="center", va="center", transform=axD2.transAxes)

        figD.suptitle("Wind Direction — Models & Agreement", y=0.98)
        figD.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(figD, clear_figure=True)


# -------------------------
# Export
# -------------------------
merged = pd.DataFrame(index=time_index)
for name, df in models.items():
    if "TWS" in df: merged[f"{name}_TWS_{speed_unit}"] = (df["TWS"]*conv)
    if "TWD" in df: merged[f"{name}_TWD_deg"]         = df["TWD"]
if speed_agree_cv_pct   is not None: merged["agree_speed_cv_pct"]   = speed_agree_cv_pct
if speed_agree_band_pct is not None: merged["agree_speed_band_pct"] = speed_agree_band_pct
if dir_agree_R_pct      is not None: merged["agree_dir_R_pct"]      = dir_agree_R_pct

csv_bytes = merged.to_csv(index_label="time").encode("utf-8")
st.download_button("Download merged & agreement CSV", data=csv_bytes,
                   file_name="meteogram_agreement_merged.csv", mime="text/csv")
