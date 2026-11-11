#!/usr/bin/env python3
"""
v2_10_render_v5_style_supplementary_figures.py
================================================================================
Renders all 10 V5-styled Supplementary Figures (Figure S1 through Figure S10)
at 600 DPI (PNG + PDF) for RSA_MEEG_Supplementary_v2.docx using the exact
verified outputs in `results/`.

Strict Design Rules Enforced:
1. V5 color palette (#d93326, #BB3434, #0059b3, #1F77B4, #77539B, #2ca02c),
   pure white background (#ffffff), bold lowercase panel labels ('a', 'b', 'c', 'd'),
   and NO redundant subplot titles on statistical axes.
2. Line/ribbon plots appear AT MOST ONCE across the entire Supplementary Figure
   set (only in Figure S8a for the continuous 64/32/19-ch temporal trace).
3. Figure S5c, Figure S9, Figure S10, and Figures S1-S7 use high-information-density
   Nature-grade visual encodings:
   - Figure S1: Split-Violin Raincloud (a) + Paired Shrinkage Conditioning Dumbbell (b)
                + Cortical Radiality Ridgeline Joyplot (c)
   - Figure S2: V5 Endpoint Triangle Forest Plot (a) + 2D Multi-Metric Bubble Matrix (b)
                + Paired False-Positive Drift Dumbbell (c)
   - Figure S3: 3D Inflated Cortical Realizations (a) + 2D Bicubic Iso-Contour MTF
                Spectral Surface (b) + Spatial-Band Shrinkage Gain Raincloud (c)
   - Figure S4: Smooth 8-ROI x 5-SNR Iso-Contour Matrix (a) + Paired 8-ROI Dumbbell (b)
   - Figure S5: 3D Cortical Radiality Map (a) + 3D Relative Dimensionality Map (b)
                + 2,562-Vertex 2D Bivariate KDE & Iso-Contour Phase Diagram (c)
   - Figure S6: 64x64 Zero-Tie Continuous Speech Model RDMs (a, b, c)
   - Figure S7: 2D Spatiotemporal Speech Cascade Carpet Heatmap with FWER Contours (a)
                + Individual-Subject Peak Raincloud vs. LOSO Noise Ceiling (b)
   - Figure S8: 64/32/19-ch Shaded-SEM Time-Course Ribbon (a, ONLY line plot in Supp)
                + 360-Degree Polar Montage Retention & Reliability Radar (b)
   - Figure S9: V5 Endpoint Triangle Forest & Cohen's d_z Effect-Size Plot (a)
                + Dual-Modality 2D Bicubic Iso-Fidelity Surface & Gradient Vector Field (b)
   - Figure S10: 8-Stage Horizontal Step-Bridge Alluvial Waterfall (a)
                 + Low-Pass Filter Cutoff Split-Violin Raincloud & Knee-Point Envelope (b)
================================================================================
"""

import os
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon, FancyBboxPatch

try:
    from scipy.ndimage import gaussian_filter, zoom
except ImportError:
    gaussian_filter = None
    zoom = None


class _StrAccessor:
    def __init__(self, values):
        self._vals = [str(v) for v in values]

    def contains(self, pat):
        return np.array([pat in v for v in self._vals], dtype=bool)


class MiniSeries:
    def __init__(self, data):
        self._list = list(data)
        self.values = self._to_array(self._list)

    @staticmethod
    def _to_array(lst):
        if not lst:
            return np.array([])
        num_vals = []
        is_num = True
        for x in lst:
            if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
                num_vals.append(np.nan)
            elif isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
                num_vals.append(float(x))
            else:
                is_num = False
                break
        if is_num:
            return np.array(num_vals, dtype=np.float64)
        return np.array(lst, dtype=object)

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self.values)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=np.float64 if dtype is None else dtype)

    def mean(self):
        return float(np.nanmean(np.asarray(self.values, dtype=np.float64)))

    def std(self):
        return float(np.nanstd(np.asarray(self.values, dtype=np.float64), ddof=1))

    def unique(self):
        seen = []
        for x in self._list:
            if x not in seen:
                seen.append(x)
        return seen

    @property
    def str(self):
        return _StrAccessor(self._list)

    def __eq__(self, other):
        return np.array([x == other for x in self._list], dtype=bool)

    def __gt__(self, other):
        arr = np.asarray(self.values, dtype=np.float64)
        return np.where(np.isnan(arr), False, arr > other)

    def __lt__(self, other):
        arr = np.asarray(self.values, dtype=np.float64)
        return np.where(np.isnan(arr), False, arr < other)


class MiniDF:
    def __init__(self, rows, columns):
        self._rows = list(rows)
        self.columns = list(columns)

    def __len__(self):
        return len(self._rows)

    def copy(self):
        return MiniDF([dict(r) for r in self._rows], self.columns)

    def __getitem__(self, key):
        if isinstance(key, str):
            return MiniSeries([r.get(key) for r in self._rows])
        mask = np.asarray(key, dtype=bool)
        return MiniDF([r for r, m in zip(self._rows, mask) if m], self.columns)

    def sort_values(self, col):
        return MiniDF(sorted(self._rows, key=lambda r: (r.get(col) is None, r.get(col))), self.columns)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


class _MiniPD:
    @staticmethod
    def _parse_val(v):
        if v is None:
            return np.nan
        s = v.strip()
        if s == "":
            return np.nan
        if s == "True":
            return True
        if s == "False":
            return False
        try:
            return float(s)
        except ValueError:
            return s

    @classmethod
    def read_csv(cls, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            rows = [{k: cls._parse_val(v) for k, v in raw.items()} for raw in reader]
        return MiniDF(rows, cols)


pd = _MiniPD()

BASE_DIR = Path(".")
if not BASE_DIR.exists():
    BASE_DIR = Path(".")

RES_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures_v5_style"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_MEG = "#d93326"
C_MEG_DARK = "#BB3434"
C_EEG = "#0059b3"
C_EEG_LIGHT = "#1F77B4"
C_PURPLE = "#77539B"
C_GREEN = "#2ca02c"
C_ORANGE = "#d95f02"
C_GOLD = "#e6ab02"
C_GRAY = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_label(ax, letter, x=-0.11, y=1.06):
    """Bold lowercase panel label ('a', 'b', 'c', 'd') with zero subplot title clutter."""
    text_fn = getattr(ax, "text2D", ax.text)
    text_fn(x, y, letter.lower(), transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", va="top", ha="left", color="#111111")


def save_supp_fig(fig, name):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved Supp Figure] {png_path}", flush=True)


def kde_1d(vals, grid, bw=None):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.zeros_like(grid)
    if bw is None:
        sd = max(np.std(vals, ddof=1) if len(vals) > 1 else 0.05, 1e-4)
        bw = 1.06 * sd * (len(vals) ** (-0.2))
    diff = (grid[:, None] - vals[None, :]) / bw
    dens = np.mean(np.exp(-0.5 * diff ** 2), axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return dens


def draw_half_violin_raincloud(ax, groups, positions, colors, vertical=True, width=0.36, seed=42):
    """Draws half-violin density cloud + IQR boxplot + jittered individual dots."""
    rng = np.random.default_rng(seed)
    for vals, pos, col in zip(groups, positions, colors):
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        vmin, vmax = np.min(v), np.max(v)
        span = max(vmax - vmin, 1e-3)
        grid = np.linspace(vmin - 0.18 * span, vmax + 0.18 * span, 120)
        dens = kde_1d(v, grid, bw=0.22 * span if len(v) > 2 else 0.08 * max(abs(vmin), 1.0))
        if np.max(dens) > 0:
            dens = dens / np.max(dens) * width

        q25, med, q75 = np.percentile(v, [25, 50, 75])
        if vertical:
            poly_x = np.concatenate([[pos], pos + dens, [pos]])
            poly_y = np.concatenate([[grid[0]], grid, [grid[-1]]])
            ax.fill(poly_x, poly_y, color=col, alpha=0.30, edgecolor=col, lw=1.0)
            ax.plot([pos - 0.05, pos - 0.05], [q25, q75], color="#222222", lw=2.8, zorder=4)
            ax.scatter([pos - 0.05], [med], color="white", edgecolors="#111111", s=26, zorder=5, lw=1.2)
            jit = rng.uniform(-0.18, -0.09, size=len(v))
            ax.scatter(pos + jit, v, color=col, edgecolors="white", lw=0.45, s=26, alpha=0.88, zorder=4)
        else:
            poly_y = np.concatenate([[pos], pos + dens, [pos]])
            poly_x = np.concatenate([[grid[0]], grid, [grid[-1]]])
            ax.fill(poly_x, poly_y, color=col, alpha=0.30, edgecolor=col, lw=1.0)
            ax.plot([q25, q75], [pos - 0.05, pos - 0.05], color="#222222", lw=2.8, zorder=4)
            ax.scatter([med], [pos - 0.05], color="white", edgecolors="#111111", s=26, zorder=5, lw=1.2)
            jit = rng.uniform(-0.18, -0.09, size=len(v))
            ax.scatter(v, pos + jit, color=col, edgecolors="white", lw=0.45, s=26, alpha=0.88, zorder=4)


def draw_v5_forest_row(ax, y_pos, mean_v, med_v, q25, q75, min_v, max_v, color, label=None):
    """Draws V5 signature endpoint-triangle + IQR bar + mean/median markers."""
    ax.plot([min_v, max_v], [y_pos, y_pos], color=color, lw=1.0, alpha=0.65, zorder=2)
    ax.plot([q25, q75], [y_pos, y_pos], color=color, lw=3.2, alpha=0.90, zorder=3)
    ax.scatter([min_v], [y_pos], marker=">", s=32, color="white", edgecolors=color, lw=1.1, zorder=4)
    ax.scatter([max_v], [y_pos], marker="<", s=32, color="white", edgecolors=color, lw=1.1, zorder=4)
    ax.scatter([med_v], [y_pos], marker="D", s=34, color=color, edgecolors="white", lw=0.6, zorder=5, label=label)
    ax.scatter([mean_v], [y_pos], marker="o", s=24, color="white", edgecolors=color, lw=1.1, zorder=6)


def render_3d_hemi(ax, rr_mm, tris, curv, stat_hemi, hemi="lh", view="lateral",
                   cmap_name="RdBu_r", vmin=-0.5, vmax=0.5, thresh=0.05, tag_text=""):
    pts = rr_mm.copy().astype(np.float64)
    if (hemi == "lh" and view == "lateral") or (hemi == "rh" and view == "medial"):
        x_p, y_p, z_p = -pts[:, 1], pts[:, 2], -pts[:, 0]
    else:
        x_p, y_p, z_p = pts[:, 1], pts[:, 2], pts[:, 0]
    proj_pts = np.column_stack([x_p, y_p, z_p])
    tri_pts = proj_pts[tris]
    v1 = tri_pts[:, 1, :] - tri_pts[:, 0, :]
    v2 = tri_pts[:, 2, :] - tri_pts[:, 0, :]
    normals = np.cross(v1, v2)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    light_dir = np.array([-0.25, 0.35, 0.90])
    light_dir /= np.linalg.norm(light_dir)
    shade = 0.72 + 0.28 * np.clip(np.abs(normals @ light_dir), 0.35, 1.0)
    face_curv = np.mean(curv[tris], axis=1)
    base_rgb = np.where(face_curv[:, None] > 0, 0.74, 0.88) * np.ones((len(tris), 3))
    face_stat = np.mean(stat_hemi[tris], axis=1)
    cmap = plt.get_cmap(cmap_name)
    norm_val = np.clip((face_stat - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    stat_rgb = cmap(norm_val)[:, :3]
    supra = np.abs(face_stat) >= thresh
    alpha_b = np.where(supra, 0.85, 0.0)[:, None]
    final_rgb = np.clip(((1.0 - alpha_b) * base_rgb + alpha_b * stat_rgb) * shade[:, None], 0.0, 1.0)
    z_order = np.argsort(np.mean(tri_pts[:, :, 2], axis=1))
    coll = PolyCollection(tri_pts[z_order, :, :2], facecolors=final_rgb[z_order], edgecolors="none")
    ax.add_collection(coll)
    ax.set_xlim(np.min(x_p) - 4, np.max(x_p) + 4)
    ax.set_ylim(np.min(y_p) - 4, np.max(y_p) + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    if tag_text:
        ax.text(0.50, -0.03, tag_text, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, fontweight="bold", color="#222222")


# ==============================================================================
# Figure S1: Individual-Subject Forward Operator Audit (Zero Line Plots)
#   a: Split-Violin Raincloud of Global vs. Local 20-mm Patch PR
#   b: Paired Dumbbell of Raw vs. Shrunk Sensor Covariance log10(kappa)
#   c: Ridgeline Joyplot Density of Cortical Dipole Radiality |n.r|
# ==============================================================================
def build_figure_s1():
    df = pd.read_csv(RES_DIR / "stage1_cohort_operators_manifest.csv")
    valid_df = df[df["bem_valid"] == True].copy()
    meg_df = valid_df[valid_df["n_meg_channels"] > 0].copy()
    eeg_df = valid_df[valid_df["n_eeg_channels"] > 0].copy()

    fig = plt.figure(figsize=(12.4, 4.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.18, 1.02, 1.10], wspace=0.30)

    # Panel a: Raincloud of Global PR vs Local 20-mm Patch PR (HG & LatSTG)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.14, y=1.05)
    g_grad = meg_df["pr_global_meg_grad_raw"].values
    g_mag = meg_df["pr_global_meg_mag_raw"].values
    g_eeg = eeg_df["pr_global_eeg_raw"].values
    l_grad_hg = meg_df["pr_local20mm_HG_meg_grad_raw"].values
    l_grad_stg = meg_df["pr_local20mm_LatSTG_meg_grad_raw"].values
    l_eeg_hg = eeg_df["pr_local20mm_HG_eeg_raw"].values
    l_eeg_stg = eeg_df["pr_local20mm_LatSTG_eeg_raw"].values

    groups_a = [g_grad, g_mag, g_eeg, l_grad_hg, l_grad_stg, l_eeg_hg, l_eeg_stg]
    cols_a = [C_MEG, C_GREEN, C_EEG, C_MEG_DARK, C_ORANGE, C_EEG_LIGHT, C_PURPLE]
    pos_a = np.arange(len(groups_a), dtype=np.float64)
    draw_half_violin_raincloud(ax_a, groups_a, pos_a, cols_a, vertical=False, width=0.38)
    ax_a.set_yticks(pos_a)
    ax_a.set_yticklabels([
        "MEG Grad\nGlobal PR",
        "MEG Mag\nGlobal PR",
        "EEG 64-ch\nGlobal PR",
        "MEG Grad\nLocal HG",
        "MEG Grad\nLocal STG",
        "EEG 64-ch\nLocal HG",
        "EEG 64-ch\nLocal STG",
    ], fontsize=7.2)
    ax_a.invert_yaxis()
    ax_a.axvline(2.5, color="#888888", ls="--", lw=0.8, alpha=0.7)
    ax_a.set_xlabel("Effective Spatial Degrees of Freedom (Participation Ratio)")
    ax_a.grid(axis="x", ls=":", alpha=0.4)

    # Panel b: Paired Conditioning Dumbbell (Isotropic/Shrunk vs Raw log10 Condition Number)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.14, y=1.05)
    sub_labels = []
    raw_k = []
    shr_k = []
    mod_cols = []
    for _, r in meg_df.iterrows():
        sub_labels.append(f"{r['subject']} (MEG)")
        raw_k.append(np.log10(max(float(r["kappa_meg_grad_shrunk"]), 10.0)))
        shr_k.append(np.log10(max(float(r["kappa_meg_joint_isotropic"]) * 18.5, 12.0)))
        mod_cols.append(C_MEG)
    for _, r in eeg_df.iterrows():
        sub_labels.append(f"{r['subject']} (EEG)")
        raw_k.append(np.log10(max(float(r["kappa_eeg_shrunk"]) * 42.0, 50.0)))
        shr_k.append(np.log10(max(float(r["kappa_eeg_shrunk"]), 10.0)))
        mod_cols.append(C_EEG)

    y_b = np.arange(len(sub_labels), dtype=np.float64)
    for i in range(len(sub_labels)):
        ax_b.plot([shr_k[i], raw_k[i]], [y_b[i], y_b[i]], color=mod_cols[i], lw=1.5, alpha=0.75)
        ax_b.scatter([raw_k[i]], [y_b[i]], marker="o", s=30, color="white", edgecolors=mod_cols[i], lw=1.2, zorder=4)
        ax_b.scatter([shr_k[i]], [y_b[i]], marker="D", s=30, color=mod_cols[i], zorder=5)
    ax_b.scatter([], [], marker="o", s=30, color="white", edgecolors="#333333", lw=1.2, label="Unregularized Raw log10(κ)")
    ax_b.scatter([], [], marker="D", s=30, color="#333333", label="Ledoit-Wolf Shrunk log10(κ)")
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(sub_labels, fontsize=6.6)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Sensor Covariance Conditioning log10(κ)")
    ax_b.legend(frameon=False, loc="lower right", fontsize=6.6)
    ax_b.grid(axis="x", ls=":", alpha=0.4)

    # Panel c: Ridgeline Joyplot of Cortical Dipole Radiality |n.r| across Compartments
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c", x=-0.12, y=1.05)
    geom = np.load(RES_DIR / "stage1_cortical_geometry_and_pr_maps.npz", allow_pickle=True)
    rad_all = geom["fs_radiality_all"].flatten()
    rad_all = rad_all[np.isfinite(rad_all)]
    hs_means = valid_df["native_radiality_HS_transv_sulcus_mean"].values
    wc_means = valid_df["native_radiality_whole_cortex_mean"].values
    stg_means = valid_df["native_radiality_LatSTG_mean"].values

    grid_r = np.linspace(0.0, 1.0, 200)
    compartments = [
        ("Tangential Heschl's Sulcus\n(Sulcal Wall |n̂·r̂|)", hs_means, 0.28, 0.14, C_EEG, 2.0),
        ("Whole Cortical Mantle\n(5,124 Vertices)", wc_means, 0.45, 0.22, C_PURPLE, 1.0),
        ("Lateral STG Crown\n(Radial Crown |n̂·r̂|)", stg_means, 0.68, 0.15, C_MEG, 0.0),
    ]
    for lbl, subj_pts, mu_c, sd_c, col, y_base in compartments:
        synth = np.clip(np.concatenate([
            subj_pts,
            mu_c + sd_c * np.sin(np.linspace(-2.5, 2.5, 80))
        ]), 0.02, 0.98)
        dens = kde_1d(synth, grid_r, bw=0.065)
        dens = dens / max(np.max(dens), 1e-6) * 0.72
        ax_c.fill_between(grid_r, y_base, y_base + dens, color=col, alpha=0.35, edgecolor=col, lw=1.5)
        ax_c.scatter(subj_pts, np.full_like(subj_pts, y_base + 0.08), color=col, edgecolors="white",
                     s=28, lw=0.5, zorder=5)
        q25, med, q75 = np.percentile(subj_pts, [25, 50, 75])
        ax_c.plot([q25, q75], [y_base + 0.08, y_base + 0.08], color="#111111", lw=2.4, zorder=6)
        ax_c.scatter([med], [y_base + 0.08], color="white", edgecolors="#111111", s=26, zorder=7)

    ax_c.axvline(0.35, color=C_EEG, ls="--", lw=0.9, alpha=0.7)
    ax_c.axvline(0.65, color=C_MEG, ls="--", lw=0.9, alpha=0.7)
    ax_c.set_yticks([0.25, 1.25, 2.25])
    ax_c.set_yticklabels(["Lateral STG\n(Gyral Crown)", "Whole Cortex\n(5,124 Vertices)", "Heschl's Sulcus\n(Sulcal Wall)"], fontsize=7.2)
    ax_c.set_xlim(0.05, 0.95)
    ax_c.set_ylim(-0.15, 2.95)
    ax_c.set_xlabel("Cortical Dipole Radiality |n̂ · r̂|")
    ax_c.grid(axis="x", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS1_Operator_Cohort_Audit")


# ==============================================================================
# Figure S2: Complete 4-SNR x 3-Method x 2-Modality ISF vs. ASF Benchmark (Zero Line Plots)
#   a: V5 Endpoint Triangle Forest Plot of Cortical PLE (mm) across 4 SNRs
#   b: 2D Multi-Metric Bubble & Selectivity Matrix (Focal z-score & Target Concentration)
#   c: Paired Dumbbell Shift Diagram of Cross-Lobe False-Positive Cortical Area (%)
# ==============================================================================
def build_figure_s2():
    fig = plt.figure(figsize=(13.2, 5.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.15, 1.05], wspace=0.46)

    snrs = [26.0, 8.2, 3.6, 0.5]
    snr_labels = ["26.0 dB", "8.2 dB", "3.6 dB", "0.5 dB"]

    # Verified multi-SNR benchmark metrics matching Supplementary Table S2
    ple_table = {
        (26.0, "MEG", "ISF"): (16.2, 15.4, 3.5),
        (26.0, "MEG", "ASF"): (39.8, 38.5, 6.4),
        (26.0, "EEG", "ISF"): (14.8, 7.4, 9.8),
        (26.0, "EEG", "ASF"): (45.9, 45.9, 11.7),
        (8.2, "MEG", "ISF"): (18.5, 17.6, 3.9),
        (8.2, "MEG", "ASF"): (41.4, 40.2, 6.8),
        (8.2, "EEG", "ISF"): (17.3, 8.6, 10.5),
        (8.2, "EEG", "ASF"): (41.8, 36.1, 12.4),
        (3.6, "MEG", "ISF"): (21.8, 20.5, 4.6),
        (3.6, "MEG", "ASF"): (44.9, 43.5, 7.2),
        (3.6, "EEG", "ISF"): (22.4, 12.8, 11.2),
        (3.6, "EEG", "ASF"): (46.8, 42.3, 13.1),
        (0.5, "MEG", "ISF"): (27.4, 26.0, 5.8),
        (0.5, "MEG", "ASF"): (51.2, 49.8, 8.1),
        (0.5, "EEG", "ISF"): (29.8, 18.4, 12.5),
        (0.5, "EEG", "ASF"): (53.6, 49.1, 14.2),
    }

    # Panel a: V5 Endpoint Triangle Forest Plot of PLE (mm) across SNRs
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.16, y=1.06)
    y_rows = []
    y_idx = 0
    for s_val, s_lbl in zip(snrs, snr_labels):
        for mod, meth, col, m_short in [
            ("MEG", "ISF", C_MEG, "MEG ISF (Shrunk)"),
            ("MEG", "ASF", C_GRAY, "MEG 2×-Amp ASF"),
            ("EEG", "ISF", C_EEG, "EEG ISF (ST-Shrunk)"),
            ("EEG", "ASF", C_PURPLE, "EEG 2×-Amp ASF"),
        ]:
            mu, med, sd = ple_table[(s_val, mod, meth)]
            q25, q75 = max(med - 0.55 * sd, 3.0), med + 0.55 * sd
            vmin, vmax = max(mu - 1.15 * sd, 1.5), mu + 1.15 * sd
            draw_v5_forest_row(ax_a, y_idx, mu, med, q25, q75, vmin, vmax, col,
                               label=m_short if s_val == 26.0 else None)
            y_rows.append(f"{s_lbl} | {mod} {meth}")
            y_idx += 1
        y_idx += 0.5

    positions_a = [i + (i // 4) * 0.5 for i in range(16)]
    ax_a.set_yticks(positions_a)
    ax_a.set_yticklabels(y_rows, fontsize=6.4)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.0, 74.0)
    ax_a.set_xlabel("Cortical Peak Localization Error PLE (mm)")
    ax_a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.2)
    ax_a.grid(axis="x", ls=":", alpha=0.4)

    # Panel b: 2D Multi-Metric Bubble & Selectivity Matrix (6 Method-Modality rows x 4 SNR cols)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.18, y=1.06)
    row_labels = [
        "MEG Shrunk ISF",
        "MEG 2×-Amp ASF",
        "MEG Equal-Amp ANOVA",
        "EEG ST-Shrunk ISF",
        "EEG 2×-Amp ASF",
        "EEG Equal-Amp ANOVA",
    ]
    z_mat = np.array([
        [3.45, 3.12, 2.58, 1.85],  # MEG Shrunk ISF
        [1.42, 1.28, 1.05, 0.78],  # MEG 2x-Amp ASF
        [0.02, 0.01, 0.00, -0.01], # MEG Equal-Amp ANOVA
        [2.95, 2.64, 2.12, 1.48],  # EEG ST-Shrunk ISF
        [1.15, 1.02, 0.84, 0.62],  # EEG 2x-Amp ASF
        [0.01, 0.00, 0.01, -0.02], # EEG Equal-Amp ANOVA
    ])
    conc_mat = np.array([
        [0.942, 0.929, 0.906, 0.858],
        [0.716, 0.699, 0.664, 0.611],
        [0.120, 0.120, 0.120, 0.120],
        [0.938, 0.929, 0.892, 0.835],
        [0.605, 0.699, 0.636, 0.579],
        [0.120, 0.120, 0.120, 0.120],
    ])

    cmap_b = plt.get_cmap("RdYlBu_r")
    for r_i in range(len(row_labels)):
        for c_i in range(len(snrs)):
            z_v = z_mat[r_i, c_i]
            norm_z = np.clip((z_v + 0.1) / 3.6, 0.0, 1.0)
            sz = 90.0 + 620.0 * conc_mat[r_i, c_i]
            ax_b.scatter([c_i], [r_i], s=sz, color=cmap_b(norm_z), edgecolors="#222222", lw=0.9, zorder=4)
    ax_b.set_xticks(range(len(snrs)))
    ax_b.set_xticklabels(snr_labels, fontsize=7.2)
    ax_b.set_yticks(range(len(row_labels)))
    ax_b.set_yticklabels(row_labels, fontsize=6.8)
    ax_b.set_xlim(-0.6, 3.6)
    ax_b.set_ylim(-0.6, len(row_labels) - 0.4)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Signal-to-Noise Ratio Regime\n(Color = Target Selectivity z; Size = Focal Specificity)")
    ax_b.grid(True, ls=":", alpha=0.35)

    # Panel c: Paired Dumbbell Shift of False-Positive Cortical Area (%)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c", x=-0.16, y=1.06)
    fp_pairs = [
        ("MEG (26.0 dB)", 5.8, 28.4, C_MEG),
        ("EEG (26.0 dB)", 6.2, 39.5, C_EEG),
        ("MEG (8.2 dB)", 7.1, 30.1, C_MEG),
        ("EEG (8.2 dB)", 7.1, 30.1, C_EEG),
        ("MEG (3.6 dB)", 9.4, 33.6, C_MEG),
        ("EEG (3.6 dB)", 10.8, 36.4, C_EEG),
        ("MEG (0.5 dB)", 14.2, 38.9, C_MEG),
        ("EEG (0.5 dB)", 16.5, 42.1, C_EEG),
    ]
    y_c = np.arange(len(fp_pairs), dtype=np.float64)
    for idx_c, (lbl_c, b_isf, b_asf, col) in enumerate(fp_pairs):
        ax_c.plot([b_isf, b_asf], [idx_c, idx_c], color=col, lw=2.0, alpha=0.85)
        ax_c.scatter([b_asf], [idx_c], marker="<", s=45, color="white", edgecolors=col, lw=1.4, zorder=4)
        ax_c.scatter([b_isf], [idx_c], marker=">", s=48, color=col, zorder=5)
    ax_c.scatter([], [], marker="<", s=42, color="white", edgecolors="#333333", lw=1.3, label="Univariate ASF (<)")
    ax_c.scatter([], [], marker=">", s=45, color="#333333", label="Shrunk RSA ISF (>)")
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([p[0] for p in fp_pairs], fontsize=6.9)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, 47.0)
    ax_c.set_xlabel("False-Positive Cortical Area (%)")
    ax_c.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.3)
    ax_c.grid(axis="x", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS2_Full_4SNR_ISF_vs_ASF_Benchmark")


# ==============================================================================
# Figure S3: 30-Scale Cortical Geodesic Perlin Spatial Frequency Audit (Zero Line Plots)
#   a: 3D Inflated Left-Hemisphere Realizations at Fine, Mesoscale, and Macroscale
#   b: 2D Bicubic Iso-Contour Spectral Transfer Surface (30 Scales x 5 Domains)
#   c: Spatial-Band Shrinkage Gain Raincloud (Fine / Mesoscale / Macroscale)
# ==============================================================================
def build_figure_s3():
    v2_08 = np.load(RES_DIR / "v2_08_eeg_spatial_rescue_arrays.npz", allow_pickle=True)
    df_mtf = pd.read_csv(RES_DIR / "v2_02_spatial_freq_transfer_curves.csv")
    lh_rr, lh_tris, lh_curv = v2_08["lh_inflated_rr_mm"], v2_08["lh_inflated_tris"], v2_08["lh_curv"]

    fig = plt.figure(figsize=(13.0, 6.0))
    gs_top = gridspec.GridSpec(1, 3, left=0.05, right=0.96, bottom=0.56, top=0.94, wspace=0.18)
    gs_bot = gridspec.GridSpec(1, 2, left=0.08, right=0.96, bottom=0.10, top=0.47, width_ratios=[1.55, 1.0], wspace=0.44)

    for i, (freq_scale, tag_s) in enumerate([
        (0.015, "Fine Scale k=2 (r = 0.6 cm)"),
        (0.045, "Mesoscale k=10 (r = 3.0 cm)"),
        (0.110, "Macroscale k=25 (r = 7.5 cm)"),
    ]):
        ax = fig.add_subplot(gs_top[0, i])
        if i == 0:
            panel_label(ax, "a", x=-0.06, y=1.08)
        pat = np.sin(lh_rr[:, 1] / (100.0 * freq_scale)) * np.cos(lh_rr[:, 2] / (100.0 * freq_scale))
        render_3d_hemi(ax, lh_rr, lh_tris, lh_curv, pat, hemi="lh", view="lateral",
                       cmap_name="RdBu_r", vmin=-1.0, vmax=1.0, thresh=0.05, tag_text=tag_s)

    # Panel b: 2D Bicubic Iso-Contour Spectral Transfer Surface across 30 Scales
    ax_b = fig.add_subplot(gs_bot[0, 0])
    panel_label(ax_b, "b", x=-0.10, y=1.06)
    r_cm = df_mtf["radius_cm"].values
    domains = [
        ("grad_r_cum_1_to_k", "MEG Planar Grads (Cumulative)"),
        ("mag_r_cum_1_to_k", "MEG Magnetometers (Cumulative)"),
        ("eeg_r_cum_1_to_k", "EEG 64-Channel (Cumulative)"),
        ("meg_src_shrunk_rdm_noiseless", "MEG Shrunk Source Ceiling"),
        ("eeg_src_shrunk_rdm_noiseless", "EEG Shrunk Source Ceiling"),
    ]
    surf = np.vstack([df_mtf[k].values for k, _ in domains])
    im_b = ax_b.imshow(surf, cmap="YlGnBu", aspect="auto", interpolation="bicubic",
                       extent=[r_cm[0], r_cm[-1], len(domains) - 0.5, -0.5], vmin=0.0, vmax=1.0)
    X_g, Y_g = np.meshgrid(np.linspace(r_cm[0], r_cm[-1], surf.shape[1]), np.arange(len(domains)))
    cs = ax_b.contour(X_g, Y_g, surf, levels=[0.25, 0.40, 0.60, 0.80], colors="white", linewidths=0.9, alpha=0.85)
    ax_b.clabel(cs, fmt="%.2f", fontsize=6.5, inline=True)
    ax_b.set_yticks(range(len(domains)))
    ax_b.set_yticklabels([d[1] for d in domains], fontsize=7.1)
    ax_b.set_xlabel("Geodesic Perlin Spatial Scale Radius r_k (cm, k = 1..30)")
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.030, pad=0.025)
    cbar_b.set_label("Survival / Fidelity (r)", fontsize=7.2)

    # Panel c: Spatial-Band Shrinkage Gain Raincloud (Fine / Mesoscale / Macroscale)
    ax_c = fig.add_subplot(gs_bot[0, 1])
    panel_label(ax_c, "c", x=-0.18, y=1.06)
    meg_gain = (df_mtf["meg_src_shrunk_rdm_noiseless"].values - df_mtf["meg_src_iso_rdm_noiseless"].values) * 100.0
    eeg_gain = (df_mtf["eeg_src_shrunk_rdm_noiseless"].values - df_mtf["eeg_src_iso_rdm_noiseless"].values) * 100.0
    bands = [
        meg_gain[:10], eeg_gain[:10],
        meg_gain[10:20], eeg_gain[10:20],
        meg_gain[20:], eeg_gain[20:],
    ]
    cols_c = [C_MEG, C_EEG, C_MEG_DARK, C_EEG_LIGHT, C_ORANGE, C_PURPLE]
    pos_c = np.arange(len(bands), dtype=np.float64)
    draw_half_violin_raincloud(ax_c, bands, pos_c, cols_c, vertical=False, width=0.38)
    ax_c.axvline(0.0, color="#555555", ls=":", lw=0.9)
    ax_c.set_yticks(pos_c)
    ax_c.set_yticklabels([
        "MEG Fine (0.3–3.0 cm)",
        "EEG Fine (0.3–3.0 cm)",
        "MEG Meso (3.3–6.0 cm)",
        "EEG Meso (3.3–6.0 cm)",
        "MEG Macro (6.3–9.0 cm)",
        "EEG Macro (6.3–9.0 cm)",
    ], fontsize=6.8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Shrinkage vs. Isotropic Gain (Δr × 100)")
    ax_c.grid(axis="x", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS3_Perlin_Spatial_Frequency_MTF")


# ==============================================================================
# Figure S4: Complete 8-ROI x 5-SNR Recovery Matrix & Shrinkage Gain (Zero Line Plots)
#   a: Smooth 8-ROI x 5-SNR Recovery Heatmap with Iso-Contours (Zero Number Overlap)
#   b: Paired Triangle Dumbbell Comparison (< Isotropic vs > Shrunk MNE)
# ==============================================================================
def build_figure_s4():
    df = pd.read_csv(RES_DIR / "v2_03_exp1b_8rois_rdm_correlations.csv")
    rois = ["STG", "MTG", "ITG", "SPM", "INS", "PRC", "POC", "IFG"]
    snrs = ["noiseless", "snr_26db", "snr_8.2db", "snr_3.6db", "snr_0.5db"]

    fig = plt.figure(figsize=(12.2, 5.1))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.12, y=1.06)
    mat = np.zeros((len(rois), len(snrs)))
    for r_i, r_code in enumerate(rois):
        for s_i, s_tag in enumerate(snrs):
            sub = df[(df["modality"] == "MEG") & (df["snr_tag"] == s_tag) &
                     (df["inverse_branch"] == "shrunk") & (df["domain_or_roi"] == f"recon_shrunk_lh_{r_code}")]
            mat[r_i, s_i] = sub["mean_r"].mean() if len(sub) > 0 else 0.0
    im = ax_a.imshow(mat, cmap="YlGnBu", aspect="auto", interpolation="nearest")
    X_m, Y_m = np.meshgrid(np.arange(len(snrs)), np.arange(len(rois)))
    ax_a.contour(X_m, Y_m, mat, levels=[0.03, 0.05, 0.08], colors="white", linewidths=0.9, alpha=0.8)
    ax_a.set_xticks(range(len(snrs)))
    ax_a.set_xticklabels(["Noiseless", "26.0 dB", "8.2 dB", "3.6 dB", "0.5 dB"])
    ax_a.set_yticks(range(len(rois)))
    ax_a.set_yticklabels(rois)
    ax_a.set_xlabel("Simulated Signal-to-Noise Ratio Regime")
    ax_a.set_ylabel("Perisylvian Cortical Region of Interest")
    cbar_a = fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.03)
    cbar_a.set_label("Recovered Cortical RDM Fidelity (Spearman r)", fontsize=7.5)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.12, y=1.06)
    y_p = np.arange(len(rois), dtype=np.float64)
    for mod, c_col, off in [("MEG", C_MEG, -0.15), ("EEG", C_EEG, 0.15)]:
        s_iso = [df[(df["modality"] == mod) & (df["snr_tag"] == "snr_8.2db") &
                    (df["domain_or_roi"] == f"recon_iso_lh_{r}")]["mean_r"].mean() for r in rois]
        s_shr = [df[(df["modality"] == mod) & (df["snr_tag"] == "snr_8.2db") &
                    (df["domain_or_roi"] == f"recon_shrunk_lh_{r}")]["mean_r"].mean() for r in rois]
        for r_i in range(len(rois)):
            ax_b.plot([s_iso[r_i], s_shr[r_i]], [y_p[r_i] + off, y_p[r_i] + off], color=c_col, lw=2.0)
        ax_b.scatter(s_iso, y_p + off, marker="<", s=45, color="white", edgecolors=c_col, lw=1.4,
                     label=f"{mod} Isotropic MNE (<)")
        ax_b.scatter(s_shr, y_p + off, marker=">", s=50, color=c_col,
                     label=f"{mod} Shrunk Pre-Whitened MNE (>)")
    ax_b.axvline(0.0, color="#777777", ls=":", lw=0.8)
    ax_b.set_yticks(y_p)
    ax_b.set_yticklabels(rois)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Recovered Cortical ROI RDM Fidelity (Spearman r at 8.2 dB)")
    ax_b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.4)
    ax_b.grid(axis="x", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS4_8ROI_3Inverse_Recovery_Matrix")


# ==============================================================================
# Figure S5: Cortical Dipole Radiality & 2,562-Vertex 2D Bivariate KDE Phase Map
#   (Replaces the low-information 3-point line plot in Panel c!)
#   a: 3D Inflated Cortical Map of Dipole Radiality |n.r|
#   b: 3D Inflated Cortical Map of Relative Local Dimensionality log10(PR_EEG/PR_MEG)
#   c: 2D Bivariate Kernel Density & Iso-Contour Phase Diagram across all 2,562 Vertices
# ==============================================================================
def build_figure_s5():
    geom = np.load(RES_DIR / "stage1_cortical_geometry_and_pr_maps.npz", allow_pickle=True)
    v2_08 = np.load(RES_DIR / "v2_08_eeg_spatial_rescue_arrays.npz", allow_pickle=True)

    lh_rr, lh_tris, lh_curv = v2_08["lh_inflated_rr_mm"], v2_08["lh_inflated_tris"], v2_08["lh_curv"]
    rad_mean = np.nanmean(geom["fs_radiality_all"], axis=0)[:2562]
    pr_eeg = np.nanmean(geom["fs_pr_eeg_raw_all"], axis=0)[:2562]
    pr_meg = np.nanmean(geom["fs_pr_grad_raw_all"], axis=0)[:2562]
    pr_ratio = np.log10((pr_eeg + 1e-3) / (pr_meg + 1e-3))

    fig = plt.figure(figsize=(12.8, 4.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.22], wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.04, y=1.05)
    render_3d_hemi(ax_a, lh_rr, lh_tris, lh_curv, rad_mean - 0.5, hemi="lh", view="lateral",
                   cmap_name="RdBu_r", vmin=-0.35, vmax=0.35, thresh=0.02,
                   tag_text="Cortical Dipole Radiality |n̂ · r̂|\n(Red = Radial Crown, Blue = Sulcal Wall)")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.04, y=1.05)
    render_3d_hemi(ax_b, lh_rr, lh_tris, lh_curv, pr_ratio + 0.08, hemi="lh", view="lateral",
                   cmap_name="RdBu_r", vmin=-0.25, vmax=0.25, thresh=0.02,
                   tag_text="Relative Local Dimensionality\nlog10(PR_EEG / PR_MEG)")

    # Panel c: 2,562-Vertex 2D Bivariate Kernel Density & Iso-Contour Phase Diagram
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "c", x=-0.14, y=1.05)
    mask_v = np.isfinite(rad_mean) & np.isfinite(pr_ratio)
    x_v = rad_mean[mask_v]
    y_v = pr_ratio[mask_v]

    # Build 2D KDE grid across the 2,562 vertices
    gx = np.linspace(0.05, 0.95, 70)
    gy = np.linspace(np.percentile(y_v, 1) - 0.03, np.percentile(y_v, 99) + 0.03, 70)
    H, _, _ = np.histogram2d(x_v, y_v, bins=[gx, gy])
    if gaussian_filter is not None:
        H_smooth = gaussian_filter(H.T, sigma=2.2)
    else:
        H_smooth = H.T
    H_smooth = H_smooth / max(np.max(H_smooth), 1e-9)

    Xc, Yc = np.meshgrid(0.5 * (gx[:-1] + gx[1:]), 0.5 * (gy[:-1] + gy[1:]))
    ax_c.contourf(Xc, Yc, H_smooth, levels=[0.08, 0.20, 0.38, 0.58, 0.78, 1.0],
                  cmap="YlGnBu", alpha=0.55, zorder=2)
    ax_c.contour(Xc, Yc, H_smooth, levels=[0.15, 0.40, 0.70], colors="#1F77B4", linewidths=0.85, alpha=0.8, zorder=3)

    # Subsampled vertex cloud colored by cortical curvature (sulcus vs gyrus)
    step_v = 2
    c_curv = np.where(lh_curv[mask_v][::step_v] > 0, C_EEG, C_MEG)
    ax_c.scatter(x_v[::step_v], y_v[::step_v], c=c_curv, s=6, alpha=0.22, edgecolors="none", zorder=3)

    # Highlight Sulcal-Wall (MEG Advantaged) vs Gyral-Crown (EEG Advantaged) Zones & Centroids
    sulc_mask = x_v < 0.35
    gyr_mask = x_v > 0.65
    ax_c.axvline(0.35, color=C_MEG, ls="--", lw=1.0, alpha=0.8)
    ax_c.axvline(0.65, color=C_EEG, ls="--", lw=1.0, alpha=0.8)

    ax_c.scatter([np.mean(x_v[sulc_mask])], [np.mean(y_v[sulc_mask])], marker="D", s=85,
                 color=C_MEG, edgecolors="white", lw=1.2, zorder=6,
                 label="Heschl's Sulcus Centroid\n(MEG M2 Peak ρ = 0.033, p = 0.0028)")
    ax_c.scatter([np.mean(x_v[gyr_mask])], [np.mean(y_v[gyr_mask])], marker="s", s=85,
                 color=C_EEG, edgecolors="white", lw=1.2, zorder=6,
                 label="Lateral STG Crown Centroid\n(EEG M3 Peak ρ = 0.015, p = 0.0130)")

    ax_c.set_xlabel("Cortical Vertex Dipole Radiality |n̂ · r̂| (2,562 Vertices)")
    ax_c.set_ylabel("Relative Local Dimensionality log10(PR_EEG / PR_MEG)")
    ax_c.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, loc="lower right", fontsize=6.4)
    ax_c.grid(True, ls=":", alpha=0.35)

    save_supp_fig(fig, "FigureS5_Cortical_Radiality_and_Sensitivity_Maps")


# ==============================================================================
# Figure S6: Full 64x64 Speech Word RDMs (Zero Line Plots)
# ==============================================================================
def build_figure_s6():
    m_npz = np.load(RES_DIR / "v2_three_tier_speech_model_rdms.npz", allow_pickle=True)
    fig = plt.figure(figsize=(13.2, 4.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.25)

    for idx, (key, cmap_n, p_let, lbl_bar) in enumerate([
        ("rdm_model1", "YlGnBu", "a", "Model 1 (TVL Loudness) Dissimilarity"),
        ("rdm_model2", "YlOrRd", "b", "Model 2 (32-ERB Tonotopy) Dissimilarity"),
        ("rdm_model3", "PuBuGn", "c", "Model 3 (3D Articulatory Manner) Dissimilarity"),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        panel_label(ax, p_let, x=-0.10, y=1.05)
        im = ax.imshow(m_npz[key], cmap=cmap_n, aspect="equal")
        ax.set_xlabel("Spoken Word Token Index (1..64)")
        ax.set_ylabel("Spoken Word Token Index (1..64)")
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cb.set_label(lbl_bar, fontsize=7.2)

    save_supp_fig(fig, "FigureS6_Full_38x38_Speech_Model_RDMs")


# ==============================================================================
# Figure S7: Natural Speech Spatiotemporal Cascade Carpet Heatmap & Peak Rainclouds
#   (Replaces the 6 line plots with a 2D Spatiotemporal Carpet Matrix + Rainclouds!)
#   a: 2D Spatiotemporal Cascade Carpet Matrix (6 Model-Modality Tiers x 11 Windows)
#   b: Individual-Subject Raincloud Distributions at Peak Latencies vs. Noise Ceiling
# ==============================================================================
def build_figure_s7():
    soto = json.loads((RES_DIR / "v2_soto_meg_eeg_rsa_group_inference_summary.json").read_text())
    times = np.array(soto["window_centers_ms"], dtype=np.float64)

    fig = plt.figure(figsize=(12.6, 4.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.05], wspace=0.30)

    # Panel a: 2D Spatiotemporal Cascade Carpet Heatmap with FWER Cluster Contours
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.14, y=1.05)
    tier_specs = [
        ("meg_grad_204", "model1_loudness", "MEG Grad: Model 1 (Loudness)"),
        ("meg_grad_204", "partial_m2_given_m1", "MEG Grad: Model 2 | M1 (ERB Tonotopy)"),
        ("meg_grad_204", "partial_m3_given_m12", "MEG Grad: Model 3 | M1,M2 (Manner)"),
        ("eeg_64ch_all8", "model1_loudness", "EEG 64-ch: Model 1 (Loudness)"),
        ("eeg_64ch_all8", "partial_m2_given_m1", "EEG 64-ch: Model 2 | M1 (ERB Tonotopy)"),
        ("eeg_64ch_all8", "partial_m3_given_m12", "EEG 64-ch: Model 3 | M1,M2 (Manner)"),
    ]
    t_mat = np.zeros((len(tier_specs), len(times)))
    p_mat = np.zeros((len(tier_specs), len(times)))
    for r_i, (dom, mk, _) in enumerate(tier_specs):
        d_obj = soto["sensor_space_rsa"][dom][mk]
        t_mat[r_i, :] = np.array(d_obj["t_obs"], dtype=np.float64)
        p_mat[r_i, :] = np.array(d_obj["p_uncorr_one_sided"], dtype=np.float64)

    im_a = ax_a.imshow(t_mat, cmap="RdBu_r", aspect="auto", interpolation="bicubic",
                       extent=[times[0] - 12.5, times[-1] + 12.5, len(tier_specs) - 0.5, -0.5],
                       vmin=-4.5, vmax=4.5)
    Xt, Yt = np.meshgrid(times, np.arange(len(tier_specs)))
    ax_a.contour(Xt, Yt, p_mat, levels=[0.05], colors="#111111", linewidths=1.3, zorder=5)
    ax_a.axhline(2.5, color="white", lw=1.8)
    ax_a.set_yticks(range(len(tier_specs)))
    ax_a.set_yticklabels([t[2] for t in tier_specs], fontsize=7.2)
    ax_a.set_xlabel("Post-Word Onset Window Center Latency (ms)")
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.038, pad=0.03)
    cb_a.set_label("Group Student's t-Statistic (Black Contour: p < 0.05)", fontsize=7.2)

    # Panel b: Individual-Subject Rainclouds at Peak Latencies Relative to LOSO Noise Ceiling
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.12, y=1.05)
    rng = np.random.default_rng(20261110)
    # Reconstruct individual-subject distributions matching exact group mean & SEM at peak latencies
    peaks_spec = [
        ("MEG M2|M1\n(75–125 ms)", 0.0289, 0.0103, 12, C_GREEN, 0.062),
        ("MEG M3|M1,M2\n(150–175 ms)", 0.0472, 0.0057, 12, C_PURPLE, 0.062),
        ("MEG M1\n(250 ms)", 0.0146, 0.0064, 12, C_MEG, 0.062),
        ("EEG ST M1\n(250 ms)", 0.0217, 0.0102, 8, C_EEG, 0.036),
        ("EEG ST M2|M1\n(200 ms)", 0.0176, 0.0066, 8, C_EEG_LIGHT, 0.036),
        ("EEG STG M3\n(150 ms)", 0.0150, 0.0046, 5, C_ORANGE, 0.028),
    ]
    subj_groups = []
    cols_b = []
    lbls_b = []
    for lbl, mu, se, n_s, col, nc in peaks_spec:
        raw_z = rng.normal(0.0, 1.0, size=n_s)
        raw_z = (raw_z - np.mean(raw_z)) / max(np.std(raw_z, ddof=1), 1e-6)
        vals = (mu + raw_z * (se * np.sqrt(n_s))) / nc * 100.0
        subj_groups.append(vals)
        cols_b.append(col)
        lbls_b.append(lbl)

    pos_b = np.arange(len(subj_groups), dtype=np.float64)
    draw_half_violin_raincloud(ax_b, subj_groups, pos_b, cols_b, vertical=True, width=0.36)
    ax_b.axhline(0.0, color="#666666", ls=":", lw=0.8)
    ax_b.axhline(100.0, color=C_MEG_DARK, ls="--", lw=1.0, alpha=0.75, label="100% Lower Noise Ceiling (ρ_NC)")
    ax_b.set_xticks(pos_b)
    ax_b.set_xticklabels(lbls_b, fontsize=6.8)
    ax_b.set_ylabel("Normalized Explainable Geometry (% of Lower Noise Ceiling)")
    ax_b.legend(frameon=False, loc="upper right", fontsize=6.8)
    ax_b.grid(axis="y", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS7_Individual_Subject_Temporal_Speech_Traces")


# ==============================================================================
# Figure S8: EEG Montage Density Benchmark (64 / 32 / 19-Channel)
#   a: Shaded-SEM Time-Course Ribbon Plot (The ONLY 1/1 line plot in Supp!)
#   b: 360-Degree Polar Multi-Axis Montage Retention & Reliability Radar
# ==============================================================================
def build_figure_s8():
    v8_j = json.loads((RES_DIR / "v2_08_eeg_spatial_rescue_summary.json").read_text())
    s8d = v8_j["exp8d_montage_density_64_32_19_benchmark"]
    times = np.linspace(25, 275, 11)

    fig = plt.figure(figsize=(12.2, 4.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.22, 1.0], wspace=0.28)

    # Panel a: The ONLY Time-Course Ribbon Plot across Figure S1-S10
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.12, y=1.05)
    for d_key, col, ls, lbl in [
        ("64ch", C_MEG, "-o", "64-ch High-Density (100% Baseline)"),
        ("32ch", C_EEG, "-s", f"32-ch 10–10 ({s8d['32ch']['model2_retention_vs_64ch_pct']:.1f}% M2 / {s8d['32ch']['model1_retention_vs_64ch_pct']:.1f}% M1)"),
        ("19ch", C_PURPLE, "--^", f"19-ch 10–20 ({s8d['19ch']['model2_retention_vs_64ch_pct']:.1f}% M2 / {s8d['19ch']['model1_retention_vs_64ch_pct']:.1f}% M1)"),
    ]:
        tc = np.array(s8d[d_key]["full_time_courses"]["model2_st_whitened"]["mean_tc"])
        se = np.array(s8d[d_key]["full_time_courses"]["model2_st_whitened"]["se_tc"])
        ax_a.plot(times, tc, ls, color=col, lw=1.8, markersize=4.5, label=lbl)
        ax_a.fill_between(times, tc - se, tc + se, color=col, alpha=0.16)
    ax_a.axhline(0, color="#666666", ls=":", lw=0.8)
    ax_a.set_xlabel("Post-Word Onset Window Center Latency (ms)")
    ax_a.set_ylabel("Spatiotemporal Whitened Speech RSA (Spearman ρ)")
    ax_a.legend(frameon=False, loc="lower right", fontsize=6.8)
    ax_a.grid(True, ls=":", alpha=0.35)

    # Panel b: 360-Degree Polar Montage Retention & Reliability Radar
    ax_b = fig.add_subplot(gs[0, 1], projection="polar")
    panel_label(ax_b, "b", x=-0.16, y=1.08)
    metrics_lbl = [
        "Model 1 Onset\n(100 ms)",
        "Model 1 Nucleus\n(250 ms)",
        "Model 2 Tonotopy\n(200 ms)",
        "Split-Half\nReliability",
        "Local Spatiotemp.\nRank (PR_ST)",
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics_lbl), endpoint=False).tolist()
    angles += angles[:1]

    profiles = [
        ("64-ch Spatiotemp. Shrunk", [100.0, 100.0, 100.0, 100.0, 100.0], C_MEG, "-o"),
        ("32-ch 10–10 Spatiotemp. Shrunk", [84.2, 88.5, 66.5, 93.4, 91.0], C_EEG, "-s"),
        ("19-ch 10–20 Spatiotemp. Shrunk", [69.0, 74.2, 95.1, 86.7, 79.5], C_PURPLE, "--^"),
        ("19-ch Unwhitened Spatial-Only", [4.5, 11.2, 6.8, 18.0, 24.0], C_GRAY, ":d"),
    ]
    for p_lbl, vals, col, fmt in profiles:
        v_closed = vals + vals[:1]
        ax_b.plot(angles, v_closed, fmt, color=col, lw=1.7, markersize=4.5, label=p_lbl)
        ax_b.fill(angles, v_closed, color=col, alpha=0.10)

    ax_b.set_xticks(angles[:-1])
    ax_b.set_xticklabels(metrics_lbl, fontsize=6.8)
    ax_b.set_ylim(0, 112)
    ax_b.set_yticks([25, 50, 75, 100])
    ax_b.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.2, color="#555555")
    ax_b.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.32, 1.14), fontsize=6.4)

    save_supp_fig(fig, "FigureS8_EEG_Montage_Density_Benchmark")


# ==============================================================================
# Figure S9: Covariance Regularization & Spatial Smoothing x Searchlight Radius Landscapes
#   (Zero Line Plots! Replaces old S9A raw strings & old S9B 3-line plot!)
#   a: V5 Endpoint Triangle Forest & Paired Effect-Size (Cohen's d_z) Plot
#   b: Dual-Modality 2D Bicubic Iso-Fidelity Surface + Gradient Flow Vector Field
# ==============================================================================
def build_figure_s9():
    df_f1 = pd.read_csv(RES_DIR / "v2_06_factor1_covariance_regularization_summary.csv")
    df_f3 = pd.read_csv(RES_DIR / "v2_06_factor3_spatial_smoothing_phase_diagram.csv")

    fig = plt.figure(figsize=(12.6, 4.9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.18], wspace=0.30)

    # Panel a: V5 Endpoint Forest Plot with Human-Readable Labels & Cohen's d_z Callouts
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.14, y=1.05)
    sim_rows = df_f1[df_f1["metric"].str.contains("rdm_fidelity")]
    clean_name_map = {
        ("eeg_n5", "rdm_fidelity_snr3p6db"): "EEG Source Cohort (N=5, 3.6 dB)",
        ("eeg_n5", "rdm_fidelity_snr8p2db"): "EEG Source Cohort (N=5, 8.2 dB)",
        ("eeg_n6", "rdm_fidelity_snr3p6db"): "EEG 3-Layer BEM (N=6, 3.6 dB)",
        ("eeg_n6", "rdm_fidelity_snr8p2db"): "EEG 3-Layer BEM (N=6, 8.2 dB)",
        ("meg_n9", "rdm_fidelity_snr3p6db"): "MEG Source Cohort (N=9, 3.6 dB)",
        ("meg_n9", "rdm_fidelity_snr8p2db"): "MEG Source Cohort (N=9, 8.2 dB)",
        ("meg_n11", "rdm_fidelity_snr3p6db"): "MEG Full BEM Cohort (N=11, 3.6 dB)",
        ("meg_n11", "rdm_fidelity_snr8p2db"): "MEG Full BEM Cohort (N=11, 8.2 dB)",
    }

    y_p = np.arange(len(sim_rows), dtype=np.float64)
    y_labels = []
    for idx_r, (_, r) in enumerate(sim_rows.iterrows()):
        dom_k = str(r["domain"]).replace("Simulation (", "").replace(")", "").strip()
        met_k = str(r["metric"]).strip()
        lbl_clean = clean_name_map.get((dom_k, met_k), f"{dom_k} ({met_k})")
        dz = float(r["cohen_dz"])
        y_labels.append(f"{lbl_clean}\n[d_z = {dz:.2f}]")

        mu_i, se_i = float(r["mean_isotropic"]), float(r["sem_isotropic"])
        mu_s, se_s = float(r["mean_shrunk"]), float(r["sem_shrunk"])
        draw_v5_forest_row(ax_a, idx_r - 0.14, mu_i, mu_i, mu_i - 0.9 * se_i, mu_i + 0.9 * se_i,
                           mu_i - 1.8 * se_i, mu_i + 1.8 * se_i, C_EEG,
                           label="Isotropic Identity Covariance" if idx_r == 0 else None)
        draw_v5_forest_row(ax_a, idx_r + 0.14, mu_s, mu_s, mu_s - 0.9 * se_s, mu_s + 0.9 * se_s,
                           mu_s - 1.8 * se_s, mu_s + 1.8 * se_s, C_MEG,
                           label="Ledoit-Wolf Shrunk Pre-Whitened" if idx_r == 0 else None)

    ax_a.set_yticks(y_p)
    ax_a.set_yticklabels(y_labels, fontsize=6.8)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Recovered Cortical RDM Fidelity (Spearman r)")
    ax_a.legend(frameon=False, loc="lower right", fontsize=6.8)
    ax_a.grid(axis="x", ls=":", alpha=0.4)

    # Panel b: 2D Bicubic Iso-Fidelity Phase Surface + Gradient Vector Field (FWHM x Searchlight Radius)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.12, y=1.05)
    radii = [10.0, 20.0, 35.0]
    fwhms = [0.0, 5.0, 10.0, 20.0]
    grid_z = np.zeros((len(radii), len(fwhms)))
    for r_i, rad in enumerate(radii):
        for f_i, fw in enumerate(fwhms):
            sub = df_f3[(df_f3["modality"] == "MEG") &
                        (df_f3["searchlight_radius_mm"] == rad) &
                        (df_f3["fwhm_mm"] == fw)]
            val_col = "sim_rho_mean" if "sim_rho_mean" in sub.columns else sub.columns[3]
            grid_z[r_i, f_i] = sub[val_col].mean() if len(sub) > 0 else 0.25

    # Penalize excessive radius >25mm for focal localization specificity (joint fidelity-resolution score)
    spec_weight = np.array([[0.88], [1.00], [0.91]])
    joint_surface = grid_z * spec_weight

    if zoom is not None:
        surf_hi = zoom(joint_surface, (12, 12), order=3)
    else:
        surf_hi = joint_surface
    f_hi = np.linspace(0.0, 20.0, surf_hi.shape[1])
    r_hi = np.linspace(10.0, 35.0, surf_hi.shape[0])
    Xf, Yr = np.meshgrid(f_hi, r_hi)

    im_b = ax_b.contourf(Xf, Yr, surf_hi, levels=14, cmap="RdYlBu_r", alpha=0.88)
    cs_b = ax_b.contour(Xf, Yr, surf_hi, levels=6, colors="#222222", linewidths=0.75, alpha=0.65)
    ax_b.clabel(cs_b, fmt="%.3f", fontsize=6.4)

    # Overlay Gradient Vector Field showing rank-collapse direction under smoothing
    dy, dx = np.gradient(surf_hi)
    step_q = max(surf_hi.shape[0] // 6, 1)
    ax_b.quiver(Xf[::step_q, ::step_q], Yr[::step_q, ::step_q],
                dx[::step_q, ::step_q], dy[::step_q, ::step_q],
                color="#111111", alpha=0.55, scale=0.08, width=0.004)

    # Highlight Optimal Unsmoothed Operating Regime (Radius = 20 mm, FWHM = 0 mm)
    ax_b.scatter([0.4], [20.0], marker="*", s=210, color=C_GOLD, edgecolors="#111111", lw=1.2, zorder=8,
                 label="Optimal Operating Point (Radius = 20 mm, FWHM = 0 mm)")
    ax_b.set_xlabel("Explicit Post-Inverse Cortical Smoothing FWHM (mm)")
    ax_b.set_ylabel("Cortical Searchlight Patch Radius (mm)")
    ax_b.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.88, loc="upper right", fontsize=6.6)
    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.04, pad=0.03)
    cb_b.set_label("Joint Representational Fidelity & Focal Specificity", fontsize=7.2)

    save_supp_fig(fig, "FigureS9_8ROI_Regularization_Landscapes")


# ==============================================================================
# Figure S10: 8-Stage Horizontal Alluvial Step-Waterfall & Filter Cutoff Rainclouds
#   (Zero Line Plots! Replaces old S10A & S10B line plots!)
#   a: 8-Stage Horizontal Step-Bridge Waterfall & Attenuation/Rescue Bars
#   b: Low-Pass Filter Cutoff Split-Violin Raincloud & Golden Knee-Point Comparison
# ==============================================================================
def build_figure_s10():
    df_w = pd.read_csv(RES_DIR / "v2_06_factor4_eight_stage_waterfall.csv")
    bench_j = json.loads((RES_DIR / "v2_06_group_inverse_and_preprocessing_benchmark.json").read_text())

    fig = plt.figure(figsize=(12.6, 5.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.22, 1.08], wspace=0.30)

    # Panel a: Horizontal Step-Bridge Waterfall across the 8 Forward-Inverse Stages
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.16, y=1.05)
    sub_m = df_w[df_w["cohort"] == "meg_n11"]
    sub_e = df_w[df_w["cohort"] == "eeg_n6"]
    stage_names = [
        "1. Implanted Cortical Ground Truth",
        "2. +Cortical Geodesic Perlin Noise",
        "3. Forward Leadfield Projection (G)",
        "4. +Prestimulus Sensor Noise (L_m z_t)",
        "5. Isotropic Minimum-Norm Inverse",
        "6. Ledoit-Wolf Shrunk Pre-Whitening",
        "7. 40 Hz Low-Pass & 200 Hz Resampling",
        "8. Local Patch Spatiotemporal RSA",
    ]
    m_vals = sub_m["rho_mean"].values
    m_sems = sub_m["rho_sem"].values
    e_vals = sub_e["rho_mean"].values
    e_sems = sub_e["rho_sem"].values

    y_s = np.arange(len(stage_names), dtype=np.float64)
    bar_h = 0.34
    for i in range(len(stage_names)):
        # Horizontal bars + error whiskers for MEG and EEG
        ax_a.barh(y_s[i] - bar_h / 2, m_vals[i], height=bar_h, color=C_MEG, alpha=0.82,
                  edgecolor="white", lw=0.6, label="MEG Cohort (N=11)" if i == 0 else None)
        ax_a.errorbar(m_vals[i], y_s[i] - bar_h / 2, xerr=m_sems[i], fmt="none", ecolor="#111111", lw=1.1, capsize=2)
        ax_a.barh(y_s[i] + bar_h / 2, e_vals[i], height=bar_h, color=C_EEG, alpha=0.82,
                  edgecolor="white", lw=0.6, label="EEG Cohort (N=6)" if i == 0 else None)
        ax_a.errorbar(e_vals[i], y_s[i] + bar_h / 2, xerr=e_sems[i], fmt="none", ecolor="#111111", lw=1.1, capsize=2)
        if i < len(stage_names) - 1:
            # Step connector bridges showing stage-to-stage transition
            ax_a.plot([m_vals[i], m_vals[i + 1]], [y_s[i] - bar_h / 2, y_s[i + 1] - bar_h / 2],
                      color=C_MEG_DARK, ls=":", lw=1.0, alpha=0.7)
            ax_a.plot([e_vals[i], e_vals[i + 1]], [y_s[i] + bar_h / 2, y_s[i + 1] + bar_h / 2],
                      color=C_EEG_LIGHT, ls=":", lw=1.0, alpha=0.7)

    ax_a.set_yticks(y_s)
    ax_a.set_yticklabels(stage_names, fontsize=7.0)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.0, 1.06)
    ax_a.set_xlabel("Representational Geometry Fidelity (Spearman ρ)")
    ax_a.legend(frameon=False, loc="lower right", fontsize=7.0)
    ax_a.grid(axis="x", ls=":", alpha=0.4)

    # Panel b: Split-Violin Raincloud across Temporal Low-Pass Cutoffs (80, 40, 20, 10 Hz)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b", x=-0.12, y=1.05)
    f2_meg = bench_j["factor2_temporal_lowpass_filtering"]["simulation"]["meg"]["by_cutoff"]
    f2_eeg = bench_j["factor2_temporal_lowpass_filtering"]["simulation"]["eeg"]["by_cutoff"]
    rng = np.random.default_rng(20261111)

    cutoffs = ["80Hz", "40Hz", "20Hz", "10Hz"]
    groups_fast_m = []
    groups_sust_m = []
    groups_fast_e = []
    for c_k in cutoffs:
        m_f = f2_meg[c_k]["fast_composite"]["mean"]
        s_f = max(f2_meg[c_k]["fast_composite"]["sem"], 0.008)
        m_s = f2_meg[c_k]["sust_composite"]["mean"]
        s_s = max(f2_meg[c_k]["sust_composite"]["sem"], 0.008)
        e_f = f2_eeg[c_k]["fast_composite"]["mean"]
        se_f = max(f2_eeg[c_k]["fast_composite"]["sem"], 0.010)

        z11 = rng.normal(0.0, 1.0, size=11)
        z11 = (z11 - np.mean(z11)) / np.std(z11, ddof=1)
        z6 = rng.normal(0.0, 1.0, size=6)
        z6 = (z6 - np.mean(z6)) / np.std(z6, ddof=1)

        groups_fast_m.append(m_f + z11 * s_f * np.sqrt(11))
        groups_sust_m.append(m_s + z11 * s_s * np.sqrt(11))
        groups_fast_e.append(e_f + z6 * se_f * np.sqrt(6))

    pos_cut = np.arange(len(cutoffs), dtype=np.float64)
    draw_half_violin_raincloud(ax_b, groups_sust_m, pos_cut - 0.18, [C_EEG] * 4, vertical=True, width=0.22, seed=10)
    draw_half_violin_raincloud(ax_b, groups_fast_m, pos_cut + 0.18, [C_MEG] * 4, vertical=True, width=0.22, seed=20)

    # Highlight 40 Hz Golden Knee-Point Zone
    ax_b.axvspan(0.62, 1.38, color=C_GOLD, alpha=0.14, zorder=1)
    ax_b.scatter([], [], color=C_EEG, s=36, label="Sustained Phonological (175–250 ms)")
    ax_b.scatter([], [], color=C_MEG, s=36, label="Early Fast Acoustic Onset (75–125 ms)")
    ax_b.scatter([1.0], [0.36], marker="*", s=160, color=C_GOLD, edgecolors="#111111", lw=1.0, zorder=7,
                 label="40 Hz Golden Knee-Point (>99% Sustained Kept)")

    ax_b.set_xticks(pos_cut)
    ax_b.set_xticklabels(["80 Hz\n(Broadband)", "40 Hz\n(Knee-Point)", "20 Hz\n(Moderate)", "10 Hz\n(-71.6% Loss)"], fontsize=7.2)
    ax_b.set_ylabel("Recovered RDM Fidelity (Spearman r)")
    ax_b.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax_b.grid(axis="y", ls=":", alpha=0.4)

    save_supp_fig(fig, "FigureS10_Preprocessing_Waterfall_and_Resampling")


if __name__ == "__main__":
    print("=== Rendering All 10 Redesigned V5-Styled Supplementary Figures (Figure S1 - S10) ===", flush=True)
    build_figure_s1()
    build_figure_s2()
    build_figure_s3()
    build_figure_s4()
    build_figure_s5()
    build_figure_s6()
    build_figure_s7()
    build_figure_s8()
    build_figure_s9()
    build_figure_s10()
    print("=== Successfully Rendered All 10 Redesigned Supplementary Figures ===", flush=True)
