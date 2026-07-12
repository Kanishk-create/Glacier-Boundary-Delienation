"""
===========================================================
  GlacioWatch: Tiled Comparative Satellite Classifier
===========================================================

Loads Sentinel-2 JP2 band images, divides the 1830x1830 grid
into a 50-tile grid configuration (10 rows × 5 cols = 50 tiles),
and runs multiclass land-type classification.

Per-tile output = ONE composite figure with 4 panels that
exactly match the 4 reference visualisation styles:

  PANEL A (top-left)  — Side-by-side: Natural Color RGB imshow
                         + Land Type Classification imshow
                         (matches Image 4: "Hyperspectral Image – Tyrol, Austria")

  PANEL B (top-right) — Hyperspectral Land Cover Classification Map
                         (matches Image 3: scatter coloured by class, legend,
                          stats annotation, Easting/Northing labels)

  PANEL C (bottom-left) — 3D Spatial-Spectral Visualization
                           (matches Image 2: X/Y/Band_1 Reflectance axes,
                            light background, full colour class spread)

  PANEL D (bottom-right) — Per-Class Distribution Maps grid
                            (matches Image 1 bottom row: grey background +
                             highlighted class pixels, 4 columns)
"""

import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import torch first to avoid WinError 1114 DLL loading conflicts
import torch
import torchvision.transforms.functional as F
import segmentation_models_pytorch as smp
import cv2

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import seaborn as sns
import joblib

sys.path.insert(0, os.path.abspath("src"))

# ── Output directories ──────────────────────────────────────────────────────
OUTPUT_DIR_1 = "website/public/images/output"
OUTPUT_DIR_2 = "output/images"
for d in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
    os.makedirs(os.path.join(d, "tiles"), exist_ok=True)

# ── Band directory (60 m) ───────────────────────────────────────────────────
BAND_DIR = (
    "processed_data/extracted/"
    "S2C_MSIL2A_20260423T053021_N0512_R105_T43RGN_20260423T084910.SAFE/"
    "GRANULE/L2A_T43RGN_A008512_20260423T053343/IMG_DATA/R60m"
)

# ── Models ──────────────────────────────────────────────────────────────────
MODEL_DIR = "output/model"
scaler   = joblib.load(f"{MODEL_DIR}/standard_scaler.pkl")
pca      = joblib.load(f"{MODEL_DIR}/pca_10components.pkl")
gb_model = joblib.load(f"{MODEL_DIR}/best_binary_glacier.pkl")
mc_model = joblib.load(f"{MODEL_DIR}/best_multiclass_model.pkl")

# Load U-Net model (PyTorch ResNet18 based)
unet_model = smp.Unet(encoder_name="resnet18", classes=1, encoder_weights=None)
unet_model.load_state_dict(torch.load("unet_model_augmented.pth", map_location="cpu"))
unet_model.eval()


# ── Band loader ─────────────────────────────────────────────────────────────
def load_band(band_name: str) -> np.ndarray:
    try:
        import rasterio
    except ImportError:
        print("[Error] rasterio not found.")
        sys.exit(1)

    filename = f"T43RGN_20260423T053021_{band_name}_60m.jp2"
    filepath = os.path.join(BAND_DIR, filename)
    if not os.path.exists(filepath):
        for root, _, files in os.walk(BAND_DIR):
            for f in files:
                if band_name in f and f.endswith(".jp2"):
                    filepath = os.path.join(root, f)
                    break
    print(f"  Loading {band_name} …")
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(float)
        return np.clip((data - 1000.0) / 10000.0, 0.0, 1.0)


# ── Pixel feature builder + classifier ─────────────────────────────────────
def build_pixel_data(t_b02, t_b03, t_b04, t_b8a, t_b11, t_b12,
                     s2_wl, target_wl, x0, y0, sample_rate):
    """Sub-sample bands, interpolate spectra, classify, return pixel dict."""
    b02s = t_b02[::sample_rate, ::sample_rate]
    b03s = t_b03[::sample_rate, ::sample_rate]
    b04s = t_b04[::sample_rate, ::sample_rate]
    b8as = t_b8a[::sample_rate, ::sample_rate]
    b11s = t_b11[::sample_rate, ::sample_rate]
    b12s = t_b12[::sample_rate, ::sample_rate]

    sh_s, sw_s = b02s.shape
    n = sh_s * sw_s

    # Pixel coordinates (tile-global image space)
    p_y = np.repeat(np.arange(sh_s), sw_s) * sample_rate + y0
    p_x = np.tile(np.arange(sw_s), sh_s)  * sample_rate + x0

    # Spectral interpolation
    flat = [b02s.ravel(), b03s.ravel(), b04s.ravel(),
            b8as.ravel(), b11s.ravel(), b12s.ravel()]
    X = np.zeros((n, len(target_wl)), dtype=np.float32)
    for i in range(n):
        s2_vals = np.array([f[i] for f in flat])
        X[i] = np.interp(target_wl, s2_wl, s2_vals)

    land_types = mc_model.predict(scaler.transform(X))

    # RGB colour per pixel (boosted reflectance, for scatter)
    rgb = np.clip(np.stack([flat[2], flat[1], flat[0]], axis=1) * 4.0, 0, 1)

    # Predict with U-Net
    img_rgb = np.stack([t_b04, t_b03, t_b02], axis=0) # shape (3, H, W)
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32)
    img_tensor = F.resize(img_tensor, [256, 384])
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0) # (1, 3, 256, 384)
    
    with torch.no_grad():
        output = unet_model(img_tensor)
        probs = torch.sigmoid(output).squeeze().numpy()
        
    probs_orig = cv2.resize(probs, (img_rgb.shape[2], img_rgb.shape[1]), interpolation=cv2.INTER_LINEAR)

    return dict(
        p_x=p_x, p_y=p_y,
        land_type=land_types,
        rgb=rgb,
        band1=flat[0],          # B02 reflectance as "Band_1"
        sh_s=sh_s, sw_s=sw_s,
        land_grid=land_types.reshape(sh_s, sw_s),
        selected_bands_data=X[:, ::20],  # every 20th band for correlation calculation
        unet_probs=probs_orig,
    )


# Helper to style black-background axes
def style_black_axes(ax):
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white", which="both", labelsize=8)

# Helper to style white-background axes
def style_white_axes(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.tick_params(colors="black", which="both", labelsize=8)


# ── Master composite plotter ────────────────────────────────────────────────
def make_composite(rgb_disp, pix, tile_idx, sh, sw):
    """
    Build a single figure with all 4 reference panels matching process.md styles.

    GridSpec layout (2 rows × 2 cols):
      [0,0] Panel A – RGB scatter + Land Type Classification scatter (side-by-side)
      [0,1] Panel B – Hyperspectral Land Cover Classification Map (scatter + Spectral palette + legend + stats box)
      [1,0] Panel C – 3D Spatial-Spectral Visualization (scatter + tab10 cmap + colorbar)
      [1,1] Panel D – Triangle Correlation Heatmap (every 20th band, crest cmap, annotations)
    """
    import pandas as pd
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    TERRAIN_CLASSES = {
        0: "Alpine Meadow",
        1: "Alpine Tundra",
        2: "Bare Rock",
        3: "Dark Rock",
        4: "Snow / Ice",
        5: "Scree / Sunlit Rock",
        6: "Valley Floor / Meadow",
        7: "Veg-Scree Mix",
    }

    unique_cls  = sorted(np.unique(pix["land_type"]))
    
    tab10       = plt.colormaps["tab10"]
    spectral_pal = sns.color_palette("Spectral", n_colors=8)

    top_h   = 14   # inches
    bot_h   = 14   # inches
    fig_w   = 32
    fig_h   = top_h + bot_h + 2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(
        f"GlacioWatch — All Visualisation Formats  |  Tile {tile_idx + 1}  |  Tyrol, Austria",
        fontsize=20, fontweight="bold", color="black", y=0.995,
    )

    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[top_h, bot_h],
        hspace=0.38,
        wspace=0.28,
        left=0.06, right=0.94,
        top=0.95,  bottom=0.05,
    )

    # Define grids for contour plotting (full-resolution tile dimensions)
    x0 = pix["p_x"].min()
    y0 = pix["p_y"].min()
    grid_x = np.arange(sw) + x0
    grid_y = np.arange(sh) + y0

    # PANEL A
    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                                   wspace=0.18, hspace=0)

    # A-left: RGB Scatter (White Background)
    ax_rgb = fig.add_subplot(gs_a[0])
    style_white_axes(ax_rgb)
    ax_rgb.scatter(pix["p_x"], pix["p_y"], c=pix["rgb"], s=2, alpha=0.9, rasterized=True)
    ax_rgb.set_title("Natural Color (RGB) Image", fontsize=13, fontweight="bold", pad=10, color="black")
    ax_rgb.set_xlabel("X Coordinate", fontsize=10, fontweight="bold", color="black")
    ax_rgb.set_ylabel("Y Coordinate", fontsize=10, fontweight="bold", color="black")
    ax_rgb.set_aspect("equal")
    ax_rgb.invert_yaxis()
    
    # Overlay UNet contour in red
    ax_rgb.contour(grid_x, grid_y, pix["unet_probs"], levels=[0.5], colors=['#ef4444'], linewidths=1.5, alpha=0.9)

    # A-right: Classification Scatter (Black Background)
    ax_cls = fig.add_subplot(gs_a[1])
    style_black_axes(ax_cls)
    scatter_cls = ax_cls.scatter(
        pix["p_x"], pix["p_y"],
        c=pix["land_type"],
        cmap="tab10",
        vmin=0, vmax=9,
        s=2, alpha=0.9,
        rasterized=True
    )
    ax_cls.set_title("Land Type Classification", fontsize=13, fontweight="bold", pad=10, color="white")
    ax_cls.set_xlabel("X Coordinate", fontsize=10, fontweight="bold", color="white")
    ax_cls.set_ylabel("Y Coordinate", fontsize=10, fontweight="bold", color="white")
    ax_cls.set_aspect("equal")
    ax_cls.invert_yaxis()

    # Discrete colorbar for Panel A (drawn on white fig background)
    cbar_a = fig.colorbar(scatter_cls, ax=ax_cls, ticks=list(range(8)), fraction=0.046, pad=0.04)
    cbar_a.set_label("Land Type Class", fontsize=10, fontweight="bold", color="black")
    cbar_a.ax.tick_params(labelsize=8, colors="black")
    cbar_a.ax.set_yticklabels([TERRAIN_CLASSES[c] for c in range(8)], fontsize=8, color="black")

    fig.text(0.02, gs[0, 0].get_position(fig).y1 + 0.005,
             "Panel A — Hyperspectral Image – Tyrol, Austria",
             fontsize=12, fontweight="bold", color="black")

    # PANEL B: Hyperspectral Land Cover Classification Map (Black Background)
    ax_b = fig.add_subplot(gs[0, 1])
    style_black_axes(ax_b)
    for cls in unique_cls:
        mask = pix["land_type"] == cls
        ax_b.scatter(
            pix["p_x"][mask], pix["p_y"][mask],
            c=[spectral_pal[cls]],
            s=3, alpha=0.8, edgecolors="none",
            label=f"{TERRAIN_CLASSES[cls]} ({mask.sum():,} px)",
            rasterized=True,
        )

    ax_b.set_title(
        "Hyperspectral Land Cover Classification Map\nTyrol Alpine Region, Austria",
        fontsize=14, fontweight="bold", pad=10, color="white"
    )
    ax_b.set_xlabel("Easting (X Coordinate in meters)", fontsize=11, fontweight="bold", color="white")
    ax_b.set_ylabel("Northing (Y Coordinate in meters)", fontsize=11, fontweight="bold", color="white")
    ax_b.set_aspect("equal")
    ax_b.invert_yaxis()
    ax_b.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, color="white")
    
    # Overlay UNet contour in red
    ax_b.contour(grid_x, grid_y, pix["unet_probs"], levels=[0.5], colors=['#ef4444'], linewidths=1.5, alpha=0.9)

    # Legend for Panel B (outside plot, drawn on white fig background)
    handles, labels = ax_b.get_legend_handles_labels()
    unet_legend_element = Line2D([0], [0], color='#ef4444', lw=2, label='U-Net Glacier Outline')
    handles.append(unet_legend_element)
    
    leg_b = ax_b.legend(
        handles=handles,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.9,
        title="Land Type Classes", title_fontsize=10,
        markerscale=3,
    )
    leg_b.get_frame().set_facecolor("#1e293b")
    leg_b.get_frame().set_edgecolor("black")
    leg_b.get_title().set_color("white")
    for text in leg_b.get_texts():
        text.set_color("white")

    # Stats box inside Panel B (on black background)
    stats_txt = (
        f"Total Pixels: {len(pix['p_x']):,}\n"
        f"Spectral Bands: 218\n"
        f"Spatial Coverage: Alpine Region"
    )
    ax_b.text(
        0.02, 0.98, stats_txt,
        transform=ax_b.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#1e293b", edgecolor="white", alpha=0.8),
        color="white"
    )

    fig.text(gs[0, 1].get_position(fig).x0, gs[0, 1].get_position(fig).y1 + 0.005,
             "Panel B — Land Cover Classification Map",
             fontsize=12, fontweight="bold", color="black")

    # PANEL C: 3D Spatial-Spectral View (Black Background)
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    ax_c.set_facecolor("black")
    ax_c.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax_c.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax_c.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax_c.xaxis.label.set_color("white")
    ax_c.yaxis.label.set_color("white")
    ax_c.zaxis.label.set_color("white")
    ax_c.title.set_color("white")
    ax_c.tick_params(colors="white", which="both", labelsize=7)
    ax_c.xaxis.pane.set_edgecolor("#555555")
    ax_c.yaxis.pane.set_edgecolor("#555555")
    ax_c.zaxis.pane.set_edgecolor("#555555")

    n_pts = len(pix["p_x"])
    rng   = np.random.default_rng(42)
    idx3d = rng.choice(n_pts, min(50_000, n_pts), replace=False)

    sc3d = ax_c.scatter(
        pix["p_x"][idx3d],
        pix["p_y"][idx3d],
        pix["band1"][idx3d],
        c=pix["land_type"][idx3d],
        cmap="tab10",
        vmin=0, vmax=9,
        s=5, alpha=0.6,
    )

    ax_c.set_xlabel("X Coordinate",      fontsize=10, fontweight="bold", labelpad=8)
    ax_c.set_ylabel("Y Coordinate",      fontsize=10, fontweight="bold", labelpad=8)
    ax_c.set_zlabel("Band_1 Reflectance", fontsize=10, fontweight="bold", labelpad=8)
    ax_c.set_title("3D Spatial-Spectral Visualization",
                   fontsize=14, fontweight="bold", pad=14)

    sm_c = plt.cm.ScalarMappable(cmap="tab10", norm=mcolors.Normalize(vmin=0, vmax=9))
    sm_c.set_array([])
    cbar_c = fig.colorbar(sm_c, ax=ax_c, ticks=list(range(8)), shrink=0.6, pad=0.1)
    cbar_c.set_label("Land Type", fontsize=9, fontweight="bold", color="black")
    cbar_c.ax.tick_params(labelsize=8, colors="black")
    cbar_c.ax.set_yticklabels([TERRAIN_CLASSES[c] for c in range(8)], fontsize=8, color="black")

    fig.text(gs[1, 0].get_position(fig).x0, gs[1, 0].get_position(fig).y1 + 0.005,
             "Panel C — 3D Spatial-Spectral View",
             fontsize=12, fontweight="bold", color="black")

    # PANEL D: Triangle Correlation Heatmap (White Background)
    ax_d = fig.add_subplot(gs[1, 1])
    style_white_axes(ax_d)

    band_names = [f"Band_{i*20 + 1}" for i in range(11)]
    df_bands = pd.DataFrame(pix["selected_bands_data"], columns=band_names)
    corr = df_bands.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cut_off = 0.25
    extreme_1 = 0.5
    extreme_2 = 0.75
    extreme_3 = 0.9

    mask |= np.abs(corr) < cut_off
    corr_masked = corr.copy()
    corr_masked[mask] = np.nan

    remove_empty = True
    if remove_empty:
        wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
        wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
        if len(wanted_cols) > 0 and len(wanted_rows) > 0:
            corr_masked = corr_masked.iloc[wanted_cols, wanted_rows]

    if corr_masked.empty or corr_masked.isna().all().all():
        ax_d.text(0.5, 0.5, "No correlations |r| ≥ 0.25 found\nin this tile.",
                  ha="center", va="center", fontsize=12, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor="wheat", edgecolor="black", alpha=0.5), color="black")
        ax_d.set_title("Triangle Correlation Heatmap (Every 20th Band)\n|r| ≥ 0.25 only",
                       fontsize=12, fontweight="bold", pad=10, color="black")
        ax_d.axis("off")
    else:
        annot = [[f"{val:.4f}"
                  + ('' if np.isnan(val) or abs(val) < extreme_1 else '\n*')
                  + ('' if np.isnan(val) or abs(val) < extreme_2 else '*')
                  + ('' if np.isnan(val) or abs(val) < extreme_3 else '*')
                  for val in row] for row in corr_masked.to_numpy()]

        sns.heatmap(corr_masked,
                    vmin=-1,
                    vmax=1,
                    annot=annot,
                    fmt="",
                    cmap="crest",
                    cbar_kws={'label': 'Correlation Coefficient'},
                    linewidths=0.5,
                    linecolor='white',
                    ax=ax_d)

        ax_d.set_title('Triangle Correlation Heatmap (Every 20th Band)\n|r| ≥ 0.25 only',
                       fontsize=12, fontweight='bold', pad=10, color="black")

        ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=45, ha='right', fontsize=8, color="black")
        ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0, fontsize=8, color="black")

        # Colorbar label styling to black
        cbar_d = ax_d.collections[0].colorbar
        cbar_d.ax.yaxis.label.set_color('black')
        cbar_d.ax.tick_params(colors='black')

        legend_text = "* = |r| ≥ 0.5\n** = |r| ≥ 0.75\n*** = |r| ≥ 0.9"
        ax_d.text(1.22, 0.5, legend_text,
                  transform=ax_d.transAxes,
                  fontsize=9,
                  verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', edgecolor='black', alpha=0.8),
                  color="black")

    fig.text(gs[1, 1].get_position(fig).x0, gs[1, 1].get_position(fig).y1 + 0.005,
             "Panel D — Triangle Correlation Heatmap",
             fontsize=12, fontweight="bold", color="black")

    return fig


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GlacioWatch — 4-Panel Composite Pipeline (50 Tiles)")
    print("=" * 60)

    b02 = load_band("B02")
    b03 = load_band("B03")
    b04 = load_band("B04")
    b8a = load_band("B8A")
    b11 = load_band("B11")
    b12 = load_band("B12")

    height, width = b02.shape
    grid_rows, grid_cols = 10, 5
    tile_h = height // grid_rows
    tile_w = width  // grid_cols

    s2_wl     = np.array([490.0, 560.0, 665.0, 842.0, 1610.0, 2190.0])
    target_wl = np.linspace(420.0, 2450.0, 218)
    sample_rate = 2

    tile_idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w

            # Band slices for this tile
            t_b02 = b02[y0:y1, x0:x1]
            t_b03 = b03[y0:y1, x0:x1]
            t_b04 = b04[y0:y1, x0:x1]
            t_b8a = b8a[y0:y1, x0:x1]
            t_b11 = b11[y0:y1, x0:x1]
            t_b12 = b12[y0:y1, x0:x1]

            sh, sw = t_b02.shape

            # Full-res RGB raster (Panel A left)
            rgb_disp = np.clip(
                np.stack([t_b04, t_b03, t_b02], axis=-1) * 4.0, 0.0, 1.0
            )

            # Classify sampled pixels
            pix = build_pixel_data(
                t_b02, t_b03, t_b04, t_b8a, t_b11, t_b12,
                s2_wl, target_wl, x0, y0, sample_rate,
            )

            # Build composite figure
            fig = make_composite(rgb_disp, pix, tile_idx, sh, sw)

            # Save
            tile_name = f"tile_{tile_idx + 1}.png"
            for out_dir in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
                fig.savefig(
                    os.path.join(out_dir, "tiles", tile_name),
                    dpi=120, bbox_inches="tight",
                    facecolor="white",
                )

            plt.close(fig)
            print(f"  ✓ {tile_name}")
            tile_idx += 1

    print("\n✅ All 50 tiles complete — each tile contains all 4 reference panels.")


if __name__ == "__main__":
    main()