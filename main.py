import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0a0a0f;
    --card: #12121a;
    --border: #1e1e2e;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --green: #10b981;
    --red: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent2);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 2px;
}

.tag-cancer  { background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.tag-image   { background: rgba(6,182,212,.15);  color: #67e8f9; border: 1px solid rgba(6,182,212,.3); }
.tag-bio     { background: rgba(16,185,129,.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,.3); }
.tag-health  { background: rgba(124,58,237,.15); color: #c4b5fd; border: 1px solid rgba(124,58,237,.3); }

.result-box {
    background: linear-gradient(135deg, rgba(124,58,237,.08), rgba(6,182,212,.08));
    border: 1px solid rgba(124,58,237,.3);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 24px !important;
    width: 100%;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

hr { border-color: var(--border) !important; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}

.hero-badge {
    display: inline-block;
    background: rgba(124,58,237,.2);
    border: 1px solid rgba(124,58,237,.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #c4b5fd;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ── U-Net Model ───────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = F.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        return torch.sigmoid(self.final(x))


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1)
    model.eval()
    return model

def preprocess(img: Image.Image, size=256):
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t

def predict(model, tensor):
    with torch.no_grad():
        return model(tensor).squeeze().numpy()

def overlay_mask(img_arr, mask, alpha=0.45, color=(124, 58, 237)):
    out = img_arr.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0.5,
                                 out[:, :, c] * (1 - alpha) + v * alpha,
                                 out[:, :, c])
    return np.clip(out, 0, 255).astype(np.uint8)

def compute_metrics(mask):
    binary = (mask > 0.5).astype(np.float32)
    area_pct = binary.mean() * 100
    return {
        "tumor_area_pct": round(area_pct, 2),
        "max_confidence": round(float(mask.max()), 3),
        "mean_confidence": round(float(mask[mask > 0.5].mean()) if binary.sum() > 0 else 0.0, 3),
        "pixel_count": int(binary.sum()),
    }

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#12121a", dpi=120)
    buf.seek(0)
    return Image.open(buf)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-badge">🧠 LGG DATASET</div>', unsafe_allow_html=True)
    st.markdown("## Brain MRI\nSegmentation")
    st.markdown("---")

    st.markdown('<div class="section-title">Tags</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="tag tag-cancer">Cancer</span>
    <span class="tag tag-image">Image</span>
    <span class="tag tag-bio">Biology</span>
    <span class="tag tag-health">Healthcare</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Model Config</div>', unsafe_allow_html=True)
    img_size = st.selectbox("Input Resolution", [128, 256, 512], index=1)
    threshold = st.slider("Mask Threshold", 0.1, 0.9, 0.5, 0.05)
    show_heatmap = st.toggle("Show Confidence Heatmap", True)
    alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.caption(
        "LGG dataset · 110 patients · TCGA lower-grade glioma collection · "
        "FLAIR abnormality segmentation masks · U-Net architecture"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Brain MRI Segmentation")
st.markdown("*FLAIR Abnormality Detection · U-Net · LGG Dataset*")
st.markdown("---")

model = load_model()

tab1, tab2, tab3 = st.tabs(["🔬 Inference", "📐 Architecture", "📊 Dataset Info"])

# ── Tab 1: Inference ──────────────────────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a brain MRI image (PNG / JPG / TIF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Input MRI", use_container_width=True)

            run_btn = st.button("⚡ Run Segmentation")
        else:
            st.info("Upload an MRI slice to begin segmentation.")
            run_btn = False

    with col_result:
        if uploaded and run_btn:
            with st.spinner("Running U-Net inference…"):
                tensor = preprocess(img, size=img_size)
                mask = predict(model, tensor)

            metrics = compute_metrics(mask)

            # ── Metric cards
            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{metrics['tumor_area_pct']}%</div>
                    <div class="metric-label">Tumor Area</div></div>""",
                    unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{metrics['max_confidence']}</div>
                    <div class="metric-label">Max Conf.</div></div>""",
                    unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{metrics['pixel_count']}</div>
                    <div class="metric-label">Pixels</div></div>""",
                    unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Visualisation grid
            img_resized = np.array(img.convert("RGB").resize((img_size, img_size)))
            binary_mask = (mask > threshold).astype(np.uint8) * 255
            overlay = overlay_mask(img_resized, mask, alpha=alpha)

            fig, axes = plt.subplots(1, 3 if not show_heatmap else 4,
                                     figsize=(14, 3.5))
            fig.patch.set_facecolor("#12121a")
            titles = ["MRI Input", "Binary Mask", "Overlay"]
            imgs_   = [img_resized, binary_mask, overlay]
            cmaps   = ["gray", "gray", None]

            if show_heatmap:
                titles.append("Confidence")
                imgs_.append(mask)
                cmaps.append("magma")

            for ax, im, title, cmap in zip(axes, imgs_, titles, cmaps):
                ax.imshow(im, cmap=cmap)
                ax.set_title(title, color="#94a3b8", fontsize=9,
                             fontfamily="monospace", pad=6)
                ax.axis("off")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1e1e2e")

            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if metrics["tumor_area_pct"] > 0:
                st.markdown(f"""<div class="result-box">
                    🔴 <b>Abnormality detected</b> — FLAIR hyperintensity covers
                    <b>{metrics['tumor_area_pct']}%</b> of the imaged region
                    ({metrics['pixel_count']} pixels).
                    Mean segmentation confidence: <b>{metrics['mean_confidence']}</b>.
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ No significant FLAIR abnormality detected.")

        elif not uploaded:
            st.markdown("""<div style='text-align:center; padding:80px 20px; color:#475569;'>
                <div style='font-size:3rem'>🧬</div>
                <div style='font-family:monospace; font-size:0.85rem; margin-top:8px;'>
                Upload an MRI to see results</div></div>""",
                unsafe_allow_html=True)


# ── Tab 2: Architecture ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### U-Net Architecture")
    st.markdown(
        "Classic encoder-decoder with skip connections, optimised for "
        "biomedical image segmentation."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Encoder (Contracting Path)**")
        for i, f in enumerate([64, 128, 256, 512]):
            st.markdown(f"- Block {i+1}: DoubleConv → {f} channels → MaxPool")
        st.markdown("- Bottleneck: DoubleConv → 1024 channels")

    with col_b:
        st.markdown("**Decoder (Expanding Path)**")
        for i, f in enumerate([512, 256, 128, 64]):
            st.markdown(f"- Block {i+1}: TransposeConv → {f} + skip → DoubleConv")
        st.markdown("- Output: 1×1 Conv → Sigmoid → Binary mask")

    st.markdown("---")
    st.markdown("**Training Recipe**")
    code = """
# ── Quick training snippet ──────────────────────────────
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

model     = UNet(in_channels=3, out_channels=1).cuda()
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

def dice_loss(pred, target, smooth=1.):
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

def bce_dice(pred, target):
    return nn.BCELoss()(pred, target) + dice_loss(pred, target)

# Training loop
for epoch in range(50):
    model.train()
    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = model(imgs)
        loss  = bce_dice(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    scheduler.step()
    """
    st.code(code, language="python")


# ── Tab 3: Dataset Info ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### LGG Segmentation Dataset")

    col1, col2, col3, col4 = st.columns(4)
    stats = [("110", "Patients"), ("3", "MRI Modalities"), ("1.06 GB", "Dataset Size"), ("8.75", "Usability Score")]
    for col, (val, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div></div>""",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
**Source:** The Cancer Imaging Archive (TCIA) · The Cancer Genome Atlas (TCGA)

**Description:** Brain MR images with manual FLAIR abnormality segmentation masks for 110 patients
from the TCGA lower-grade glioma collection. Each patient has at least a fluid-attenuated inversion
recovery (FLAIR) sequence and genomic cluster data.

**Structure:**
```
kaggle_3m/
├── TCGA_CS_xxxx_xxxxxxxx/   # One folder per patient
│   ├── *_<slice>_mask.tif   # Binary segmentation mask
│   └── *_<slice>.tif        # Corresponding MRI slice
└── data.csv                  # Genomic + clinical metadata
```

**License:** CC BY-NC-SA 4.0
    """)

    st.markdown("**Key Papers**")
    st.markdown("""
- Buda et al., *Computers in Biology and Medicine*, 2019 — deep learning shape feature extraction for LGG
- Mazurowski et al., *Journal of Neuro-Oncology*, 2017 — radiogenomics of lower-grade glioma
    """)
