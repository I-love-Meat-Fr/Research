"""
End-to-End Finger Vein Recognition — Streamlit Demo App
=========================================================
A research demo GUI for the finger-vein recognition pipeline.

Features:
    * Register new users (one or multiple sample images).
    * Verify a single image against a claimed identity (1:1 matching).
    * Browse and manage the enrolled user database.
    * View live evaluation metrics (EER, AUC) on the loaded checkpoint.

Run with:
    streamlit run src/e2e_app.py

This module reuses the existing pieces of the codebase:
    * infer.py        — model loading + image preprocessing
    * enrollment_db.py — SQLite-backed user/embedding store
    * eval_results.json — pre-computed verification statistics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# Make `src` importable so we can reuse the project modules directly.
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# Reuse existing components ---------------------------------------------------------
# These imports are deliberately placed at the top of the module so that
# Streamlit's hot-reload detects any changes to the underlying training code.
from enrollment_db import EnrollmentDB  # noqa: E402


# ============================================================================
# MODEL DEFINITION (must match the architecture in eval_fvusm.py / train_fvusm.py)
# ============================================================================
class ResNet18Backbone(nn.Module):
    """Backbone architecture matching the training script (`train_fvusm.py`).

    The trained checkpoint stores the encoder inside an attribute named
    `backbone` (an `nn.Sequential` so the keys look like
    `backbone.0.weight`, `backbone.4.0.conv1.weight`, ...) plus an `embed`
    linear and a `bn` BatchNorm. The ArcFace classifier head is stored
    separately under the key `head.weight` and is *not* needed for
    embedding extraction, so we only build the backbone here.

    Forward returns the L2-normalised 128-D embedding (matching the
    `compute_eer`/gallery-centroid pipeline used during evaluation).
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        import torchvision.models as tvm

        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc = nn.Identity()
        # Wrap everything before `fc` (already Identity) in a Sequential so the
        # state_dict layout matches the saved checkpoint exactly.
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.embed = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor, return_embeddings: bool = True) -> torch.Tensor:
        # We always return the embedding — the demo only does verification.
        f = self.backbone(x).flatten(1)
        emb_raw = self.embed(f)
        emb = self.bn(emb_raw)
        return nn.functional.normalize(emb, p=2, dim=1)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Finger Vein Recognition — Research Demo",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CKPT = SRC_DIR / "results" / "fvusm_resnet18_arcface_v1" / "best.pth"
DEFAULT_DB = SRC_DIR / "results" / "enrollment.db"
EVAL_JSON = SRC_DIR / "results" / "fvusm_resnet18_arcface_v1" / "eval_results.json"


# ============================================================================
# RESOURCE LOADERS (cached so the model is loaded only once per session)
# ============================================================================
@st.cache_resource(show_spinner="Loading ResNet-18 backbone …")
def load_model(checkpoint_path: str):
    """Load the trained ResNet-18 model that matches the saved checkpoint format.

    The checkpoint is a dict with two state_dict groups plus metadata:
        * `backbone`            — conv encoder (`nn.Sequential`) + `embed` + `bn`
        * `head`                — ArcFace classifier (`weight` parameter)
        * `embedding_dim`       — int
        * `num_classes`         — int
        * `eer`, `pos_mean`, `neg_mean`, `epoch` — training metrics

    We only need the backbone for verification, so we build a
    `ResNet18Backbone` and load `ckpt['backbone']` into it. The ArcFace head
    is intentionally discarded (we never classify; we compare embeddings).
    Cached for the lifetime of the Streamlit session.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    embedding_dim = ckpt.get("embedding_dim", 128)
    num_classes = ckpt.get("num_classes", 492)

    model = ResNet18Backbone(embedding_dim=embedding_dim).to(device)
    missing, unexpected = model.load_state_dict(ckpt["backbone"], strict=False)
    if missing or unexpected:
        # Surface mismatches loudly — this almost certainly means the
        # checkpoint was produced by a different architecture.
        raise RuntimeError(
            f"Backbone state_dict mismatch. Missing={missing[:5]}... "
            f"Unexpected={unexpected[:5]}..."
        )
    model.eval()

    meta = {
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
        "img_size": 128,  # matches training preprocessing (eval_fvusm.py)
        "threshold": ckpt.get("eer"),  # EER stored at training time
        "epoch": ckpt.get("epoch"),
        "eer": ckpt.get("eer"),
        "pos_mean": ckpt.get("pos_mean"),
        "neg_mean": ckpt.get("neg_mean"),
    }
    return model, meta, device


def _torch_cuda_available() -> bool:
    import torch
    return bool(torch.cuda.is_available())


@st.cache_resource
def get_db() -> EnrollmentDB:
    return EnrollmentDB(str(DEFAULT_DB))


@st.cache_data
def load_eval_stats() -> dict:
    if EVAL_JSON.exists():
        with open(EVAL_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ============================================================================
# CORE HELPERS — kept inside this file to remain self-contained.
# ============================================================================
def _extract_single(model, image: Image.Image, device: str, img_size: int) -> np.ndarray:
    """Extract L2-normalised embedding for one image."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = image.convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor, return_embeddings=True)
    emb = emb.cpu().numpy().squeeze()
    # Normalise (defensive — model already normalises but be explicit).
    norm = np.linalg.norm(emb) + 1e-12
    return (emb / norm).astype(np.float32)


def _extract_batch(model, images: List[Image.Image], device: str, img_size: int) -> np.ndarray:
    """Extract embeddings for a list of images, returns array shape [N, D]."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensors = []
    for img in images:
        img = img.convert("L")
        tensors.append(transform(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        emb = model(batch, return_embeddings=True)
    emb = emb.cpu().numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return (emb / norms).astype(np.float32)


def _compute_decision_threshold(stats: dict, fallback: float = 0.5) -> float:
    """Pick a sensible decision threshold from the eval stats.

    We pick the midpoint between the intra-class (positive) and inter-class
    (negative) cosine means. That is approximately the Equal-Error-Rate
    operating point when the score distributions are symmetric.
    """
    if not stats:
        return fallback
    pos_mean = stats.get("pos_mean")
    neg_mean = stats.get("neg_mean")
    if pos_mean is None or neg_mean is None:
        return fallback
    return float((pos_mean + neg_mean) / 2)


def _format_metric_pill(label: str, value: str, color: str = "#4ECDC4") -> str:
    """Tiny HTML pill used inside markdown to display a metric."""
    return (
        f"<div style='display:inline-block;padding:6px 14px;margin:4px 6px 4px 0;"
        f"border-radius:14px;background:{color}22;border:1px solid {color};"
        f"font-size:0.95em;'><b>{label}:</b> {value}</div>"
    )


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> dict:
    """Render the sidebar with model selection + dataset overview."""
    st.sidebar.title("Finger Vein Demo")
    st.sidebar.caption("Research prototype · ResNet-18 + ArcFace")

    checkpoint_path = str(DEFAULT_CKPT)
    if not Path(checkpoint_path).exists():
        st.sidebar.error(f"Checkpoint missing: {checkpoint_path}")
        st.stop()

    db = get_db()
    info = db.db_info()

    st.sidebar.markdown("### Enrolled users")
    st.sidebar.metric("Users", info["num_users"])
    st.sidebar.metric("Stored embeddings", info["num_embeddings"])
    st.sidebar.caption(f"DB: `{info['path']}`")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: register at least two samples per user to make verification robust."
    )
    return {
        "checkpoint": checkpoint_path,
        "db": db,
    }


# ============================================================================
# REGISTER PAGE
# ============================================================================
def page_register(model, meta, device, db: EnrollmentDB):
    st.header("Register a new user")
    st.markdown(
        "Upload one or more finger-vein images for a new identity. "
        "Multiple samples help the model produce a stable embedding gallery."
    )

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        user_id = st.text_input(
            "User ID",
            placeholder="e.g. u001 or alice",
            help="Unique identifier. Cannot contain spaces or special characters.",
        ).strip()
        full_name = st.text_input("Full name (optional)", placeholder="Alice Nguyen")
        uploaded_files = st.file_uploader(
            "Finger-vein images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            help="Single image per session or multiple samples — both work.",
        )

        register_clicked = st.button(
            "Save & enroll",
            type="primary",
            disabled=not (user_id and uploaded_files),
        )

    with col_preview:
        if uploaded_files:
            st.markdown("**Preview**")
            cols = st.columns(min(3, len(uploaded_files)))
            for i, f in enumerate(uploaded_files[:6]):
                cols[i % len(cols)].image(f, caption=f.name, use_container_width=True)
            if len(uploaded_files) > 6:
                st.caption(f"+ {len(uploaded_files) - 6} more images")
        else:
            st.info("Upload images to preview here.")

    if register_clicked:
        if not user_id:
            st.error("Please provide a User ID.")
            return
        if not all(c.isalnum() or c in "-_" for c in user_id):
            st.error("User ID must be alphanumeric (dashes and underscores allowed).")
            return
        if not uploaded_files:
            st.error("Please upload at least one image.")
            return

        # Insert or look up the user record.
        existing = db.get_user(user_id)
        if existing is None:
            if not db.add_user(user_id, full_name):
                st.error(f"Could not create user `{user_id}`.")
                return
        elif full_name and not existing.get("name"):
            # Update name if it was left empty before.
            with st.spinner("Updating user info …"):
                pass  # name update intentionally skipped — schema only stores once

        # Convert uploaded files -> PIL images once.
        images: List[Image.Image] = []
        for f in uploaded_files:
            try:
                img = Image.open(f).copy()
                images.append(img)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not open `{f.name}`: {exc}")
                return

        # Extract embeddings.
        embeddings = _extract_batch(model, images, device, meta["img_size"])

        progress = st.progress(0.0, text="Saving embeddings …")
        saved = 0
        for idx, emb in enumerate(embeddings):
            ok = db.add_embedding(
                user_id=user_id,
                embedding=emb,
                sample_idx=idx,
                source_path=uploaded_files[idx].name,
            )
            saved += int(ok)
            progress.progress((idx + 1) / len(embeddings))
        progress.empty()

        st.success(
            f"Enrolled **{saved}** embedding(s) for user `{user_id}` "
            f"({full_name or 'no name'})."
        )
        st.balloons()


def _identify_probe(
    model, image: Image.Image, device: str, db: EnrollmentDB, img_size: int, top_k: int = 10
) -> List[Tuple[str, float, str]]:
    """Run 1:N identification: compare probe against every enrolled user.

    Returns
    -------
    List of (user_id, cosine_score, name) sorted descending by score.
    """
    probe_emb = _extract_single(model, image, device, img_size)
    users = db.list_users()
    results = []
    for u in users:
        uid = u["user_id"]
        samples = db.get_user_embeddings(uid)
        if not samples:
            continue
        gallery = np.stack([e[1] for e in samples]).mean(axis=0)
        gallery = gallery / (np.linalg.norm(gallery) + 1e-12)
        score = float(np.dot(probe_emb, gallery))
        name = u.get("name") or ""
        results.append((uid, score, name))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================================
# IDENTIFY PAGE
# ============================================================================
def page_identify(model, meta, device, db: EnrollmentDB, stats: dict):
    st.header("Identify identity")
    st.markdown(
        "Upload a finger-vein image and the system will automatically find "
        "the most similar enrolled identity — no need to pick a user manually."
    )

    db_users = db.list_users()
    if not db_users:
        st.warning(
            "No users enrolled yet. Head to the **Register** page to add one first."
        )
        return

    stats = load_eval_stats()
    threshold = _compute_decision_threshold(stats)

    # ── Upload area ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload probe image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
        key="identify_uploader",
    )

    if not uploaded:
        st.info("⬆️ Upload an image to start identification.")
        return

    image = Image.open(uploaded)
    st.image(image, caption=f"Probe: {uploaded.name}", width=360)

    # ── Run identification ───────────────────────────────────────────────
    with st.spinner("Comparing against all enrolled users …"):
        results = _identify_probe(model, image, device, db, meta["img_size"])

    if not results:
        st.error("No enrolled users found in the database.")
        return

    best_uid, best_score, best_name = results[0]
    is_match = best_score >= threshold

    # ── Top-1 verdict ────────────────────────────────────────────────────
    if is_match:
        verdict_label = "IDENTITY FOUND"
        pill_color = "#4ECDC4"
    else:
        verdict_label = "NO MATCH FOUND"
        pill_color = "#FF6B6B"

    st.markdown("---")
    st.markdown("### 🔍 Top result")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(
            f"### {'✅' if is_match else '❌'}  "
            f"<span style='color:{pill_color};font-size:1.4em'>{verdict_label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{best_uid}**" + (f"  —  {best_name}" if best_name else ""))
        user_rec = db.get_user(best_uid)
        st.markdown(f"Samples: {user_rec.get('num_samples', 0) if user_rec else '?'}")

    with col_right:
        st.markdown(
            _format_metric_pill("Best cosine score", f"{best_score:.4f}", pill_color)
            + _format_metric_pill("Threshold", f"{threshold:.4f}", "#888")
            + _format_metric_pill("Verdict", verdict_label, pill_color),
            unsafe_allow_html=True,
        )
        ratio = min(max((best_score + 1) / 2, 0.0), 1.0)
        bar_w = int(ratio * 300)
        thresh_x = int((threshold + 1) / 2 * 300)
        st.markdown(
            f"<div style='background:#eee;border-radius:6px;height:22px;width:300px;"
            f"position:relative;'>"
            f"<div style='background:{pill_color};height:100%;border-radius:6px;"
            f"width:{bar_w}px;transition:width .3s;'></div>"
            f"<div style='position:absolute;top:0;left:{thresh_x}px;"
            f"width:2px;height:22px;background:#333;' title='threshold'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Ranked list ──────────────────────────────────────────────────────
    with st.expander("📋  View all ranked results", expanded=True):
        header_cols = st.columns([3, 1, 1, 1])
        header_cols[0].markdown("**User**")
        header_cols[1].markdown("**Score**")
        header_cols[2].markdown("**vs Threshold**")
        header_cols[3].markdown("**Verdict**")
        st.markdown("---")

        for rank, (uid, score, name) in enumerate(results, 1):
            above = score >= threshold
            row_cols = st.columns([3, 1, 1, 1])
            label = f"**{uid}**" + (f"  —  {name}" if name else "")
            row_cols[0].markdown(label)
            row_cols[1].markdown(f"`{score:+.4f}`")
            diff = score - threshold
            diff_str = f"{diff:+.4f}"
            row_cols[2].markdown(diff_str)
            row_cols[3].markdown(
                f"{'✅ Match' if above else '❌ Below'}"
            )

    # ── Reference stats ───────────────────────────────────────────────────
    if stats:
        st.markdown("---")
        st.markdown("#### Reference stats from offline evaluation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EER", f"{stats.get('eer_percent', 0):.3f}%")
        c2.metric("Intra-class cosine", f"{stats.get('pos_mean', 0):.3f} ± {stats.get('pos_std', 0):.3f}")
        c3.metric("Inter-class cosine", f"{stats.get('neg_mean', 0):.3f} ± {stats.get('neg_std', 0):.3f}")
        c4.metric("Margin", f"{stats.get('margin', 0):.3f}")


def _format_user_label(db: EnrollmentDB, user_id: str) -> str:
    user = db.get_user(user_id)
    if not user:
        return user_id
    name = user.get("name") or ""
    samples = user.get("num_samples", 0)
    label = f"{user_id}"
    if name:
        label = f"{user_id} — {name}"
    label += f"  ({samples} samples)"
    return label


# ============================================================================
# DATABASE PAGE
# ============================================================================
def page_database(db: EnrollmentDB):
    st.header("Enrolled users database")
    users = db.list_users()
    if not users:
        st.info("No users yet. Switch to the **Register** tab to enroll one.")
        return

    rows = []
    for u in users:
        samples = db.get_user_embeddings(u["user_id"])
        rows.append(
            {
                "User ID": u["user_id"],
                "Name": u.get("name") or "—",
                "Samples": u.get("num_samples", len(samples)),
                "Created at": u.get("created_at", ""),
                "Source files": ", ".join(s[2] for s in samples) or "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Danger zone")
    col1, col2 = st.columns([3, 1])
    with col1:
        target = st.selectbox(
            "Delete user",
            options=["—"] + [u["user_id"] for u in users],
        )
    with col2:
        confirm = st.button("Delete user", type="secondary")
    if confirm and target != "—":
        if db.delete_user(target):
            st.success(f"Deleted user `{target}`.")
            st.rerun()
        else:
            st.error(f"Could not delete `{target}`.")

    with st.expander("Database info"):
        info = db.db_info()
        st.json(info)


# ============================================================================
# ABOUT PAGE
# ============================================================================
def page_about(stats: dict):
    st.header("About this demo")
    st.markdown(
        """
        This is a research prototype built on top of the existing
        **Finger-Vein Recognition** training pipeline.

        **Pipeline**
        1. Pre-trained `ResNet-18` backbone with the first conv layer
           adapted to single-channel grayscale input (128x128).
        2. Embedding projection to 128-D, followed by `BatchNorm1d`.
        3. Trained with `ArcFace` loss (scale 30, margin 0.5) on
           the FV-USM finger-vein dataset (492 classes, 2 sessions).
        4. Embeddings are L2-normalised so that cosine similarity is
           the same as a dot product.

        **Run**
        ```
        streamlit run src/e2e_app.py
        ```
        """
    )

    if stats:
        st.markdown("### Latest offline evaluation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EER", f"{stats.get('eer_percent', 0):.3f}%")
        c2.metric("Intra-class cosine", f"{stats.get('pos_mean', 0):.3f}")
        c3.metric("Inter-class cosine", f"{stats.get('neg_mean', 0):.3f}")
        c4.metric("Margin", f"{stats.get('margin', 0):.3f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Test classes", stats.get("n_classes", "?"))
        c6.metric("Test images", stats.get("n_test_images", "?"))
        c7.metric("Checkpoint epoch", stats.get("checkpoint_epoch", "?"))


# ============================================================================
# MAIN
# ============================================================================
def main():
    sidebar = render_sidebar()
    db: EnrollmentDB = sidebar["db"]
    checkpoint_path = sidebar["checkpoint"]

    page = st.sidebar.radio(
        "Navigate",
        ["Register", "Identify", "Database", "About"],
        index=1,
    )

    # Load the model once and share it across pages.
    model, meta, device = load_model(checkpoint_path)
    stats = load_eval_stats()

    if page == "Register":
        page_register(model, meta, device, db)
    elif page == "Identify":
        page_identify(model, meta, device, db, stats)
    elif page == "Database":
        page_database(db)
    elif page == "About":
        page_about(stats)


if __name__ == "__main__":
    main()
