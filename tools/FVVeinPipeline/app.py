"""
Finger Vein E2E Pipeline — Streamlit App
=========================================
Clean, self-contained E2E pipeline for finger vein recognition:
  1. ENROLLMENT: save 3 images → create embedding → store in DB
  2. IDENTIFICATION: import query image → cosine match → identify person

Usage:
  cd tools/FVVeinPipeline
  streamlit run app.py

No proprietary DLLs required. Runs on ResNet18 + ArcFace deep learning model.
"""

from __future__ import annotations

import io
import sys
import tempfile
import traceback
from pathlib import Path

# Resolve services from relative path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from services.deep_learning import DeepLearningEngine
from services.enrollment_db import EnrollmentDB
from services.sdk_decryptor import SDKDecryptor, is_encrypted_sdk_jpg


# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="Finger Vein E2E Pipeline",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parent.parent.parent
    / "src" / "results" / "baseline_resnet18_arcface_expD" / "best_model.pth"
)
DEFAULT_DB_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src" / "results" / "enrollment.db"
)
DEFAULT_DECRYPTOR = (
    r"D:\WorkSpace\Research\Research\MFFV\tools\DecryptNir\publish\DecryptNir.exe"
)
# Allowed "Max samples per user" choices in the Enrollment form.
# Option A: exactly 1 real image → 11 augmented variants = 12 samples per user.
ALLOWED_MAX_SAMPLES = [1]
DEFAULT_THRESHOLD = 0.40
IMG_PREVIEW_W = 200


# ============================================================================
# Helpers
# ============================================================================

def load_image(uploaded_file) -> Image.Image | None:
    """Load an uploaded file as grayscale PIL image."""
    try:
        img = Image.open(uploaded_file)
        if img.mode != "L":
            img = img.convert("L")
        return img
    except Exception:
        return None


def load_image_smart(
    uploaded_file,
    decryptor: SDKDecryptor | None,
) -> tuple[Image.Image | None, str]:
    """
    Load an uploaded file, transparently handling SDK-encrypted .jpg files.

    Returns:
        (image, source_label) where source_label describes what happened
        ("decrypted via SDK" | "plain JPEG" | "failed").
    """
    # Spool the upload to a temp path so we can sniff the header.
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        # Encrypted SDK file?
        if is_encrypted_sdk_jpg(tmp_path) and decryptor is not None:
            try:
                decrypted = decryptor.decrypt_file(tmp_path)
                img = Image.open(decrypted)
                if img.mode != "L":
                    img = img.convert("L")
                return img, "decrypted via SDK"
            except Exception as e:
                return None, f"failed (decrypt error: {e})"

        # Plain JPEG/PNG — try Pillow directly.
        try:
            img = Image.open(tmp_path)
            if img.mode != "L":
                img = img.convert("L")
            return img, "plain JPEG"
        except Exception:
            return None, "failed"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def render_thumbnail(img: Image.Image, caption: str = "") -> None:
    """Render a small image preview in the Streamlit column."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, width=IMG_PREVIEW_W)


def render_query_image(img: Image.Image, caption: str = "") -> None:
    """Render a larger image for query display."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, width=400)


def score_gauge(score: float, threshold: float) -> go.Figure:
    """Gauge chart for a single similarity score."""
    color = "#2ECC71" if score >= threshold else "#E74C3C"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [-1, threshold], "color": "#FADBD8"},
                    {"range": [threshold, 1.0], "color": "#D5F5E3"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold,
                },
            },
            number={"valueformat": ".4f", "font": {"size": 28}},
            title={"text": "Similarity", "font": {"size": 16}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ============================================================================
# Cached model loader
# ============================================================================

@st.cache_resource(show_spinner="⏳ Loading deep learning model...")
def load_engine(checkpoint_path: str) -> DeepLearningEngine:
    """Load and cache the deep learning engine."""
    return DeepLearningEngine(checkpoint_path, device="auto")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar() -> tuple[DeepLearningEngine, EnrollmentDB, dict, SDKDecryptor | None]:
    """
    Render sidebar and return (engine, db, settings_dict, decryptor).

    `settings_dict` carries per-mode thresholds so the same verification
    threshold (0.42) does NOT leak into identification (where it would
    wrongly accept unknown persons who merely happen to be similar to a
    gallery user).
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Checkpoint
        checkpoint = st.text_input(
            "Checkpoint path",
            value=DEFAULT_CHECKPOINT,
            help="Path to ResNet18+ArcFace checkpoint (.pth)",
        )
        if not Path(checkpoint).exists():
            st.error("⚠️ Checkpoint not found!")
            st.stop()

        # Load engine
        try:
            engine = load_engine(checkpoint)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        st.success(
            f"✅ Model loaded: {engine.meta.get('embedding_dim', '?')}D embeddings, "
            f"{engine.device.upper()}"
        )
        if engine.meta.get("eer") is not None:
            st.caption(f"Trained EER: {engine.meta['eer']*100:.2f}%")

        # ---------------------------------------------------------------
        # Thresholds — TWO separate knobs
        # ---------------------------------------------------------------
        default_t = float(engine.meta.get("threshold") or DEFAULT_THRESHOLD)

        st.markdown("### 🎚️ Decision Thresholds")
        st.caption(
            "Verification (1:1) and Identification (1:N) need **different** "
            "thresholds because Identification is open-set — it must reject "
            "unknown persons, not just pick the closest match."
        )

        verify_threshold = st.slider(
            "Verification threshold (1:1)",
            min_value=0.0,
            max_value=1.0,
            value=default_t,
            step=0.01,
            help=(
                "1:1 matching. Score ≥ threshold → GENUINE. "
                f"Default = {default_t:.2f} (EER)."
            ),
            key="verify_threshold",
        )

        # Identification threshold defaults HIGHER than verification because
        # we need to resist impostors who happen to resemble a gallery user.
        ident_threshold = st.slider(
            "Identification threshold (1:N)",
            min_value=0.0,
            max_value=1.0,
            value=max(default_t + 0.10, 0.55),
            step=0.01,
            help=(
                "1:N matching (top-1 score). Higher than verification to "
                "reject unknown persons."
            ),
            key="ident_threshold",
        )

        margin_threshold = st.slider(
            "Margin threshold (top1 − top2)",
            min_value=0.0,
            max_value=0.5,
            value=0.08,
            step=0.01,
            help=(
                "Minimum gap between top-1 and top-2 scores. If top-1 is not "
                "clearly better than top-2, treat the query as UNKNOWN even "
                "if the score exceeds the threshold."
            ),
            key="margin_threshold",
        )

        # Ratio threshold = top1 / top2. Catches the case where every score
        # is bunched in a tight band (e.g. all 0.90-0.92): absolute margin
        # is tiny but the ratio can still discriminate the true user.
        ratio_threshold = st.slider(
            "Ratio threshold (top1 / top2)",
            min_value=1.00,
            max_value=1.20,
            value=1.02,
            step=0.005,
            help=(
                "Minimum RELATIVE ratio top1/top2. e.g. 1.02 means top-1 must "
                "be at least 2% higher than top-2. This is the most important "
                "gate for stopping impostor matches when all scores are "
                "clustered (margin is near 0 but top1 is clearly above top2)."
            ),
            key="ratio_threshold",
        )

        # Quick visual hint for the user
        if ident_threshold < verify_threshold + 0.05:
            st.warning(
                "⚠️ Identification threshold is too close to verification "
                "threshold — unknown persons may be wrongly accepted."
            )

        settings = {
            "verify_threshold": verify_threshold,
            "ident_threshold": ident_threshold,
            "margin_threshold": margin_threshold,
            "ratio_threshold": ratio_threshold,
        }

        # Back-compat: legacy single-threshold call sites use this key
        settings["threshold"] = ident_threshold

        st.divider()

        # Database
        db_path = st.text_input(
            "Enrollment DB path",
            value=DEFAULT_DB_PATH,
            help="SQLite database for enrolled users",
        )
        db = EnrollmentDB(db_path)

        info = db.db_info()
        st.markdown("### 📊 Gallery")
        st.metric("Enrolled users", info["num_users"])
        st.metric("Total samples", info["num_embeddings"])

        st.divider()
        st.markdown("### 🔓 SDK Decryptor (Option A)")
        decryptor_path = st.text_input(
            "DecryptNir.exe path",
            value=DEFAULT_DECRYPTOR,
            help="Path to DecryptNir.exe — handles raw SDK .jpg files",
        )
        decryptor: SDKDecryptor | None = None
        if Path(decryptor_path).exists():
            try:
                decryptor = SDKDecryptor(decryptor_path)
                st.success("✅ Decryptor ready")
            except Exception as e:
                st.warning(f"Decryptor load failed: {e}")
        else:
            st.caption(
                "_DecryptNir.exe not found — encrypted uploads will be "
                "rejected; only plain JPEG/PNG accepted._"
            )

        st.divider()
        st.markdown("### ⚠️ Danger Zone")
        if st.button("🗑️ Clear All Data", type="secondary"):
            db.clear_all()
            st.cache_resource.clear()
            st.success("Database cleared.")
            st.rerun()

        st.caption(
            f"DB: `{db_path}`\n"
            f"Size: {info['size_bytes'] / 1024:.1f} KB"
        )

    return engine, db, settings, decryptor


# ============================================================================
# Tab 1: Enrollment
# ============================================================================

def render_image_grid(images: list[Image.Image], captions: list[str] = None):
    """Render images in a responsive grid."""
    n = len(images)
    cols = st.columns(min(n, 3))
    for i, img in enumerate(images):
        with cols[i % 3]:
            cap = captions[i] if captions and i < len(captions) else f"Sample {i+1}"
            render_thumbnail(img, cap)


def tab_enrollment(engine: DeepLearningEngine, db: EnrollmentDB, decryptor: SDKDecryptor | None):
    st.subheader("📁 Enrollment — Register a New User")
    st.markdown(
        "Upload **1 finger vein image** — the system will generate **11 strong "
        "augmented variants** to create a diverse embedding distribution. "
        "Each enrollment saves 12 samples (1 real + 11 augmented) per user. "
        "Files are auto-decrypted if they come straight from the ENG SDK."
    )

    col_form, col_preview = st.columns([1, 2], gap="medium")

    with col_form:
        with st.form("enrollment_form", clear_on_submit=False):
            user_id = st.text_input(
                "User ID",
                placeholder="e.g. person_001",
                help="Unique identifier for this person",
            ).strip()

            name = st.text_input(
                "Name (optional)",
                placeholder="e.g. Nguyen Van A",
            ).strip()

            uploaded = st.file_uploader(
                "Upload 1 finger vein image (JPEG/PNG or raw SDK .jpg)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                accept_multiple_files=False,
                help=(
                    "Upload a single finger vein image. "
                    "The system will generate 11 augmented variants automatically. "
                    "Plain images, or encrypted files straight from the ENG SDK. "
                    "Encrypted files are auto-detected (32-byte magic header) and "
                    "decrypted via DecryptNir.exe."
                ),
            )

            submitted = st.form_submit_button(
                "💾 Enroll User (1 Real + 11 Augmented)",
                type="primary",
                use_container_width=True,
            )

        # Status messages
        if submitted:
            if not user_id:
                st.error("❌ User ID is required.")
                return
            if not uploaded:
                st.error("❌ Please upload exactly 1 finger vein image.")
                return

            # Load image (auto-decrypt if needed)
            img, src_label = load_image_smart(uploaded, decryptor)
            if img is None:
                st.error(f"❌ Could not load image: {src_label}")
                return

            # Extract real + 11 augmented embeddings
            with st.spinner("Extracting embedding + generating 11 augmented variants..."):
                try:
                    results = engine.enroll_with_augmentation(img, n_augmentations=11)
                except Exception as e:
                    st.error(f"❌ Feature extraction failed: {e}")
                    st.code(traceback.format_exc())
                    return

            # Store in DB
            db.upsert_user(user_id, name)
            saved = 0
            for i, (emb, is_aug, aug_config) in enumerate(results):
                if db.add_embedding(user_id, emb, sample_idx=i, source_path=uploaded.name, is_augmented=int(is_aug), aug_config=aug_config):
                    saved += 1

            if saved == len(results):
                st.success(f"Enrolled `{user_id}`: 1 real + 11 augmented = 12 samples")
            else:
                st.warning(
                    f"⚠️ **Enrolled `{user_id}`** — only {saved}/{len(results)} "
                    f"sample(s) saved (some failed)."
                )
            st.cache_resource.clear()
            st.rerun()

    with col_preview:
        if uploaded:
            st.markdown("**Preview**")
            img, src_label = load_image_smart(uploaded, decryptor)
            if img:
                render_thumbnail(img, f"{uploaded.name}\n({src_label})")

    # Enrolled users table
    st.divider()
    st.subheader("👥 Enrolled Users")
    users = db.list_users()
    if not users:
        st.info("No users enrolled yet. Use the form above to register the first user.")
    else:
        df = pd.DataFrame(users)
        df.columns = ["User ID", "Name", "Created At", "Samples"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_detail, col_emb = st.columns([1, 2])
        with col_detail:
            selected = st.selectbox(
                "Inspect user",
                options=[u["user_id"] for u in users],
            )
        embs = db.get_user_embeddings(selected)
        center = db.get_user_center(selected)
        real_count = sum(1 for e in embs if e[3] == 0)  # is_augmented == 0
        aug_count = sum(1 for e in embs if e[3] == 1)   # is_augmented == 1
        with col_detail:
            st.markdown(f"**{selected}**")
            if embs:
                st.write(f"Total samples: {len(embs)} (1 real + 11 augmented)")
                st.write(f"Embedding dim: {embs[0][1].shape[0]}")
                st.write(f"Center norm: {np.linalg.norm(center):.4f}")

                norms = [np.linalg.norm(e[1]) for e in embs]
                st.write(f"Sample norms: {[f'{n:.4f}' for n in norms]}")
        with col_emb:
            if embs:
                arr = np.stack([e[1] for e in embs])
                st.dataframe(
                    pd.DataFrame(arr).describe().T,
                    use_container_width=True,
                )


# ============================================================================
# Tab 2: Identification
# ============================================================================

def tab_identification(
    engine: DeepLearningEngine,
    db: EnrollmentDB,
    settings: dict,
    decryptor: SDKDecryptor | None,
):
    st.subheader("🎯 Identification — Scan & Match")
    st.markdown(
        "Upload a **query finger vein image** to identify who it belongs to. "
        "Encrypted SDK files are auto-decrypted on the fly."
    )

    # Pull thresholds from the settings dict (set by the sidebar)
    ident_threshold = settings["ident_threshold"]
    margin_threshold = settings["margin_threshold"]
    ratio_threshold = settings.get("ratio_threshold", 1.02)

    users = db.list_users()
    if not users:
        st.warning(
            "⚠️ No enrolled users. Go to the **Enrollment** tab to register the first person."
        )
        return

    gallery = db.get_all_centers()
    gallery_raw = db.get_all_embeddings_raw()
    if not gallery:
        st.warning("No embeddings in database. Please enroll users first.")
        return

    st.markdown(f"**Gallery**: {len(gallery)} enrolled users")

    # Show the active identification settings so the user knows which knobs
    # are in effect right now.
    st.info(
        f"Active identification settings -> "
        f"score >= {ident_threshold:.2f}, "
        f"margin >= {margin_threshold:.2f}, "
        f"ratio >= {ratio_threshold:.3f}"
    )

    col_query, col_results = st.columns([1, 2], gap="medium")

    with col_query:
        uploaded = st.file_uploader(
            "Upload query image (plain or encrypted SDK .jpg)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="A single finger vein image to identify",
        )

        if uploaded:
            img, src_label = load_image_smart(uploaded, decryptor)
            if img:
                render_query_image(img, f"Query - {uploaded.name} ({src_label})")
            else:
                st.error(f"Could not load image ({src_label}).")

        # ---- Strategy selector ----
        strategy = st.radio(
            "Matching strategy",
            options=[
                "Center (mean)",
                "Re-rank (max sample)",
                "Re-rank (top-2 mean)",
                "Mahalanobis (open-set)",
                "Mahalanobis + Re-rank",
            ],
            index=1,
            help=(
                "Center: compare query to the mean embedding of each user. "
                "Re-rank (max): first pick top candidates via centers, then "
                "re-score them by their BEST raw sample vs the query. "
                "Mahalanobis: statistical open-set — measures how far the "
                "query is from each user's MEAN+COVARIANCE region and runs "
                "a chi-square test to reject unknown persons with a real "
                "p-value. Best when imposter scores overlap with genuine."
            ),
        )
        strategy_key = {
            "Center (mean)": "center",
            "Re-rank (max sample)": "rerank_max",
            "Re-rank (top-2 mean)": "rerank_top2",
            "Mahalanobis (open-set)": "mahalanobis",
            "Mahalanobis + Re-rank": "mahalanobis_rerank",
        }[strategy]

        # Edge case: when gallery has only 1 user, the slider would have
        # min == max (both 1) which Streamlit rejects. Force min=1 in that
        # case (slider is also disabled when strategy=center).
        n_gallery = len(gallery)
        rerank_min = 2 if n_gallery >= 2 else 1
        rerank_max = max(rerank_min + 1, min(20, n_gallery))
        rerank_default = min(5, rerank_max)

        rerank_top_n = st.slider(
            "Re-rank top-N candidates",
            min_value=rerank_min,
            max_value=rerank_max,
            value=rerank_default,
            help="How many center candidates to re-rank against raw samples.",
            disabled=(strategy_key == "center"),
        )

        # Adaptive mode: when all scores are bunched in a low band (e.g.
        # mean=0.35) the query is genuinely far from every gallery user
        # but if top-1 is the highest we should still accept it instead of
        # rejecting every genuine user with a hard ratio gate.
        adaptive = st.checkbox(
            "Adaptive thresholds (recommended)",
            value=True,
            help=(
                "Relax ratio/margin gates when all scores are in a low "
                "tight band (mean < 0.6). True impostors in this regime "
                "are still rejected because top1 is usually >> threshold."
            ),
        )

        top_k_min = 1
        top_k_max = max(top_k_min + 1, min(10, len(gallery)))
        top_k_default = min(3, top_k_max)

        top_k = st.slider(
            "Show top-K results",
            min_value=top_k_min,
            max_value=top_k_max,
            value=top_k_default,
        )

        # ---- Mahalanobis-specific controls ----
        # Only shown when a Mahalanobis strategy is picked. Defaults are
        # conservative; users can tighten alpha to reduce false accepts.
        if strategy_key in ("mahalanobis", "mahalanobis_rerank"):
            with st.expander("🧪 Mahalanobis settings", expanded=False):
                mahal_alpha = st.slider(
                    "Significance level (alpha)",
                    min_value=0.001,
                    max_value=0.10,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help=(
                        "Chi-square significance threshold. A user is "
                        "ACCEPTED only if chi2.sf(d_M^2, df) >= 1 - alpha. "
                        "Smaller alpha = stricter = fewer false accepts. "
                        "0.01 is a good default."
                    ),
                    key="mahal_alpha",
                )
                mahal_min_cosine = st.slider(
                    "Minimum cosine (coarse floor)",
                    min_value=-1.0,
                    max_value=0.8,
                    value=-1.0,
                    step=0.05,
                    help=(
                        "If cosine(query, user_center) < this value the "
                        "user is rejected outright without even running "
                        "the chi-square test. Use -1.0 to disable the "
                        "floor (pure Mahalanobis) or 0.3-0.5 to add a "
                        "cosine safety belt."
                    ),
                    key="mahal_min_cosine",
                )
        else:
            # Fallbacks for the non-Mahalanobis strategies
            mahal_alpha = 0.01
            mahal_min_cosine = -1.0

        run_btn = st.button(
            "Identify",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    with col_results:
        if not run_btn:
            st.info("👆 Upload a query image and click **Identify** to start.")
            return

        img, src_label = load_image_smart(uploaded, decryptor)
        if img is None:
            st.error(f"Could not process the query image ({src_label}).")
            return

        with st.spinner("⏳ Extracting query embedding..."):
            try:
                query_emb = engine.extract(img)
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                st.code(traceback.format_exc())
                return

        with st.spinner(f"Searching {len(gallery)} users ({strategy})..."):
            if strategy_key == "center":
                results = engine.identify(
                    query_emb,
                    gallery,
                    threshold=ident_threshold,
                    margin_threshold=margin_threshold,
                    ratio_threshold=ratio_threshold,
                    top_k=len(gallery),
                    adaptive=adaptive,
                )
            elif strategy_key == "rerank_max":
                results = engine.identify_with_reranking(
                    query_emb,
                    gallery_raw=gallery_raw,
                    centers=gallery,
                    threshold=ident_threshold,
                    margin_threshold=margin_threshold,
                    ratio_threshold=ratio_threshold,
                    top_k=len(gallery),
                    rerank_top_n=rerank_top_n,
                    aggregation="max",
                    adaptive=adaptive,
                )
            elif strategy_key == "rerank_top2":
                results = engine.identify_with_reranking(
                    query_emb,
                    gallery_raw=gallery_raw,
                    centers=gallery,
                    threshold=ident_threshold,
                    margin_threshold=margin_threshold,
                    ratio_threshold=ratio_threshold,
                    top_k=len(gallery),
                    rerank_top_n=rerank_top_n,
                    aggregation="topk_mean",
                    adaptive=adaptive,
                )
            elif strategy_key == "mahalanobis":
                # Statistical open-set identification: per-user mean +
                # covariance + chi-square test.
                user_stats = db.get_all_user_stats()
                results = engine.identify_with_mahalanobis(
                    query_emb,
                    user_stats,
                    alpha=mahal_alpha,
                    min_cosine=mahal_min_cosine,
                    top_k=len(gallery),
                )
            else:  # mahalanobis_rerank
                # Two-stage: re-rank on raw samples, then verify the
                # top-N candidates with Mahalanobis + chi-square.
                user_stats = db.get_all_user_stats()
                results = engine.identify_with_mahalanobis_rerank(
                    query_emb,
                    gallery_raw=gallery_raw,
                    user_stats=user_stats,
                    alpha=mahal_alpha,
                    min_cosine=mahal_min_cosine,
                    top_k=len(gallery),
                    rerank_top_n=rerank_top_n,
                    aggregation="max",
                )

        top_results = results[:top_k]
        best = top_results[0] if top_results else None

        # Mahalanobis strategies return different fields than the cosine
        # strategies (no "margin"/"ratio"; instead "p_value", "mahal_sq",
        # "rejection"). Branch on strategy_key so the UI keeps its shape.
        is_mahal = strategy_key in ("mahalanobis", "mahalanobis_rerank")

        if best and best["is_match"]:
            if is_mahal:
                p_val = best.get("p_value", 0.0)
                d_m = best.get("mahalanobis", 0.0)
                rej = best.get("rejection") or "none"
                st.success(
                    f"## ✅ **IDENTIFIED: `{best['user_id']}`**\n"
                    f"Score: **{best.get('score', best.get('cosine', 0.0)):.4f}**  "
                    f"·  Mahalanobis d_M: **{d_m:.4f}**  "
                    f"·  p-value: **{p_val:.4f}**  (≥ {1.0 - mahal_alpha:.3f})"
                )
            else:
                best_ratio = best.get("ratio")
                ratio_str = f"{best_ratio:.4f}" if isinstance(best_ratio, (int, float)) and best_ratio != float("inf") else "∞"
                st.success(
                    f"## ✅ **IDENTIFIED: `{best['user_id']}`**\n"
                    f"Score: **{best['score']:.4f}**  "
                    f"(≥ {ident_threshold})  ·  "
                    f"Margin: **{best.get('margin', 0):.4f}**  "
                    f"(≥ {margin_threshold})  ·  "
                    f"Ratio: **{ratio_str}**  "
                    f"(≥ {ratio_threshold:.3f})"
                )
        else:
            if is_mahal:
                best_score = best.get("score", best.get("cosine", 0.0)) if best else 0.0
                best_user = best["user_id"] if best else "N/A"
                best_p = best.get("p_value", 0.0) if best else 0.0
                best_dm = best.get("mahalanobis", 0.0) if best else 0.0
                best_rej = best.get("rejection") if best else "no_candidates"
                st.error(
                    f"## ❌ **UNKNOWN PERSON**\n"
                    f"Best candidate: `{best_user}` @ cosine {best_score:.4f}, "
                    f"d_M={best_dm:.4f}, p-value={best_p:.4f} "
                    f"(need p ≥ {1.0 - mahal_alpha:.3f})"
                )
                with st.expander("Why was this rejected?", expanded=True):
                    reason_map = {
                        "cosine": f"Cosine too low (< {mahal_min_cosine:.2f}). Query is far from every enrolled user.",
                        "chi2": f"Chi-square test failed (p-value < {1.0 - mahal_alpha:.3f}). Query falls outside this user's Mahalanobis region.",
                    }
                    st.write(reason_map.get(best_rej, f"Rejection reason: {best_rej}"))
                    st.caption(
                        "Mahalanobis identification rejects unknown persons "
                        "statistically: d_M^2 follows chi-square(df={dim}) "
                        "under H0='query belongs to this user'. If p < 1-alpha, "
                        "we reject H0 and call it UNKNOWN."
                        .format(dim=engine.meta.get("embedding_dim", 128))
                    )
            else:
                best_score = best["score"] if best else 0.0
                best_user = best["user_id"] if best else "N/A"
                best_margin = best.get("margin") if best else None
                best_ratio = best.get("ratio") if best else None
                ratio_str = f"{best_ratio:.4f}" if isinstance(best_ratio, (int, float)) and best_ratio != float("inf") else "N/A"
                reason = best.get("reason", "Unknown") if best else "No gallery users"
                st.error(
                    "## ❌ **UNKNOWN PERSON**\n"
                    f"Best candidate: `{best_user}` @ {best_score:.4f} "
                    f"(margin={f'{best_margin:.4f}' if best_margin is not None else 'N/A'}, "
                    f"ratio={ratio_str})"
                )
                with st.expander("Why was this rejected?", expanded=True):
                    st.write(reason)
                    regime = best.get("regime") or "fixed"
                    score_mean = best.get("score_mean")
                    if score_mean is not None:
                        st.caption(
                            f"Regime: `{regime}` (mean score across all users = {score_mean:.4f}). "
                            "Identification uses 3 adaptive gates: similarity, "
                            "margin, and ratio. The gates relax when all scores "
                            "are in a low tight band so genuine users in unseen "
                            "poses are not falsely rejected."
                        )

        st.divider()

        # Top-K results table — show the columns that make sense for each
        # strategy. Classic strategies use "score"; Mahalanobis uses cosine
        # + p_value + mahalanobis.
        df_results = pd.DataFrame(top_results)
        if is_mahal:
            cols = [c for c in ["user_id", "cosine", "mahalanobis", "p_value", "is_match"] if c in df_results.columns]
            df_display = df_results[cols].copy()
            rename = {
                "user_id": "User ID",
                "cosine": "Cosine",
                "mahalanobis": "Mahalanobis d_M",
                "p_value": "p-value",
                "is_match": "Decision",
            }
            df_display = df_display.rename(columns=rename)
            df_display["Decision"] = df_display["Decision"].map(
                {True: "✅ ACCEPTED", False: "❌ REJECTED"}
            )
        else:
            df_display = df_results[["user_id", "score", "is_match"]].copy()
            df_display.columns = ["User ID", "Similarity", "Decision"]
            df_display["Decision"] = df_display["Decision"].map(
                {True: "✅ ACCEPTED", False: "❌ REJECTED"}
            )
        st.markdown(f"**Top-{top_k} candidates**")
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Bar chart of all gallery scores
        all_df = pd.DataFrame(results)
        if is_mahal:
            cols_all = [c for c in ["user_id", "cosine", "mahalanobis", "p_value", "is_match"] if c in all_df.columns]
            all_df_display = all_df[cols_all].copy()
            rename_all = {
                "user_id": "User ID",
                "cosine": "Cosine",
                "mahalanobis": "Mahalanobis d_M",
                "p_value": "p-value",
                "is_match": "Decision",
            }
            all_df_display = all_df_display.rename(columns=rename_all)
            all_df_display["Decision"] = all_df_display["Decision"].map(
                {True: "✅ ACCEPTED", False: "❌ REJECTED"}
            )
            # Use cosine as the y-axis for the bar chart (familiar metric).
            fig_bar = px.bar(
                all_df_display.head(min(20, len(all_df_display))),
                x="User ID",
                y="Cosine",
                color="Decision",
                color_discrete_map={"✅ ACCEPTED": "#2ECC71", "❌ REJECTED": "#E74C3C"},
                title="All Gallery Cosine Scores (top 20)",
            )
            fig_bar.add_hline(
                y=mahal_min_cosine,
                line_dash="dash",
                line_color="black",
                annotation_text=f"min_cosine={mahal_min_cosine:.2f}",
            )
        else:
            all_df_display = all_df[["user_id", "score", "is_match"]].copy()
            all_df_display.columns = ["User ID", "Similarity", "Decision"]
            all_df_display["Decision"] = all_df_display["Decision"].map(
                {True: "✅ ACCEPTED", False: "❌ REJECTED"}
            )
            fig_bar = px.bar(
                all_df_display.head(min(20, len(all_df_display))),
                x="User ID",
                y="Similarity",
                color="Decision",
                color_discrete_map={"✅ ACCEPTED": "#2ECC71", "❌ REJECTED": "#E74C3C"},
                title="All Gallery Scores (top 20)",
            )
            fig_bar.add_hline(
                y=ident_threshold,
                line_dash="dash",
                line_color="black",
                annotation_text=f"threshold={ident_threshold:.2f}",
            )
        fig_bar.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        all_scores = [r.get("score", r.get("cosine", 0.0)) for r in results]
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=all_scores,
                nbinsx=20,
                name="Gallery",
                marker_color="#4ECDC4",
                opacity=0.8,
            )
        )
        if best:
            fig_hist.add_trace(
                go.Scatter(
                    x=[best.get("score", best.get("cosine", 0.0))],
                    y=[0],
                    mode="markers",
                    marker=dict(size=18, color="red", symbol="star"),
                    name="Query",
                )
            )
        if is_mahal:
            fig_hist.add_vline(
                x=mahal_min_cosine,
                line_dash="dash",
                line_color="black",
                annotation_text=f"min_cosine={mahal_min_cosine:.2f}",
            )
            fig_hist.update_layout(
                title="Cosine Score Distribution",
                xaxis_title="Cosine Similarity",
                yaxis_title="# Users",
                template="plotly_white",
                height=280,
            )
        else:
            fig_hist.add_vline(
                x=ident_threshold,
                line_dash="dash",
                line_color="black",
                annotation_text=f"threshold={ident_threshold:.2f}",
            )
            fig_hist.update_layout(
                title="Score Distribution",
                xaxis_title="Cosine Similarity",
                yaxis_title="# Users",
                template="plotly_white",
                height=280,
            )
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================================
# Main
# ============================================================================

def main():
    st.title("🩸 Finger Vein E2E Pipeline")
    st.caption(
        "End-to-end biometric pipeline — enrollment → identification. "
        "Powered by ResNet18 + ArcFace deep learning. "
        "Option A: no legacy .dat — raw SDK files are auto-decrypted."
    )
    st.markdown("---")

    # Sidebar
    engine, db, settings, decryptor = render_sidebar()

    # Tabs
    tab_enroll, tab_ident = st.tabs([
        "📁 Enrollment",
        "🎯 Identification",
    ])

    with tab_enroll:
        tab_enrollment(engine, db, decryptor)

    with tab_ident:
        tab_identification(engine, db, settings, decryptor)

    st.markdown("---")
    st.caption(
        "Finger Vein E2E Pipeline · Deep Learning (ResNet18 + ArcFace) · "
        "Built with Streamlit · Option A"
    )


if __name__ == "__main__":
    main()
