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
MIN_SAMPLES = 3
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

def render_sidebar() -> tuple[DeepLearningEngine, EnrollmentDB, float, SDKDecryptor | None]:
    """Render sidebar and return (engine, db, threshold, decryptor)."""
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

        # Threshold
        default_t = float(engine.meta.get("threshold") or DEFAULT_THRESHOLD)
        threshold = st.slider(
            "Decision threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_t,
            step=0.01,
            help="Score ≥ threshold → MATCH",
        )

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

    return engine, db, threshold, decryptor


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
        f"Upload **at least {MIN_SAMPLES} finger vein images** for one finger/person. "
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
                f"Upload {MIN_SAMPLES}+ sample images (JPEG/PNG or raw SDK .jpg)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                accept_multiple_files=True,
                help=(
                    "Plain images, or encrypted files straight from the ENG SDK. "
                    "Encrypted files are auto-detected (32-byte magic header) and "
                    "decrypted via DecryptNir.exe."
                ),
            )

            submitted = st.form_submit_button(
                "💾 Enroll User",
                type="primary",
                use_container_width=True,
            )

        # Status messages
        if submitted:
            if not user_id:
                st.error("❌ User ID is required.")
                return
            if not uploaded or len(uploaded) < MIN_SAMPLES:
                st.error(
                    f"❌ Please upload at least {MIN_SAMPLES} images. "
                    f"You uploaded {len(uploaded) if uploaded else 0}."
                )
                return

            # Load images (auto-decrypt if needed)
            images: list[tuple[Image.Image, str]] = []
            failed: list[str] = []
            for uf in uploaded:
                img, src_label = load_image_smart(uf, decryptor)
                if img is None:
                    failed.append(uf.name)
                else:
                    images.append((img, f"{uf.name} ({src_label})"))

            if failed:
                st.warning(f"⚠️ Could not load: {', '.join(failed)}")

            if len(images) < MIN_SAMPLES:
                st.error(
                    f"❌ Only {len(images)} valid images. Need at least {MIN_SAMPLES}."
                )
                return

            # Extract embeddings
            with st.spinner(f"🔄 Extracting embeddings from {len(images)} images..."):
                try:
                    pil_images = [img for img, _ in images]
                    embeddings = engine.extract_batch(pil_images)
                except Exception as e:
                    st.error(f"❌ Feature extraction failed: {e}")
                    st.code(traceback.format_exc())
                    return

            # Store in DB
            db.upsert_user(user_id, name)
            saved = 0
            for i, (emb, (img, src_name)) in enumerate(zip(embeddings, images)):
                if db.add_embedding(user_id, emb, sample_idx=i, source_path=src_name):
                    saved += 1

            st.success(
                f"✅ **Enrolled `{user_id}`** — {saved}/{len(images)} samples saved."
            )
            st.cache_resource.clear()
            st.rerun()

    with col_preview:
        if uploaded:
            st.markdown(f"**Preview — {len(uploaded)} images selected**")
            rows = [uploaded[i: i + 3] for i in range(0, len(uploaded), 3)]
            for row in rows:
                cols = st.columns(len(row))
                for col_idx, uf in enumerate(row):
                    with cols[col_idx]:
                        img, src_label = load_image_smart(uf, decryptor)
                        if img:
                            render_thumbnail(img, f"{uf.name}\n({src_label})")

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
        with col_detail:
            st.markdown(f"**{selected}**")
            if embs:
                st.write(f"Samples: {len(embs)}")
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
    threshold: float,
    decryptor: SDKDecryptor | None,
):
    st.subheader("🎯 Identification — Scan & Match")
    st.markdown(
        "Upload a **query finger vein image** to identify who it belongs to. "
        "Encrypted SDK files are auto-decrypted on the fly."
    )

    users = db.list_users()
    if not users:
        st.warning(
            "⚠️ No enrolled users. Go to the **Enrollment** tab to register the first person."
        )
        return

    gallery = db.get_all_centers()
    if not gallery:
        st.warning("No embeddings in database. Please enroll users first.")
        return

    st.markdown(f"**Gallery**: {len(gallery)} enrolled users")

    col_query, col_results = st.columns([1, 2], gap="medium")

    with col_query:
        uploaded = st.file_uploader(
            "📤 Upload query image (plain or encrypted SDK .jpg)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="A single finger vein image to identify",
        )

        if uploaded:
            img, src_label = load_image_smart(uploaded, decryptor)
            if img:
                render_query_image(img, f"Query — {uploaded.name} ({src_label})")
            else:
                st.error(f"Could not load image ({src_label}).")

        top_k = st.slider(
            "Show top-K results",
            min_value=1,
            max_value=min(10, len(gallery)),
            value=3,
        )

        run_btn = st.button(
            "🔍 Identify",
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

        with st.spinner(f"🔎 Searching {len(gallery)} users..."):
            results = engine.identify(
                query_emb,
                gallery,
                threshold=threshold,
                top_k=len(gallery),
            )

        top_results = results[:top_k]
        best = top_results[0] if top_results else None

        if best and best["is_match"]:
            st.success(
                f"## ✅ **IDENTIFIED: `{best['user_id']}`**\n"
                f"Score: **{best['score']:.4f}**  (≥ {threshold})"
            )
        else:
            st.error(
                "## ❌ **UNKNOWN PERSON**\n"
                f"Best match: `{best['user_id'] if best else 'N/A'}` "
                f"({best['score']:.4f} if best else 'N/A') ≤ {threshold}"
            )

        st.divider()

        df_results = pd.DataFrame(top_results)
        df_results.columns = ["User ID", "Similarity", "Match?"]
        df_results["Match?"] = df_results["Match?"].map(
            {True: "✅ MATCH", False: "❌ NO MATCH"}
        )
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        all_df = pd.DataFrame(results)
        all_df.columns = ["User ID", "Similarity", "Match?"]
        all_df["Match?"] = all_df["Match?"].map(
            {True: "✅ MATCH", False: "❌ NO MATCH"}
        )

        fig_bar = px.bar(
            all_df.head(min(20, len(all_df))),
            x="User ID",
            y="Similarity",
            color="Match?",
            color_discrete_map={"✅ MATCH": "#2ECC71", "❌ NO MATCH": "#E74C3C"},
            title="All Gallery Scores (top 20)",
        )
        fig_bar.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"threshold={threshold:.2f}",
        )
        fig_bar.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        all_scores = [r["score"] for r in results]
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
                    x=[best["score"]],
                    y=[0],
                    mode="markers",
                    marker=dict(size=18, color="red", symbol="star"),
                    name="Query",
                )
            )
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"threshold={threshold:.2f}",
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
    engine, db, threshold, decryptor = render_sidebar()

    # Tabs
    tab_enroll, tab_ident = st.tabs([
        "📁 Enrollment",
        "🎯 Identification",
    ])

    with tab_enroll:
        tab_enrollment(engine, db, decryptor)

    with tab_ident:
        tab_identification(engine, db, threshold, decryptor)

    st.markdown("---")
    st.caption(
        "Finger Vein E2E Pipeline · Deep Learning (ResNet18 + ArcFace) · "
        "Built with Streamlit · Option A"
    )


if __name__ == "__main__":
    main()
