"""
Finger Vein Recognition — End-to-End Streamlit App
==================================================
Multi-tab app: Training Dashboard, Verify Pair, Batch Verify, Enrollment, Identification.

Usage:
    cd src
    streamlit run e2e_app.py

Architecture:
    Tab 1: Training Dashboard    (charts & history)
    Tab 2: Single Pair Verify   (upload 2 imgs → similarity → verdict)
    Tab 3: Batch Verify          (CSV → metrics table + download)
    Tab 4: Enrollment            (multi-sample upload → save embeddings to SQLite)
    Tab 5: Identification         (1 query img → top-K matches against DB)

Dependencies:
    streamlit, plotly, torch, torchvision, Pillow, numpy, pandas
    + src/infer.py, src/dashboard.py, src/enrollment_db.py
"""
import sys
import io
import csv
import json
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Ensure src dir is importable when run via `streamlit run`
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torch

# Local imports (sibling modules in src/)
from infer import (
    load_checkpoint,
    build_model,
    preprocess_image,
    extract_embedding,
    compute_similarity,
    verify_pair,
)
from enrollment_db import EnrollmentDB


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Finger Vein Recognition E2E",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_RESULTS_DIR = "results"
DEFAULT_DB_PATH = "results/enrollment.db"
DEFAULT_CHECKPOINT = "results/baseline_resnet18_arcface_expD/best_model.pth"


# ============================================================================
# CACHED MODEL LOADER
# ============================================================================

@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(checkpoint_path: str):
    """Load model + metadata. Cached across reruns.

    Returns: (model, meta, device)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta, state_dict = load_checkpoint(checkpoint_path, device)
    model = build_model(meta, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta, device


def discover_checkpoints(results_dir: str = DEFAULT_RESULTS_DIR) -> Dict[str, str]:
    """Return {display_name: full_path} for all best_model.pth files."""
    found = {}
    root = Path(results_dir)
    if not root.exists():
        return found
    for pth in root.rglob("best_model.pth"):
        if pth.is_file():
            display = str(pth.relative_to(root)).replace("\\", "/")
            found[display] = str(pth)
    return found


# ============================================================================
# SHARED HELPERS
# ============================================================================

def render_image_with_caption(img: Image.Image, caption: str, width: int = 250):
    """Render PIL image with caption."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, width=width)


def load_uploaded_image(uploaded_file) -> Optional[Image.Image]:
    """Load image from Streamlit UploadedFile, convert to grayscale."""
    try:
        img = Image.open(uploaded_file)
        if img.mode != 'L':
            img = img.convert('L')
        return img
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


def similarity_gauge(score: float, threshold: float) -> go.Figure:
    """Gauge chart showing similarity score vs threshold."""
    color = '#2ECC71' if score > threshold else '#E74C3C'
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={'reference': threshold, 'increasing': {'color': '#2ECC71'}, 'decreasing': {'color': '#E74C3C'}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [-1, threshold], 'color': '#FADBD8'},
                {'range': [threshold, 1.0], 'color': '#D5F5E3'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': threshold,
            },
        },
        number={'valueformat': '.4f', 'font': {'size': 32}},
        title={'text': 'Cosine Similarity', 'font': {'size': 18}},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def verdict_badge(verdict: str, score: float, threshold: float) -> str:
    """Markdown badge for verdict."""
    if verdict == 'GENUINE':
        return f"### :green[MATCH — score {score:.4f} > threshold {threshold:.4f}]"
    elif verdict == 'IMPOSTOR':
        return f"### :red[NO MATCH — score {score:.4f} < threshold {threshold:.4f}]"
    else:
        return f"### :orange[TIE — score {score:.4f} = threshold {threshold:.4f}]"


# ============================================================================
# TAB 1: TRAINING DASHBOARD (logic from dashboard.py)
# ============================================================================

@st.cache_data
def load_results(results_dir: str = DEFAULT_RESULTS_DIR) -> List[Dict]:
    results = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        json_file = model_dir / "results.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['model_dir'] = model_dir.name
                    data['results_path'] = str(json_file)
                    results.append(data)
            except Exception:
                pass
    return results


def render_training_chart(history: List[Dict], metric: str) -> go.Figure:
    df = pd.DataFrame(history)
    fig = go.Figure()
    if metric == "eer":
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=[h * 100 for h in df['val_eer']],
            mode='lines+markers', name='Val EER (%)',
            line=dict(color='#FF6B6B', width=3), marker=dict(size=10),
        ))
        fig.update_yaxes(title="EER (%)")
    elif metric == "auc":
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['val_auc'],
            mode='lines+markers', name='Val AUC',
            line=dict(color='#4ECDC4', width=3), marker=dict(size=10),
        ))
        fig.update_yaxes(title="AUC", range=[0.8, 1.01])
    elif metric == "loss":
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['train_loss'],
            mode='lines+markers', name='Train Loss',
            line=dict(color='#FF6B6B', width=3), marker=dict(size=10),
        ))
        fig.update_yaxes(title="Loss")
    fig.update_layout(
        title=f"Training Progress — {metric.upper()}",
        xaxis_title="Epoch", hovermode="x unified",
        template="plotly_white", height=400,
    )
    return fig


def render_comparison_chart(results: List[Dict], metric: str) -> Optional[go.Figure]:
    df_data = []
    for r in results:
        model_name = r.get('model', 'Unknown')
        if r.get('arcface'):
            model_name += " + ArcFace"
        if metric == "eer":
            value = r.get('final_test', {}).get('eer', 0) * 100
        elif metric == "auc":
            value = r.get('final_test', {}).get('auc', 0)
        elif metric == "hter":
            value = r.get('final_test', {}).get('hter', 0) * 100
        else:
            continue
        df_data.append({'Model': model_name, metric.upper(): value, 'Epochs': r.get('epochs', 0)})

    df = pd.DataFrame(df_data)
    if df.empty:
        return None
    if metric in ("eer", "hter"):
        df = df.sort_values(metric.upper(), ascending=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    fig = px.bar(df, x='Model', y=metric.upper(), color='Model',
                 color_discrete_sequence=colors, title=f"Model Comparison — {metric.upper()}")
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig.update_layout(showlegend=False, template="plotly_white", height=400)
    return fig


def tab_training_dashboard(results_dir: str):
    results = load_results(results_dir)

    if not results:
        st.warning("⚠️ No training results found.")
        st.code(f"python baseline_resnet.py --epochs 50 --arcface --device cuda", language="bash")
        return

    # Sidebar selectors
    with st.sidebar:
        st.markdown("### 📊 Dashboard Settings")
        metric = st.selectbox("Metric", ["eer", "auc", "loss"], key="dash_metric")

    model_names = [
        f"{r.get('model', 'Unknown')}" + (" + ArcFace" if r.get('arcface') else "")
        for r in results
    ]
    with st.sidebar:
        selected_model = st.selectbox("Select Model", model_names, key="dash_model")

    selected = None
    for r, name in zip(results, model_names):
        if name == selected_model:
            selected = r
            break

    if not selected:
        return

    col1, col2, col3, col4 = st.columns(4)
    eer = selected.get('final_test', {}).get('eer', 0) * 100
    auc = selected.get('final_test', {}).get('auc', 0)
    hter = selected.get('final_test', {}).get('hter', 0) * 100
    epochs = selected.get('epochs', 0)
    col1.metric("Test EER", f"{eer:.2f}%")
    col2.metric("Test AUC", f"{auc:.4f}")
    col3.metric("Test HTER", f"{hter:.2f}%")
    col4.metric("Epochs", epochs)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Training Progress")
        if selected.get('history'):
            st.plotly_chart(render_training_chart(selected['history'], metric), use_container_width=True)
    with col_r:
        st.subheader("🎯 Model Comparison")
        fig = render_comparison_chart(results, metric)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Training History")
    df = pd.DataFrame(selected.get('history', []))
    if not df.empty:
        if 'val_eer' in df.columns:
            df['val_eer'] = df['val_eer'].apply(lambda x: f"{x*100:.2f}%")
            df['val_hter'] = df['val_hter'].apply(lambda x: f"{x*100:.2f}%")
            df['val_auc'] = df['val_auc'].apply(lambda x: f"{x:.4f}")
            df['train_loss'] = df['train_loss'].apply(lambda x: f"{x:.4f}")
        if 'train_acc' in df.columns:
            df['train_acc'] = df['train_acc'].apply(lambda x: f"{x*100:.1f}%")
        if 'time_sec' in df.columns:
            df['time_sec'] = df['time_sec'].apply(lambda x: f"{x:.1f}s")
        st.dataframe(df, use_container_width=True)


# ============================================================================
# TAB 2: SINGLE PAIR VERIFY
# ============================================================================

def tab_single_pair(model, meta, device, threshold: float):
    st.subheader("🔍 Verify a Single Pair")
    st.markdown(
        "Upload two finger vein images to compute similarity and decide "
        ":green[Genuine] vs :red[Impostor]."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Image 1**")
        uploaded1 = st.file_uploader(
            "First image", type=["png", "jpg", "jpeg", "bmp"],
            key="pair_img1", label_visibility="collapsed",
        )
    with col2:
        st.markdown("**Image 2**")
        uploaded2 = st.file_uploader(
            "Second image", type=["png", "jpg", "jpeg", "bmp"],
            key="pair_img2", label_visibility="collapsed",
        )

    # Preview
    if uploaded1:
        img1 = load_uploaded_image(uploaded1)
        if img1:
            render_image_with_caption(img1, f"Image 1 — {uploaded1.name}")
    if uploaded2:
        img2 = load_uploaded_image(uploaded2)
        if img2:
            render_image_with_caption(img2, f"Image 2 — {uploaded2.name}")

    if uploaded1 and uploaded2:
        if st.button("🔎 Verify Pair", type="primary"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t1:
                    img1 = load_uploaded_image(uploaded1)
                    img1.save(t1.name)
                    p1 = t1.name
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t2:
                    img2 = load_uploaded_image(uploaded2)
                    img2.save(t2.name)
                    p2 = t2.name

                result = verify_pair(model, p1, p2, threshold, meta['img_size'], device)
                score = result['score']
                verdict = result['verdict']

                # Verdict badge
                st.markdown("---")
                st.markdown(verdict_badge(verdict, score, threshold))

                col_gauge, col_meta = st.columns([1, 1])
                with col_gauge:
                    st.plotly_chart(similarity_gauge(score, threshold), use_container_width=True)
                with col_meta:
                    st.markdown("#### Details")
                    st.json({
                        'cosine_similarity': round(score, 6),
                        'threshold': round(threshold, 6),
                        'verdict': verdict,
                        'embedding_dim': meta['embedding_dim'],
                        'image_size': meta['img_size'],
                        'trained_eer': f"{meta['eer']*100:.2f}%" if meta.get('eer') else 'N/A',
                    })

                # Cleanup
                Path(p1).unlink(missing_ok=True)
                Path(p2).unlink(missing_ok=True)
            except Exception as e:
                st.error(f"Verification failed: {e}")
                st.code(traceback.format_exc())


# ============================================================================
# TAB 3: BATCH VERIFY
# ============================================================================

def _compute_batch_metrics(rows: List[Dict], model, threshold: float, img_size: int, device) -> List[Dict]:
    out = []
    for row in rows:
        p1, p2 = row.get('img1'), row.get('img2')
        label = row.get('label')
        if not p1 or not p2:
            out.append({**row, 'score': None, 'verdict': 'ERROR', 'error': 'missing path'})
            continue
        try:
            r = verify_pair(model, p1, p2, threshold, img_size, device)
            row_out = {
                **row,
                'score': round(r['score'], 6),
                'verdict': r['verdict'],
            }
            if label is not None:
                # Convert label string/int to ground-truth genuine (1) / impostor (0)
                if isinstance(label, str):
                    gt = 1 if label.strip().lower() in ('1', 'genuine', 'match', 'true', 'same') else 0
                else:
                    gt = int(label)
                pred = 1 if r['verdict'] == 'GENUINE' else 0
                row_out['gt'] = gt
                row_out['pred'] = pred
                row_out['correct'] = (gt == pred)
            out.append(row_out)
        except Exception as e:
            out.append({**row, 'score': None, 'verdict': 'ERROR', 'error': str(e)})
    return out


def tab_batch_verify(model, meta, device, threshold: float):
    st.subheader("📋 Batch Verification from CSV")
    st.markdown(
        "Upload a CSV with columns `img1`, `img2` (and optional `label` where "
        "`1`/genuine vs `0`/impostor)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    use_sample = st.checkbox(
        "Or load sample pairs (synthetic from FV-USM subset)",
        value=False,
    )

    rows: List[Dict] = []
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            required = {'img1', 'img2'}
            if not required.issubset(df_in.columns):
                st.error(f"CSV must contain columns: {required}. Found: {set(df_in.columns)}")
                return
            rows = df_in.to_dict('records')
            st.success(f"Loaded {len(rows)} pairs from CSV")
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")
            return

    if use_sample:
        # Auto-pick a few pairs from FV-USM
        fvusm_root = Path("../data/external/Published_database_FV-USM_Dec2013/1st_session/extractedvein")
        if fvusm_root.exists():
            persons = sorted([d for d in fvusm_root.iterdir() if d.is_dir()])[:6]
            for i, p in enumerate(persons):
                imgs = sorted(p.glob("*.jpg"))[:2]
                if len(imgs) >= 2:
                    rows.append({'img1': str(imgs[0]), 'img2': str(imgs[1]), 'label': 1})
            # Impostor pairs
            for i in range(min(3, len(persons) - 1)):
                p1 = persons[i]
                p2 = persons[i + 1]
                imgs1 = sorted(p1.glob("*.jpg"))[:1]
                imgs2 = sorted(p2.glob("*.jpg"))[:1]
                if imgs1 and imgs2:
                    rows.append({'img1': str(imgs1[0]), 'img2': str(imgs2[0]), 'label': 0})
            st.info(f"Generated {len(rows)} sample pairs")
        else:
            st.error("FV-USM not found; cannot generate samples.")

    if rows and st.button("▶️ Run Batch Verification", type="primary"):
        progress = st.progress(0.0)
        results = []
        n = len(rows)
        for i, row in enumerate(rows):
            results.extend(_compute_batch_metrics([row], model, threshold, meta['img_size'], device))
            progress.progress((i + 1) / n)
        progress.empty()

        df_out = pd.DataFrame(results)
        st.markdown("### 📊 Results")
        st.dataframe(df_out, use_container_width=True)

        # Metrics summary if labels present
        if 'gt' in df_out.columns and 'pred' in df_out.columns:
            valid = df_out.dropna(subset=['gt', 'pred'])
            if len(valid) > 0:
                tp = ((valid['gt'] == 1) & (valid['pred'] == 1)).sum()
                fp = ((valid['gt'] == 0) & (valid['pred'] == 1)).sum()
                tn = ((valid['gt'] == 0) & (valid['pred'] == 0)).sum()
                fn = ((valid['gt'] == 1) & (valid['pred'] == 0)).sum()
                far = fp / max(1, fp + tn)
                frr = fn / max(1, tp + fn)
                hter = (far + frr) / 2
                acc = (tp + tn) / max(1, len(valid))

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{acc*100:.2f}%")
                c2.metric("FAR", f"{far*100:.2f}%")
                c3.metric("FRR", f"{frr*100:.2f}%")
                c4.metric("HTER", f"{hter*100:.2f}%")
                c5.metric("Pairs", len(valid))

                # Score distribution by class
                fig = px.histogram(
                    valid, x='score', color=valid['gt'].map({1: 'Genuine', 0: 'Impostor'}),
                    nbins=30, barmode='overlay', opacity=0.7,
                    title="Score Distribution by Class",
                )
                fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                              annotation_text=f"threshold={threshold:.4f}")
                st.plotly_chart(fig, use_container_width=True)

        # Download
        csv_buf = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results CSV",
            csv_buf,
            file_name=f"verification_results_{threshold:.4f}.csv",
            mime="text/csv",
        )


# ============================================================================
# TAB 4: ENROLLMENT
# ============================================================================

def get_db(db_path: str) -> EnrollmentDB:
    return EnrollmentDB(db_path)


def tab_enrollment(db: EnrollmentDB, model, meta, device):
    st.subheader("📁 User Enrollment")
    st.markdown(
        "Register a new user by uploading multiple finger vein samples. "
        "Embeddings are stored in SQLite for later identification."
    )

    with st.sidebar:
        st.markdown("### 📁 Enrollment")
        with st.expander("DB Info"):
            st.json(db.db_info())

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Register new user**")
        user_id = st.text_input("User ID", placeholder="e.g. person_001", key="enr_userid")
        name = st.text_input("Name (optional)", key="enr_name")
        uploaded_files = st.file_uploader(
            "Upload sample images (≥3 recommended)",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="enr_upload",
        )

        if uploaded_files:
            st.write(f"Selected: {len(uploaded_files)} images")

        enroll_btn = st.button("💾 Enroll User", type="primary", disabled=not (user_id and uploaded_files))

        if enroll_btn:
            if not user_id.strip():
                st.error("User ID required.")
            else:
                if db.add_user(user_id.strip(), name.strip()):
                    st.success(f"Created user '{user_id}'.")
                else:
                    st.info(f"User '{user_id}' already exists — adding new samples.")

                saved = 0
                failed = 0
                tmp_paths = []
                try:
                    for i, uf in enumerate(uploaded_files):
                        img = load_uploaded_image(uf)
                        if img is None:
                            failed += 1
                            continue
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                            img.save(t.name)
                            tmp_paths.append(t.name)
                        t1 = preprocess_image(t.name, meta['img_size'], device)
                        emb = extract_embedding(model, t1)
                        ok = db.add_embedding(
                            user_id.strip(),
                            emb,
                            sample_idx=i,
                            source_path=uf.name,
                        )
                        if ok:
                            saved += 1
                        else:
                            failed += 1
                finally:
                    for p in tmp_paths:
                        Path(p).unlink(missing_ok=True)

                st.success(f"Enrolled {saved} samples (failed: {failed}).")
                st.rerun()

    with col2:
        st.markdown("**Enrolled Users**")
        users = db.list_users()
        if not users:
            st.info("No users enrolled yet.")
        else:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)

            # Manage existing users
            st.markdown("---")
            st.markdown("**Manage**")
            user_to_delete = st.selectbox(
                "Select user to delete",
                options=[u['user_id'] for u in users],
                key="enr_delete_select",
            )
            if st.button("🗑️ Delete Selected User"):
                if db.delete_user(user_to_delete):
                    st.success(f"Deleted '{user_to_delete}'.")
                    st.rerun()
                else:
                    st.error("Delete failed.")

            if st.button("⚠️ Clear All Users"):
                db.clear_all()
                st.success("Cleared all users.")
                st.rerun()

            # Inspect embeddings for one user
            st.markdown("---")
            inspect_user = st.selectbox(
                "Inspect user embeddings",
                options=[u['user_id'] for u in users],
                key="enr_inspect",
            )
            embs = db.get_user_embeddings(inspect_user)
            if embs:
                st.write(f"{inspect_user}: {len(embs)} samples, dim={len(embs[0][1])}")
                arr = np.stack([e[1] for e in embs])
                st.dataframe(pd.DataFrame(arr).describe().T, use_container_width=True)


# ============================================================================
# TAB 5: IDENTIFICATION (1:N)
# ============================================================================

def tab_identification(db: EnrollmentDB, model, meta, device, threshold: float):
    st.subheader("🎯 Identification (1:N Matching)")
    st.markdown(
        "Upload a query finger vein image and find the closest enrolled users "
        "by cosine similarity."
    )

    users = db.list_users()
    if not users:
        st.warning("No enrolled users. Go to Enrollment tab first.")
        return

    # Build gallery: {user_id: mean_embedding}
    gallery = {}
    for u in users:
        embs = db.get_user_embeddings(u['user_id'])
        if embs:
            arr = np.stack([e[1] for e in embs])
            # Mean + re-normalize for robust center
            mean = arr.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            gallery[u['user_id']] = mean

    if not gallery:
        st.warning("Enrolled users have no embeddings.")
        return

    st.markdown(f"**Gallery**: {len(gallery)} users")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Query image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="ident_query",
        )
        if uploaded:
            img = load_uploaded_image(uploaded)
            if img:
                render_image_with_caption(img, f"Query — {uploaded.name}")
        top_k = st.slider("Top-K", min_value=1, max_value=min(10, len(gallery)), value=3)

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            img = load_uploaded_image(uploaded)
            img.save(t.name)
            tmp = t.name
        try:
            t1 = preprocess_image(tmp, meta['img_size'], device)
            query_emb = extract_embedding(model, t1)

            scores = []
            for uid, gallery_emb in gallery.items():
                score = float(np.dot(query_emb, gallery_emb))
                scores.append((uid, score))
            scores.sort(key=lambda x: x[1], reverse=True)

            top = scores[:top_k]

            with col2:
                st.markdown("**Top Matches**")
                df_scores = pd.DataFrame([
                    {'User ID': uid, 'Similarity': round(s, 6),
                     'Verdict': '✓ MATCH' if s > threshold else '✗ NO MATCH'}
                    for uid, s in top
                ])
                st.dataframe(df_scores, use_container_width=True)

                # Bar chart of top-K scores
                fig = px.bar(
                    df_scores, x='User ID', y='Similarity',
                    color='Verdict',
                    color_discrete_map={'✓ MATCH': '#2ECC71', '✗ NO MATCH': '#E74C3C'},
                    title="Top-K Similarity Scores",
                )
                fig.add_hline(y=threshold, line_dash="dash", line_color="black",
                              annotation_text=f"threshold={threshold:.4f}")
                st.plotly_chart(fig, use_container_width=True)

            # Best match verdict
            best_uid, best_score = top[0]
            st.markdown("---")
            if best_score > threshold:
                st.success(
                    f"### ✅ Best Match: **{best_uid}** "
                    f"(similarity {best_score:.4f} > threshold {threshold:.4f})"
                )
            else:
                st.error(
                    f"### ❌ No Confident Match — "
                    f"Best: {best_uid} ({best_score:.4f}) ≤ threshold {threshold:.4f}"
                )

            # Score distribution
            all_scores = [s for _, s in scores]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=all_scores, nbinsx=20, name='Gallery scores', marker_color='#4ECDC4',
            ))
            fig2.add_trace(go.Scatter(
                x=[best_score], y=[0], mode='markers',
                marker=dict(size=18, color='red', symbol='star'),
                name='Query', showlegend=True,
            ))
            fig2.add_vline(x=threshold, line_dash="dash", line_color="black",
                           annotation_text=f"threshold={threshold:.4f}")
            fig2.update_layout(
                title="Gallery Score Distribution",
                xaxis_title="Cosine Similarity",
                yaxis_title="# Users",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)
        finally:
            Path(tmp).unlink(missing_ok=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🩸 Finger Vein Recognition — End-to-End App")
    st.caption("Upload images, verify, enroll, identify. Built on ResNet18 + ArcFace.")
    st.markdown("---")

    # ----------------------------------------------------------------
    # Sidebar: model + threshold + DB path
    # ----------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚙️ Global Settings")

        results_dir = st.text_input("Results dir", value=DEFAULT_RESULTS_DIR, key="results_dir")
        ckpts = discover_checkpoints(results_dir)
        if not ckpts:
            st.error("No checkpoints found.")
            ckpt_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
        else:
            # Put default expD at top if exists
            sorted_keys = sorted(ckpts.keys(), key=lambda k: (0 if 'expD' in k else 1, k))
            default_idx = 0
            ckpt_label = st.selectbox("Checkpoint", sorted_keys, index=default_idx)
            ckpt_path = ckpts[ckpt_label]

        # Load model (cached)
        try:
            model, meta, device = get_cached_model(ckpt_path)
            st.success(f"Model loaded: dim={meta['embedding_dim']}, classes={meta['num_classes']}")
            if meta.get('threshold') is not None:
                st.caption(f"Trained threshold: {meta['threshold']:.4f}")
            if meta.get('eer') is not None:
                st.caption(f"Trained EER: {meta['eer']*100:.2f}%")
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            st.stop()

        default_threshold = float(meta.get('threshold') or 0.95)
        threshold = st.slider(
            "Decision threshold",
            min_value=0.0, max_value=1.0, value=default_threshold, step=0.01,
            help="Cosine similarity > threshold → GENUINE.",
        )

        db_path = st.text_input("Enrollment DB path", value=DEFAULT_DB_PATH)
        db = get_db(db_path)

    # ----------------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------------
    tabs = st.tabs([
        "📊 Training Dashboard",
        "🔍 Single Pair Verify",
        "📋 Batch Verify",
        "📁 Enrollment",
        "🎯 Identification",
    ])

    with tabs[0]:
        tab_training_dashboard(results_dir)

    with tabs[1]:
        tab_single_pair(model, meta, device, threshold)

    with tabs[2]:
        tab_batch_verify(model, meta, device, threshold)

    with tabs[3]:
        tab_enrollment(db, model, meta, device)

    with tabs[4]:
        tab_identification(db, model, meta, device, threshold)

    st.markdown("---")
    st.caption("E2E Finger Vein App · Built with Streamlit · ResNet18 + ArcFace")


if __name__ == "__main__":
    main()
