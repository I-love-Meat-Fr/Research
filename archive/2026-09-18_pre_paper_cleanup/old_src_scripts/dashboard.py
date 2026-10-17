"""
Finger Vein Recognition Training Dashboard
==========================================
Real-time visualization of training progress with Streamlit.

Usage:
    source venv/bin/activate
    streamlit run src/dashboard.py

Author: Research Team
Date: 2026-08-10
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Finger Vein Recognition Dashboard",
    page_icon="🔬",
    layout="wide",
)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_results(results_dir: str = "results"):
    """Load all results from the results directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return []
    
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        json_file = model_dir / "results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['model_dir'] = model_dir.name
                data['results_path'] = str(json_file)
                results.append(data)
    
    return results


def create_training_chart(history: list, metric: str = "eer"):
    """Create training progress chart."""
    df = pd.DataFrame(history)
    df['epoch'] = df['epoch']
    
    fig = go.Figure()
    
    if metric == "eer":
        fig.add_trace(go.Scatter(
            x=df['epoch'], 
            y=[h * 100 for h in df['val_eer']],
            mode='lines+markers',
            name='Val EER (%)',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10)
        ))
        fig.update_yaxes(title="EER (%)", range=[0, min(20, max(df['val_eer']) * 100 + 5)])
    elif metric == "auc":
        fig.add_trace(go.Scatter(
            x=df['epoch'], 
            y=df['val_auc'],
            mode='lines+markers',
            name='Val AUC',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=10)
        ))
        fig.update_yaxes(title="AUC", range=[0.8, 1.01])
    elif metric == "loss":
        fig.add_trace(go.Scatter(
            x=df['epoch'], 
            y=df['train_loss'],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10)
        ))
        fig.update_yaxes(title="Loss")
    
    fig.update_layout(
        title=f"Training Progress - {metric.upper()}",
        xaxis_title="Epoch",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    
    return fig


def create_comparison_chart(results: list, metric: str = "eer"):
    """Create model comparison chart."""
    df_data = []
    
    for r in results:
        model_name = f"{r.get('model', 'Unknown')}"
        if r.get('arcface'):
            model_name += " + ArcFace"
        
        if metric == "eer":
            value = r.get('final_test', {}).get('eer', 0) * 100
        elif metric == "auc":
            value = r.get('final_test', {}).get('auc', 0)
        elif metric == "hter":
            value = r.get('final_test', {}).get('hter', 0) * 100
        
        df_data.append({
            'Model': model_name,
            metric.upper(): value,
            'Epochs': r.get('epochs', 0),
        })
    
    df = pd.DataFrame(df_data)
    
    if df.empty:
        return None
    
    # Sort by metric value
    if metric == "eer" or metric == "hter":
        df = df.sort_values(metric.upper(), ascending=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    fig = px.bar(
        df, 
        x='Model', 
        y=metric.upper(),
        color='Model',
        color_discrete_sequence=colors,
        title=f"Model Comparison - {metric.upper()}"
    )
    
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        height=400,
    )
    
    # Add value labels on bars
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    
    return fig


def create_roc_curve_placeholder():
    """Create placeholder for ROC curve."""
    fig = go.Figure()
    
    # Simulated ROC curve
    fpr = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    tpr = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.85, 0.95, 0.98, 0.99, 1.0]
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name='Model',
        line=dict(color='#4ECDC4', width=3),
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', width=1, dash='dash'),
    ))
    
    fig.update_layout(
        title="ROC Curve (Placeholder)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=400,
    )
    
    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    # Header
    st.title("🔬 Finger Vein Recognition Dashboard")
    st.markdown("---")
    
    # Load results
    results_dir = st.sidebar.text_input("Results Directory", value="results")
    results = load_results(results_dir)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("---")
    
    if not results:
        st.warning("⚠️ No results found. Run training first:")
        st.code("""
source venv/bin/activate
python src/baseline_resnet.py --epochs 10 --arcface --device cuda
        """, language="bash")
        return
    
    # Model selector
    st.sidebar.subheader("📊 Model Selection")
    model_names = [f"{r.get('model', 'Unknown')}" + (" + ArcFace" if r.get('arcface') else "") 
                   for r in results]
    selected_model = st.sidebar.selectbox("Select Model", model_names)
    
    # Find selected result
    selected_result = None
    for r in results:
        model_name = f"{r.get('model', 'Unknown')}" + (" + ArcFace" if r.get('arcface') else "")
        if model_name == selected_model:
            selected_result = r
            break
    
    # Metrics selector
    st.sidebar.subheader("📈 Metric")
    metric = st.sidebar.selectbox("Select Metric", ["eer", "auc", "loss"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📁 Results: {len(results)} models")
    
    # =========================================================================
    # Main Content
    # =========================================================================
    
    if selected_result:
        # Model info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            eer = selected_result.get('final_test', {}).get('eer', 0) * 100
            st.metric("Test EER", f"{eer:.2f}%", delta=f"{-eer + 15:.2f}%" if eer < 15 else None)
        
        with col2:
            auc = selected_result.get('final_test', {}).get('auc', 0)
            st.metric("Test AUC", f"{auc:.4f}")
        
        with col3:
            hter = selected_result.get('final_test', {}).get('hter', 0) * 100
            st.metric("Test HTER", f"{hter:.2f}%")
        
        with col4:
            epochs = selected_result.get('epochs', 0)
            st.metric("Epochs", epochs)
        
        st.markdown("---")
        
        # Training charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📈 Training Progress")
            if selected_result.get('history'):
                fig = create_training_chart(selected_result['history'], metric)
                st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("🎯 Model Comparison")
            fig = create_comparison_chart(results, metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed metrics table
        st.subheader("📋 Training History")
        
        df = pd.DataFrame(selected_result.get('history', []))
        df['epoch'] = df['epoch']
        df['val_eer'] = df['val_eer'].apply(lambda x: f"{x*100:.2f}%")
        df['val_hter'] = df['val_hter'].apply(lambda x: f"{x*100:.2f}%")
        df['val_auc'] = df['val_auc'].apply(lambda x: f"{x:.4f}")
        df['train_loss'] = df['train_loss'].apply(lambda x: f"{x:.4f}")
        df['train_acc'] = df['train_acc'].apply(lambda x: f"{x*100:.1f}%")
        df['time_sec'] = df['time_sec'].apply(lambda x: f"{x:.1f}s")
        
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # ROC Curve placeholder
        st.subheader("📉 ROC Curve")
        fig = create_roc_curve_placeholder()
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Summary view when no model selected
        st.subheader("📊 All Models Summary")
        
        summary_data = []
        for r in results:
            model_name = f"{r.get('model', 'Unknown')}"
            if r.get('arcface'):
                model_name += " + ArcFace"
            
            summary_data.append({
                'Model': model_name,
                'Epochs': r.get('epochs', 0),
                'Test EER (%)': f"{r.get('final_test', {}).get('eer', 0) * 100:.2f}",
                'Test AUC': f"{r.get('final_test', {}).get('auc', 0):.4f}",
                'Test HTER (%)': f"{r.get('final_test', {}).get('hter', 0) * 100:.2f}",
                'Batch Size': r.get('batch_size', 0),
                'LR': r.get('lr', 0),
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_comparison_chart(results, "eer")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_comparison_chart(results, "auc")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
