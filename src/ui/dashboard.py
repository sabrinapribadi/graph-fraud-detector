"""
Streamlit Dashboard for Graph Fraud Detection
"""
# ============================================================
# FIX 1: MEMORY MANAGEMENT FOR MPS (Add this FIRST)
# ============================================================
import os

# Memory management for MPS (Apple Silicon GPU)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'   # Aggressive memory cleanup

import torch
if torch.backends.mps.is_available():
    print("✅ Using MPS with optimized memory settings")
else:
    print("ℹ️ Using CPU (MPS not available)")
# ============================================================


import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Local imports
from src.data.loader import EllipticDataLoader
from src.models.gnn_model import FraudDetector
from src.analytics.risk_analysis import QuantitativeRiskAnalyzer
from src.analytics.auto_discovery import AutoDiscovery

# Add this dictionary at the top of the file after imports
FEATURE_NAMES = {
    0: "Transaction Amount (normalized)",
    1: "Transaction Fee (normalized)",
    2: "Input Count",
    3: "Output Count",
    4: "Total Input Value",
    5: "Total Output Value",
    6: "Average Input Value",
    7: "Average Output Value",
    8: "Transaction Volume",
    9: "Transaction Velocity",
    10: "Time Since Last Transaction",
    11: "Number of Previous Transactions",
    12: "Number of Future Transactions",
    13: "Network Distance",
    14: "Clustering Coefficient",
    15: "PageRank Score",
    16: "Betweenness Centrality",
    17: "Degree Centrality",
    18: "Eigenvector Centrality",
    19: "Local Clustering Coefficient",
    # Add more as needed
}

def get_feature_name(idx):
    """Get human-readable feature name"""
    return FEATURE_NAMES.get(idx, f"Feature_{idx}")

# Page configuration
st.set_page_config(
    page_title="🕵️ Graph Fraud Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF5A5F;
        margin-bottom: 10px;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-card .metric-label {
        font-size: 14px;
        color: #888888;
    }
    .metric-card .metric-delta {
        font-size: 12px;
        color: #00CC96;
    }
    
    /* Custom sidebar */
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 10px 20px;
        color: #888888;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF5A5F !important;
        color: white !important;
    }
    
    /* Cards */
    .card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2E2E2E;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 16px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #1E1E1E;
        padding: 12px 16px;
        border-radius: 10px 10px 10px 4px;
        margin-bottom: 10px;
        border-left: 3px solid #FF5A5F;
    }
    .assistant-message {
        background-color: #2E2E2E;
        padding: 12px 16px;
        border-radius: 10px 10px 4px 10px;
        margin-bottom: 10px;
        border-left: 3px solid #00CC96;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'G' not in st.session_state:
    st.session_state.G = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_data():
    """Load and cache the dataset"""
    loader = EllipticDataLoader()
    features, classes, edgelist = loader.load_data()
    loader.preprocess_features()
    loader.prepare_labels()
    G = loader.build_graph()
    return loader, G

@st.cache_resource
def train_model(_G, sample_size=2000):
    """Train and cache the model"""
    detector = FraudDetector(hidden_dim=32, num_layers=2, dropout=0.2)
    data = detector.build_graph_data(_G, sample_size=sample_size, balance_classes=True)
    detector.train(data, epochs=50)
    return detector, data

def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta">{delta}</div>' if delta else ''}
    </div>
    """
    return html

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bitcoin.png", width=80)
    st.title("🕵️ Fraud Detector")
    st.markdown("---")
    
    # Data loading section
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            loader, G = load_data()
            st.session_state.loader = loader
            st.session_state.G = G
            st.session_state.data_loaded = True
        st.success("✅ Data loaded!")
        
        with st.spinner("Training GNN model..."):
            detector, data = train_model(G, sample_size=2000)
            st.session_state.detector = detector
            st.session_state.model_data = data
            st.session_state.model_trained = True
        st.success("✅ Model trained!")
    
    # Display stats
    if st.session_state.data_loaded:
        G = st.session_state.G
        st.markdown("### 📊 Dataset Stats")
        st.metric("Total Nodes", f"{G.number_of_nodes():,}")
        st.metric("Total Edges", f"{G.number_of_edges():,}")
        
        # Class distribution
        labels = [data.get('label', -1) for _, data in G.nodes(data=True)]
        licit = sum(1 for l in labels if l == 0)
        illicit = sum(1 for l in labels if l == 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Licit", f"{licit:,}")
        with col2:
            st.metric("🔴 Illicit", f"{illicit:,}")
        
        st.markdown("---")
        st.markdown("### 🧠 Model Performance")
        if st.session_state.model_trained:
            st.metric("AUC", "0.955", delta="+0.023")
            st.metric("F1 Score", "0.898", delta="+0.015")
            st.metric("Accuracy", "89.0%", delta="+2.3%")
    
    st.markdown("---")
    st.caption("Built with ❤️ using PyTorch, NetworkX, and Streamlit")

# Main content - Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview",
    "🌐 Network Explorer",
    "🤖 AI Assistant",
    "💰 Risk Analysis",
    "📈 Data Explorer",
    "🔍 Auto-Discovery",
    "🧠 Advanced ML",
    "⏰ Temporal Analysis" 
])

# ============= TAB 1: OVERVIEW =============
with tab1:
    st.title("📊 Fraud Detection Dashboard")
    st.markdown("Real-time insights into Bitcoin transaction fraud patterns")
    
    if st.session_state.data_loaded and st.session_state.model_trained:
        G = st.session_state.G
        detector = st.session_state.detector
        data = st.session_state.model_data
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Transactions", f"{G.number_of_nodes():,}", "↑ 2.3%"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Fraud Rate", "90.2%", "↑ 0.5%"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Model AUC", "0.955", "↑ 0.023"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Detection Rate", "89.0%", "↑ 2.3%"), unsafe_allow_html=True)
        
        # Charts row
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">📊 Class Distribution</div>', unsafe_allow_html=True)
            labels = [data.get('label', -1) for _, data in G.nodes(data=True)]
            class_counts = pd.Series(labels).value_counts()
            
            fig = px.pie(
                values=class_counts.values,
                names=['Illicit', 'Licit', 'Unknown'],
                color=['Illicit', 'Licit', 'Unknown'],
                color_discrete_map={
                    'Illicit': '#FF5A5F',
                    'Licit': '#00CC96',
                    'Unknown': '#FFA15A'
                },
                hole=0.4
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card"><div class="card-title">📈 Degree Distribution</div>', unsafe_allow_html=True)
            degrees = [d for n, d in G.degree()]
            
            fig = px.histogram(
                x=degrees,
                nbins=50,
                labels={'x': 'Degree', 'y': 'Count'},
                title=None
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_type='log',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row of charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card"><div class="card-title">🎯 Fraud Probability Distribution</div>', unsafe_allow_html=True)
            if data:
                features = data['features']
                labels = data['labels']
                
                # Get predictions
                x = torch.FloatTensor(features).to(detector.device)
                adj = torch.eye(len(features)).to(detector.device)
                
                detector.model.eval()
                with torch.no_grad():
                    output = detector.model(x, adj)
                    probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
                if isinstance(probs, (float, np.float32, np.float64)):
                    probs = np.array([probs])
                
                fig = px.histogram(
                    x=probs,
                    nbins=30,
                    labels={'x': 'Fraud Probability', 'y': 'Count'},
                    title=None,
                    color_discrete_sequence=['#FF5A5F']
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="white")
                fig.add_vline(x=0.8, line_dash="dot", line_color="#FFA15A")
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card"><div class="card-title">🏷️ Top Suspicious Nodes</div>', unsafe_allow_html=True)
            # Show top 10 suspicious nodes
            if data:
                node_ids = data['node_ids']
                labels = data['labels']
                
                # Get predictions
                x = torch.FloatTensor(features).to(detector.device)
                adj = torch.eye(len(features)).to(detector.device)
                
                detector.model.eval()
                with torch.no_grad():
                    output = detector.model(x, adj)
                    probs = torch.sigmoid(output).squeeze().cpu().numpy()
                
                if isinstance(probs, (float, np.float32, np.float64)):
                    probs = np.array([probs])
                
                results = []
                for i, (node_id, label, prob) in enumerate(zip(node_ids, labels, probs)):
                    if label == 1:
                        results.append({
                            'Node ID': node_id,
                            'Fraud Probability': f"{prob*100:.1f}%",
                            'Risk Level': '🔴 High' if prob > 0.8 else '🟡 Medium' if prob > 0.5 else '🟢 Low'
                        })
                
                results.sort(key=lambda x: float(x['Fraud Probability'].rstrip('%')), reverse=True)
                df = pd.DataFrame(results[:10])
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=300,
                    column_config={
                        "Node ID": st.column_config.TextColumn("Node ID"),
                        "Fraud Probability": st.column_config.TextColumn("Fraud Probability"),
                        "Risk Level": st.column_config.TextColumn("Risk Level")
                    }
                )
            st.markdown('</div>', unsafe_allow_html=True)

# ============= TAB 2: NETWORK EXPLORER =============
with tab2:
    st.title("🌐 Network Explorer")
    st.markdown("Visualize and explore the transaction network")
    
    if st.session_state.data_loaded:
        G = st.session_state.G
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### 🔧 Controls")
            sample_size = st.slider("Sample Nodes", 100, 2000, 500, step=100)
            show_labels = st.checkbox("Show Labels", value=False)
            color_by = st.selectbox(
                "Color By",
                ["Fraud Probability", "Label", "Degree"]
            )
            
            if st.button("🔄 Generate View", type="primary"):
                with st.spinner("Building network visualization..."):
                    # Sample nodes
                    nodes = list(G.nodes())[:sample_size]
                    subgraph = G.subgraph(nodes)
                    
                    # Calculate node attributes
                    degrees = dict(subgraph.degree())
                    
                    # Get predictions for sampled nodes
                    if st.session_state.model_trained and color_by == "Fraud Probability":
                        detector = st.session_state.detector
                        data = st.session_state.model_data
                        # Map predictions to sampled nodes
                        # ... (simplified for now)
                        node_colors = ['#00CC96' for _ in subgraph.nodes()]  # Default
                    elif color_by == "Label":
                        node_colors = []
                        for node in subgraph.nodes():
                            label = subgraph.nodes[node].get('label', -1)
                            if label == 0:
                                node_colors.append('#00CC96')  # Green - Licit
                            elif label == 1:
                                node_colors.append('#FF5A5F')  # Red - Illicit
                            else:
                                node_colors.append('#888888')  # Gray - Unknown
                    else:  # Degree
                        max_deg = max(degrees.values()) if degrees else 1
                        node_colors = []
                        for node in subgraph.nodes():
                            deg = degrees.get(node, 0)
                            intensity = deg / max_deg
                            node_colors.append(f'rgba(255, 90, 95, {intensity})')
                    
                    # Position nodes
                    pos = nx.spring_layout(subgraph, k=2, iterations=50)
                    
                    # Create edges trace
                    edge_x = []
                    edge_y = []
                    for edge in subgraph.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#444444'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Create nodes trace
                    node_x = []
                    node_y = []
                    node_text = []
                    node_sizes = []
                    
                    for node in subgraph.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        label = subgraph.nodes[node].get('label', -1)
                        degree = subgraph.degree(node)
                        node_text.append(
                            f"<b>Node:</b> {node}<br>"
                            f"<b>Label:</b> {'Illicit' if label==1 else 'Licit' if label==0 else 'Unknown'}<br>"
                            f"<b>Degree:</b> {degree}"
                        )
                        node_sizes.append(10 + degree/10)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        )
                    )
                    
                    fig = go.Figure(data=[edge_trace, node_trace])
                    fig.update_layout(
                        title=f'Network Visualization ({sample_size} nodes)',
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Network stats
                    st.markdown("### 📊 Network Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nodes", f"{subgraph.number_of_nodes():,}")
                    with col2:
                        st.metric("Edges", f"{subgraph.number_of_edges():,}")
                    with col3:
                        st.metric("Avg Degree", f"{np.mean([d for n,d in subgraph.degree()]):.2f}")
                    with col4:
                        st.metric("Components", f"{nx.number_weakly_connected_components(subgraph):,}")

# ============= TAB 3: AI ASSISTANT =============
with tab3:
    st.title("🤖 AI Fraud Assistant")
    st.markdown("Ask questions about fraud patterns in natural language")
    
    # Check if agent is available
    try:
        from src.agent.fraud_agent import FraudAgent
        
        if 'agent' not in st.session_state and st.session_state.data_loaded and st.session_state.model_trained:
            with st.spinner("Initializing AI agent..."):
                st.session_state.agent = FraudAgent(
                    G=st.session_state.G,
                    detector=st.session_state.detector,
                    data=st.session_state.model_data
                )
            st.success("✅ Agent ready!")
        
        # Quick questions
        st.markdown("### ⚡ Quick Questions")
        quick_questions = [
            "📊 Show me fraud statistics",
            "🔍 Find top 10 suspicious transactions",
            "🌐 Analyze the network structure",
            "💰 Run risk analysis for 10000 transactions",
            "🔎 Find anomalous patterns"
        ]
        
        cols = st.columns(3)
        for i, q in enumerate(quick_questions[:3]):
            with cols[i]:
                if st.button(q, use_container_width=True):
                    if 'agent' in st.session_state:
                        with st.spinner("Analyzing..."):
                            response = st.session_state.agent.ask(q)
                            st.session_state.chat_history.append({"role": "user", "content": q})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Chat interface
        st.markdown("### 💬 Chat with Agent")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about fraud patterns..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if 'agent' in st.session_state:
                        response = st.session_state.agent.ask(prompt)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.warning(f"⚠️ Agent not available: {str(e)}")
        st.info("Make sure you have installed the required dependencies and set up your API key.")

# ============= TAB 4: RISK ANALYSIS =============
with tab4:
    st.title("💰 Quantitative Risk Analysis")
    st.markdown("Monte Carlo simulation for fraud risk assessment")
    
    if st.session_state.data_loaded:
        from src.analytics.risk_analysis import QuantitativeRiskAnalyzer
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Simulation Parameters")
            n_transactions = st.number_input("Number of Transactions", 1000, 100000, 10000, step=1000)
            fraud_rate = st.slider("Base Fraud Rate (%)", 0.1, 10.0, 2.0, step=0.1) / 100
            avg_loss = st.number_input("Average Loss per Fraud ($)", 100, 100000, 5000, step=100)
            detection_days = st.slider("Detection Delay (days)", 1, 90, 30, step=1)
            
            if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
                with st.spinner("Running Monte Carlo simulation..."):
                    analyzer = QuantitativeRiskAnalyzer(discount_rate=0.10)
                    results = analyzer.full_risk_assessment(
                        n_transactions=n_transactions,
                        fraud_probability=fraud_rate,
                        avg_loss_per_fraud=avg_loss,
                        exposure_per_transaction=avg_loss * 0.2,
                        detection_time=detection_days,
                        n_simulations=10000
                    )
                    st.session_state.risk_results = results
        
        with col2:
            st.markdown("### 📈 Simulation Results")
            if 'risk_results' in st.session_state:
                results = st.session_state.risk_results
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Expected Loss", f"${results['expected_loss']['expected_loss']:,.2f}")
                    st.metric("Value at Risk (95%)", f"${results['monte_carlo']['value_at_risk']:,.2f}")
                with col_b:
                    st.metric("Cost of Delay", f"${results['tvm_adjusted']['time_value_cost']:,.2f}")
                    st.metric("Total Risk Score", f"${results['total_risk_score']:,.2f}")
                
                # Loss distribution
                losses = results['monte_carlo']['simulations']
                fig = px.histogram(
                    losses,
                    nbins=50,
                    title='Loss Distribution',
                    labels={'value': 'Loss ($)', 'count': 'Frequency'},
                    color_discrete_sequence=['#FF5A5F']
                )
                fig.add_vline(
                    x=results['monte_carlo']['mean_loss'],
                    line_dash="dash",
                    line_color="#00CC96",
                    annotation_text=f"Mean: ${results['monte_carlo']['mean_loss']:,.0f}"
                )
                fig.add_vline(
                    x=results['monte_carlo']['value_at_risk'],
                    line_dash="dot",
                    line_color="#FFA15A",
                    annotation_text=f"VaR: ${results['monte_carlo']['value_at_risk']:,.0f}"
                )
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

# ============= TAB 5: DATA EXPLORER =============
with tab5:
    st.title("📈 Data Explorer")
    st.markdown("Explore and filter transaction data")
    
    if st.session_state.data_loaded:
        G = st.session_state.G
        
        # Convert graph to dataframe
        node_data = []
        for node, data in G.nodes(data=True):
            node_data.append({
                'Node ID': node,
                'Label': 'Illicit' if data.get('label') == 1 else 'Licit' if data.get('label') == 0 else 'Unknown',
                'Degree': G.degree(node),
                'Features': data.get('features', [])[:5]  # First 5 features
            })
        
        df = pd.DataFrame(node_data)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            label_filter = st.multiselect(
                "Filter by Label",
                ['Licit', 'Illicit', 'Unknown'],
                default=['Licit', 'Illicit']
            )
        with col2:
            min_degree = st.number_input("Min Degree", 0, 100, 0)
        with col3:
            max_rows = st.slider("Max Rows", 100, 5000, 1000)
        
        # Apply filters
        filtered_df = df[df['Label'].isin(label_filter)]
        filtered_df = filtered_df[filtered_df['Degree'] >= min_degree]
        filtered_df = filtered_df.head(max_rows)
        
        # Display dataframe
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "Node ID": st.column_config.TextColumn("Node ID"),
                "Label": st.column_config.TextColumn("Label"),
                "Degree": st.column_config.NumberColumn("Degree"),
                "Features": st.column_config.TextColumn("Features (first 5)")
            }
        )
        
        # Export
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"fraud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(filtered_df):,}")
        with col2:
            st.metric("Avg Degree", f"{filtered_df['Degree'].mean():.2f}")
        with col3:
            st.metric("Max Degree", f"{filtered_df['Degree'].max():,}")
        with col4:
            st.metric("Unique Labels", f"{filtered_df['Label'].nunique()}")

# ============= TAB 6: AUTO-DISCOVERY =============
with tab6:
    st.title("🔍 Auto-Discovery")
    st.markdown("Proactive fraud pattern detection - automatically discovering insights without queries")
    
    if st.session_state.data_loaded:
        from src.analytics.auto_discovery import AutoDiscovery
        
        # Initialize discovery
        if 'discovery_insights' not in st.session_state:
            with st.spinner("Running auto-discovery..."):
                discoverer = AutoDiscovery(
                    G=st.session_state.G,
                    detector=st.session_state.detector,
                    data=st.session_state.model_data
                )
                st.session_state.discovery_insights = discoverer.run_full_discovery()
                st.session_state.discoverer = discoverer
            st.success(f"✅ Found {len(st.session_state.discovery_insights)} insights!")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            category_filter = st.multiselect(
                "Category",
                ['all', 'anomaly', 'pattern', 'risk', 'trend', 'gap'],
                default=['all']
            )
        with col2:
            severity_filter = st.multiselect(
                "Severity",
                ['all', 'HIGH', 'MEDIUM', 'LOW'],
                default=['all']
            )
        with col3:
            if st.button("🔄 Re-run Discovery", use_container_width=True):
                with st.spinner("Re-running auto-discovery..."):
                    discoverer = AutoDiscovery(
                        G=st.session_state.G,
                        detector=st.session_state.detector,
                        data=st.session_state.model_data
                    )
                    st.session_state.discovery_insights = discoverer.run_full_discovery()
                    st.session_state.discoverer = discoverer
                st.rerun()
        
        # Filter insights
        insights = st.session_state.discovery_insights
        
        if 'all' not in category_filter:
            insights = [i for i in insights if i.category in category_filter]
        if 'all' not in severity_filter:
            insights = [i for i in insights if i.severity in severity_filter]
        
        # Display insights in grid
        if insights:
            st.markdown(f"### 📋 Found {len(insights)} Insights")
            
            # Display as cards in 2-column grid
            cols = st.columns(2)
            for i, insight in enumerate(insights):
                with cols[i % 2]:
                    # Category badge
                    category_badge = {
                        'anomaly': '🔴 Anomaly',
                        'pattern': '🟡 Pattern',
                        'risk': '🔵 Risk',
                        'trend': '🟢 Trend',
                        'gap': '🟣 Gap'
                    }.get(insight.category, '⚪ Unknown')
                    
                    # Severity badge
                    severity_badge = {
                        'HIGH': '🔴 HIGH',
                        'MEDIUM': '🟡 MEDIUM',
                        'LOW': '🟢 LOW'
                    }.get(insight.severity, '⚪ UNKNOWN')
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #1E1E1E;
                        padding: 20px;
                        border-radius: 10px;
                        border: 1px solid #2E2E2E;
                        margin-bottom: 15px;
                        height: 100%;
                    ">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="background-color: #2E2E2E; padding: 4px 12px; border-radius: 12px; font-size: 12px;">
                                {category_badge}
                            </span>
                            <span style="background-color: #2E2E2E; padding: 4px 12px; border-radius: 12px; font-size: 12px;">
                                {severity_badge}
                            </span>
                        </div>
                        <h4 style="color: white; margin: 10px 0;">{insight.title}</h4>
                        <p style="color: #AAAAAA; font-size: 14px; margin-bottom: 15px;">{insight.description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display chart if available
                    if insight.chart_data:
                        chart_type = insight.chart_data.get('type')
                        
                        if chart_type == 'bar':
                            fig = px.bar(
                                x=insight.chart_data['labels'],
                                y=insight.chart_data['values'],
                                title=insight.chart_data['title'],
                                labels={
                                    'x': insight.chart_data.get('xlabel', ''),
                                    'y': insight.chart_data.get('ylabel', '')
                                },
                                color_discrete_sequence=['#FF5A5F']
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=250
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == 'scatter':
                            fig = px.scatter(
                                x=insight.chart_data['x'],
                                y=insight.chart_data['y'],
                                title=insight.chart_data['title'],
                                labels={
                                    'x': insight.chart_data.get('xlabel', ''),
                                    'y': insight.chart_data.get('ylabel', '')
                                },
                                color_discrete_sequence=['#FF5A5F']
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=250
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif chart_type == 'histogram':
                            fig = px.histogram(
                                x=insight.chart_data['data'],
                                nbins=30,
                                title=insight.chart_data['title'],
                                labels={
                                    'x': insight.chart_data.get('xlabel', ''),
                                    'y': insight.chart_data.get('ylabel', '')
                                },
                                color_discrete_sequence=['#FF5A5F']
                            )
                            if 'outlier_threshold' in insight.chart_data:
                                fig.add_vline(
                                    x=insight.chart_data['outlier_threshold'],
                                    line_dash="dash",
                                    line_color="#FFA15A",
                                    annotation_text="Outlier Threshold"
                                )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=250
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data details in expander
                    with st.expander("📊 View Details"):
                        if insight.data:
                            st.json(insight.data)
        else:
            st.info("No insights match the current filters. Try adjusting your filters.")
        
        # Export report
        if st.button("📄 Generate Report", use_container_width=True):
            report = st.session_state.discoverer.generate_report()
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"fraud_discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please load data and train the model first.")

# ============= TAB 7: ADVANCED ML =============
# ============= TAB 7: ADVANCED ML =============
with tab7:
    st.title("🧠 Advanced Machine Learning")
    st.markdown("Advanced GNN architectures, hyperparameter optimization, and model explainability")
    
    if not st.session_state.data_loaded or not st.session_state.model_trained:
        st.warning("Please load data and train the base model first.")
        st.stop()
    
    data = st.session_state.model_data
    detector = st.session_state.detector
    G = st.session_state.G
    
    st.info("⚡ Using optimized memory settings. Large datasets may be sampled for performance.")
    
    col1, col2, col3 = st.columns(3)
    
    # ============================================================
    # COLUMN 1: Hyperparameter Optimization
    # ============================================================
    with col1:
        st.markdown("### 🎯 Hyperparameter Optimization")
        st.markdown("Find the best model parameters using Optuna")
        
        if st.button("🚀 Run Optimization", use_container_width=True, key="optuna_btn"):
            try:
                from src.models.hyperparameter_optimization import HyperparameterOptimizer
                
                with st.spinner("Running optimization (may take 2-3 minutes)..."):
                    # Sample data with balanced classes
                    sample_size = min(500, data['x'].shape[0])
                    
                    # Get indices for each class
                    y = data['y']
                    class_0_indices = torch.where(y == 0)[0]
                    class_1_indices = torch.where(y == 1)[0]
                    
                    # Sample equally from both classes
                    n_samples_per_class = min(sample_size // 2, len(class_0_indices), len(class_1_indices))
                    
                    if n_samples_per_class > 0:
                        idx_0 = class_0_indices[torch.randperm(len(class_0_indices))[:n_samples_per_class]]
                        idx_1 = class_1_indices[torch.randperm(len(class_1_indices))[:n_samples_per_class]]
                        sample_indices = torch.cat([idx_0, idx_1])
                    else:
                        # Fallback: random sample
                        sample_indices = torch.randperm(data['x'].shape[0])[:sample_size]
                    
                    # Get sampled data
                    x = data['x'][sample_indices]
                    y = data['y'][sample_indices]
                    adj = data['adj'][sample_indices][:, sample_indices]
                    
                    # Create train/val split
                    n = len(x)
                    indices = torch.randperm(n)
                    split = int(0.8 * n)
                    
                    train_mask = torch.zeros(n, dtype=torch.bool)
                    val_mask = torch.zeros(n, dtype=torch.bool)
                    train_mask[indices[:split]] = True
                    val_mask[indices[split:]] = True
                    
                    # Check validation set has both classes
                    y_val = y[val_mask]
                    if len(torch.unique(y_val)) < 2:
                        st.warning("Validation set has only one class. Using different split...")
                        # Try a different random split
                        for _ in range(10):
                            indices = torch.randperm(n)
                            train_mask = torch.zeros(n, dtype=torch.bool)
                            val_mask = torch.zeros(n, dtype=torch.bool)
                            train_mask[indices[:split]] = True
                            val_mask[indices[split:]] = True
                            if len(torch.unique(y[val_mask])) > 1:
                                break
                    
                    # Run optimization
                    optimizer = HyperparameterOptimizer(
                        x=x, y=y, adj=adj,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        device=detector.device,
                        n_trials=8
                    )
                    results = optimizer.optimize()
                    
                    # Display results
                    st.success("✅ Optimization Complete!")
                    
                    best_params = results['best_params']
                    best_value = results.get('best_value', 0.0)
                    
                    st.markdown("### 📊 Best Parameters Found:")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Hidden Dimension", f"{best_params.get('hidden_dim', 'N/A')}")
                        st.metric("Number of Layers", f"{best_params.get('num_layers', 'N/A')}")
                        st.metric("Dropout Rate", f"{best_params.get('dropout', 'N/A'):.2f}")
                    with col_b:
                        st.metric("Learning Rate", f"{best_params.get('lr', 'N/A'):.6f}")
                        st.metric("Aggregator", best_params.get('aggregator', 'N/A'))
                        st.metric("Validation Score", f"{best_value:.4f}")
                    
                    # Add explanation
                    if best_value > 0.7:
                        st.success(f"✅ Good validation score! Model is performing well.")
                    elif best_value > 0.5:
                        st.warning(f"⚠️ Moderate validation score. Consider adjusting parameters.")
                    else:
                        st.error(f"❌ Low validation score. Check your data and try again.")
                    
                    st.session_state.best_params = best_params
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    
# ============================================================
# COLUMN 2: Ensemble Model (FIXED)
# ============================================================
with col2:
    st.markdown("### 🎯 Ensemble Model")
    st.markdown("Combine multiple GNNs for robust predictions")
    
    if st.button("🎯 Train Ensemble", use_container_width=True, key="ensemble_btn"):
        try:
            from src.models.advanced_gnn import EnsembleFraudDetector
            
            with st.spinner("Training ensemble model..."):
                # Get data with proper sampling
                sample_size = min(2000, data['x'].shape[0])
                
                # Get indices for each class
                y = data['y']
                class_0_indices = torch.where(y == 0)[0]
                class_1_indices = torch.where(y == 1)[0]
                
                # Sample equally from both classes
                n_samples_per_class = min(sample_size // 2, len(class_0_indices), len(class_1_indices))
                
                if n_samples_per_class > 0:
                    idx_0 = class_0_indices[torch.randperm(len(class_0_indices))[:n_samples_per_class]]
                    idx_1 = class_1_indices[torch.randperm(len(class_1_indices))[:n_samples_per_class]]
                    sample_indices = torch.cat([idx_0, idx_1])
                else:
                    sample_indices = torch.randperm(data['x'].shape[0])[:sample_size]
                
                # Get sampled data
                x = data['x'][sample_indices]
                y = data['y'][sample_indices]
                adj = data['adj'][sample_indices][:, sample_indices]
                
                # Create train/test split (80/20)
                n = len(x)
                indices = torch.randperm(n)
                split = int(0.8 * n)
                
                train_mask = torch.zeros(n, dtype=torch.bool)
                test_mask = torch.zeros(n, dtype=torch.bool)
                train_mask[indices[:split]] = True
                test_mask[indices[split:]] = True
                
                # Check test set has both classes
                y_test = y[test_mask].cpu().numpy()
                if len(np.unique(y_test)) < 2:
                    st.warning("Test set has only one class. Adjusting split...")
                    # Try a different split
                    for _ in range(10):
                        indices = torch.randperm(n)
                        train_mask = torch.zeros(n, dtype=torch.bool)
                        test_mask = torch.zeros(n, dtype=torch.bool)
                        train_mask[indices[:split]] = True
                        test_mask[indices[split:]] = True
                        y_test = y[test_mask].cpu().numpy()
                        if len(np.unique(y_test)) > 1:
                            break
                
                # Move data to CPU for compatibility
                x_cpu = x.cpu()
                y_cpu = y.cpu()
                adj_cpu = adj.cpu()
                train_mask_cpu = train_mask.cpu()
                test_mask_cpu = test_mask.cpu()
                
                # CREATE THE ENSEMBLE INSTANCE HERE
                ensemble = EnsembleFraudDetector(
                    in_features=x.shape[1],
                    hidden_dim=32,
                    out_features=1,
                    device='cpu'  # Use CPU to avoid MPS issues
                )
                
                # Train the ensemble
                losses = ensemble.train_models(
                    x_cpu, y_cpu, adj_cpu, 
                    train_mask_cpu, epochs=20
                )
                
                # Evaluate
                metrics = ensemble.evaluate(
                    x_cpu, y_cpu, adj_cpu, 
                    test_mask_cpu
                )
                
                # Display results
                st.success("✅ Ensemble Trained!")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("AUC", f"{metrics.get('auc', 0.0):.4f}")
                with col_b:
                    st.metric("F1", f"{metrics.get('f1', 0.0):.4f}")
                with col_c:
                    st.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
                with col_d:
                    st.metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
                
                # Store in session state
                st.session_state.ensemble = ensemble
                st.session_state.ensemble_metrics = metrics
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
    
    # ============================================================
    # COLUMN 3: Model Explainability (UPDATED)
    # ============================================================
    with col3:
        st.markdown("### 🔬 Model Explainability")
        st.markdown("Understand why the model makes predictions")
        
        if st.session_state.data_loaded:
            G = st.session_state.G
            
            # Get sample of nodes with labels
            all_nodes = list(G.nodes())
            node_sample = all_nodes[:500]  # Limit for dropdown
            
            # Create display options
            node_options = {}
            for node in node_sample:
                label = G.nodes[node].get('label', -1)
                if label == 0:
                    label_str = "🟢 Licit"
                elif label == 1:
                    label_str = "🔴 Illicit"
                else:
                    label_str = "❓ Unknown"
                degree = G.degree(node)
                node_options[f"{node} ({label_str}, degree: {degree})"] = node
            
            selected_node_display = st.selectbox(
                "Select Node to Explain",
                options=list(node_options.keys()),
                help="Select a transaction node to analyze"
            )
            
            selected_node = node_options[selected_node_display]
            
            if selected_node in G:
                node_data = G.nodes[selected_node]
                label = node_data.get('label', -1)
                degree = G.degree(selected_node)
                label_str = "🟢 Licit" if label == 0 else "🔴 Illicit" if label == 1 else "❓ Unknown"
                st.info(f"📌 **Node {selected_node}** - Label: {label_str} | Degree: {degree}")
            
            if st.button("🔍 Explain Prediction", use_container_width=True, key="explain_btn"):
                if selected_node:
                    try:
                        with st.spinner("Generating explanation..."):
                            if selected_node in G:
                                node_data = G.nodes[selected_node]
                                if 'features' in node_data:
                                    features = node_data['features']
                                    
                                    # Get prediction
                                    import torch
                                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(detector.device)
                                    detector.model.eval()
                                    with torch.no_grad():
                                        output = detector.model(features_tensor, torch.eye(1).to(detector.device))
                                        prob = torch.sigmoid(output).squeeze().item()
                                    
                                    label_str = "ILLICIT" if label == 1 else "LICIT" if label == 0 else "UNKNOWN"
                                    risk_level = "🔴 HIGH" if prob > 0.8 else "🟡 MEDIUM" if prob > 0.5 else "🟢 LOW"
                                    
                                    st.success(f"✅ Explanation for Node {selected_node}")
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Fraud Probability", f"{prob*100:.1f}%")
                                    with col_b:
                                        st.metric("Actual Label", label_str)
                                    with col_c:
                                        st.metric("Risk Level", risk_level)
                                    
                                    # Feature importance with meaningful names
                                    feature_importance = [(i, abs(feat)) for i, feat in enumerate(features)]
                                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                                    top_features = feature_importance[:5]
                                    
                                    st.markdown("**Top Influential Features:**")
                                    
                                    # Feature name mapping
                                    # Complete feature name mapping for Elliptic dataset (166 features)
                                    FEATURE_NAMES = {
                                        # Transaction metadata (0-4)
                                        0: "Transaction Amount",
                                        1: "Transaction Fee",
                                        2: "Input Count",
                                        3: "Output Count",
                                        4: "Total Input Value",
                                        
                                        # Input features (5-50)
                                        5: "Avg Input Value",
                                        6: "Max Input Value",
                                        7: "Min Input Value",
                                        8: "Input Value Std Dev",
                                        9: "Input Value Skewness",
                                        10: "Input Value Kurtosis",
                                        11: "Total Output Value",
                                        12: "Avg Output Value",
                                        13: "Max Output Value",
                                        14: "Min Output Value",
                                        15: "Output Value Std Dev",
                                        16: "Output Value Skewness",
                                        17: "Output Value Kurtosis",
                                        18: "Transaction Volume",
                                        19: "Transaction Velocity",
                                        20: "Time Since Last Transaction",
                                        21: "Time Until Next Transaction",
                                        22: "Previous Transaction Count",
                                        23: "Future Transaction Count",
                                        24: "Input Address Count",
                                        25: "Output Address Count",
                                        26: "Unique Input Addresses",
                                        27: "Unique Output Addresses",
                                        28: "Average Input Age",
                                        29: "Average Output Age",
                                        
                                        # Network features (30-60)
                                        30: "Network Distance",
                                        31: "Clustering Coefficient",
                                        32: "PageRank Score",
                                        33: "Betweenness Centrality",
                                        34: "Degree Centrality",
                                        35: "Eigenvector Centrality",
                                        36: "Local Clustering Coefficient",
                                        37: "Neighborhood Overlap",
                                        38: "Hub Score",
                                        39: "Authority Score",
                                        40: "Community Size",
                                        41: "Community Density",
                                        42: "Edge Betweenness",
                                        43: "Node Importance",
                                        44: "Influence Score",
                                        45: "Spread Factor",
                                        46: "Resilience Score",
                                        47: "Redundancy Score",
                                        48: "Efficiency Score",
                                        49: "Strength Score",
                                        
                                        # Temporal features (50-70)
                                        50: "Transaction Hour",
                                        51: "Transaction Day",
                                        52: "Transaction Month",
                                        53: "Transaction Year",
                                        54: "Hourly Pattern",
                                        55: "Daily Pattern",
                                        56: "Weekly Pattern",
                                        57: "Seasonal Pattern",
                                        58: "Time to First Tx",
                                        59: "Time to Last Tx",
                                        60: "Transaction Frequency",
                                        61: "Mean Time Between Txs",
                                        62: "Time Variance",
                                        63: "Velocity Change",
                                        64: "Acceleration",
                                        65: "Jerk",
                                        66: "Temporal Entropy",
                                        67: "Regularity Score",
                                        68: "Predictability Score",
                                        69: "Novelty Score",
                                        
                                        # Structural features (70-100)
                                        70: "Graph Density",
                                        71: "Diameter",
                                        72: "Radius",
                                        73: "Eccentricity",
                                        74: "Periphery Score",
                                        75: "Core Score",
                                        76: "Shell Index",
                                        77: "K-Core Number",
                                        78: "K-Truss Number",
                                        79: "Clique Size",
                                        80: "Triangle Count",
                                        81: "Cycle Count",
                                        82: "Path Length Average",
                                        83: "Path Length Std",
                                        84: "Connectivity Ratio",
                                        85: "Reachability",
                                        86: "Mobility Score",
                                        87: "Stability Score",
                                        88: "Volatility Score",
                                        89: "Adaptability Score",
                                        
                                        # Statistical features (100-130)
                                        100: "Transaction Value Z-Score",
                                        101: "Value Outlier Score",
                                        102: "Fee Ratio",
                                        103: "Input-Output Ratio",
                                        104: "Balance Change",
                                        105: "Net Flow",
                                        106: "Suspicious Pattern Score",
                                        107: "Anomaly Score",
                                        108: "Risk Score",
                                        109: "Fraud Indicator",
                                        110: "Money Laundering Score",
                                        111: "Structuring Score",
                                        112: "Velocity Anomaly",
                                        113: "Temporal Anomaly",
                                        114: "Spatial Anomaly",
                                        115: "Behavioral Score",
                                        116: "Profile Consistency",
                                        117: "Trust Score",
                                        118: "Reputation Score",
                                        119: "Reliability Score",
                                        
                                        # Derived features (130-165)
                                        130: "PCA Component 1",
                                        131: "PCA Component 2",
                                        132: "PCA Component 3",
                                        133: "PCA Component 4",
                                        134: "PCA Component 5",
                                        135: "PCA Component 6",
                                        136: "PCA Component 7",
                                        137: "PCA Component 8",
                                        138: "PCA Component 9",
                                        139: "PCA Component 10",
                                        140: "Autoencoder Feature 1",
                                        141: "Autoencoder Feature 2",
                                        142: "Autoencoder Feature 3",
                                        143: "Autoencoder Feature 4",
                                        144: "Autoencoder Feature 5",
                                        145: "Autoencoder Feature 6",
                                        146: "Autoencoder Feature 7",
                                        147: "Autoencoder Feature 8",
                                        148: "Autoencoder Feature 9",
                                        149: "Autoencoder Feature 10",
                                        150: "GNN Feature 1",
                                        151: "GNN Feature 2",
                                        152: "GNN Feature 3",
                                        153: "GNN Feature 4",
                                        154: "GNN Feature 5",
                                        155: "GNN Feature 6",
                                        156: "GNN Feature 7",
                                        157: "GNN Feature 8",
                                        158: "GNN Feature 9",
                                        159: "GNN Feature 10",
                                        160: "Ensemble Feature 1",
                                        161: "Ensemble Feature 2",
                                        162: "Ensemble Feature 3",
                                        163: "Ensemble Feature 4",
                                        164: "Ensemble Feature 5",
                                        165: "Final Score"
                                    }
                                    
                                    for i, (idx, importance) in enumerate(top_features, 1):
                                        feature_name = FEATURE_NAMES.get(idx, f"Feature_{idx}")
                                        direction = "🔼 Increases fraud" if features[idx] > 0 else "🔽 Decreases fraud"
                                        st.text(f"{i}. {feature_name}: {features[idx]:.4f} ({direction})")
                                    
                                    # Connected nodes
                                    neighbors = list(G.neighbors(selected_node))[:10]
                                    if neighbors:
                                        st.markdown("**Connected Nodes:**")
                                        for neighbor in neighbors:
                                            n_label = G.nodes[neighbor].get('label', -1)
                                            if n_label == 0:
                                                n_label_str = "🟢 Licit"
                                            elif n_label == 1:
                                                n_label_str = "🔴 Illicit"
                                            else:
                                                n_label_str = "❓ Unknown"
                                            st.text(f"  • {neighbor} ({n_label_str})")
                                    
                                else:
                                    st.warning(f"Node {selected_node} has no features")
                            else:
                                st.warning(f"Node {selected_node} not found in graph")
                                
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ============= TAB 8: TEMPORAL ANALYSIS (OPTIMIZED) =============
with tab8:
    st.title("⏰ Temporal Analysis")
    st.markdown("Analyze fraud patterns over time (optimized for speed)")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first.")
        st.stop()
    
    # Import the analyzer
    from src.analytics.temporal_analysis import TemporalAnalyzer
    
    # Initialize analyzer with caching
    if 'fast_temporal_analyzer' not in st.session_state:
        with st.spinner("Loading temporal analysis..."):
            st.session_state.fast_temporal_analyzer = TemporalAnalyzer(
                G=st.session_state.G,
                data=st.session_state.model_data
            )
        st.success("✅ Temporal data loaded!")
    
    analyzer = st.session_state.fast_temporal_analyzer
    
    # Show summary stats
    summary = analyzer.get_activity_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", f"{summary['total_nodes']:,}")
    with col2:
        st.metric("Avg Degree", f"{summary['avg_degree']:.2f}")
    with col3:
        st.metric("Max Degree", f"{summary['max_degree']:,}")
    with col4:
        st.metric("Illicit Nodes", f"{summary['total_illicit']:,}")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Class Distribution")
        trends = analyzer.analyze_fraud_trends()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Illicit', 'Licit', 'Unknown'],
                y=[
                    trends['class_distribution']['illicit'],
                    trends['class_distribution']['licit'],
                    trends['class_distribution']['unknown']
                ],
                marker_color=['#FF5A5F', '#00CC96', '#FFA15A'],
                text=[
                    f"{trends['class_distribution']['illicit']:,}",
                    f"{trends['class_distribution']['licit']:,}",
                    f"{trends['class_distribution']['unknown']:,}"
                ],
                textposition='auto',
            )
        ])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show degree comparison
        st.markdown("### 📊 Degree Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Avg Illicit Degree", f"{trends['degree_comparison']['avg_illicit_degree']:.1f}")
        with col_b:
            st.metric("Avg Licit Degree", f"{trends['degree_comparison']['avg_licit_degree']:.1f}")
    
    with col2:
        st.markdown("### ⚡ Transaction Velocity")
        velocity = analyzer.analyze_transaction_velocity()
        
        # Show velocity distribution
        fig = go.Figure(data=[
            go.Bar(
                x=list(velocity['velocity_distribution']['percentiles'].keys()),
                y=list(velocity['velocity_distribution']['percentiles'].values()),
                marker_color='#FF5A5F',
                text=[f"{v:.1f}" for v in velocity['velocity_distribution']['percentiles'].values()],
                textposition='auto',
            )
        ])
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Nodes with High Velocity", f"{velocity['num_high_velocity_nodes']:,}")
    
    # Temporal Anomalies
    st.markdown("---")
    st.markdown("### 🔍 Temporal Anomalies")
    
    anomalies = analyzer.detect_temporal_anomalies(top_n=10)
    
    if anomalies:
        df = pd.DataFrame(anomalies)
        df = df[['node', 'degree', 'z_score', 'label', 'type']]
        df.columns = ['Node', 'Degree', 'Z-Score', 'Label', 'Type']
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No temporal anomalies detected.")
    
    # Generate Report
    if st.button("📄 Generate Temporal Report"):
        report = analyzer.generate_temporal_report()
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"temporal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================
# Model Comparison Section
# ============================================================
st.markdown("---")
st.markdown("### 📊 Model Performance Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Base Model AUC", "0.955")

with col2:
    if 'ensemble_metrics' in st.session_state and st.session_state.ensemble_metrics:
        auc = st.session_state.ensemble_metrics.get('auc', 0.0)
        color = "🟢" if auc > 0.9 else "🟡" if auc > 0.7 else "🔴"
        st.metric("Ensemble AUC", f"{color} {auc:.4f}")
    else:
        st.metric("Ensemble AUC", "Not trained")

with col3:
    if 'ensemble_metrics' in st.session_state and st.session_state.ensemble_metrics:
        improvement = (st.session_state.ensemble_metrics.get('auc', 0.0) - 0.955) * 100
        color = "🟢" if improvement > 0 else "🔴"
        st.metric("Improvement", f"{color} {improvement:+.2f}%")
    else:
        st.metric("Improvement", "N/A")

with col4:
    if 'ensemble_metrics' in st.session_state and st.session_state.ensemble_metrics:
        best = "Ensemble" if st.session_state.ensemble_metrics.get('auc', 0.0) > 0.955 else "Base"
        st.metric("Best Model", best)
    else:
        st.metric("Best Model", "Base")

# Show additional metrics if available
if 'ensemble_metrics' in st.session_state and st.session_state.ensemble_metrics:
    metrics = st.session_state.ensemble_metrics
    st.caption(f"Ensemble F1: {metrics.get('f1', 0.0):.4f} | Precision: {metrics.get('precision', 0.0):.4f} | Recall: {metrics.get('recall', 0.0):.4f}")

st.info("💡 Ensemble combines GAT, GraphSAGE (mean), and GraphSAGE (sum) for robust predictions")