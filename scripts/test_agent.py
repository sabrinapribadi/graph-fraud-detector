"""
Test the Fraud Agent
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data.loader import EllipticDataLoader
from src.models.gnn_model import FraudDetector
from src.agent.fraud_agent import FraudAgent

print("="*60)
print("TESTING FRAUD AGENT")
print("="*60)

# Load data
print("\n📊 Loading data...")
loader = EllipticDataLoader()
features, classes, edgelist = loader.load_data()
loader.preprocess_features()
loader.prepare_labels()
G = loader.build_graph()
print(f"✅ Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Train model
print("\n🧠 Training model...")
detector = FraudDetector(hidden_dim=32, num_layers=2, dropout=0.2)
data = detector.build_graph_data(G, sample_size=2000, balance_classes=True)
detector.train(data, epochs=50)
print("✅ Model trained!")

# Create agent
print("\n🤖 Creating agent...")
agent = FraudAgent(G, detector, data)
print("✅ Agent ready!")

# Test questions
questions = [
    "Show me fraud statistics",
    "What are the top 5 most suspicious transactions?",
    "Analyze the network structure",
    "Predict fraud probability for node 5530458",
    "Run a risk analysis for 5000 transactions with 2% fraud rate",
    "Find anomalous patterns"
]

print("\n" + "="*60)
print("TESTING AGENT QUESTIONS")
print("="*60)

for i, q in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {q}")
    print('-'*60)
    try:
        response = agent.ask(q)
        print(response)
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Agent testing complete!")