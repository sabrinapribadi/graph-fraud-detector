"""
Validate the downloaded Elliptic Bitcoin dataset
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_dataset():
    """Load and validate the Elliptic dataset files"""
    data_path = Path("data/raw/elliptic_bitcoin_dataset")
    
    logger.info("Loading Elliptic dataset...")
    
    # Load features - 203,769 nodes, 165 features
    features = pd.read_csv(
        data_path / "elliptic_txs_features.csv", 
        header=None
    )
    logger.info(f"✅ Features loaded: {features.shape}")
    
    # Load classes
    classes = pd.read_csv(
        data_path / "elliptic_txs_classes.csv"
    )
    logger.info(f"✅ Classes loaded: {classes.shape}")
    
    # Load edgelist
    edgelist = pd.read_csv(
        data_path / "elliptic_txs_edgelist.csv",
        header=None,
        names=["source", "target", "timestamp"]
    )
    logger.info(f"✅ Edgelist loaded: {edgelist.shape}")
    
    # Validation checks
    print("\n" + "="*50)
    print("DATASET VALIDATION REPORT")
    print("="*50)
    
    print(f"\n📊 Features:")
    print(f"  - Nodes: {features.shape[0]:,}")
    print(f"  - Features per node: {features.shape[1]-1}")  # -1 for node ID
    
    print(f"\n📊 Classes:")
    class_counts = classes['class'].value_counts()
    for class_name, count in class_counts.items():
        pct = (count / len(classes)) * 100
        print(f"  - {class_name}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📊 Edgelist:")
    print(f"  - Edges: {len(edgelist):,}")
    print(f"  - Timestamp range: {edgelist['timestamp'].min()} → {edgelist['timestamp'].max()}")
    
    # Check for missing data
    print(f"\n🔍 Data Quality:")
    print(f"  - Features with nulls: {features.isnull().any().sum()}")
    print(f"  - Classes with nulls: {classes.isnull().any().sum()}")
    print(f"  - Edgelist with nulls: {edgelist.isnull().any().sum()}")
    
    # Sample node IDs
    sample_nodes = features.iloc[:5, 0].tolist()
    print(f"\n📝 Sample node IDs: {sample_nodes[:5]}")
    
    print("\n✅ Validation complete!")
    
    return features, classes, edgelist

if __name__ == "__main__":
    validate_dataset()