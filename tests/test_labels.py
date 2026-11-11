"""Label mapping follows the Elliptic dataset: class "1" = illicit (4,545), class "2" = licit (42,019)."""
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

CLASSES = project_root / "data" / "processed" / "classes.parquet"


@pytest.mark.skipif(not CLASSES.exists(), reason="processed Elliptic classes not available")
def test_illicit_is_the_minority_class_as_published():
    from src.data.loader import EllipticDataLoader
    loader = EllipticDataLoader()
    loader.load_data()
    loader.preprocess_features()   # sets node_ids, as in scripts/precompute_api_cache.py
    loader.prepare_labels()
    counts = loader.classes["label"].value_counts().to_dict()
    assert counts[1] == 4545 and counts[0] == 42019 and counts[-1] == 157205
