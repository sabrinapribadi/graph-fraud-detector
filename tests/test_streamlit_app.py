"""Streamlit app (R26, app template v2): first screen, every page with its business question, and no owner key; offline.

The app loads the committed Elliptic parquet files and trains a small GraphSAGE once (cached), so the first test takes ~20 s.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.utils.measured import measured  # noqa: E402

APP = str(ROOT / "src" / "ui" / "dashboard.py")
PAGES = ["overview", "explore", "network", "discovery", "models", "risk", "stress", "strategy", "agent"]


def app(page="overview"):
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception
    return at


def body(at) -> str:
    return " ".join(e.proto.body for e in at.get("html"))


def test_first_screen_kpis_brief_and_pages():
    at = app()
    m = measured()
    assert [x.value for x in at.metric][:3] == [f"{m['rf_f1']:.3f}", f"{m['rf_recall']:.0%}", f"{m['gs_auc']:.3f}"]
    for w in ["What", "Why", "Who", "Where", "When it fails", "How"]:          # the 5W1H brief (R22)
        assert f'class="f-w">{w}<' in body(at), w
    assert any("12 hours" in c.value for c in at.caption)
    assert [b.key for b in at.sidebar.button] == [f"nav_{k}" for k in PAGES]


def test_every_page_opens_with_its_business_question():
    for key in PAGES[1:]:
        assert "Business question" in body(app(key)), key


def test_assistant_defaults_to_free_tools_and_never_reads_the_owner_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk" + "-" + "owner-key-must-not-be-used")   # fake, built at runtime
    at = app("agent")
    assert at.radio(key="chat_mode").value.startswith("Free")
    assert "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] != "sk" + "-" + "owner-key-must-not-be-used"
    src = (ROOT / "src" / "ui" / "dashboard.py").read_text()
    assert "st.secrets" not in src and 'os.environ.pop("OPENAI_API_KEY", None)' in src.splitlines()[1]


def test_free_mode_answers_from_the_tools():
    at = app("agent")
    at.button(key="quick3_0").click().run()                                    # "Fraud statistics"
    assert not at.exception
    answer = " ".join(x.value for m in at.chat_message for x in m.markdown)
    assert "Illicit: 4,545" in answer and "Licit: 42,019" in answer


def test_frame_is_the_shared_copy():
    """The app frame is identical in every portfolio app; change it everywhere at once (checksum of the shared copy)."""
    import hashlib
    assert hashlib.sha256((ROOT / "src" / "ui" / "frame.py").read_bytes()).hexdigest()[:16] == "385e92329f20405a"
