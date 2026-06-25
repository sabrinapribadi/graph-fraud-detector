
## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit, Plotly, NetworkX |
| **API** | FastAPI, Uvicorn |
| **ML** | PyTorch, scikit-learn, Optuna |
| **Graph** | NetworkX, GraphSAGE, GAT |
| **Agent** | LangChain, OpenAI GPT-4o-mini |
| **Quant Finance** | Monte Carlo, TVM, VaR |
| **Deployment** | Docker, Render |
| **Language** | Python 3.12 |

## 📦 Installation

### Prerequisites
- Python 3.12
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/sabrinapribadi/graph-fraud-detector.git
cd graph-fraud-detector

# Install dependencies
poetry install

# Run the dashboard
PYTHONPATH=. poetry run streamlit run src/ui/dashboard.py

# Run the API server
PYTHONPATH=. poetry run python scripts/run_api.py