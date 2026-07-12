# Blockers, Lessons Learned & Solutions

Project: Graph Fraud Detector
Scope: All sprints from initial deployment through Streamlit Community Cloud migration
Author: Sabrina Pribadi

---

## Summary Table

| # | Blocker | Sprint | Severity | Resolution |
|---|---------|--------|----------|------------|
| 1 | OOM crash loop on Render Starter (512 MB) | 28–29 | Critical | Migrated dashboard to Streamlit Community Cloud |
| 2 | `poetry install` non-determinism caused heavier builds | 28 | High | Switched to `pip install -r requirements.txt` with pinned versions |
| 3 | Float64 memory spike during parquet load | 28 | High | Float32 downcast + manual z-score instead of StandardScaler |
| 4 | Render instance type upgrade silently did not apply | 28 | High | Abandoned upgrade path; migrated to free alternative instead |
| 5 | Corporate network (IOH) blocks database ports | 28 | Medium | Used web UIs (Supabase SQL Editor, MongoDB Atlas) over HTTPS |
| 6 | Supabase transaction-mode pooler requires `autocommit=True` | 28 | Medium | Set `conn.autocommit = True` before every query |
| 7 | Accidentally broke Network Explorer AI feature | 28 | Medium | Immediately reverted; root cause: removed shared node attribute |
| 8 | `docs/index.html` silently excluded by `.gitignore` | 29 | Medium | Added `!docs/index.html` negation rule |
| 9 | README shows broken alt text instead of screenshot | 29 | Low | Removed `![alt text](image.png)` — `*.png` was gitignored |
| 10 | Streamlit Cloud deployed Python 3.14, skipped all packages | 29 | High | Added `runtime.txt` with `python-3.12` |
| 11 | Docker base image CVE warnings | 28–29 | Low | Added `apt-get upgrade -y` to both Dockerfiles |

---

## Blocker 1 — OOM Crash Loop on Render Starter (512 MB)

**Context:** Sprints 28–29. The Streamlit dashboard was deployed to Render's Starter plan ($7/month, 512 MB RAM).

**Symptom:** Render sent repeated "exceeded memory limit" emails. The dashboard bounced between "Deployed" and "Failed service" in a continuous crash loop. First-visit response was always a 502.

**Root cause — why it was a loop, not a one-time crash:**
The crash happened *during* the first page load, before `@st.cache_resource` could persist anything. On restart, the process had to load everything from scratch again — and crashed again. The cache never survived long enough to help.

**Memory profile (Starter tier):**

| Component | RAM |
|-----------|-----|
| Python + Streamlit + other imports | ~120 MB |
| `import torch` | ~180 MB |
| `from src.models.gnn_model import FraudDetector` (torch-geometric) | ~50 MB |
| Other libraries (LangChain, ChromaDB, Prophet, etc.) | ~110 MB |
| **Import baseline** | **~460 MB** |
| `load_data()` — parquet read + StandardScaler float64 peak | +272 MB |
| `build_graph()` — NetworkX graph with 203k nodes | +136 MB |
| **Peak during first page load** | **~868 MB** |

**Solutions attempted:**

1. Switched `poetry install` → `pip install -r requirements.txt` (pinned, reproducible builds) — did not fix OOM but eliminated non-determinism
2. Float32 downcast + manual z-score in `loader.py` — reduced peak by ~136 MB, still not enough
3. Attempted instance type upgrade to Standard (2 GB, $25/month) — UI appeared to accept but the badge stayed on "Starter"; upgrade did not apply

**Final solution:** Migrated the dashboard to **Streamlit Community Cloud** (free, 1 GB RAM). The Render dashboard service was suspended. Cost dropped from $14/month (two Render services) to $7/month (API only).

**Lesson learned:** When a stack's steady-state memory requirement is close to the instance limit, any headroom gets consumed by startup overhead and GC pressure. The fix is not to optimize hot paths — it is to right-size the instance or choose a platform designed for the workload. Streamlit Community Cloud is purpose-built for Streamlit apps and gives 1 GB free; Render Starter is not the right fit for heavy ML dashboards.

---

## Blocker 2 — `poetry install` Non-Determinism Caused Heavier Builds

**Context:** Sprint 28. After adding `psycopg2-binary` to `pyproject.toml`, the Render dashboard deployment started pulling heavier package versions.

**Symptom:** The same codebase that deployed fine before Sprint 28 began OOMing immediately after adding one new dependency.

**Root cause:** `poetry install` without a committed `poetry.lock` file re-resolves the entire dependency tree on each build. Adding `psycopg2-binary` changed the resolution graph and caused Poetry to pull a newer (heavier) torch or torch-geometric build.

**Solution:** Replaced `poetry install` in `Dockerfile` with `pip install --no-cache-dir -r requirements.txt`, where `requirements.txt` contains exact pinned versions generated from the locked Poetry environment on the developer's machine.

```dockerfile
# Before
RUN poetry install --no-interaction --no-ansi

# After
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```

**Lesson learned:** For production Docker builds, always pin dependencies. `poetry install` without a lock file is non-deterministic across CI runs. Generate `requirements.txt` once from a known-good environment (`pip freeze > requirements.txt`) and commit it. Every build is then bit-for-bit reproducible.

---

## Blocker 3 — Float64 Memory Spike During Parquet Load

**Context:** Sprint 28. Even after switching to pinned requirements, the app continued to OOM during data loading.

**Symptom:** App crashed approximately 14 seconds after the log line "Loading from parquet cache..." — consistent with the time needed to read and preprocess the parquet files.

**Root cause — two hidden float64 allocations:**

1. `pd.read_parquet()` reads float columns as float64 by default. For 203,769 nodes × 166 features, this is `203769 × 166 × 8 bytes = 271 MB`.

2. `StandardScaler.fit_transform()` silently upcasts float32 input back to float64 internally, creating a second 271 MB intermediate array — even when the input DataFrame has been manually downcast.

**Solution in `src/data/loader.py`:**

```python
# After parquet load — downcast immediately
f64_cols = self.features.select_dtypes(include="float64").columns
if len(f64_cols):
    self.features[f64_cols] = self.features[f64_cols].astype(np.float32)

# Replace StandardScaler with manual float32 z-score
mean = features.mean(axis=0)          # stays float32
std  = features.std(axis=0)           # stays float32
std[std == 0] = 1.0
features_scaled = (features - mean) / std   # stays float32
```

**Memory saved:** ~272 MB peak reduction (eliminated the StandardScaler intermediate allocation).

**Lesson learned:** scikit-learn preprocessing functions are not dtype-transparent. `StandardScaler`, `MinMaxScaler`, and similar transformers upcast to float64 regardless of input dtype. When memory is a constraint, implement the equivalent operation manually in numpy to stay in float32 throughout. Always profile dtype at each step of a preprocessing pipeline — silent upcasting is a common source of unexpected memory spikes.

---

## Blocker 4 — Render Instance Type Upgrade Silently Did Not Apply

**Context:** Sprint 28. After OOM continued despite code fixes, the attempt was made to upgrade the Render dashboard from Starter ($7/month, 512 MB) to Standard ($25/month, 2 GB).

**Symptom:** The "Pick an Instance Type" screen showed Standard selected and "Save Changes" active. After saving, the dashboard's service badge still showed "Starter" and OOM crashes continued.

**Root cause:** Unclear — the Render UI accepted the input but the change did not propagate. Possibly a billing or workspace plan restriction.

**Solution:** Rather than debugging Render's billing system, evaluated free alternatives. Streamlit Community Cloud provides 1 GB RAM for free for public repos. Migrated there instead. Saved $18/month compared to the Standard upgrade.

**Lesson learned:** When a platform upgrade is not applying as expected, check whether there is a free alternative that meets the requirements before paying. For Streamlit-based data apps, Streamlit Community Cloud is almost always the better hosting choice compared to general-purpose PaaS providers.

---

## Blocker 5 — Corporate Network (IOH) Blocks Database Ports

**Context:** Sprint 28. Attempted to connect DBeaver to Supabase PostgreSQL to verify the gold layer schema.

**Symptom:**
- Port 6543 (Supabase transaction-mode pooler): "Connection attempt timed out"
- Port 5432 (direct connection): "No route to host"
- Port 27017 (MongoDB Atlas): Also blocked

**Root cause:** Indosat Ooredoo Hutchison (IOH) corporate network firewall blocks outbound connections on non-standard ports. Only HTTP (80) and HTTPS (443) are permitted.

**Solution:** Used web-based database UIs instead of desktop clients:
- **Supabase:** SQL Editor and Table Editor at `app.supabase.com` — fully functional over HTTPS
- **MongoDB Atlas:** Data Explorer at `cloud.mongodb.com` — fully functional over HTTPS

Both UIs allow full schema inspection, query execution, and data browsing without requiring a direct port connection.

**Lesson learned:** Always design database access workflows with network restriction fallbacks. Web-based database UIs (Supabase, MongoDB Atlas, PlanetScale, Neon) provide equivalent functionality to desktop clients and work on any network that allows HTTPS. When working in a corporate environment, assume non-HTTP ports are blocked until proven otherwise.

---

## Blocker 6 — Supabase Transaction-Mode Pooler Requires `autocommit=True`

**Context:** Sprint 28. After implementing `src/db/postgres.py` with `psycopg2.pool.ThreadedConnectionPool`, write operations to Supabase silently failed.

**Symptom:** `upsert_daily_alert()` calls completed without error but no rows appeared in `fraud_alert_daily`. No exception was raised.

**Root cause:** Supabase's transaction-mode pooler (port 6543) does not support the `BEGIN`/`COMMIT` transaction lifecycle that psycopg2 uses by default. When psycopg2 opens a connection, it immediately issues an implicit `BEGIN`. The pooler cannot maintain transaction state across connection hops, so it rejects the transaction silently.

**Solution:** Set `conn.autocommit = True` on each connection retrieved from the pool before executing any query:

```python
conn = _pool.getconn()
try:
    conn.autocommit = True   # required for Supabase transaction-mode pooler
    with conn.cursor() as cur:
        cur.execute(sql, params)
finally:
    _pool.putconn(conn)
```

**Lesson learned:** Supabase's pooler operates in transaction mode, not session mode. Any psycopg2 code written against a standard PostgreSQL server will need `autocommit=True` when targeting Supabase's pooler port (6543). Read the pooler mode documentation before writing connection pool code — the silent failure mode (no exception, no rows written) makes this very hard to diagnose otherwise.

---

## Blocker 7 — Accidentally Broke Network Explorer AI Feature

**Context:** Sprint 28. While optimising memory usage, `features=self.node_features[i]` was removed from the node attribute assignment in `EllipticDataLoader.build_graph()` to reduce the NetworkX graph's memory footprint.

**Symptom:** Network Explorer tab's "AI-Predicted Label" colour mode stopped working. All nodes rendered the same colour regardless of model prediction.

**Root cause:** `dashboard.py` line 844 reads `_sub2.nodes[_sn2].get("features")` to retrieve each node's feature vector for per-subgraph GNN inference. Removing the `features` attribute from node data broke this lookup silently — `.get()` returned `None` instead of raising an error, causing the inference branch to be skipped.

**Solution:** Immediately reverted the removal. The `features` attribute is a reference to a row of `self.node_features` (a numpy array view), not a copy, so the actual memory overhead is minimal — only the Python dict entry, not a data duplicate.

**Lesson learned:** When removing data attributes for memory optimisation, grep the entire codebase for every consumer of that attribute first. Silent `None` returns from `.get()` make missing attribute bugs invisible at runtime — the feature simply does not render rather than raising an exception. Write a test or at least a log warning when a critical attribute is absent.

---

## Blocker 8 — `docs/index.html` Silently Excluded by `.gitignore`

**Context:** Sprint 29. After creating `docs/index.html` for the GitHub Pages demo, `git status` did not show the file as untracked.

**Symptom:** `git status` output omitted `docs/index.html` entirely. The file existed on disk but was invisible to git.

**Root cause:** `.gitignore` contained a top-level `*.html` rule intended to exclude build artifacts. This rule matched `docs/index.html`, causing git to silently ignore it. Because `.gitignore` suppresses untracked file listing, there was no warning — the file simply did not appear.

**Diagnosis command:**
```bash
git check-ignore -v docs/index.html
# Output: .gitignore:42:*.html   docs/index.html
```

**Solution:** Added a negation rule immediately after the `*.html` line:

```gitignore
*.html
!docs/index.html
```

**Lesson learned:** When a file does not appear in `git status`, always run `git check-ignore -v <path>` before assuming git is broken. Broad wildcard rules in `.gitignore` silently swallow specific files that should be tracked. Use negation rules (`!path/to/file`) to re-include exceptions, and add a comment explaining why the exception exists.

---

## Blocker 9 — README Shows Broken Alt Text Instead of Screenshot

**Context:** Sprint 29. The README contained `![alt text](image.png)` referencing a dashboard screenshot.

**Symptom:** GitHub rendered "alt text" as plain text beside the badge row instead of showing an image. Looked broken and unprofessional on the repo front page.

**Root cause:** `.gitignore` contained `*.png`, which excluded `image.png` from the repository. The markdown image reference was committed but the image file itself was not, so GitHub served the alt text fallback.

**Solution:** Removed the broken image reference from README.md. A dashboard screenshot is available at `graph-fraud-detector.streamlit.app` for any reviewer who wants to see the UI.

**Lesson learned:** Before committing a markdown image reference, verify the file is not excluded by `.gitignore`. Either add a `.gitignore` negation rule for the specific image, or host the image externally (e.g., upload to the GitHub repo's Issues section to get a permanent CDN URL) and reference the CDN URL in the markdown.

---

## Blocker 10 — Streamlit Cloud Deployed Python 3.14, Skipped All Packages

**Context:** Sprint 29. After deploying to Streamlit Community Cloud, the app crashed immediately with `ModuleNotFoundError: No module named 'plotly'`.

**Symptom:** The deployment log path showed `/home/adminuser/venv/lib/python3.14/` — Streamlit Cloud had selected Python 3.14 as the runtime. Every package in `requirements.txt` has a `; python_version == "3.12"` environment marker, so pip evaluated all markers as `False` and installed nothing.

**Root cause:** `requirements.txt` was generated by Poetry on a Python 3.12 machine. Poetry adds environment markers to every line to record which Python version the lock was resolved for. When Streamlit Cloud used Python 3.14 (its default at time of deployment), all markers resolved to `False` and the install was a no-op.

**Solution:** Added `runtime.txt` to the repository root:

```
python-3.12
```

Streamlit Community Cloud reads this file and provisions the matching Python version before installing dependencies.

**Lesson learned:** Poetry-generated `requirements.txt` files are not portable across Python versions — they embed environment markers that bind every package to the resolver's Python version. Always add a `runtime.txt` (Streamlit Cloud) or `.python-version` (other platforms) when deploying a Poetry-locked `requirements.txt`. Without it, a platform version upgrade silently voids the entire dependency installation.

---

## Blocker 11 — Docker Base Image CVE Warnings

**Context:** Sprints 28–29. VS Code's Docker extension flagged `python:3.12-slim` with "1 critical and 2 high vulnerabilities."

**Symptom:** Warning banner in VS Code: "python:3.12-slim contains 1 critical and 2 high vulnerabilities in system packages."

**Root cause:** The `python:3.12-slim` base image ships with a fixed snapshot of Debian system packages. Security patches released after the image was built are not automatically applied.

**Solution:** Added `apt-get upgrade -y` to both Dockerfiles:

```dockerfile
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

This patches all Debian system packages to their latest available versions during the Docker build, without changing the Python base image version.

**Lesson learned:** Slim base images trade security patching for image size. In production, either use a regularly updated base image tag, add `apt-get upgrade -y` to the build step, or implement a CI pipeline that rebuilds images on a schedule (e.g., weekly) to pull in OS security patches. For a portfolio project, `apt-get upgrade -y` during build is the simplest fix.

---

## Cross-Cutting Lessons

| Theme | Lesson |
|-------|--------|
| **Memory budgeting** | Measure baseline import memory before writing any data loading code. `import torch` alone costs ~180 MB. Add data and model memory on top to estimate peak before choosing an instance size. |
| **Deployment platform fit** | Match the platform to the workload. General-purpose PaaS (Render Starter) is a poor fit for 500 MB+ Python ML stacks. Purpose-built platforms (Streamlit Community Cloud) provide more RAM for the same cost (free). |
| **Dependency pinning** | Never rely on `poetry install` without a lock file in CI/CD. Pin exact versions in `requirements.txt` for reproducible, predictable builds. |
| **Silent failures** | Database writes (`autocommit`), git exclusions (`.gitignore`), and Python markers (`; python_version`) all fail silently. Verify each with a positive check: read back a written row, run `git check-ignore`, check the deployment log's Python path. |
| **Attribute contracts** | Any node/edge attribute used by downstream code is an implicit contract. Document these contracts and write guard clauses (`.get()` with a not-None check + a log warning) rather than relying on silent `None` fallbacks. |
| **Rollback instinct** | When an optimisation breaks an unrelated feature, revert immediately rather than trying to patch forward. The feature worked before; the optimisation was not worth the regression. Revert first, then find a safer approach. |
