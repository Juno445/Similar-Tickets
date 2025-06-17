# Semantic-Ticket-Search

Detect and automatically merge **duplicate Freshservice tickets** using state-of-the-art sentence embeddings, FAISS Approximate-Nearest-Neighbours (ANN) search and department-aware filtering with domain-specific heuristics.

The repository contains everything needed to **train** a similarity index from your historical tickets **and serve** a low-latency API that can be wired to a Freshservice webhook.

---

## ✨  Features

• **End-to-end pipeline**: data harvest → embedding → ANN index → real-time duplicate detection  
• **Language-agnostic** – powered by [Sentence-Transformers](https://www.sbert.net/)  
• **⚡ <10 ms query latency** with in-memory FAISS  
• **Hot-reload** of new snapshots without downtime  
• **Smart merge workaround** implemented via Freshservice REST API (adds notes + closes duplicate)  
• **Department-aware duplicate detection** – only compares tickets within the same department  
• **Human-in-the-loop review console** with Streamlit UI for manual validation  
• **Active learning** with pair labeling and confidence tracking  
• **Immutable snapshots** with atomic updates and rollback capability  
• **Prometheus metrics** & structured, colourised logging  
• **100% reproducible** via Docker Compose ‑ no local Python required  
• **Extensive test-suite** (pytest) & modular design for rapid iteration

---

## 🏗️  Architecture

```text
┌────────────────────────────────────┐
│            Trainer (batch)         │
│                                    │
│ 1. Fetch N days of tickets         │
│ 2. Embed unseen tickets            │
│ 3. Build FAISS IP index            │
│ 4. Write immutable snapshot        │
└──────────────┬─────────────────────┘
               │  shared volume (/shared)
               ▼
┌────────────────────────────────────┐
│            Serve (FastAPI)         │
│                                    │
│ • /webhook – duplicate check       │
│ • /review – human review UI        │
│ • /candidates – active learning    │
│ • /label – pair feedback           │
│ • /reloadIndex – hot reload        │
│ • /metrics – Prometheus            │
│ • /healthz – readiness             │
└────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│       Review Console (Streamlit)   │
│                                    │
│ • Manual duplicate validation      │
│ • Confidence-based filtering       │
│ • Active learning feedback         │
│ • Batch review workflows           │
└────────────────────────────────────┘
```

**Immutable Snapshots** are laid out as:
```
/shared/
  ├─ snapshots/2024-06-01T12-00-00Z/
  │    ├─ vectors.faiss   # binary FAISS index
  │    └─ meta.json       # tickets + metadata
  ├─ snapshots/2024-06-01T18-30-15Z/
  │    ├─ vectors.faiss   # newer snapshot
  │    └─ meta.json       # with more tickets
  └─ current  →  snapshots/2024-06-01T18-30-15Z/  # symlink flipped atomically
```
Serve mounts `/shared` **read-only**; the symlink ensures zero-downtime upgrades and easy rollbacks.

---

## 📂  Directory layout (top-level)

| Path | Purpose |
|------|---------|
| `src/` | Library code shared by trainer & API |
| `src/similarity/` | Core similarity detection & active learning |
| `src/freshservice/` | Freshservice API client & merge logic |
| `src/ann_index.py` | FAISS index loader with hot-reload |
| `src/review_console.py` | Streamlit-based manual review UI |
| `src/webhook_server.py` | FastAPI server with all endpoints |
| `trainer/` | Batch job entry-point & Dockerfile |
| `serve/` | Startup script & Dockerfile for FastAPI |
| `tests/` | pytest test-suite |
| `requirements.txt` | Runtime & training dependencies |
| `dev-requirements.txt` | Development & testing dependencies |
| `docker-compose.yml` | One-command deployment |

---

## 🚀  Quick start (Docker Compose)

1. Copy `.env.example` → `.env` and fill **at least**:
   ```env
   FS_DOMAIN=acme.freshservice.com
   FS_API_KEY=xxxxxxxxxxxxxxxx
   GROUP_IDS=1234,5678
   ```
2. Run:
   ```bash
   docker compose up --build
   ```
   • The **trainer** runs once, writes the first snapshot and exits.  
   • The **api** container waits until the snapshot is present, then exposes HTTP :8000.

### Minimal curl example
```bash
curl -X POST http://localhost:8000/webhook \
     -H "Content-Type: application/json" \
     -d '{"ticket_id": 809188}'
```
Response:
```json
{"merged": true}
```

---

## ⚙️  Configuration reference

All options are supplied via environment variables (Docker reads them from `.env`).

| Variable | Description | Default |
|----------|-------------|---------|
| FS_DOMAIN | Freshservice sub-domain (without https) | – *(required)* |
| FS_API_KEY / FS_API_KEY_FILE | API key or path to file containing the key | – *(required)* |
| GROUP_IDS / GROUP_ID | Comma-separated list of ticket group IDs | – *(required)* |
| INDEX_PATH | Shared volume for snapshots | `/shared` |
| DAYS_BACK | Look-back window for trainer | `60` |
| EMBEDDING_MODEL | sentence-transformers model name | `all-MiniLM-L6-v2` |
| SIMILARITY_THRESHOLD | Probability cut-off for automatic merge (serve) | `0.9` |
| RELOAD_TOKEN | Secret header for `/reloadIndex` | *(unset)* |
| **WEBHOOK_SECRET** | **HMAC secret for webhook authentication** | ***(unset)*** |
| **HF_TOKEN** | **Hugging Face access token (avoids rate limits)** | ***(unset)*** |
| **HF_TOKEN_FILE** | **Path to file containing HF token** | ***(unset)*** |
| **MODEL_CACHE_DIR** | **Directory for caching downloaded models** | **`/app/models`** |
| FS_PER_PAGE | Pagination size when fetching tickets | `100` |
| FS_MAX_PAGES | Max pages per trainer run | `10` |
| UNMERGED_LOG | Path for tickets that were *not* auto-merged | `/data/unmerged_tickets.log` |
| **REVIEW_USER** | **Username for review console attribution** | **`anonymous`** |
| **REVIEW_SIM_LOWER** | **Lower similarity threshold for review candidates** | **`0.65`** |
| **REVIEW_SIM_UPPER** | **Upper similarity threshold for review candidates** | **`0.90`** |
| **REVIEW_MAX** | **Maximum candidates to show in review console** | **`100`** |
| **REVIEW_DAYS_BACK** | **Days to look back for review console tickets** | **`60`** |

### 🤗 Hugging Face Authentication & Model Caching

To avoid rate limiting and enable offline model usage, configure Hugging Face authentication:

#### Option 1: Environment Variable
```bash
# Add to your .env file
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

#### Option 2: Token File
```bash
# Store token in a file (useful for Docker secrets)
echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxx" > /path/to/hf_token.txt
HF_TOKEN_FILE=/path/to/hf_token.txt
```

#### Model Caching
Models are automatically cached after first download:
```bash
# Default cache directory (will be created if it doesn't exist)
MODEL_CACHE_DIR=/app/models

# In Docker, mount this as a volume for persistence:
# volumes:
#   - model_cache:/app/models
```

**Benefits:**
- **Avoids rate limits** when downloading models from Hugging Face
- **Faster startup** after first run (models loaded from cache)
- **Offline usage** once models are cached
- **Bandwidth savings** in production deployments

### 🔐 Webhook Security

When `WEBHOOK_SECRET` is set, the `/webhook` endpoint requires HMAC-SHA256 authentication:

```bash
# Generate signature
payload='{"ticket_id": 12345}'
signature=$(echo -n "$payload" | openssl dgst -sha256 -hmac "$WEBHOOK_SECRET" -hex | cut -d" " -f2)

# Send authenticated request
curl -X POST http://localhost:8000/webhook \
     -H "Content-Type: application/json" \
     -H "X-Webhook-Signature: $signature" \
     -d "$payload"
```

If `WEBHOOK_SECRET` is unset, authentication is **disabled** (backward compatibility).

---

## 🏋️‍♀️  Trainer details

Run manually (outside Docker) with your local Python:
```bash
pip install -r requirements.txt
python -m trainer.train_freshservice
```
• Only *new* tickets are embedded – the script loads previous metadata for incremental updates.  
• After finishing, `current` symlink is updated atomically.  
• Log output is written to console; failures are non-zero exit codes (ideal for CI schedulers).

### Adding custom heuristics
Open `src/freshservice/freshservice_client.py` and tweak `_calculate_duplicate_probability()` or adjust `ignore_subject_phrases` in `src/config.py`.

---

## 🌐  Serve / API endpoints

| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/webhook` | `{ "ticket_id": 123 }` | Fetches ticket, compares to ANN index within same department, merges if probability ≥ threshold. Returns `{ "merged": bool }`. |
| GET | `/review` | – | Human-in-the-loop review UI (HTML page). |
| GET | `/candidates` | `?max_pairs=25` | Returns potential duplicate pairs for manual review (JSON). |
| POST | `/label` | `{ "ticket1_id": "123", "ticket2_id": "456", "is_duplicate": true }` | Label a ticket pair for active learning. |
| POST | `/reloadIndex` | – | Reload snapshot & embedding model. Requires header `X-Reload-Token`. |
| GET | `/healthz` | – | Readiness probe, returns vector count. |
| GET | `/metrics` | – | Prometheus/OpenMetrics exposition. |

### Hot-reload flow
1. Trainer writes new snapshot.
2. External process calls `/reloadIndex` (or rely on watchdog).  
3. Serve clears internal lru-caches and memory-maps the new FAISS file – all in <200 ms.

## 🏢  Department-Aware Duplicate Detection

The system automatically enforces **department boundaries** when detecting potential duplicates. This ensures data privacy and more relevant results:

### How it works
- **Automatic extraction**: Department IDs are extracted from Freshservice API responses (`department_id` field)
- **Similarity filtering**: Only tickets within the same department are compared for similarity
- **Cross-department isolation**: Tickets from different departments never influence each other's duplicate detection
- **Backward compatibility**: Tickets without department IDs fall back to global comparison

### Benefits
- **🔒 Data Privacy**: Prevents cross-department information leakage
- **🎯 Relevant Results**: More accurate duplicate detection within organizational boundaries  
- **⚡ Performance**: Reduced search space leads to faster similarity calculations
- **📊 Better Accuracy**: Department context improves probability calculations

### Configuration
Department filtering is **automatic** and requires no additional configuration. The system:
- Reads `department_id` from Freshservice ticket API responses
- Gracefully handles tickets with missing or null department IDs
- Maintains full backward compatibility with existing deployments

## 🕵️  Human-in-the-Loop Review Console

The system includes a **Streamlit-based review console** for manual validation of potential duplicates that fall below the automatic merge threshold. This enables continuous improvement through active learning.

### Features
- **Confidence-based filtering**: Shows tickets with similarity between configurable thresholds (default: 65%-90%)
- **Session persistence**: Tracks already-reviewed pairs within a session to avoid duplicates
- **One-click actions**: Approve merges or mark as not-duplicates with single button clicks
- **Real-time feedback**: Labels are immediately fed back to the training system
- **Batch workflows**: Process multiple candidates efficiently in sequence

### Running the Review Console
```bash
# Install additional dependencies
pip install streamlit

# Set environment variables for the console
export REVIEW_USER="your-name"
export REVIEW_SIM_LOWER="0.65"    # Lower similarity threshold
export REVIEW_SIM_UPPER="0.90"    # Upper similarity threshold  
export REVIEW_MAX="100"           # Max candidates to show
export REVIEW_DAYS_BACK="60"      # Days to look back for tickets

# Launch the console
streamlit run src/review_console.py --server.port 8501
```

The console will be available at `http://localhost:8501` and integrates with the same Freshservice configuration as the main system.

## 🤖  Automatic Ticket Merging

**Yes, the system automatically merges tickets** when configured with a Freshservice webhook. Here's how it works:

### Automatic Merge Process
1. **Webhook Trigger**: Freshservice sends a webhook when a new ticket is created
2. **Similarity Analysis**: The system compares the new ticket against existing tickets in the same department
3. **Probability Calculation**: Uses cosine similarity + domain heuristics (email matching, department context)
   - 85% similarity score + 10% email bonus + 5% department bonus
4. **Automatic Merge**: If probability ≥ threshold (default 90%), tickets are merged automatically
5. **Merge Implementation**: Adds private notes to both tickets and closes the duplicate

### Merge Mechanism
Since Freshservice doesn't have a native merge API, the system implements a **merge workaround**:
- Adds explanatory private note to the winning (original) ticket  
- Adds reference note to the duplicate ticket pointing to the master
- Closes the duplicate ticket (status = 5)
- All future correspondence happens on the winning ticket

### Safety Features
- **High Threshold**: Default 90% confidence prevents false positives
- **Department Isolation**: Only merges tickets within the same department
- **Audit Trail**: All merges are logged with detailed probability scores
- **Manual Override**: Unmerged tickets (below threshold) are logged for human review

### Configuration
```bash
SIMILARITY_THRESHOLD=0.9    # 90% confidence required for auto-merge
WEBHOOK_SECRET=your_secret  # Optional webhook authentication
```

**Note**: Automatic merging only occurs when tickets are submitted via the `/webhook` endpoint. The trainer and review console do not perform automatic merges.

---

## 📊  Observability

• **Logs** – Unified stdlib + Loguru formatter. Colourised in console, JSON-less plain text in files.  
• **Metrics** – Two counters + one histogram exported via `prometheus_client`; no background threads.

---

## 🧪  Tests

```bash
pip install -r requirements.txt -r dev-requirements.txt
pytest -q
```
The suite uses pytest + coverage and mocks HTTP calls – no network dependency.

---

## 🔄  CI/CD Pipeline

### Pipeline Jobs

1. **Test Suite** - Runs tests across Python 3.10, 3.11, and 3.12
   - Installs dependencies with pip caching
   - Executes pytest with coverage reporting (75% minimum)
   - Uploads coverage to Codecov

2. **Code Quality** - Enforces code standards
   - Runs pre-commit hooks (Black, isort, flake8)
   - Type checking with mypy
   - Security scanning with Bandit

3. **Docker Build** - Validates containerization
   - Builds both trainer and serve images
   - Uses Docker layer caching for speed
   - Validates docker-compose configuration

4. **Security Scan** - Vulnerability detection
   - Trivy filesystem scanning
   - Results uploaded to GitHub Security tab

### Local Development Setup

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run quality checks locally
pre-commit run --all-files

# Run tests with coverage
pytest --cov=src --cov=trainer --cov-report=html

# Type checking
mypy src/ trainer/

# Security scan
bandit -r src/ trainer/
```

All tools are configured via `pyproject.toml` for consistency across environments.

---

## 🤝  Contributing

1. Fork & clone, create a branch.  
2. Adhere to **PEP-8** and run `pytest` before pushing.  
3. For new modules write unit tests under `tests/`.  
4. Open a pull request – CI will run lint + tests.

Feel free to open issues for bugs or feature requests. PRs are warmly welcomed!

---

## 📝  License

Apache 2.0 – see `LICENSE`. Commercial use, distribution and modification are permitted.

---

## ❤️  Acknowledgements

• [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)  
• [FAISS](https://github.com/facebookresearch/faiss)  
• [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/)  
• [Loguru](https://github.com/Delgan/loguru)
