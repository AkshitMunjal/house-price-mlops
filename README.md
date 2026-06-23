# 🏡 End-to-End MLOps System — Bengaluru House Price Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker" />
  <img src="https://img.shields.io/badge/AWS-EC2%20%7C%20S3-FF9900?style=flat-square&logo=amazonaws" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat-square" />
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions" />
  <img src="https://img.shields.io/badge/DagsHub-Tracking-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=flat-square&logo=prometheus" />
  <img src="https://img.shields.io/badge/Grafana-Dashboards-F46800?style=flat-square&logo=grafana" />
</p>

> **A production-grade MLOps system** — not just a model, but a full ML lifecycle covering cloud-based data ingestion, experiment tracking, containerized deployment, CI/CD automation, Prometheus + Grafana monitoring, S3 prediction logging, concept drift detection, and conditional retraining.

🔗 **DagsHub Project:** [https://dagshub.com/AkshitMunjal/house-price-mlops](https://dagshub.com/AkshitMunjal/house-price-mlops)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Engineering Decisions](#-engineering-decisions--production-constraints)
- [Data Pipeline](#-data-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Model Training & Evaluation](#-model-training--evaluation)
- [MLOps Components](#-mlops-components)
- [FastAPI Inference Service](#-fastapi-inference-service)
- [Dockerized Architecture](#-dockerized-architecture)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring & Logging](#-monitoring--logging)
- [Conditional Retraining](#-conditional-retraining)
- [Scheduled Orchestration](#-scheduled-orchestration)
- [Tech Stack](#-tech-stack)
- [Performance Benchmarks](#-performance-benchmarks)
- [Key Engineering Learnings](#-key-engineering-learnings)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 Overview

This project goes well beyond a Kaggle-style notebook. It simulates what a real ML engineering team builds and maintains in production — a **reliable, observable, and automatically retrained** ML system deployed on cloud infrastructure.

**Target Users:** Home buyers, sellers, real estate agents, and property analysts in Bengaluru.

**Core Engineering Goals:**

| Goal | Approach |
|---|---|
| Real-time inference | FastAPI + lazy-loaded model caching |
| Reproducible training | DVC + MLflow experiment tracking |
| Cloud-native data ingestion | AWS S3 → DVC pipeline stages |
| Safe deployments | Two-container separation (inference vs. retraining) |
| Operational reliability | CI/CD + conditional retraining + drift detection |
| Production observability | Prometheus + Grafana + S3 prediction logging |
| Low-cost infrastructure | AWS EC2 Free Tier + lightweight cron orchestration |

---

## 🏗️ System Architecture

<img width="2816" height="1536" alt="ml_project_system_architecture" src="https://github.com/user-attachments/assets/cd43d10c-a6cd-49a0-bfe1-1ef555d726d3" />

```
                        ┌─────────────────────────────────┐
                        │           Developer              │
                        │       Pushes Code to GitHub      │
                        └────────────┬────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────────┐
                        │      GitHub Actions (CI/CD)      │
                        │  Tests → Build → Push → Deploy   │
                        └────────────┬────────────────────┘
                                     │
                    ┌────────────────┴──────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌───────────────────┐               ┌─────────────────────┐
        │   DockerHub        │               │  MLflow + DagsHub   │
        │  (Image Registry)  │               │  (Model Registry)   │
        └─────────┬─────────┘               └──────────┬──────────┘
                  │                                     │
                  ▼                                     │
        ┌─────────────────────────────────────────────┐│
        │              AWS EC2 Instance               ││
        │                                             ││
        │  ┌──────────────────┐  ┌─────────────────┐  ││
        │  │ Inference         │  │ Retraining      │◄─┘│
        │  │ Container         │  │ Container       │   │
        │  │ (FastAPI API)     │  │ (cron-triggered)│   │
        │  └────────┬─────────┘  └────────┬────────┘   │
        │           │                     │             │
        └───────────┼─────────────────────┼─────────────┘
                    │                     │
                    ▼                     ▼
             ┌────────────┐        ┌────────────────┐
             │  AWS S3     │        │ MLflow Registry │
             │ (Prediction │        │ (@production    │
             │  Logging)   │        │  alias)         │
             └────────────┘        └────────────────┘
```

---

## 📁 Project Structure

```
machine_learning_application/
│
├── .dvc/                          # DVC cache and remote config
├── .github/
│   └── workflows/                 # GitHub Actions CI/CD pipeline definitions
│
├── artifacts/                     # Saved model artifacts and encoders
├── data/
│   ├── raw/                       # Original unprocessed dataset
│   └── processed/                 # Cleaned and feature-engineered data
│
├── frontend/                      # UI for interacting with the prediction API
├── mlruns/                        # MLflow local experiment tracking store
├── notebooks/                     # Exploratory data analysis and prototyping
│
├── src/
│   ├── api/                       # FastAPI app — routes, schemas, model loader
│   ├── data/                      # Data ingestion and cleaning scripts
│   ├── features/                  # Feature engineering and transformation logic
│   ├── monitoring/                # Drift detection and prediction logging
│   ├── pipelines/                 # End-to-end training and inference pipelines
│   └── training/                  # Model training, hyperparameter tuning, evaluation
│
├── tests/                         # Unit and integration tests
├── venv/                          # Python virtual environment
│
├── .dockerignore                  # Docker build exclusions
├── .dvcignore                     # DVC tracking exclusions
├── .env                           # Environment variables (not committed)
├── .gitignore                     # Git exclusions
├── Dockerfile                     # Inference container definition
├── Dockerfile.retraining          # Retraining container definition
├── dvc.lock                       # DVC pipeline lock file
├── dvc.yaml                       # DVC pipeline stage definitions
├── main.py                        # Application entry point
├── mlflow.db                      # Local MLflow SQLite tracking store
├── pytest.ini                     # Pytest configuration
├── README.md
└── requirements.txt               # Python dependencies
```

---

## ⚙️ Engineering Decisions & Production Constraints

### Why XGBoost Over Deep Learning?

This was a deliberate, reasoned decision — not a default:

| Factor | XGBoost | Deep Learning |
|---|---|---|
| Dataset size (~13K rows) | ✅ Ideal fit | ❌ High overfitting risk |
| Inference latency | ✅ ~90–130 ms | ❌ Higher overhead |
| Infrastructure cost | ✅ CPU-only | ❌ GPU required |
| Deployment complexity | ✅ Simple serialization | ❌ Larger dependencies |
| Tabular performance | ✅ Consistently strong | ❌ No clear advantage |

> **Decision:** Production efficiency and reliability over unnecessary model complexity.

---

### Why Cron Over Airflow?

| Factor | Linux Cron | Apache Airflow |
|---|---|---|
| Memory overhead | ✅ Minimal | ❌ ~500 MB+ baseline |
| Setup complexity | ✅ Zero dependencies | ❌ Significant config overhead |
| Fit for scale | ✅ Right-sized for project | ❌ Overkill at this scale |
| EC2 Free Tier compatibility | ✅ Yes | ❌ Resource-constrained |

> **Decision:** Lightweight orchestration appropriate for project scale. Airflow is listed as a future upgrade path.

---

### Latency Budget

```
Target: < 100 ms per inference request

Achieved:
  Warm inference:   ~90–130 ms  ✅
  Cold start:       Several seconds (MLflow artifact load — one-time)

Optimizations applied:
  → Lazy model loading with in-memory cache
  → Preprocessing pipeline reuse during inference
  → Dynamic feature schema alignment
  → Inference and retraining workloads isolated in separate containers
```

---

## 🔧 Data Pipeline

### Cloud-Based Data Ingestion

Rather than relying on manual local file loading, the raw dataset is ingested directly from **AWS S3** through a dedicated ingestion pipeline — a production-standard pattern for reproducibility and portability.

```
AWS S3 (raw dataset)
        ↓
download_data_pipeline.py
        ↓
data/raw/bengaluru_house_prices.csv
        ↓
DVC-tracked pipeline stage (dvc.yaml)
```

**Benefits of this approach:**
- Reproducible across any environment — no manual file transfers
- Centralized dataset management with versioning via DVC
- Mirrors real production data ingestion patterns

---

### Data Cleaning & Validation

The dataset (~8,600–13,000 records) required extensive **domain-informed cleaning** rather than naive statistical filtering.

**Key Decisions:**

- Missing value imputation with business logic, not blind mean/median fills
- Inconsistent data type correction (e.g., range-format square footage like `1200–1500`)
- BHK and bathroom feature extraction from raw text fields
- Unrealistic pricing pattern removal (e.g., price per sqft outliers assessed per location)
- Localized outlier removal — global IQR filtering was deliberately avoided to preserve realistic property variations across Bengaluru's diverse micro-markets

---

## 🧠 Feature Engineering

| Feature | Description |
|---|---|
| `bhk` | Extracted from free-text size field |
| `total_sqft_normalized` | Handles range inputs, inconsistent units |
| `price_per_sqft` | Used for consistency validation and outlier detection |
| `location_grouped` | High-cardinality locations bucketed via frequency threshold |
| Location OHE vectors | One-hot encoded after grouping to reduce sparsity |

**Location Grouping Strategy:** Multiple frequency thresholds were evaluated to balance sparsity reduction against information loss. The final threshold was chosen empirically.

---

## 📊 Model Training & Evaluation

### Training Pipeline

```
Raw Data
   ↓
Cleaning + Validation
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
GridSearchCV (Hyperparameter Tuning)
   ↓
Cross-Validation
   ↓
Evaluation (multiple metrics + segments)
   ↓
MLflow Logging + Artifact Persistence
```

### Hyperparameter Search Space (GridSearchCV)

```python
{
  "learning_rate":       [0.01, 0.05, 0.1],
  "max_depth":           [3, 5, 7],
  "n_estimators":        [100, 300, 500],
  "subsample":           [0.7, 0.8, 1.0],
  "colsample_bytree":    [0.7, 0.8, 1.0]
}
```

### Evaluation Philosophy

Standard R² alone is a misleading proxy for production model quality. This project used:

- **Segment-wise evaluation** — separate metrics for low/mid/high price range properties
- **Percentage error analysis** — more interpretable for stakeholders than absolute RMSE
- **Root-cause analysis** on extreme failures — traced to feature-target inconsistencies and rare-pattern properties
- **Scale-preserved evaluation** — metrics computed on both log-transformed and original price scales

> **Trade-off accepted:** Slightly lower R² in exchange for significantly reduced extreme prediction failures and improved reliability on edge-case properties.

---

## 📦 MLOps Components

### MLflow + DagsHub Integration

🔗 [View Experiments on DagsHub](https://dagshub.com/AkshitMunjal/house-price-mlops)

| Capability | Detail |
|---|---|
| Experiment tracking | All runs logged with parameters, metrics, tags |
| Artifact logging | Models, preprocessors, and feature schemas versioned |
| Model registry | Centralized registry with versioned model entries |
| Production governance | `@production` alias for controlled model promotion |
| Remote tracking | DagsHub-hosted MLflow tracking server |

### Model Governance Flow

```
Retrained Candidate Model
         ↓
   Logged to MLflow
         ↓
   Manual Review (metrics comparison)
         ↓
   Promote → @production alias
         ↓
   Inference container loads @production model
```

This prevents automatic promotion of degraded models into production.

---

## 🚀 FastAPI Inference Service

The prediction API is built with FastAPI and exposes a clean REST interface.

**Endpoint:** `POST /predict`

**Capabilities:**
- Input validation via Pydantic schemas
- Dynamic feature alignment to match training schema
- Preprocessing pipeline reuse (identical to training transforms)
- Cached model loading (loaded once, reused per request)
- JSON prediction response with metadata

**Consistent preprocessing** between training and inference is enforced by reusing the same serialized pipeline artifact — eliminating training-serving skew.

---

## 🐳 Dockerized Architecture

Two containers with isolated responsibilities:

### Container 1 — Inference (`Dockerfile`)

```
Responsibilities:
  ├── Serve FastAPI prediction API
  ├── Load @production model from MLflow registry
  ├── Run preprocessing pipeline
  └── Log predictions to AWS S3
```

### Container 2 — Retraining (`Dockerfile.retraining`)

```
Responsibilities:
  ├── Evaluate production model on fresh data
  ├── Trigger retraining if performance degraded
  ├── Log new experiment to MLflow
  └── Register retrained model for human review
```

**Benefits of separation:**
- Inference container is never blocked by retraining jobs
- Retraining failures do not affect serving availability
- Independent scaling and deployment of each workload
- Cleaner operational boundaries

---

## 🔄 CI/CD Pipeline

```
Developer pushes to main branch
            ↓
    GitHub Actions triggered
            ↓
    ┌──────────────────┐
    │  Run Test Suite  │  ← pytest with pytest.ini config
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ Build Docker Image│
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Push to DockerHub│
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Deploy to EC2   │  ← SSH + docker pull + restart
    └──────────────────┘
```

Every push to `main` results in a fully tested, containerized, and deployed update — with zero manual deployment steps.

---

## 📡 Monitoring & Logging

The system implements a multi-layer monitoring strategy covering both infrastructure metrics and ML-specific observability.

### Prometheus + Grafana (Infrastructure & API Monitoring)

After deploying the FastAPI inference service, **Prometheus** was integrated to scrape runtime metrics from the application:

| Prometheus Metric | What It Tracks |
|---|---|
| Request count | Total inference requests served |
| API latency | Per-request response time distribution |
| Response times | P50 / P95 / P99 percentile breakdown |
| Inference monitoring stats | Application-level custom metrics |

**Grafana dashboards** provide real-time visualization of:
- API performance and latency trends over time
- System health and resource utilization
- Production monitoring metrics with alerting capability

This gives real-time visibility into inference behavior and operational stability post-deployment.

### AWS S3 Prediction Logging (ML Observability)

All prediction metadata is uploaded to **AWS S3** for durable, queryable storage. This logging layer supports:

| Component | Implementation |
|---|---|
| Prediction logging | Inputs, outputs, latency, and timestamps per request |
| Log storage | AWS S3 — durable and queryable at scale |
| Concept drift evaluation | Production model performance monitored against recent data distribution |
| Model health checks | Run as part of the conditional retraining workflow |
| Future drift detection | S3 logs feed directly into retraining dataset generation |

---

## 🔁 Conditional Retraining

Retraining is **not triggered blindly on a schedule**. The system evaluates whether retraining is actually warranted:

```
Scheduled trigger (cron)
         ↓
Evaluate production model on recent data
         ↓
         ├── Performance OK?  →  Exit. No retraining.
         │
         └── Performance degraded?
                    ↓
             Retrain candidate model
                    ↓
             Log to MLflow registry
                    ↓
             Await manual promotion review
```

This prevents unnecessary retraining noise and avoids inadvertently deploying worse models.

---

## ⏰ Scheduled Orchestration

The retraining pipeline is triggered via **Linux cron** on the EC2 instance.

```bash
# Example cron entry — runs retraining evaluation nightly
0 2 * * * docker run --rm house-price-retraining:latest
```

**Why not Airflow?**
At this project's scale, Airflow's memory footprint (~500 MB+) would exhaust the EC2 free-tier instance. Cron provides reliable scheduling with near-zero overhead. Airflow migration is documented in the future roadmap.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML Framework | XGBoost, Scikit-learn |
| API Framework | FastAPI + Uvicorn |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Remote Tracking Server | DagsHub |
| Cloud Compute | AWS EC2 (Free Tier) |
| Cloud Storage | AWS S3 |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Orchestration | Linux cron |
| Monitoring & Dashboards | Prometheus + Grafana |
| Testing | pytest |
| Version Control | Git + GitHub |

---

## 📈 Performance Benchmarks

| Metric | Value |
|---|---|
| Warm Inference Latency | ~90–130 ms |
| Cold Start Latency | Several seconds (one-time model load) |
| Target Latency SLA | < 100 ms |
| Deployment Platform | AWS EC2 Free Tier |
| Deployment Type | Dockerized FastAPI |
| Dataset Size | ~8,600–13,000 records |

---

## 🔭 Future Roadmap

| Feature | Description |
|---|---|
| Real-time drift detection | Stream S3 prediction logs into automated drift detection pipeline |
| Automated model promotion | Remove manual promotion step when confidence thresholds are met |
| Kubernetes deployment | Horizontal scaling, rolling updates, and self-healing pods |
| Terraform IaC | Fully reproducible infrastructure provisioning via code |
| Airflow orchestration | Migrate from cron for complex multi-step DAG-based workflows |
| Canary deployments | Gradual traffic shifting for safer, lower-risk model rollouts |
| Feature store | Centralized feature registry for consistent train/serve feature access |

---

## 💡 Key Engineering Learnings

This project required solving real production ML challenges beyond model accuracy:

- **Training-serving skew prevention** — enforced by serializing and reusing preprocessing pipelines at inference time
- **Container isolation strategy** — separating inference and retraining workloads for operational independence and availability
- **Cloud-native data ingestion** — S3-based pipeline ingestion instead of local file loading for full environment portability
- **Right-sized infrastructure** — matching orchestration complexity to actual project requirements (cron over Airflow)
- **Model governance** — using MLflow aliases to prevent unreviewed or degraded models from reaching production
- **Multi-layer monitoring** — combining Prometheus/Grafana for infrastructure visibility with S3 logs for ML observability
- **Evaluation depth** — segment-wise and percentage-error analysis reveals far more than aggregate R² alone
- **Cloud deployment debugging** — navigating EC2 networking, Docker networking, and environment parity issues

---

## 🧾 Conclusion

This project demonstrates what production ML engineering actually looks like:

```
S3 Data Ingestion → DVC Pipeline → Feature Engineering → Experiment Tracking
                                          ↓
                    Model Training → Evaluation → Registry → Deployment
                                          ↓
              Prometheus/Grafana → Prediction Logging → Drift Monitoring → Conditional Retraining
                                          ↓
                         CI/CD → Containerized → Cloud-Deployed → Observable
```

The focus was building a system that is **reliable, maintainable, and production-ready** — not just maximizing a benchmark metric on a notebook.

---

<p align="center">
  Built with production engineering principles · Deployed on AWS · Tracked on <a href="https://dagshub.com/AkshitMunjal/house-price-mlops">DagsHub</a>
</p>
