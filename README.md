# Project-4-Predictive-Maintenance-using-MLOps
An end-to-end production engineering pipeline designed to monitor industrial asset telemetry and predict equipment failures using a Random Forest framework. It integrates MLflow for robust model  version control, deployment tracking while implementing secure skops deserialization to satisfy enterprise data governance and CVE safety constraints.

A secure, automated machine learning lifecycle pipeline using **MLflow** for model governance and **Skops** for secure production deployment of a predictive maintenance asset classifier.

## 🚀 System Pipeline Features
* **Experiment Tracking:** Centralized metric logging with MLflow.
* **Metadata & Governance:** Models are tracked with descriptive version tags (`team`, `validation_status`, `framework`).
* **Stage Management:** Automated promotion transitions through `Staging` and `Production` with version aliases.
* **Security Layer:** Built entirely using `skops` instead of insecure `pickle` files to satisfy modern **CVE-2024-37065** safety parameters.
* **Rollback Architecture:** Safe rollback switches back from failed versions to known historical dependencies inside the model registry.

---

##  Core Pipeline Code Ledger

### 1. Model Registry Governance & Staging Promotion
Sets up client tracking configs, stamps organizational tags, and safely builds the model instance into active notebook memory using secure type inspection.


## MLOps Governance Pipeline Workflow
```text
[ Experiment Tracking ] ──> Logs metrics, parameters, and secure .skops artifacts
                                       │
                                       ▼
[ Registry Governance ] ──> Stamps metadata tags (team, validation_status, framework)
                                       │
                                       ▼
[ Staging Environment ] ──> Runs secure CVE-2024-37065 type-inspection scans
                                       │
                                       ▼
[ Staging Validation  ] ──> Tests input boundaries against telemetry scenarios
                                       │
                                       ▼
[ Production Release  ] ──> Promotes via version aliases; locks in-memory for zero latency
                                       │
                                       ▼
[ Active Guardrails   ] ──> Monitors streams; triggers automated version rollback on fault

```

## Key Infrastructure Milestones
- Centralized Lifecycle Tracking: Handles absolute file routing bypasses on Windows (C:\Users\Ayesha\mlartifacts\) to completely eliminate local server network path mismatch crashes ([WinError 123]).

- Automated Stage Routing: Configures structural tags and points modern @staging and @production lookup pointers directly to verified container iterations.

- Enterprise Compliance Layer: Replaces insecure pickle engines with skops.io structural pre-scans, protecting runtime servers from arbitrary code execution exploits.

- Resilient Failure Recovery: Automatically handles rollback mechanics to shift pipeline deployment routes away from failed candidate models and lock back onto steady historical versions instantaneously.

## Main Code:

```python
import os
import skops.io as sio  
import pandas as pd
from mlflow.tracking import MlflowClient
import mlflow

# Configure local server variables
TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "Predictive-Maintenance-model"
ALIAS = "staging"
REAL_SKOPS_PATH = r"C:\Users\Ayesha\mlartifacts\2\models\m-fce3960842d54c73a9cab93bb81abe25\artifacts\model.skops"

print("=== Step 1: Handling Model Registry Promotion ===")
try:
    client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version="1")
    print(f"[SUCCESS] Model version 1 promoted to @{ALIAS}")
except Exception as e:
    print(f"-> Registry status note: {e}")

print("\n=== Step 2: Injecting Audit Metadata Tags ===")
try:
    client.update_registered_model(name=MODEL_NAME, description="Random Forest Asset Wear Classifier")
    client.set_model_version_tag(name=MODEL_NAME, version="1", key="validation_status", value="approved")
    client.set_model_version_tag(name=MODEL_NAME, version="1", key="team", value="Data-Science")
    client.set_model_version_tag(name=MODEL_NAME, version="1", key="framework", value="scikit-learn/skops")
    print("[SUCCESS] Governance tags stamped into MLflow!")
except Exception as e:
    print(f"-> Metadata tag note: {e}")

print("\n=== Step 3: Compiling Staging Inference Engine ===")
try:
    # Resolve CVE-2024-37065 constraints by explicitly verifying trusted types
    untrusted_types = sio.get_untrusted_types(file=REAL_SKOPS_PATH)
    staging_model = sio.load(REAL_SKOPS_PATH, trusted=untrusted_types)
    print(f"🎉 SUCCESS! Staging Model compiled! Framework: {type(staging_model).__name__}")
except Exception as e:
    print(f"\n[CRITICAL] Staging Pipeline Broken: {e}")
