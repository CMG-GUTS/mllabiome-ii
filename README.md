<img src="assets/favicon.svg" width="80" height="80" alt="mllabiome-ii icon">

Interactive local inference app for developed and trained microbiota models.

This repository contains the application code only. Model ZIP artifacts are intentionally distributed separately as GitHub Release assets.

## What is included

- FastAPI backend for packaged MPMA-E model inference
- Interactive frontend for model selection, upload, batch prediction, and local explanations
- Local SHAP/LIME explainability for the selected sample
- Export tools for creating deployment model ZIPs from completed experiments

## Requirements

- Python 3.11+
- Node.js 20–25
- GitHub CLI, optional but recommended for downloading model release assets

## Clone and install

```bash
git clone https://github.com/CMG-GUTS/mllabiome-ii.git mllabiome-ii
cd mllabiome-ii
python -m venv .venv
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
```

Install frontend dependencies:

```bash
cd app/frontend
npm install
cd ../..
```

## Download model artifacts from GitHub Releases

Model artifacts are expected in:

```text
app/backend/production_models/
```

Recommended download command after the `models-v0.1.0` release has been published:

```bash
scripts/download_models_from_release.sh CMG-GUTS/mllabiome-ii models-v0.1.0
```

Expected model asset filenames:

```text
ibs_mpma_e_model_v0.1.0.zip
scz_mpma_e_model_v0.1.0.zip
crc_mpma_e_model_v0.1.0.zip
ghs_mpma_e_model_v0.1.0.zip
ibd_mpma_e_model_v0.1.0.zip
dm_mpma_e_model_v0.1.0.zip
```

Manual download is also fine: place the ZIPs in `app/backend/production_models/`, then restart the backend.

## Run backend

From the repository root:

```bash
source .venv/bin/activate
python app/run_backend.py --host 0.0.0.0 --port 8000
```

Check model discovery:

```bash
curl http://localhost:8000/api/inference/models
```

## Run frontend

In a second terminal:

```bash
cd app/frontend
npm install
npm run dev
```
Open the localhost link accordingly to the displayed output in the terminal.

## Input formats

The app accepts wide sample-by-feature CSV/TSV tables. Benchmark files should contain a `sample_id` column plus taxonomic feature columns. Object model packages may also accept MetaPhlAn-style profile tables with taxonomic features in the first column and samples in the remaining columns.

A small smoke-test file is included at:

```text
app/test_ibs_sample.tsv
app/backend/test_ibs_sample.tsv
```

## Explainability

The inference UI provides local SHAP and LIME explanations for the selected sample. For interactivity on high-dimensional microbiome profiles, the backend first screens local candidate taxa and then computes SHAP/LIME on the top candidate set. 


## Optional result chat

The `/api/chat` endpoint is available for local result interpretation through Ollama. Inference and explainability do not require Ollama. Configure with `OLLAMA_URL` and `OLLAMA_MODEL` if needed.
