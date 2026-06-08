# ESG Risk Dashboard

Streamlit dashboard for uploading ESG or sustainability PDF reports, extracting ESG-related sentences, and presenting category-based risk signals without exposing numeric trust or risk scores.

## Project Layout

```text
app.py                         Streamlit entrypoint
data/
  vpesg_4k_train_1000.json     Training/reference peer data
  samples/                     Sample ESG PDF reports
esg_dashboard/
  core/                        Model, scoring, RAG, taxonomy, config
  data/                        PDF/text processing pipeline
  ui/                          Shared Streamlit UI helpers
models/                        Local deployed model artifacts
notebooks/                     Training and experiment notebooks
```

## Dashboard Features

- PDF upload and ESG sentence extraction
- Promise, evidence, timeline, and evidence-quality classification
- Merged risk levels: High, Medium, Low, Neutral
- Risk statistics and sentence-level results
- Peer comparison by risk-level distribution
- ESG issue summary and issue-level detail view
- Issue risk matrix for categorical model outputs
- Audit action feed and related evidence paragraphs
- CSV export for issue summaries

## Run

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

## Model Assets

The Streamlit app expects the deployed model artifacts under:

```text
models/final_roberta_task13_A_stable_baseline/
```

This folder should contain the RoBERTa checkpoints, tokenizer, config, and RAG retrieval assets exported from the final training notebook.

## Notebooks

- `notebooks/Untitled1.ipynb`: earlier experiment/reference notebook
- `notebooks/final(修改).ipynb`: final training/export notebook

The app no longer depends on notebooks at runtime; they are kept as training and reproduction references.
