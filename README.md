# ESG Hybrid Trust Dashboard

Streamlit app for uploading ESG or sustainability PDF reports, extracting sentence-level disclosures, and evaluating them with the Hybrid v4 task structure from `Untitled1.ipynb`.

The notebook pipeline is represented in the app as:

- Task 1 `promise_status`: whether the sentence contains a commitment or target
- Task 2 `verification_timeline`: already, within 2 years, 2-5 years, longer than 5 years, or N/A
- Task 3 `evidence_status`: whether supporting evidence is present
- Task 4 `evidence_quality`: Clear, Not Clear, Misleading, or N/A

The current implementation in `esg_dashboard/hybrid_model.py` is an offline deterministic inference layer that preserves the notebook output contract. It can be replaced later with the trained RoBERTa checkpoint and GPT/RAG calls from the notebook without changing the Streamlit dashboard shape.

## Dashboard Features

- PDF upload and text extraction
- Sentence splitting for Chinese and English disclosures
- ESG category and topic grouping
- Overall trust score and greenwashing risk
- Greenwashing radar
- Peer benchmarking
- Active audit feed
- PDF viewer
- AI analysis panel
- Milestone timeline
- CSV export

## Run

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

## Model Integration Notes

`Untitled1.ipynb` describes a Hybrid v4 model:

- RoBERTa handles `promise_status` and `evidence_status`
- GPT/RAG handles `verification_timeline` and `evidence_quality`
- The final score uses official weights:
  - `promise_status`: 0.20
  - `evidence_status`: 0.30
  - `evidence_quality`: 0.35
  - `verification_timeline`: 0.15

To move from the offline rule layer to the trained model, add the checkpoint/API implementation behind `HybridESGAnalyzer.predict_one()` while keeping the returned fields unchanged.
