# TB and cancer CT: research prototype

A small local app for the MSc Data Science thesis *Explainable Artificial Intelligence for Interpreting CT Imaging and Clinical Narratives in Tuberculosis and Cancer Characterization. A Study in an Ethiopia Hospital.*

**It is a research tool, not a medical device.** It must not be used to diagnose, treat, triage or make decisions about any patient.

## What it does
| Step | Works today | Needs a trained model |
|---|---|---|
| View a DICOM series or PNG/JPG slices with the right CT window for the body region | yes | no |
| Read a short clinical narrative with the thesis's transparent rules: tidy text, TB / Cancer / Indeterminate / Other, certainty, organ system, text with diagnosis words hidden | yes | no |
| Probabilities for TB / Cancer / Indeterminate / Other from CT (and from the narrative, and a fused result) | no | **yes** |
| Grad-CAM heat maps, a weight-randomization sanity check, and the words moving the text model | no | **yes** |

Without a model package the app says so and shows no prediction. It does not generate placeholder results.

## Install and run (on a hospital-approved computer)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run.sh                                               # or: streamlit run app.py
```
The app listens on `127.0.0.1` only and makes no network calls. Nothing is uploaded anywhere.

## Getting a model package
1. Run Part B of the Colab notebook (`TB_Cancer_XAI_Thesis_Colab.ipynb`) on your approved CT data.
2. Run the last cell (B11, `export_model_package`). It writes `ct_resnet18.pt`, `text_model.joblib` (optional) and `model_card.json`.
3. Copy the three files into `model_package/`. Restart the app.

`model_card.json` carries the classes, CT windows, fusion weight, the abstain threshold (default 0.60) and the **test results** the app shows beside every output. Write real numbers, from the held-out patients, and keep the limitations list.

## Safety behaviour built in
- A confirmation that data are de-identified is required before anything is processed.
- DICOM header identifiers are never displayed. The app warns if headers still carry them.
- Low confidence (below the threshold) and "Indeterminate" are always routed to a person.
- The heat-map tab includes a weight-randomization check. If the map barely changes when the model is scrambled, the app tells you not to rely on it.
- Optional local audit log (time, model version, top class, confidence, a short hash of the first slice). No images, no names.

## Tests and what is verified
`python tests/test_core.py` checks the narrative rules, DICOM reading (synthetic series with scrambled slice order), body-region windowing, the cautious decision rule, fusion, and that the app refuses to predict without a model. The Streamlit app was also loaded headlessly in inspection mode with no errors.

**Not yet verified:** the PyTorch paths (model loading, Grad-CAM, randomization check) could not be run where this was built because PyTorch would not install. The tests for them are included and run automatically when PyTorch is present. Run them once on your machine before you trust the heat maps.

## Before any use beyond research
Held-out testing on patients not used in training, external validation at another site, clinician review of explanations, ethics approval, and the regulatory and data-protection steps required in Ethiopia for software that influences care.
