Put the files exported by the notebook (cell B11) here:

- `ct_resnet18.pt` : CT model weights (ResNet-18, 4 classes)
- `text_model.joblib` : narrative model (TF-IDF + logistic regression), optional
- `model_card.json` : classes, CT windows, fusion weight, abstain threshold, test results, limitations

Without these files the app runs in inspection mode and does not predict.
