"""Loading a trained model package and turning slices into probabilities.
If no package is present the app stays in inspection mode: it never invents a prediction."""
import json, os
import numpy as np

CLASSES_DEFAULT = ["TB", "Cancer", "Indeterminate", "Other"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def torch_available():
    try:
        import torch, torchvision  # noqa: F401
        return True
    except Exception:
        return False

def load_package(pkg_dir):
    """Returns dict with keys: card, ct_model (or None), text_model (or None), messages."""
    out = {"card": None, "ct_model": None, "text_model": None, "messages": []}
    card_path = os.path.join(pkg_dir, "model_card.json")
    if not os.path.exists(card_path):
        out["messages"].append("No model_card.json found. Inspection mode only.")
        return out
    out["card"] = json.load(open(card_path))
    ckpt = os.path.join(pkg_dir, out["card"].get("ct_checkpoint", "ct_resnet18.pt"))
    if os.path.exists(ckpt):
        if torch_available():
            import torch
            from torchvision import models
            n = len(out["card"].get("classes", CLASSES_DEFAULT))
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(512, n)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            out["ct_model"] = m
        else:
            out["messages"].append("PyTorch is not installed, so the CT model cannot be loaded.")
    else:
        out["messages"].append(f"CT checkpoint not found at {ckpt}.")
    tpath = os.path.join(pkg_dir, out["card"].get("text_model", "text_model.joblib"))
    if os.path.exists(tpath):
        try:
            import joblib
            out["text_model"] = joblib.load(tpath)
        except Exception as e:
            out["messages"].append(f"Text model could not be loaded: {type(e).__name__}")
    return out

def predict_ct(model, rgb_slices, size=224):
    """Mean of per-slice softmax probabilities -> (record_probs, per_slice_probs)."""
    import torch
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = torch.stack([tf(Image.fromarray(s)) for s in rgb_slices])
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).numpy()
    return p.mean(axis=0), p

def text_probs(text_pipeline, text, classes=CLASSES_DEFAULT):
    """Map a fitted sklearn pipeline's probabilities onto the full class list."""
    p = np.zeros(len(classes))
    proba = text_pipeline.predict_proba([text])[0]
    for c, v in zip(text_pipeline.classes_, proba):
        p[int(c)] = v
    return p

def fuse(ct_p, tx_p, w_text):
    if not 0 <= w_text <= 1:
        raise ValueError("w_text must be between 0 and 1")
    return w_text * np.asarray(tx_p) + (1 - w_text) * np.asarray(ct_p)

def decide(probs, classes=CLASSES_DEFAULT, threshold=0.60):
    """Turns probabilities into a cautious statement. Low confidence and 'Indeterminate' always go to a human."""
    probs = np.asarray(probs, dtype=float)
    k = int(probs.argmax()); top, conf = classes[k], float(probs[k])
    review = conf < threshold or top == "Indeterminate"
    if review:
        msg = f"No firm characterization. Leaning '{top}' ({conf:.0%}); needs review by a clinician."
    else:
        msg = f"Most consistent with '{top}' ({conf:.0%}). This is a research output, not a diagnosis."
    return {"top": top, "confidence": conf, "needs_human_review": bool(review), "message": msg}
