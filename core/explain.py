"""Explanations: Grad-CAM for the CT model, word contributions for the text model, and a sanity check."""
import numpy as np

def gradcam(model, x, target_class=None):
    """x: tensor (3,H,W) already normalised. Returns (cam in 0..1 of size h x w, class index)."""
    import torch
    model.eval()
    layer = model.layer4[-1]
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        out = model(x.unsqueeze(0))
        c = int(out.argmax(1)) if target_class is None else int(target_class)
        model.zero_grad()
        out[0, c].backward()
        wts = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((wts * store["a"]).sum(1))[0]
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    return cam, c

def randomization_check(model, x):
    """Spearman correlation between the real map and a map from a copy with re-initialised layer4 and fc.
    Close to 1 means the map does not depend on what the model learned, so do not trust it."""
    import copy
    from scipy.stats import spearmanr
    real, c = gradcam(model, x)
    m2 = copy.deepcopy(model)
    for mod in list(m2.layer4.modules()) + [m2.fc]:
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    rand, _ = gradcam(m2, x, target_class=c)
    return float(spearmanr(real.ravel(), rand.ravel()).correlation)

def overlay(rgb_slice, cam, size=224, alpha=0.45):
    """Blend a heat map onto the slice. Returns uint8 image."""
    from PIL import Image
    import matplotlib
    base = np.array(Image.fromarray(rgb_slice).resize((size, size)))
    cam_img = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    heat = (matplotlib.colormaps["jet"](cam_img)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * base + alpha * heat).astype(np.uint8)

def text_contributions(pipeline, text, class_index, top=6):
    """Which words push the text model toward a class (linear model: weight x tf-idf)."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    names = tfidf.get_feature_names_out()
    vec = tfidf.transform([text])
    row = list(clf.classes_).index(class_index)
    contrib = vec.multiply(clf.coef_[row]).tocoo()
    items = [(names[j][3:], float(v)) for j, v in zip(contrib.col, contrib.data) if names[j].startswith("w__")]
    items.sort(key=lambda t: -abs(t[1]))
    return items[:top]
