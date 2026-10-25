import os, sys, io, json, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import ct_io, model_io, explain
from core.narrative import analyze_narrative

def synthetic_dicom_dir(n=30):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
    d = tempfile.mkdtemp(); suid = generate_uid(); paths = []
    for k in range(n):
        meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = CTImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(os.path.join(d, f"s{k:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientName = "Test^Patient"; ds.PatientID = "X1"; ds.Modality = "CT"; ds.SeriesInstanceUID = suid
        ds.Rows = ds.Columns = 64; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"; ds.RescaleSlope = 1; ds.RescaleIntercept = -1024
        ds.ImagePositionPatient = [0, 0, float(n - k)]; ds.InstanceNumber = k
        ds.PixelData = np.random.default_rng(k).integers(0, 2000, (64, 64)).astype(np.int16).tobytes()
        ds.save_as(ds.filename); paths.append(ds.filename)
    return d, paths

def test_narrative():
    r = analyze_narrative("chest", "r/o active TB")
    assert r["category"] == "Indeterminate" and r["suspected_for"] == "TB" and r["organ_system"] == "Thoracic"
    assert "tb" not in r["text_masked"].replace("[mask]", "")
    assert analyze_narrative("abdp ct", "ovarian ca")["category"] == "Cancer"
    assert analyze_narrative("triphasic ct", "hepatic mass")["category"] == "Indeterminate"      # a mass is not cancer
    assert analyze_narrative("hrct", "post tb fibrosis")["tb_stage"] == "post-TB sequelae"
    assert analyze_narrative("neck + chest", "x")["organ_system"] == "Multi-region"

def test_ct_io():
    d, paths = synthetic_dicom_dir()
    try:
        vol, info = ct_io.load_dicom_series(paths)
        assert vol.shape == (30, 64, 64) and info["header_has_identifiers"] is True
        assert "PatientName" not in json.dumps(info) and "X1" not in json.dumps(info)
        blobs = [open(p, "rb").read() for p in paths]          # same thing from uploaded bytes
        vol2, _ = ct_io.load_dicom_series(blobs); assert np.allclose(vol, vol2)
        rgb = ct_io.to_rgb_slices(vol, "Thoracic", 16); assert len(rgb) == 16 and rgb[0].shape == (64, 64, 3) and rgb[0].dtype == np.uint8
        try: ct_io.load_dicom_series([b"not dicom"]); assert False
        except ValueError: pass
    finally:
        shutil.rmtree(d)

def test_decision_and_fusion():
    d = model_io.decide([0.1, 0.8, 0.05, 0.05]); assert d["top"] == "Cancer" and not d["needs_human_review"]
    assert model_io.decide([0.4, 0.3, 0.2, 0.1])["needs_human_review"]                          # low confidence
    assert model_io.decide([0.1, 0.1, 0.7, 0.1])["needs_human_review"]                          # indeterminate always reviewed
    f = model_io.fuse([1, 0, 0, 0], [0, 1, 0, 0], 0.25); assert np.allclose(f, [0.75, 0.25, 0, 0])

def test_no_model_means_no_prediction():
    tmp = tempfile.mkdtemp()
    try:
        p = model_io.load_package(tmp); assert p["ct_model"] is None and p["card"] is None
        json.dump({"name": "x", "version": "0"}, open(os.path.join(tmp, "model_card.json"), "w"))
        p = model_io.load_package(tmp); assert p["ct_model"] is None            # card but no checkpoint
    finally:
        shutil.rmtree(tmp)

def test_torch_path():
    if not model_io.torch_available():
        print("  (skipped: torch not installed)"); return
    import torch
    from torchvision import models
    tmp = tempfile.mkdtemp()
    try:
        m = models.resnet18(weights=None); m.fc = torch.nn.Linear(512, 4)
        torch.save(m.state_dict(), os.path.join(tmp, "ct_resnet18.pt"))
        json.dump({"name": "toy", "version": "0", "classes": model_io.CLASSES_DEFAULT, "image_size": 64}, open(os.path.join(tmp, "model_card.json"), "w"))
        pkg = model_io.load_package(tmp); assert pkg["ct_model"] is not None
        rgb = [np.random.default_rng(i).integers(0, 255, (64, 64, 3)).astype(np.uint8) for i in range(5)]
        p, per = model_io.predict_ct(pkg["ct_model"], rgb, 64); assert p.shape == (4,) and abs(p.sum() - 1) < 1e-5 and per.shape == (5, 4)
        from PIL import Image
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(model_io.IMAGENET_MEAN, model_io.IMAGENET_STD)])
        x = tf(Image.fromarray(rgb[0]))
        cam, c = explain.gradcam(pkg["ct_model"], x); assert cam.ndim == 2 and 0 <= cam.min() and cam.max() <= 1.0001
        assert explain.overlay(rgb[0], cam, 64).shape == (64, 64, 3)
        rho = explain.randomization_check(pkg["ct_model"], x); assert -1 <= rho <= 1
    finally:
        shutil.rmtree(tmp)

def test_text_contributions():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    X = ["pulmonary tb", "post tb fibrosis", "ovarian cancer", "colon cancer", "hepatic mass", "pulmonary mass"] * 3
    y = [0, 0, 1, 1, 2, 2] * 3
    pipe = Pipeline([("tfidf", FeatureUnion([("w", TfidfVectorizer()), ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)))])),
                     ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    words = [w for w, _ in explain.text_contributions(pipe, "pulmonary tb", 0)]
    assert "tb" in words
    p = model_io.text_probs(pipe, "ovarian cancer"); assert p.shape == (4,) and p.argmax() == 1 and p[3] == 0

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
    print("ALL APP TESTS PASSED")
