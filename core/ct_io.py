"""Reading CT data safely: DICOM series or ordinary image slices -> windowed 8-bit slices.
Identifiers in DICOM headers are never returned for display."""
import io
import numpy as np

WINDOWS = {"Thoracic": (-600, 1500), "Abdomen/Pelvis": (40, 400), "Head and neck": (40, 400),
           "Spine/MSK": (400, 1800), "Multi-region": (40, 400), "Other": (40, 400)}

def window_to_uint8(img, center, width):
    lo, hi = center - width / 2.0, center + width / 2.0
    return (np.clip((np.asarray(img, dtype=np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

def pick_slice_indices(n, k):
    """k evenly spaced slices, ignoring the first and last 10 % of the scan."""
    trim = int(0.10 * n)
    return np.unique(np.linspace(trim, max(trim, n - 1 - trim), k).round().astype(int))

def _bytes(f):
    return f if isinstance(f, (bytes, bytearray)) else (f.getvalue() if hasattr(f, "getvalue") else f.read())

def load_dicom_series(files):
    """files: list of paths, bytes or uploaded-file objects. Returns (volume_in_HU, info) for the longest CT series.
    info never contains patient identifiers."""
    import pydicom
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f if isinstance(f, str) else io.BytesIO(_bytes(f)), force=True)
        except Exception:
            continue
        if hasattr(ds, "PixelData") and str(getattr(ds, "Modality", "")) == "CT" and hasattr(ds, "SeriesInstanceUID"):
            datasets.append(ds)
    if not datasets:
        raise ValueError("No readable CT DICOM slices found in the upload.")
    by_series = {}
    for ds in datasets:
        by_series.setdefault(str(ds.SeriesInstanceUID), []).append(ds)
    series = max(by_series.values(), key=len)
    def zkey(d):
        try:    return float(d.ImagePositionPatient[2])
        except Exception: return float(getattr(d, "InstanceNumber", 0))
    series.sort(key=zkey)
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0)) for d in series])
    carries_ids = any(str(getattr(series[0], t, "")).strip() for t in ("PatientName", "PatientBirthDate"))
    info = {"n_slices": len(series), "n_series_in_upload": len(by_series), "rows": int(vol.shape[1]), "cols": int(vol.shape[2]),
            "header_has_identifiers": bool(carries_ids), "hu_min": float(vol.min()), "hu_max": float(vol.max())}
    return vol, info

def load_image_slices(files):
    """PNG/JPG slices that are already windowed. Returns a list of uint8 grayscale arrays."""
    from PIL import Image
    out = []
    for f in files:
        img = Image.open(f if isinstance(f, str) else io.BytesIO(_bytes(f))).convert("L")
        out.append(np.array(img))
    if not out:
        raise ValueError("No readable image slices found.")
    return out

def to_rgb_slices(volume_or_slices, organ_system="Thoracic", k=16, already_windowed=False):
    """-> list of H x W x 3 uint8 arrays ready for the model."""
    if already_windowed:
        sl = list(volume_or_slices)
        idx = pick_slice_indices(len(sl), min(k, len(sl))) if len(sl) > k else range(len(sl))
        gray = [sl[i] for i in idx]
    else:
        c, w = WINDOWS.get(organ_system, WINDOWS["Other"])
        vol = volume_or_slices
        gray = [window_to_uint8(vol[i], c, w) for i in pick_slice_indices(len(vol), k)]
    return [np.stack([g, g, g], axis=-1) for g in gray]
