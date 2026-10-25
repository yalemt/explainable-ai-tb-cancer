"""Narrative rules: the same code as Part A of the Colab notebook (kept in sync by copy).
Turns a short clinical text into a tidy version, a rule-based label with a certainty tier,
an organ system (from the examination type) and leakage-masked versions of the text."""
import re

SUSPECTED_AS_INDETERMINATE = True
MERGE_ABD_PELVIS = True

TEXT_FIXES = [
    (r"\br\s*/\s*o\b|\br\.o\b|\bro\b", "rule out"),
    (r"\bca\b", "cancer"), (r"\bhcc\b", "hepatocellular carcinoma"),
    (r"\bhx\b", "history"), (r"\bgb\b", "gallbladder"), (r"\bgi\b", "gastrointestinal"),
    (r"\brt\b", "right"), (r"\blt\b", "left"), (r"\bpul\b", "pulmonary"), (r"\bsec\b", "secondary"),
    (r"\btumer\b|\btumour\b", "tumor"),
    (r"\bmetstasis\b|\bmetastasis\b|\bmetastatic\b", "metastasis"),
    (r"\besphagal\b|\beshophagial\b|\besophagial\b|\besphageal\b", "esophageal"),
    (r"\blarengeal\b|\blarngal\b|\blaryngeal\b", "laryngeal"),
    (r"\btyroid\b|\bthyriod\b", "thyroid"),
    (r"\btraphasic\b|\btriphsic\b|\btriphassic\b|\btph\b", "triphasic"),
    (r"\bplural\b", "pleural"), (r"\bsqumous\b", "squamous"),
    (r"\bnasoparngeal\b", "nasopharyngeal"), (r"\breterostrnal\b|\bretrostrnal\b", "retrosternal"),
    (r"\bglobus\b", "glomus"), (r"\bblader\b", "bladder"), (r"\bappendicial\b", "appendiceal"),
    (r"\bsondylitis\b|\bspondylosis\b", "spondylitis"), (r"\bcorpulmonale\b", "cor pulmonale"),
    (r"\bcarotted\b", "carotid"), (r"\badx\b", "adnexal"), (r"\bobtruction\b", "obstruction"),
    (r"\bmasess\b|\bmassess\b", "masses"), (r"\bad\b", "adrenal"),
]
EXAM_FIXES = [
    (r"\ba\s*/\s*p\b|\babd\s*/\s*p\b|\babdp\b|\bap\b|\babdomino\s*pelvic\b|\babdominopelvic\b", "abdomen pelvis"),
    (r"\babdo\b|\babd\b|\babdominal\b", "abdomen"),
    (r"\bctu\b", "ct urography"), (r"\bpns\b", "paranasal sinus"), (r"\btl\b", "thoracolumbar"),
    (r"\bcy\b", "ct"),
    (r"\btraphasic\b|\btriphsic\b|\btriphassic\b|\btph\b", "triphasic"),
]

def normalize(s, fixes):
    s = str(s).lower()
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    for pat, rep in fixes:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9? ]", " ", s)           # keep '?' (uncertainty marker)
    return re.sub(r"\s+", " ", s).strip()


TB_PAT    = r"\btb\b|tubercul|\bpott"
CA_PAT    = r"cancer|carcinom|malignan|metasta|\btumor|neoplas|lymphoma|sarcoma|hepatocellular"
UNC_PAT   = r"rule out|suspect|probable|possible|\?|query"
MASS_PAT  = r"\bmass(es)?\b|\blesion|\bnodule|\bcyst"
SEQ_PAT   = r"post tb|tb fibrosis|fibrosis|history of tb|bronchiectasis|sequela|old tb|healed"

def label_text(t):
    has_tb, has_ca = bool(re.search(TB_PAT, t)), bool(re.search(CA_PAT, t))
    has_unc, has_mass = bool(re.search(UNC_PAT, t)), bool(re.search(MASS_PAT, t))
    review, suspected_for = False, ""
    if has_tb and has_ca:
        cat, conf, review = "Indeterminate", "uncertain", True
    elif has_tb or has_ca:
        disease = "TB" if has_tb else "Cancer"
        if has_unc and SUSPECTED_AS_INDETERMINATE:
            cat, conf, suspected_for = "Indeterminate", "uncertain", disease
        else:
            cat, conf = disease, ("suspected" if has_unc else "confirmed")
    elif has_unc or has_mass:
        cat, conf = "Indeterminate", "uncertain"
    else:
        cat, conf = "Other", "confirmed"
    stage = ""
    if cat == "TB" or suspected_for == "TB":
        stage = "post-TB sequelae" if re.search(SEQ_PAT, t) else "active/unspecified TB"
    return {"category": cat, "confidence": conf, "needs_review": review,
            "suspected_for": suspected_for, "tb_stage": stage}


LABEL_MAP = {"TB": 0, "Cancer": 1, "Indeterminate": 2, "Other": 3}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

ORGAN_PATTERNS = {
    "Thoracic":        r"chest|hrct|thora|lung|mediastin|pulmon",
    "Abdomen/Pelvis":  r"abdomen|pelvi|triphasic|liver|hepat|urograph|gallbladder|colon|kidney|renal",
    "Head and neck":   r"neck|brain|head|paranasal|sinus|thyroid|laryn",
    "Spine/MSK":       r"spine|thoracolumbar|spondyl|vertebr",
}
def organ_from_exam(e):
    hits = [k for k, p in ORGAN_PATTERNS.items() if re.search(p, e)]
    if len(hits) == 1: return hits[0]
    if len(hits) > 1:  return "Spine/MSK" if "Spine/MSK" in hits else "Multi-region"
    return "Other"

def organ_from_exam(e):
    hits = [k for k, p in ORGAN_PATTERNS.items() if re.search(p, e)]
    if len(hits) == 1: return hits[0]
    if len(hits) > 1:  return "Spine/MSK" if "Spine/MSK" in hits else "Multi-region"
    return "Other"

LEAK_PAT = (r"\btb\b|tubercul\w*|cancer\w*|carcinom\w*|malignan\w*|metasta\w*|\btumou?rs?\b|neoplas\w*|"
            r"lymphoma|sarcoma|adenocarcinoma|hepatocellular")
SITE_PAT = (r"lung\w*|pulmon\w*|chest|thora\w*|mediastin\w*|pleur\w*|liver|hepat\w*|gallbladder|ovar\w*|adnex\w*|uter\w*|"
            r"cervi\w*|renal|kidney|bladder|colon\w*|rect\w*|esophag\w*|breast|thyroid|neck|laryn\w*|naso\w*|brain|"
            r"spin\w*|pancrea\w*|adrenal|appendic\w*|abdom\w*|pelvi\w*|splen\w*|gastro\w*|sinus|nasal")

def mask_terms(t, pat):
    return re.sub(r"\s+", " ", re.sub(pat, " [MASK] ", t)).strip()


def analyze_narrative(exam_type: str, clinical_text: str) -> dict:
    """Everything the app shows about one narrative. Pure rules, no model, no data leaves the machine."""
    text_norm = normalize(clinical_text, TEXT_FIXES)
    exam_norm = normalize(exam_type, EXAM_FIXES)
    lab = label_text(text_norm)
    masked = mask_terms(text_norm, LEAK_PAT)
    return {"text_norm": text_norm, "exam_norm": exam_norm, "organ_system": organ_from_exam(exam_norm),
            "category": lab["category"], "confidence": lab["confidence"], "suspected_for": lab["suspected_for"],
            "tb_stage": lab["tb_stage"], "needs_review": lab["needs_review"],
            "text_masked": masked, "text_masked_site": mask_terms(masked, SITE_PAT)}
