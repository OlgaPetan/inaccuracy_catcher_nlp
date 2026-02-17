import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


# =========================
# AMENITIES
# =========================

HIGH_INTENT_AMENITIES = {
    "Comfort & Wellness": [
        "Hot tub", "Sauna", "Gym", "Exercise equipment",
        "Fireplace", "Indoor fireplace", "Fire pit",
    ],
    "Convenience & Functionality": [
        "Dedicated workspace", "EV charger",
        "Free parking on premises", "Free street parking",
        "Paid parking on premises", "Paid parking off premises",
    ],
    "Leisure & Outdoor": [
        "Pool", "Beach access", "Lake access", "Waterfront",
        "Patio or balcony", "BBQ grill", "Outdoor kitchen",
        "Outdoor dining area", "Resort access", "Ski-in/Ski-out",
    ],
    "Entertainment & Experience": [
        "Pool table", "Ping pong table", "Arcade games", "Game console",
        "Board games", "Movie theater", "Theme room", "Mini golf",
        "Climbing wall", "Bowling alley", "Laser tag", "Hockey rink",
        "Boat slip", "Skate ramp", "Life size games", "Bikes", "Kayak",
    ],
    "Scenic Views": [
        "Ocean view", "Sea view", "Beach view", "Lake view", "River view",
        "Bay view", "Mountain view", "Valley view", "City skyline view",
        "City view", "Garden view", "Pool view", "Park view", "Courtyard view",
        "Resort view", "Vineyard view", "Desert view", "Water view",
    ],
}

LOW_PRIORITY_AMENITIES = {
    "Secondary/basic": [
        "Room-darkening shades", "Body soap", "Shampoo", "Conditioner", "Shower gel",
        "Dishes and silverware", "Baking sheet", "Barbecue utensils", "Blender",
        "Bread maker", "Coffee", "Cooking basics", "Dining table", "Wine glasses",
        "Trash compactor", "Ethernet connection", "Pocket wifi", "Outlet covers",
        "Table corner guards", "Window guards", "Bidet", "Bathtub",
        "Single level home", "Cleaning available during stay", "Kitchenette",
        "Laundromat nearby", "Carbon monoxide alarm", "Smoke alarm",
        "Fire extinguisher", "First aid kit", "Ceiling fan", "Portable fans",
        "Extra pillows and blankets", "Hangers", "Mosquito net", "Bed linens",
        "Drying rack for clothing", "Clothing storage", "Cleaning products",
        "Air conditioning", "Dryer", "Essentials", "Heating", "Hot water",
        "Kitchen", "TV", "Washer", "Wifi", "Oven", "Microwave", "Stove",
        "Refrigerator", "Freezer", "Mini fridge", "Rice maker", "Toaster",
        "Dishwasher", "Coffee maker", "Private entrance", "Luggage dropoff allowed",
        "Long term stays allowed", "Hair dryer", "Iron", "Safe", "Crib",
        "High chair", "Children’s books and toys", "Baby bath", "Baby monitor",
        "Baby safety gates", "Changing table", "Pack ’n play / Travel crib",
        "Babysitter recommendations", "Children’s dinnerware",
    ]
}


# =========================
# UTILITIES
# =========================

def safe_df(df: pd.DataFrame):
    """Robust Streamlit display handles conversion failures."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True)
    except TypeError:
        try:
            st.dataframe(df)
        except Exception:
            st.dataframe(df.astype(str))
    except Exception:
        st.dataframe(df.astype(str))

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_key(k: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", (k or "").lower()).strip("_")

def is_review_key(k: str) -> bool:
    kn = normalize_key(k)
    return ("review" in kn) or ("reviews" in kn)

def is_indexed_path(k: str) -> bool:
    return bool(re.search(r"(^|\.)\d+(\.|$)", k))

def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )

def highlight_html(text: str, needle: str, max_chars: int = 380) -> str:
    t = text or ""
    n = (needle or "").strip()
    if not t:
        return "<em>(empty)</em>"
    if not n:
        return escape_html(t[:max_chars])

    tl = t.lower()
    nl = n.lower()

    idx = tl.find(nl)
    if idx < 0 and len(nl) >= 25:
        idx = tl.find(nl[:25])
    if idx < 0:
        return escape_html(t[:max_chars])

    start = max(0, idx - max_chars // 3)
    end = min(len(t), idx + len(n) + (max_chars // 3) * 2)
    snippet = t[start:end]

    local_idx = idx - start
    local_end = min(len(snippet), local_idx + len(n))

    pre = escape_html(snippet[:local_idx])
    mid = escape_html(snippet[local_idx:local_end])
    post = escape_html(snippet[local_end:])

    return f"{pre}<mark>{mid}</mark>{post}"

def flatten_json(data: Any, parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def rec(obj: Any, path: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                nk = f"{path}{sep}{k}" if path else str(k)
                rec(v, nk)
        elif isinstance(obj, list):
            out[path] = obj
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    rec(v, f"{path}{sep}{i}")
        else:
            out[path] = obj

    rec(data, parent)
    return out

def coerce_amenities(v):
    out = []

    # Your exact structure: dict of categories -> list[str]
    if isinstance(v, dict):
        for _, items in v.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.strip():
                        out.append(item.strip())

    # Also allow list[str] just in case some JSONs differ
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())

    # de-dupe, preserve order
    seen = set()
    deduped = []
    for a in out:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    return deduped


def detect_title_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if isinstance(v, str):
            kn = normalize_key(k)
            if kn in {"title", "listing_title", "name"} or kn.endswith("_title"):
                return k
    cands = []
    for k, v in flat.items():
        if isinstance(v, str):
            s = v.strip()
            if 10 <= len(s) <= 140 and "\n" not in s:
                cands.append((len(s), k))
    return sorted(cands)[0][1] if cands else None

def detect_amenities_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if normalize_key(k) == "amenities" and isinstance(v, list) and all(isinstance(x, str) for x in v):
            return k
    for k, v in flat.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v) and "amenit" in normalize_key(k):
            return k
    return None

def detect_house_rules_key(flat: Dict[str, Any]) -> Optional[str]:
    for k, v in flat.items():
        if isinstance(v, str) and "house_rules" in normalize_key(k) and len(v.strip()) >= 30:
            return k
    return None

def detect_text_keys(flat: Dict[str, Any], min_len: int = 120) -> List[str]:
    keys = []
    for k, v in flat.items():
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= min_len or ("\n" in s) or (len(re.split(r"(?<=[.!?])\s+", s)) >= 2 and len(s) >= 80):
                keys.append(k)
    return sorted(keys)

PREFERRED_TEXT_NORMAL_KEYS = {
    "summary", "the_space", "guest_access", "other_things_to_note", "description"
}

def choose_editable_text_keys(flat: Dict[str, Any], title_key: Optional[str], house_rules_key: Optional[str]) -> List[str]:
    preferred = []
    for k, v in flat.items():
        if isinstance(v, str) and normalize_key(k) in PREFERRED_TEXT_NORMAL_KEYS:
            preferred.append(k)

    detected = detect_text_keys(flat)
    out = []
    for k in preferred + detected:
        if k == title_key:
            continue
        if k == house_rules_key:
            continue
        if is_review_key(k):
            continue
        if k not in out:
            out.append(k)
    return out

def split_sentences(text: str) -> List[str]:
    t = normalize_ws(text)
    if not t:
        return []
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]

_NUM_WORDS = {
    "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20
}
def word_to_int(w: str) -> Optional[int]:
    return _NUM_WORDS.get(re.sub(r"[^a-z]", "", (w or "").lower()))

def parse_int_maybe(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"\d{1,3}", s):
            return int(s)
        if re.fullmatch(r"[A-Za-z]+", s):
            return word_to_int(s)
    return None

def extract_number_near(text: str, idx: int, window: int = 40) -> Optional[int]:
    if not text:
        return None
    start = max(0, idx - window)
    snippet = text[start:idx]
    m = re.search(r"(\d{1,3})\s*$", snippet)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    toks = re.findall(r"[A-Za-z']+", snippet.lower())
    if toks:
        return word_to_int(toks[-1])
    return None


# =========================
# REVIEWS (READ-ONLY DISPLAY + INCLUDED IN CHECKING)
# =========================

def build_readonly_reviews(flat: Dict[str, Any]) -> Dict[str, str]:
    readonly: Dict[str, str] = {}
    keys = []
    for k, v in flat.items():
        if not is_review_key(k):
            continue
        if is_indexed_path(k):
            continue
        if isinstance(v, (str, list)):
            keys.append(k)

    for k in sorted(set(keys)):
        v = flat.get(k)
        if isinstance(v, str):
            txt = v.strip()
            if txt:
                readonly[k] = txt[:12000]
        elif isinstance(v, list):
            parts: List[str] = []
            for item in v[:200]:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for _, vv in item.items():
                        if isinstance(vv, str) and vv.strip():
                            parts.append(vv.strip())
            if parts:
                readonly[k] = "\n\n".join(parts)[:12000]
    return readonly


# =========================
# AMENITY (DETERMINISTIC)
# =========================

AMENITY_SYNONYMS = {
    "hot tub": ["hot tub", "jacuzzi", "spa tub", "whirlpool"],
    "pool": ["pool", "swimming pool"],
    "bbq grill": ["bbq", "barbecue", "bbq grill", "grill"],
    "dedicated workspace": ["dedicated workspace", "workspace", "work desk", "desk"],
    "ev charger": ["ev charger", "electric vehicle charger", "tesla charger"],
    "game console": ["game console", "ps5", "playstation", "xbox", "nintendo switch"],
    "movie theater": ["movie theater", "home theater", "cinema room"],
    "fire pit": ["fire pit", "fire-pit"],
    "indoor fireplace": ["indoor fireplace", "fireplace"],
}

def canon_amenity(a: str) -> str:
    a0 = normalize_ws(a).lower()
    for canon, syns in AMENITY_SYNONYMS.items():
        if a0 == canon or a0 in [s.lower() for s in syns]:
            return canon
    return a0

def all_known_amenities() -> Tuple[Dict[str, str], set, set]:
    display = {}
    high = set()
    low = set()
    for _, items in HIGH_INTENT_AMENITIES.items():
        for it in items:
            c = canon_amenity(it)
            high.add(c)
            display.setdefault(c, it)
    for _, items in LOW_PRIORITY_AMENITIES.items():
        for it in items:
            c = canon_amenity(it)
            low.add(c)
            display.setdefault(c, it)
    return display, high, low

def find_amenity_hits(text: str, canon: str) -> List[Tuple[int, int, str]]:
    t = text or ""
    syns = AMENITY_SYNONYMS.get(canon, [canon])
    hits = []
    for s in syns:
        escaped = re.escape(s).replace("\\ ", r"[\s\-]+")
        pat = re.compile(r"(?i)\b" + escaped + r"\b")
        for m in pat.finditer(t):
            hits.append((m.start(), m.end(), t[m.start():m.end()]))
    hits.sort(key=lambda x: x[0])
    return hits


# =========================
# EMBEDDINGS MATCHER
# =========================

class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        av = self.encode([a])[0]
        bv = self.encode([b])[0]
        return float(av @ bv)

    def best_match(self, query: str, sentences: List[str]) -> Tuple[float, str]:
        if not sentences:
            return 0.0, ""
        qv = self.encode([query])[0]
        sv = self.encode(sentences)
        best_i = 0
        best = float(qv @ sv[0])
        for i in range(1, len(sentences)):
            sc = float(qv @ sv[i])
            if sc > best:
                best = sc
                best_i = i
        return best, sentences[best_i]


# =========================
# GROUND TRUTH MAPPING 
# =========================

CANON_DESC = {
    "max_guests": "Maximum number of guests allowed in the listing (guest capacity)",
    "bedrooms": "Number of bedrooms in the listing",
    "bathrooms": "Number of bathrooms in the listing",
    "property_type": "Property type / listing type (apartment, house, villa, cottage, etc.)",
    "pets_allowed": "Whether pets are allowed (true/false)",
    "extra_guest_fee": "Extra guest fee amount or additional guest fee configuration",
}

BED_KEYWORDS = {"king", "queen", "double", "full", "twin", "single", "sofa", "bunk", "murphy", "crib"}

def map_ground_truth(flat: Dict[str, Any], matcher: SemanticMatcher, exclude_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    candidates = []
    for k, v in flat.items():
        if k in exclude_keys:
            continue
        if isinstance(v, (dict, list)):
            continue
        if isinstance(v, str) and len(v) >= 160:
            continue
        rep = f"{k}: {v}"
        candidates.append((k, rep))

    gt: Dict[str, Dict[str, Any]] = {}
    for canon, desc in CANON_DESC.items():
        best = (0.0, None, None)
        for k, rep in candidates:
            sc = matcher.similarity(rep, desc)
            if sc > best[0]:
                best = (sc, k, rep)
        if best[1] is not None and best[0] >= 0.55:
            gt[canon] = {"key": best[1], "value": flat.get(best[1]), "confidence": float(best[0])}
        else:
            gt[canon] = {"key": None, "value": None, "confidence": 0.0}

    bed_dict_key = None
    bed_dict_val = None
    for k, v in flat.items():
        if k in exclude_keys:
            continue
        if isinstance(v, dict):
            keys = " ".join([normalize_key(str(x)) for x in v.keys()])
            if any(b in keys for b in BED_KEYWORDS):
                bed_dict_key = k
                bed_dict_val = v
                break
    gt["sleeping_arrangements"] = {"key": bed_dict_key, "value": bed_dict_val, "confidence": 0.7 if bed_dict_key else 0.0}

    return gt


# =========================
# TEXT CLAIM EXTRACTION
# =========================

PARKING_CONTEXT = re.compile(r"(?i)\b(parking|cars?|vehicles?)\b")
HOT_TUB_CONTEXT = re.compile(r"(?i)\b(hot tub|jacuzzi|spa)\b")

GUEST_PAT = re.compile(r"(?i)\b(sleeps|sleep|accommodates|accommodate|max(?:imum)?|up to|capacity)\b.{0,25}\b(\d{1,3}|[A-Za-z]+)\b")
BEDROOM_PAT = re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(?:bedroom|bedrooms)\b")
BATHROOM_PAT = re.compile(r"(?i)\b(\d{1,3}|[A-Za-z]+)\s+(?:bathroom|bathrooms|baths?)\b")
BED_PAT = re.compile(r"(?i)\b(\d{1,2}|[A-Za-z]+)\s+(king|queen|double|full|twin|single|sofa bed|bunk|murphy|crib)\s+beds?\b")

def extract_guest_capacity_mentions(text: str) -> List[Tuple[int, str]]:
    out = []
    t = text or ""
    for m in GUEST_PAT.finditer(t):
        raw = m.group(2)
        n = parse_int_maybe(raw)
        if n is None:
            continue
        span = t[m.start():m.end()]
        if PARKING_CONTEXT.search(span) or HOT_TUB_CONTEXT.search(span):
            continue
        out.append((n, span.strip()))
    return out

def extract_bedroom_mentions(text: str) -> List[Tuple[int, str]]:
    out = []
    t = text or ""
    for m in BEDROOM_PAT.finditer(t):
        n = parse_int_maybe(m.group(1))
        if n is not None:
            out.append((n, m.group(0).strip()))
    return out

def extract_bathroom_mentions(text: str) -> List[Tuple[int, str]]:
    out = []
    t = text or ""
    for m in BATHROOM_PAT.finditer(t):
        n = parse_int_maybe(m.group(1))
        if n is not None:
            out.append((n, m.group(0).strip()))
    return out

def extract_sleeping_arrangements(text: str) -> List[Tuple[str, int, str]]:
    out = []
    t = text or ""
    for m in BED_PAT.finditer(t):
        n = parse_int_maybe(m.group(1))
        if n is None:
            continue
        bed = m.group(2).lower()
        out.append((bed, n, m.group(0).strip()))
    return out

def detect_pet_mention(text: str) -> Optional[str]:
    t = text or ""
    if re.search(r"(?i)\b(pets? allowed|pet-friendly|pet friendly|bring your pet)\b", t):
        return "allowed"
    if re.search(r"(?i)\b(no pets|pets? not allowed|no animals)\b", t):
        return "not_allowed"
    return None

def detect_private_shared(text: str) -> Dict[str, Optional[str]]:
    t = text or ""
    out = {"pool": None, "hot tub": None}
    if re.search(r"(?i)\bprivate\s+pool\b", t):
        out["pool"] = "private"
    elif re.search(r"(?i)\bshared\s+pool\b", t):
        out["pool"] = "shared"
    if re.search(r"(?i)\bprivate\s+(hot tub|jacuzzi|spa)\b", t):
        out["hot tub"] = "private"
    elif re.search(r"(?i)\bshared\s+(hot tub|jacuzzi|spa)\b", t):
        out["hot tub"] = "shared"
    return out

PROPERTY_TYPES = {
    "apartment": ["apartment", "flat", "condo", "condominium", "studio"],
    "house": ["house", "home"],
    "villa": ["villa"],
    "cottage": ["cottage", "cabin", "cabins"],
}

def detect_property_types(text: str) -> List[str]:
    t = text or ""
    found = []
    for canon, syns in PROPERTY_TYPES.items():
        for s in syns:
            if re.search(rf"(?i)\b{s}\b", t):
                found.append(canon)
                break
    return sorted(set(found))

EXTRA_GUEST_FEE_PAT = re.compile(r"(?i)\b(extra guest|additional guest)\b|\bper guest\b|\bguest fee\b|\$\s*\d+")

def mentions_extra_guest_fee(text: str) -> bool:
    return bool(EXTRA_GUEST_FEE_PAT.search(text or ""))


# =========================
# CHECKS
# =========================

def run_checks(
    flat: Dict[str, Any],
    title: str,
    texts: Dict[str, str],
    amenities_selected: List[str],
    exclusive_keys: List[str],
    matcher: SemanticMatcher,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    exclude_keys = list(texts.keys())
    gt = map_ground_truth(flat, matcher, exclude_keys=exclude_keys)
    corpus = {"title": title, **texts}

    gt_max = parse_int_maybe(gt.get("max_guests", {}).get("value"))
    if gt_max is not None:
        for field, text in corpus.items():
            for n, span in extract_guest_capacity_mentions(text):
                if n != gt_max:
                    issues.append({
                        "issue_type": "Max capacity mismatch",
                        "severity": "high",
                        "field": field,
                        "claim": str(n),
                        "ground_truth": str(gt_max),
                        "evidence": span,
                        "reason": f"{field} suggests max guests is {n}, but structured max_guests is {gt_max}."
                    })

    gt_bed = parse_int_maybe(gt.get("bedrooms", {}).get("value"))
    if gt_bed is not None:
        for field, text in corpus.items():
            for n, span in extract_bedroom_mentions(text):
                if n != gt_bed:
                    issues.append({
                        "issue_type": "Room count mismatch",
                        "severity": "high",
                        "field": field,
                        "claim": str(n),
                        "ground_truth": str(gt_bed),
                        "evidence": span,
                        "reason": f"{field} implies {n} bedrooms, but structured bedrooms is {gt_bed}."
                    })

    gt_bath = parse_int_maybe(gt.get("bathrooms", {}).get("value"))
    if gt_bath is not None:
        for field, text in corpus.items():
            for n, span in extract_bathroom_mentions(text):
                if n != gt_bath:
                    issues.append({
                        "issue_type": "Bathroom count mismatch",
                        "severity": "high",
                        "field": field,
                        "claim": str(n),
                        "ground_truth": str(gt_bath),
                        "evidence": span,
                        "reason": f"{field} implies {n} bathrooms, but structured bathrooms is {gt_bath}."
                    })

    gt_beds = gt.get("sleeping_arrangements", {}).get("value")
    if isinstance(gt_beds, dict) and gt_beds:
        norm_gt = {str(k).lower(): parse_int_maybe(v) for k, v in gt_beds.items()}
        for field, text in corpus.items():
            for bed, n, span in extract_sleeping_arrangements(text):
                if bed in norm_gt and norm_gt[bed] is not None and n != norm_gt[bed]:
                    issues.append({
                        "issue_type": "Sleeping arrangement mismatch",
                        "severity": "high",
                        "field": field,
                        "claim": f"{n} {bed} beds",
                        "ground_truth": f"{norm_gt[bed]} {bed} beds",
                        "evidence": span,
                        "reason": f"{field} mentions {n} {bed} beds, but structured sleeping arrangement has {norm_gt[bed]}."
                    })

    amenity_labels = " ".join([a.lower() for a in amenities_selected])
    gt_pool = "shared" if "shared pool" in amenity_labels else ("private" if "private pool" in amenity_labels else None)
    gt_ht = "shared" if ("shared hot tub" in amenity_labels or "shared jacuzzi" in amenity_labels) else ("private" if ("private hot tub" in amenity_labels or "private jacuzzi" in amenity_labels) else None)

    for field, text in corpus.items():
        sh = detect_private_shared(text)
        if gt_pool and sh["pool"] and sh["pool"] != gt_pool:
            issues.append({
                "issue_type": "Shared vs private amenity mismatch",
                "severity": "high",
                "field": field,
                "claim": f"{sh['pool']} pool",
                "ground_truth": f"{gt_pool} pool",
                "evidence": "private pool" if sh["pool"] == "private" else "shared pool",
                "reason": f"{field} implies a {sh['pool']} pool, but amenities list indicates a {gt_pool} pool."
            })
        if gt_ht and sh["hot tub"] and sh["hot tub"] != gt_ht:
            issues.append({
                "issue_type": "Shared vs private amenity mismatch",
                "severity": "high",
                "field": field,
                "claim": f"{sh['hot tub']} hot tub",
                "ground_truth": f"{gt_ht} hot tub",
                "evidence": "private hot tub" if sh["hot tub"] == "private" else "shared hot tub",
                "reason": f"{field} implies a {sh['hot tub']} hot tub, but amenities list indicates a {gt_ht} hot tub."
            })

    display, high_set, low_set = all_known_amenities()
    selected = {canon_amenity(a) for a in amenities_selected}

    for canon in sorted(high_set | low_set):
        mentioned = False
        max_count = None
        max_field = None
        max_evidence = None

        for field, text in corpus.items():
            hits = find_amenity_hits(text, canon)
            if hits:
                mentioned = True
                for s, e, _ in hits:
                    n = extract_number_near(text, s)
                    if n is not None and n >= 2:
                        if max_count is None or n > max_count:
                            max_count = n
                            max_field = field
                            max_evidence = (text[max(0, s-35):min(len(text), e+35)]).strip()

        if mentioned and canon not in selected:
            sev = "high" if canon in high_set else "low"
            issues.append({
                "issue_type": "Amenity mentioned but not selected",
                "severity": sev,
                "field": "title" if find_amenity_hits(title, canon) else "text",
                "claim": display.get(canon, canon),
                "ground_truth": "Not in Amenities list",
                "evidence": display.get(canon, canon),
                "reason": f"Text mentions '{display.get(canon, canon)}' but it is not present in the Amenities ground truth."
            })

        if max_count is not None and canon in selected:
            issues.append({
                "issue_type": "Amenity count mismatch",
                "severity": "medium" if canon in high_set else "low",
                "field": max_field or "text",
                "claim": f"{max_count}× {display.get(canon, canon)}",
                "ground_truth": f"Amenities includes '{display.get(canon, canon)}' (single selection)",
                "evidence": max_evidence or display.get(canon, canon),
                "reason": f"{max_field or 'Text'} suggests {max_count} {display.get(canon, canon)}(s), but the amenities ground truth only indicates the amenity is selected (no multiple units)."
            })

    all_text = " ".join([title] + list(texts.values())).lower()
    for canon in sorted(selected):
        if not find_amenity_hits(all_text, canon):
            issues.append({
                "issue_type": "Amenity selected but not mentioned",
                "severity": "low",
                "field": "amenities",
                "claim": display.get(canon, canon),
                "ground_truth": "Selected in amenities",
                "evidence": display.get(canon, canon),
                "reason": f"Amenities includes '{display.get(canon, canon)}' but it is not mentioned in the title/text fields."
            })

    high_in_title = []
    for canon in sorted(high_set):
        if find_amenity_hits(title, canon):
            high_in_title.append(display.get(canon, canon))
    if len(high_in_title) > 3:
        issues.append({
            "issue_type": "Title amenity stuffing",
            "severity": "medium",
            "field": "title",
            "claim": f"{len(high_in_title)} high-priority amenities",
            "ground_truth": "<= 3 high-priority amenities in title",
            "evidence": ", ".join(high_in_title[:6]),
            "reason": f"Title includes {len(high_in_title)} high-priority amenities. Flag titles that contain more than 3."
        })

    pets_val = gt.get("pets_allowed", {}).get("value")
    pets_bool = None
    if isinstance(pets_val, bool):
        pets_bool = pets_val
    elif isinstance(pets_val, str):
        t = pets_val.strip().lower()
        if t in {"true", "yes", "1"}:
            pets_bool = True
        elif t in {"false", "no", "0"}:
            pets_bool = False

    if pets_bool is not None:
        for field, text in corpus.items():
            mention = detect_pet_mention(text)
            if mention == "allowed" and pets_bool is False:
                issues.append({
                    "issue_type": "Pet friendliness conflict",
                    "severity": "high",
                    "field": field,
                    "claim": "Pets allowed",
                    "ground_truth": "Pets not allowed",
                    "evidence": "pet friendly / pets allowed",
                    "reason": f"{field} implies pets are allowed, but ground truth indicates pets are not allowed."
                })
            if mention == "not_allowed" and pets_bool is True:
                issues.append({
                    "issue_type": "Pet friendliness conflict",
                    "severity": "high",
                    "field": field,
                    "claim": "Pets not allowed",
                    "ground_truth": "Pets allowed",
                    "evidence": "no pets / pets not allowed",
                    "reason": f"{field} implies pets are not allowed, but ground truth indicates pets are allowed."
                })

    gt_pt = gt.get("property_type", {}).get("value")
    gt_pt_norm = str(gt_pt).strip().lower() if gt_pt is not None else ""
    if gt_pt_norm:
        for field, text in corpus.items():
            mentions = detect_property_types(text)
            if mentions and gt_pt_norm not in mentions:
                issues.append({
                    "issue_type": "Property type inconsistency",
                    "severity": "medium",
                    "field": field,
                    "claim": ", ".join(mentions),
                    "ground_truth": gt_pt_norm,
                    "evidence": mentions[0],
                    "reason": f"{field} describes the property as {', '.join(mentions)}, but structured property type is '{gt_pt_norm}'."
                })

    fee_val = gt.get("extra_guest_fee", {}).get("value")
    fee_num = parse_int_maybe(fee_val)
    for field, text in corpus.items():
        if mentions_extra_guest_fee(text) and (fee_num is None or fee_num == 0):
            issues.append({
                "issue_type": "Extra guest fee inconsistency",
                "severity": "medium",
                "field": field,
                "claim": "Extra guest fee mentioned",
                "ground_truth": "No extra guest fee configured in structured fields",
                "evidence": "extra guest / additional guest / $",
                "reason": f"{field} mentions an extra/additional guest fee, but the structured fields do not show a configured extra_guest_fee."
            })

    # Exclusive leakage (semantic)
    for ex_key in exclusive_keys:
        ex_text = str(flat.get(ex_key, "") or "")
        ex_sents = split_sentences(ex_text)[:12]
        if not ex_sents:
            continue

        for field, text in corpus.items():
            if field == ex_key:
                continue
            sents = split_sentences(text)
            if not sents:
                continue

            for rule in ex_sents:
                if len(rule) < 18:
                    continue

                if rule.lower() in text.lower():
                    issues.append({
                        "issue_type": "Exclusive text leakage",
                        "severity": "medium",
                        "field": field,
                        "claim": rule[:140],
                        "ground_truth": f"Should only appear in {ex_key}",
                        "evidence": rule[:140],
                        "reason": f"A sentence from {ex_key} appears in {field}. {ex_key} content should not be repeated in other fields."
                    })
                    continue

                sc, best = matcher.best_match(rule, sents)
                if sc >= 0.86:
                    issues.append({
                        "issue_type": "Exclusive text leakage",
                        "severity": "medium",
                        "field": field,
                        "claim": rule[:140],
                        "ground_truth": f"Should only appear in {ex_key}",
                        "evidence": best[:140],
                        "reason": f"Content from {ex_key} appears (semantically) in {field}. Similarity={sc:.2f}. {ex_key} content should not be repeated in other fields."
                    })

    merged = []
    seen = set()
    for it in issues:
        key = (
            it.get("issue_type", ""),
            it.get("field", ""),
            (it.get("evidence", "") or "")[:80],
            (it.get("claim", "") or "")[:80],
            (it.get("ground_truth", "") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)

    return merged, gt


# =========================
# STREAMLIT UI
# =========================

def load_json_upload(upload) -> Any:
    raw = upload.read()
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))

def issues_to_df(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["severity", "issue_type", "field", "reason", "claim", "ground_truth", "evidence"]
    if not issues:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(issues)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def sev_rank(s: str) -> int:
    s = (s or "").lower()
    return {"high": 0, "medium": 1, "low": 2}.get(s, 3)

def render_issue_cards(issues: List[Dict[str, Any]], title: str, texts: Dict[str, str]):
    if not issues:
        st.success("No issues found (with current settings).")
        return

    combined_all = title + "\n\n" + "\n\n".join([v for v in texts.values() if v])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for it in issues:
        grouped.setdefault(it.get("field", "text") or "text", []).append(it)

    for field, items in grouped.items():
        st.subheader(f"Field: {field} ({len(items)})")

        if field == "title":
            base = title
        elif field == "amenities":
            base = combined_all
        else:
            base = texts.get(field, "")
            if not base:
                base = combined_all

        for it in sorted(items, key=lambda x: (sev_rank(x.get("severity")), x.get("issue_type", ""))):
            sev = (it.get("severity") or "low").lower()
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(sev, "⚪")
            with st.expander(f"{icon} {it.get('issue_type','Issue')} — {str(it.get('reason',''))[:90]}"):
                st.markdown(f"**Reason:** {it.get('reason','')}")
                if it.get("ground_truth"):
                    st.markdown(f"**Ground truth:** `{it.get('ground_truth')}`")
                if it.get("claim"):
                    st.markdown(f"**Claim:** `{it.get('claim')}`")
                ev = it.get("evidence") or ""
                st.markdown("**Evidence (highlighted):**")
                st.markdown(
                    f"<div style='padding:10px;border:1px solid #ddd;border-radius:10px'>{highlight_html(base, ev)}</div>",
                    unsafe_allow_html=True,
                )

def as_display_value(v: Any) -> str:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return "" if v is None else str(v)

def main():
    st.set_page_config(page_title="JSON Discrepancy Checker — Embeddings", layout="wide")
    st.title("JSON Discrepancy Checker — Local embeddings semantic")

    st.sidebar.header("Upload JSON")
    upload = st.sidebar.file_uploader("JSON file", type=["json"])
    use_sample = st.sidebar.checkbox("Use sample JSON", value=False)

    run_live = st.sidebar.checkbox("Re-check automatically while editing", value=True)

    st.sidebar.header("Embeddings model")
    model_name = st.sidebar.text_input("sentence-transformers model", value="all-MiniLM-L6-v2")

    if use_sample and not upload:
        data = {
            "title": "Cozy villa with 2 hot tubs, private pool, sauna, and EV charger — sleeps 4",
            "max_guests": 6,
            "bedrooms": 2,
            "bathrooms": 1,
            "property_type": "apartment",
            "Amenities": ["Hot tub", "Shared pool", "Wifi"],
            "house_rules": "No pets. No parties. Quiet hours after 10pm.",
            "description": "Sleeps 4. Two hot tubs and a private pool. Pet friendly! Perfect villa getaway.",
            "summary": "A stylish apartment with shared pool. Extra guest fee applies.",
            "reviews": [{"text": "Great place!"}, {"text": "Loved the pool."}]
        }
    elif upload:
        try:
            data = load_json_upload(upload)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            return
    else:
        st.info("Upload a JSON file (or enable sample JSON).")
        return

    if isinstance(data, list):
        st.warning("JSON root is a list. Using first item.")
        data = data[0] if data else {}

    if not isinstance(data, dict):
        st.error("JSON root must be an object (dict).")
        return

    flat = flatten_json(data)

    # Auto-detect (no Field Selection UI)
    title_key = detect_title_key(flat)
    amenities_key = detect_amenities_key(flat)
    house_rules_key = detect_house_rules_key(flat)
    exclusive_keys = [house_rules_key] if house_rules_key else []
    text_keys = choose_editable_text_keys(flat, title_key=title_key, house_rules_key=house_rules_key)
    readonly_reviews = build_readonly_reviews(flat)

    title_val = str(flat.get(title_key, "")) if title_key else ""
    amenities_raw = data.get("amenities")  # your JSON is nested, don't use flat for this
    amenities_val = coerce_amenities(amenities_raw)


    st.header("Editable listing text")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader("Title")
        title_edit = st.text_input("Title", value=title_val, key="__title")

        st.subheader("Amenities (ground truth)")
        st.write(amenities_val if amenities_val else "—")

    with c2:
        edited_texts: Dict[str, str] = {}
        st.subheader("Editable text fields")
        if not text_keys:
            st.info("No editable text fields detected in this JSON.")
        for k in text_keys:
            edited_texts[k] = st.text_area(k, value=str(flat.get(k, "")), height=160, key=f"__txt_{k}")

        if readonly_reviews:
            st.subheader("Read-only review fields (not editable)")
            for k, txt in readonly_reviews.items():
                try:
                    st.text_area(k, value=txt, height=160, key=f"__ro_{k}", disabled=True)
                except TypeError:
                    st.markdown(f"**{k} (read-only)**")
                    st.code(txt[:4000])

    all_texts_for_checking = {**edited_texts, **readonly_reviews}

    def run_once():
        matcher = SemanticMatcher(model_name=model_name)
        issues, gt = run_checks(
            flat=flat,
            title=title_edit,
            texts=all_texts_for_checking,
            amenities_selected=amenities_val,
            exclusive_keys=exclusive_keys,
            matcher=matcher,
        )
        st.session_state.setdefault("runs", [])
        st.session_state["runs"].append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "issues": issues,
            "gt": gt,
            "title": title_edit,
            "texts": all_texts_for_checking,
            "model": model_name,
        })
        st.session_state["current"] = st.session_state["runs"][-1]

    if run_live:
        run_once()
    else:
        if st.button("Run checker"):
            run_once()

    current = st.session_state.get("current")
    if not current:
        return

    issues = current["issues"]
    gt = current["gt"]

    st.header("Results")

    counts = {"high": 0, "medium": 0, "low": 0}
    for it in issues:
        counts[(it.get("severity") or "low").lower()] = counts.get((it.get("severity") or "low").lower(), 0) + 1
    m1, m2, m3 = st.columns(3)
    m1.metric("High", counts.get("high", 0))
    m2.metric("Medium", counts.get("medium", 0))
    m3.metric("Low", counts.get("low", 0))

    st.subheader("Canonical field mapping (embeddings best-effort)")
    rows = []
    for canon, d in gt.items():
        rows.append({
            "canonical": str(canon),
            "json_key": as_display_value(d.get("key")),
            "value": as_display_value(d.get("value")),
            "confidence": str(round(float(d.get("confidence", 0.0)), 3)),
        })
    safe_df(pd.DataFrame(rows))

    df = issues_to_df(issues).copy()
    df = df.sort_values(by=["severity", "issue_type"], key=lambda s: s.map(sev_rank), ascending=True)
    st.subheader("Issues table")
    safe_df(df)

    st.download_button("Download issues (CSV)", df.to_csv(index=False).encode("utf-8"), "issues.csv", "text/csv")
    st.download_button("Download issues (JSON)", json.dumps(issues, ensure_ascii=False, indent=2).encode("utf-8"), "issues.json", "application/json")

    st.subheader("Issue details (highlights + reasons)")
    render_issue_cards(issues, title_edit, all_texts_for_checking)

    st.sidebar.header("Runs")
    st.sidebar.write(f"{len(st.session_state.get('runs', []))} run(s) this session")
    if st.sidebar.button("Clear runs"):
        st.session_state["runs"] = []
        st.session_state["current"] = None
        st.rerun()


if __name__ == "__main__":
    main()
