"""Deterministic triage for the patient consult: answer parsing, severity level, specialist.

Pure functions, no model and no network: the *safety-critical* decisions (is this an emergency, how soon should the patient
be seen, which kind of doctor) never depend on a small language model's judgement. The model only writes the explanation.
"""
import re
from dataclasses import dataclass, field

from . import safety

LEVELS = ("self_care", "see_doctor", "urgent", "emergency")
LEVEL_INFO = {
    "self_care":  {"label": "Minor - self-care is reasonable", "window": "See a doctor if it lasts beyond a few days or gets worse"},
    "see_doctor": {"label": "See a doctor soon",               "window": "Book an appointment within the next 2-3 days"},
    "urgent":     {"label": "See a doctor today",              "window": "Visit a doctor or urgent-care clinic within 24 hours"},
    "emergency":  {"label": "Emergency",                       "window": "Call your local emergency number (112 / 911 / 108) now"},
}

# ---------- answer parsing ----------
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "few": 3, "couple": 2, "several": 4}
_UNIT_DAYS = {"hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
_DUR = re.compile(r"\b(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple(?: of)?|several)\s*"
                  r"(hour|day|week|month|year)s?\b", re.I)


_AGE_PHRASE = re.compile(r"\b\d{1,3}\s*(?:years?|yrs?|y\.?o\.?)(?:[\s-]*old)\b|\b(?:age|aged)\s*(?:is\s*)?\d{1,3}\b|\b\d{1,3}\s*(?:years?|yrs?)\s*(?:of age)\b|(?<!for )(?<!past )(?<!last )(?<!over )(?<!about )(?<!around )(?<!nearly )(?<!since )\b(?:1[3-9]|[2-9]\d|1[01]\d)\s*(?:years?|yrs?)\b(?!\s*(?:ago|back|now))", re.I)


def parse_duration_days(text: str) -> float | None:
    """'3 days' -> 3, 'for two weeks' -> 14, 'since yesterday' -> 1, 'since this morning' -> 0.3."""
    t = _AGE_PHRASE.sub(" ", text.lower())               # "I am 39 years old" is an age, not a 39-year illness
    m = _DUR.search(t)
    if m:
        q = m.group(1).replace(" of", "")
        n = float(q) if q.replace(".", "").isdigit() else _NUM_WORDS.get(q, 1)
        return round(n * _UNIT_DAYS[m.group(2)], 2)
    if re.search(r"since (yesterday|last night)|\byesterday\b", t):
        return 1.0
    if re.search(r"since (this )?(morning|today)|\btoday\b|this morning|just now|few hours", t):
        return 0.3
    if re.search(r"since (last|the) (week|weekend)", t):
        return 7.0
    return None


def parse_trend(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(worse|worsen\w*|spreading|getting bad|increas\w*|more painful|not (getting )?better|not improving|deteriorat\w*)\b", t):
        return "worse"
    if re.search(r"\b(better|improv\w*|settling|reducing|less)\b", t):
        return "better"
    if re.search(r"\b(same|unchanged|constant|staying|no change)\b", t):
        return "same"
    return None


_SEV_WORDS = [(r"\b(unbearable|excruciating|worst|agonis\w*)\b", 10), (r"\bsevere|terrible|intense|very (bad|painful)\b", 8),
              (r"\bmoderate|medium\b", 5), (r"\bmild|slight|little|minor\b", 2)]


def parse_severity(text: str, bare_ok: bool = False) -> int | None:
    """0-10 score from '6/10', '6 out of 10', 'pain is 7', or (when a score was just asked for) a bare number / word."""
    t = text.lower()
    m = re.search(r"\b(10|[0-9])\s*(?:/|out of)\s*10\b", t) or re.search(r"\b(?:pain|severity|score|level|about|around|like)\s*(?:is|of|at)?\s*(10|[0-9])\b", t)
    if m:
        return int(m.group(1))
    if bare_ok:
        m = re.fullmatch(r"\D*\b(10|[0-9])\b\D*", t)
        if m:
            return int(m.group(1))
        for w, i in _NUM_WORDS.items():
            if i <= 10 and re.search(rf"\b{w}\b", t) and w not in ("a", "an", "few", "couple", "several"):
                return i
    t = strip_negated(t)                                      # "it is not severe" / "not very painful" are not severe
    for rx, val in _SEV_WORDS:
        if re.search(rx, t):
            return val
    return None


_DEPENDENTS = r"(?:son|daughter|child|kid|baby|boy|girl|wife|husband|mother|father|mom|dad|friend|brother|sister)"


def parse_age(text: str, bare_ok: bool = False) -> int | None:
    """The PATIENT's age. 'I'm 39' beats '... a 5 year old son'; 'my son is 5' is the patient when a parent describes their child."""
    t = text.lower()
    m = (re.search(r"\bi(?:'m| am)\s+(\d{1,3})\b(?!\s*(?:kg|cm|%|lbs?|st\b|weeks?|days?|hours?|minutes?|mins?))", t)
         or re.search(r"\b(?:my )?(?:son|daughter|child|kid|baby)\s+(?:is\s+)?(\d{1,2})\b(?!\s*(?:kg|cm|%))", t))
    if not m:
        child = None
        for g in re.finditer(r"\b(\d{1,3})\s*(?:years?|yrs?|y\.?o\.?|yo)(?:[\s-]*old)?\b", t):
            if re.match(rf"\s*{_DEPENDENTS}\b", t[g.end(): g.end() + 14]):         # '5 year old son': a relative's age, not the patient's
                child = child or g; continue
            m = g
            break
        m = m or child
    if not m:
        m = re.search(r"\b(?:age|aged)\s*(?:is\s*)?(\d{1,3})\b", t)
    if not m and bare_ok:
        m = re.match(r"^\D{0,12}(\d{1,3})\b", t)
    if m and 0 <= int(m.group(1)) <= 120:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,2})\s*(?:months?|mos?)\s*old\b", t)
    return 0 if m else None


_RISKS = {
    "diabetes": r"diabet|insulin|sugar (disease|problem)",
    "heart disease": r"heart (disease|problem|failure|attack)|cardiac|coronary|stent|bypass|pacemaker|hypertension|high blood pressure|\bbp\b",
    "lung disease": r"asthma|copd|emphysema|chronic bronchitis|lung disease",
    "kidney disease": r"kidney (disease|failure|problem)|dialysis|\bckd\b",
    "weakened immunity": r"cancer|chemo\w*|\bhiv\b|immuno(?:suppress\w*|compromis\w*|deficien\w*)|transplant|steroid (?:tablets?|injections?|therapy|treatment|medication)|on steroids|long[- ]term steroids|lupus|on biologics",
    "liver disease": r"liver (disease|cirrhosis|problem)|cirrhosis|hepatitis",
    "pregnancy": r"pregnan\w*|expecting (?:a )?(?:baby|child)|\b\d+ (weeks|months) pregnant",
    "stomach ulcer or gastritis": r"ulcer|gastritis|\bgerd\b|acid reflux|hyperacidity|gastric (problem|pain)",
    "bleeding disorder": r"haemophilia|hemophilia|bleeding (disorder|problem)|low platelets?|thrombocytopeni\w*",
}


_THIRD_PARTY = re.compile(
    r"\b(?:mother|mom|mum|father|dad|parents?|brother|sister|siblings?|grand\w+|aunt|uncle|cousin|wife|husband|partner|girlfriend|boyfriend|friend|colleague|"
    r"neighbou?r|in-laws?|relatives?)\b|\bfamily\b|\bruns? in\b|\bhereditary\b|\bgenetic\b", re.I)
_CLAUSES = re.compile(r"[.;,\n]|\s+(?:and|but|while|whereas)\s+(?=i\b|i'm\b|i am\b|i have\b|my own\b)", re.I)


def own_clauses(text: str) -> str:
    """The parts of what the patient said that are about THEM: 'my mother has diabetes, but I have asthma' -> 'but I have asthma'.
    Family history and other people's illnesses (mother, wife, 'runs in the family'...) are not the patient's own conditions."""
    return " . ".join(c for c in _CLAUSES.split(text) if c and c.strip() and not _THIRD_PARTY.search(c))


def parse_risks(text: str) -> set[str]:
    """Conditions the patient says THEY have. Denials ('no diabetes', 'I do not have any existing disease') and other people's
    conditions (family history, 'my wife is pregnant') are not counted."""
    t = strip_negated(own_clauses(text.lower()))
    out = set()
    for name, rx in _RISKS.items():
        if re.search(rx, t):
            out.add(name)
    return out


_TEMP_NUM = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d)?)\s*(?:°|º|degrees?|deg\b)?\s*[a-z]{0,10}(?![\d/])", re.I)
_TEMP_CONTEXT = re.compile(r"temp|fever|thermomet|reading|degree|°|º|fahr|farh|far\w*ght|celsius|centigrade", re.I)
_NOT_A_TEMP_UNIT = re.compile(r"^(?:years?|yrs?|days?|weeks?|months?|hours?|hrs?|mins?|kg|kgs|mg|ml|mcg|bpm|%|cm|lbs?|pounds|percent)\b", re.I)


def parse_temperature_c(text: str, bare_ok: bool = False) -> float | None:
    """Highest body temperature mentioned, in Celsius: '102 degree fahrenheit' -> 38.9, 'temp was 101' -> 38.3, '38.5 C' -> 38.5.

    Fahrenheit (93-109) and Celsius (34-43) ranges do not overlap, so the unit can be inferred (and misspellings such as
    'farheight' do not matter). A number only counts as a temperature when fever/temperature/degree words are nearby."""
    t = text.lower()
    best: float | None = None
    for m in _TEMP_NUM.finditer(t):
        tail = t[m.end(1): m.end(1) + 14].strip()
        if _NOT_A_TEMP_UNIT.match(tail) or _NOT_A_TEMP_UNIT.match(t[m.end():m.end() + 12].strip()):
            continue
        if re.search(r"pulse|heart ?rate|\bbp\b|pressure|sugar|glucose|oxygen|spo2|weight|\bhr\b|\brate\b|systolic|diastolic", t[max(0, m.start() - 16): m.start()]):
            continue                                                 # "my pulse is 100": a vital sign that is not a temperature
        ctx = t[max(0, m.start() - 45): m.end() + 25]
        if not bare_ok and not _TEMP_CONTEXT.search(ctx):          # bare_ok: they were just asked for the reading
            continue
        v = float(m.group(1))
        if 93 <= v <= 109:
            c = (v - 32) * 5 / 9
        elif 34 <= v <= 43:
            c = v
        else:
            continue
        best = c if best is None else max(best, c)
    return round(best, 1) if best is not None else None


def f_from_c(c: float) -> int:
    return round(c * 9 / 5 + 32)


_ALLERGY_TAGS = {
    "nsaid": r"ibuprofen|brufen|diclofenac|naproxen|aspirin|ecosprin|disprin|nsaids?|nimesulide|painkillers?|anti-?inflammator\w*",
    "paracetamol": r"paracetamol|acetaminophen|crocin|dolo|tylenol|calpol",
    "penicillin": r"penicillin|amoxicillin|amoxyclav|augmentin|ampicillin",
    "sulfa": r"sulfa|sulpha|sulfonamide|septran|cotrimoxazole",
    "antihistamine": r"cetirizine|loratadine|antihistamine|levocetirizine|fexofenadine",
}


def parse_allergies(text: str) -> set[str]:
    """Drug classes the patient says they are allergic to ('allergic to penicillin and aspirin' -> {penicillin, nsaid})."""
    low = text.lower()
    out: set[str] = set()
    # "I have no known allergies except penicillin": the exception IS an allergy, so read it before the denial is removed
    for m in re.finditer(r"\b(?:no|not|nil|none)\b[^.;]{0,40}?\ballerg\w*[^.;]{0,25}?\b(?:except|other than|apart from|besides)\b([^.;]*)", low):
        for tag, rx in _ALLERGY_TAGS.items():
            if re.search(rx, m.group(1)):
                out.add(tag)
    # A DENIAL IS NOT AN ALLERGY: "I am not allergic to any medicine", "no known drug allergies", "NKDA", "never been allergic to...".
    # Remove every negated statement first, then read what is left.
    t = strip_negated(re.sub(r"\bnkda\b", "no allergies", low))
    for m in re.finditer(r"allerg\w*|hypersensitiv\w*|reaction to|rash (?:from|with|after)", t):
        span = t[max(0, m.start() - 60): m.end() + 80]
        for tag, rx in _ALLERGY_TAGS.items():
            if re.search(rx, span):
                out.add(tag)
        if re.search(r"allerg\w*\s+(?:to\s+)?(?:most|many|all|several|multiple)\s+(?:medicines?|drugs?|tablets?)", span):
            out.add("many")
    return out


_MED_TAGS = {
    "anticoagulant": r"warfarin|acenocoumarol|apixaban|rivaroxaban|dabigatran|heparin|clopidogrel|ticagrelor|prasugrel|blood thinners?|anticoagulants?",
    "aspirin": r"aspirin|ecosprin|disprin",
    "nsaid": r"ibuprofen|diclofenac|naproxen|brufen|nimesulide|etoricoxib|ketorolac|nsaids?",
    "bp_kidney": r"lisinopril|enalapril|ramipril|losartan|telmisartan|olmesartan|valsartan|furosemide|lasix|hydrochlorothiazide|spironolactone|"
                 r"diuretics?|water pills?|ace inhibitors?",
    "lithium_methotrexate": r"lithium|methotrexate",
    "antidepressant": r"sertraline|fluoxetine|escitalopram|citalopram|paroxetine|venlafaxine|duloxetine|amitriptyline|ssri|antidepressants?",
    "paracetamol": r"paracetamol|acetaminophen|crocin|dolo\b|tylenol|calpol|combiflam|sinarest|cold (?:and|&) flu tablets?",
}


def parse_medicines(text: str) -> set[str]:
    """Categories of regular medicines the patient says they take ('I take warfarin and amlodipine' -> {anticoagulant})."""
    low = text.lower()
    # medicines they have STOPPED are not current ("I stopped taking warfarin last year", "used to be on aspirin")
    low = re.sub(r"\b(?:stopped|quit|discontinued|used to (?:take|be on|use)|no longer (?:take|taking|on|use)|previously (?:took|was on)|was (?:on|taking))\b[^.;]{0,50}", " ", low)
    low = re.sub(r"\b(?:not|never|no)\s+(?:known\s+)?allerg\w*[^,.;]*", " ", low)   # a denied allergy clause is not a medicine
    t = strip_negated(strip_allergy_clauses(low))              # "I take no aspirin", "not on blood thinners" are not medicines
    return {tag for tag, rx in _MED_TAGS.items() if re.search(rx, t)}


def strip_allergy_clauses(t: str) -> str:
    """Drop 'allergic to X' clauses so an allergy to aspirin is not mistaken for taking aspirin."""
    return re.sub(r"(?:allerg\w*|hypersensitiv\w*|reaction to)[^.;]{0,70}", " ", t)


# ---------- negation ("no rash and no stiff neck" must not count as a rash and a stiff neck) ----------
_CUES = {"no", "not", "without", "denies", "denied", "never", "nor", "neither", "negative"}
_CUES_STRICT = {"no", "without", "denies", "denied", "never", "nor"}      # emergency check: "not breathing" must stay a flag
_BREAK = {"but", "however", "although", "though", "except", "yet", "because", "while", "whereas", "since"}
_NOT_DENIAL = {"can't", "cant", "couldn't", "won't", "wouldn't"}           # "can't breathe" is an inability, not a denial
_HAVE_NOT = re.compile(r"\b(?:do|does|did|have|has|had)\s+not\s+(?:have|had|feel|felt|get|got|experience|any)\b|"
                       r"\b(?:don't|doesn't|didn't|haven't|hasn't|hadn't)\s+(?:have|had|feel|felt|get|got|experience|any)\b", re.I)
_TOKEN = re.compile(r"[\w'/-]+|[,.;:!?]")


def strip_negated(text: str, *, strict: bool = False, window: int = 5) -> str:
    """Remove the words a negation cue applies to ('no rash and no stiff neck' -> '').

    Scope: up to `window` words after the cue, ended by punctuation, a contrast word (but/however...), or 'and' (a new
    clause, which may hold a real symptom: 'no fever and blood in stool'). It continues across 'or' ('no fever or chills').
    strict=True is for the emergency check and only honours explicit denials (no/without/denies/never, 'do not have'),
    so 'he is not breathing' and 'I can't breathe' are never hidden.
    """
    cues = _CUES_STRICT if strict else _CUES
    text = _HAVE_NOT.sub(" no ", text)
    toks = _TOKEN.findall(text)
    out: list[str] = []
    scope, i = 0, 0
    while i < len(toks):
        t, lw = toks[i], toks[i].lower()
        is_word = t[0].isalnum() or t[0] == "'"
        if is_word and (lw in cues or (not strict and lw.endswith("n't") and lw not in _NOT_DENIAL)):
            scope = window
        elif scope and not is_word:
            scope = 0
            out.append(t)
        elif scope and lw in _BREAK:
            scope = 0
            out.append(t)
        elif scope and lw == "and":
            nxt = toks[i + 1].lower() if i + 1 < len(toks) else ""
            if nxt in cues or (not strict and nxt.endswith("n't")):
                pass                                           # 'no rash and no stiff neck': the second cue restarts the scope
            else:
                scope = 0
                out.append(t)
        elif scope:
            scope -= 1                                         # dropped: inside a negation's scope
        else:
            out.append(t)
        i += 1
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(out))


# ---------- red-flag / moderate-flag symptom phrases (beyond safety.py's emergency list) ----------
_URGENT = [
    (r"blood (in|with) (my )?(stool|urine|vomit|sputum|phlegm)|vomit(ing)? blood|coughing (up )?blood|black,? (tarry )?stools?|blood in (my )?(poop|motion)", "blood in stool, urine, vomit or sputum"),
    (r"stiff neck|neck stiffness", "stiff neck with other symptoms"),
    (r"worst headache|thunderclap", "sudden, worst-ever headache"),
    (r"jaundice|yellow(ing|ish)? (of )?(my )?(skin|eyes)", "yellowing of skin or eyes"),
    (r"(can'?t|cannot|unable to|not able to) (keep|hold) (down )?(fluids|water|food)|persistent vomiting|vomiting (for|since) (more than )?\w+ days", "unable to keep fluids down"),
    (r"no urine|not (passing|peeing) urine|haven'?t (urinated|peed)", "not passing urine"),
    (r"(high|very high) fever", "high fever"),               # numeric temperatures are handled in assess() via parse_temperature_c
    (r"testicular pain|testicle pain|sudden (scrotal|testicular)", "sudden testicular pain"),
    (r"severe (abdominal|stomach|belly) pain", "severe abdominal pain"),
    (r"(pregnan\w*).{0,40}(bleed\w*|severe pain|cramp\w*)|(bleed\w*|severe pain).{0,40}pregnan", "bleeding or severe pain in pregnancy"),
    (r"sudden (loss of )?vision|lost (my )?vision|vision (loss|going)", "sudden vision change"),
    (r"\bconfusion\b|disorient\w+|(very|suddenly|newly|increasingly|become|became|becoming) confused|drowsy and hard to wake", "confusion"),
    (r"faint(ed|ing)|blacked out", "fainting"),
    (r"dehydrat\w+|very dry mouth", "signs of dehydration"),
    (r"(rash|spots).{0,30}(fever)|(fever).{0,30}(rash|spots)", "fever with rash"),
]
_SOON = [
    (r"unexplained weight loss|losing weight (without|for no)", "unexplained weight loss"),
    (r"night sweats", "night sweats"),
    (r"\blump\b|swelling that (is )?(not|won'?t)", "a lump or persistent swelling"),
    (r"pus|discharge|infect\w+", "possible infection"),
    (r"burning (while|when|during) (urinat|peeing)|painful urination|frequent urination", "urinary symptoms"),
    (r"palpitation|racing heart|irregular heart", "palpitations"),
    (r"numbness|tingling|weakness (in|of) (my )?(arm|leg|hand|foot)", "numbness or limb weakness"),
    (r"wheez\w+|short(ness)? of breath|difficulty breathing|breathless", "breathing difficulty"),
    (r"fever.{0,40}(3|4|5|6|7|three|four|five) days|(3|4|5|6|7|three|four|five) days.{0,40}fever", "fever for several days"),
    (r"persistent cough|cough.{0,25}(weeks|month)", "persistent cough"),
    (r"blood pressure|sugar level", "monitoring-related concern"),
    (r"depress\w+|hopeless|panic attack|can'?t sleep|anxiety", "mental-health symptoms"),
]


@dataclass
class Facts:
    text: str = ""                       # everything the patient has written, joined
    symptom_text: str = ""               # only what they said about the problem (not their medical-history answer): picks the specialist
    duration_days: float | None = None
    trend: str | None = None
    severity: int | None = None
    age: int | None = None
    risks: set[str] = field(default_factory=set)
    temp_c: float | None = None          # highest temperature reported (Celsius)
    allergies: set[str] = field(default_factory=set)         # drug classes ('nsaid', 'penicillin', ...)
    medicines: set[str] = field(default_factory=set)         # regular medicines by class ('anticoagulant', ...)
    allergies_asked: bool = False        # the allergy/medicine question has been answered (even with 'none')
    slots: dict[str, bool] = field(default_factory=dict)     # symptom slots: True = has it, False = explicitly does not (see clinic.py)


@dataclass
class Assessment:
    level: str
    score: int
    reasons: list[str]
    specialty: str
    complex: bool                        # drives which knowledge source answers (see consult.py)
    possible: list[dict] = field(default_factory=list)       # conditions that fit the symptoms (possibilities, not a diagnosis)

    @property
    def info(self) -> dict:
        return LEVEL_INFO[self.level]


# ---------- specialist ----------
SPECIALTIES: dict[str, dict] = {
    "Cardiologist":       {"osm": ["cardio"], "kw": r"palpitat|heart|chest|blood pressure|hypertens|cholesterol|irregular heartbeat|swollen (ankle|feet)"},
    "Neurologist":        {"osm": ["neuro"], "kw": r"headache|migraine|seizure|numb|tingl|dizz|vertigo|tremor|memory|paraly|nerve|faint"},
    "Dermatologist":      {"osm": ["dermat"], "kw": r"\brash|itch|skin|acne|eczema|hives|psoria|\bmole|hair (loss|fall)|dandruff|fungal|ringworm|\bboil|\bwart|pimple"},
    "Gastroenterologist": {"osm": ["gastro"], "kw": r"stomach|abdom|belly|vomit|nausea|diarrh|constipat|acid|reflux|heartburn|bloat|\bgas\b|stool|liver|jaundice|ulcer|indigestion|\bmotion"},
    "ENT specialist":     {"osm": ["otolaryng", "ent", "ear_nose"], "kw": r"\bear\b|\bears\b|earache|throat|tonsil|sinus|\bnose\b|nasal|hearing|hoarse|snoring|blocked nose|runny nose"},
    "Orthopedist":        {"osm": ["orthop", "trauma"], "kw": r"joint|knee|back pain|shoulder|fracture|\bbone|sprain|neck pain|\bhip\b|ankle|wrist|muscle pain|arthritis|\bspine|slip disc"},
    "Pulmonologist":      {"osm": ["pulmon", "respirat", "chest"], "kw": r"cough|breath|wheez|asthma|lung|phlegm|sputum|bronch"},
    "Gynecologist":       {"osm": ["gyn", "obstet"], "kw": r"period|menstru|pregnan|vagina|pelvic|breast|pcos|menopaus|white discharge|missed (a )?period"},
    "Urologist":          {"osm": ["urolog"], "kw": r"urin|bladder|kidney stone|prostate|testic|erectile|penis|\buti\b|burning (while|when) peeing"},
    "Ophthalmologist":    {"osm": ["ophthal", "eye"], "kw": r"\beye\b|\beyes\b|vision|eyesight|blurr|red eye|eyelid|watering eyes"},
    "Psychiatrist":       {"osm": ["psychiat", "psycholog"], "kw": r"anxiety|depress|panic|stress|insomnia|can'?t sleep|mood|hallucinat|hopeless"},
    "Endocrinologist":    {"osm": ["endocrin", "diabet"], "kw": r"diabet|thyroid|blood sugar|weight gain|hormon"},
    "Dentist":            {"osm": ["dent"], "kw": r"tooth|teeth|\bgum|dental|\bjaw|cavity|toothache"},
    "Nephrologist":       {"osm": ["nephro", "kidney"], "kw": r"kidney (disease|failure|function)|dialysis|creatinine"},
}
DEFAULT_SPECIALTY = {"name": "General Physician", "osm": ["general", "internal", "family", "primary"]}
_CHILD = re.compile(r"\b(my )?(baby|infant|newborn|toddler|child|kid|son|daughter)\b", re.I)


def pick_specialty(text: str, age: int | None = None) -> tuple[str, list[str]]:
    """Specialist whose keywords match best; Pediatrician for children; General Physician when nothing stands out."""
    if (age is not None and age < 13) or _CHILD.search(text):
        return "Pediatrician", ["paediatr", "pediatr", "child"]
    best, hits = DEFAULT_SPECIALTY["name"], 0
    for name, spec in SPECIALTIES.items():
        n = len(re.findall(spec["kw"], text, re.I))
        if n > hits:
            best, hits = name, n
    return (best, SPECIALTIES[best]["osm"]) if hits else (DEFAULT_SPECIALTY["name"], DEFAULT_SPECIALTY["osm"])


# ---------- the assessment ----------
def assess(f: Facts) -> Assessment:
    reasons: list[str] = []
    text = f.text

    flags = safety.emergency_flags(strip_negated(text, strict=True) + " I am")
    text = strip_negated(text)                           # everything below matches symptoms the patient HAS, not ones they deny
    symptoms = strip_negated(f.symptom_text) if f.symptom_text else text
    from . import clinic                                   # (clinic imports this module, hence the local import)
    slot_flags = clinic.red_flags(f)                       # warning signs the patient confirmed with a plain "yes"
    flags += [label for lvl, label in slot_flags if lvl == "emergency"]
    if flags:
        spec, _ = pick_specialty(symptoms, f.age)
        return Assessment("emergency", 99, [f"Emergency warning sign: {x}" for x in dict.fromkeys(flags)], spec, True)

    score = 0
    urgent_hit = False
    for lvl, label in slot_flags:
        if lvl == "urgent":
            score, urgent_hit = score + 3, True
            reasons.append(f"Warning sign: {label}")
    if f.slots.get("breath_short") is True and not any("breath" in r.lower() for r in reasons):
        score += 2
        reasons.append("Needs a clinician's check: shortness of breath")
    for rx, label in _URGENT:
        if re.search(rx, text, re.I):
            score, urgent_hit = score + 3, True
            reasons.append(f"Warning sign: {label}")
    for rx, label in _SOON:
        if re.search(rx, text, re.I):
            score += 2
            reasons.append(f"Needs a clinician's check: {label}")
    temp = f.temp_c if f.temp_c is not None else parse_temperature_c(f.text)
    if temp is not None:
        label = f"{f_from_c(temp)}°F ({temp:.1f}°C)"
        if temp >= 39.4:                                     # 103°F and above
            score, urgent_hit = score + 3, True
            reasons.append(f"High fever: {label}")
        elif temp >= 38.0:
            score += 1
            reasons.append(f"Fever: {label}")
        if temp >= 38.0 and f.age is not None and f.age < 2:
            score, urgent_hit = score + 3, True
            reasons.append("Fever in a baby under 2 needs to be seen the same day")
    if f.severity is not None:
        add = 3 if f.severity >= 8 else 2 if f.severity >= 5 else 1 if f.severity >= 3 else 0
        score += add
        if add:
            reasons.append(f"Symptom severity reported as {f.severity}/10")
    if f.duration_days is not None:
        if f.duration_days >= 14:
            score += 2
            reasons.append(f"Lasting {round(f.duration_days)} days")
        elif f.duration_days >= 7:
            score += 1
            reasons.append(f"Lasting about {round(f.duration_days)} days")
    if f.trend == "worse" and (f.duration_days or 0) >= 1:
        score += 1
        reasons.append("Getting worse rather than better")
    vulnerable = f.age is not None and (f.age < 2 or f.age >= 70)
    if vulnerable and score >= 1:
        score += 1
        reasons.append(f"Age {f.age} (higher risk)")
    if f.risks and score >= 1:
        score += 1
        reasons.append("Existing condition(s): " + ", ".join(sorted(f.risks)))
    if "pregnancy" in f.risks and score >= 1:
        score += 1

    if urgent_hit or score >= 6 or (f.severity is not None and f.severity >= 9):
        level = "urgent"
    elif score >= 3:
        level = "see_doctor"
    else:
        level = "self_care"
    if not reasons:
        reasons = ["No warning signs reported; symptoms sound mild"]

    spec, _ = pick_specialty(symptoms, f.age)
    st = clinic.state(f)
    if spec == "Neurologist" and st.get("fever") and not any(st.get(k) for k in ("neuro_deficit", "stiff_neck", "worst_headache", "aura", "head_one_sided")):
        spec = "General Physician"                             # headache with fever: a general physician looks for the cause first
    complex_case = level != "self_care" or bool(f.risks) or (f.duration_days or 0) >= 14
    return Assessment(level, score, reasons, spec, complex_case, clinic.differential(f))
