"""Clinical knowledge for the digital-doctor interview: what the patient has told us (symptom "slots"), which conditions that
points to (a small differential), and which answers are red flags.

Everything is transparent rules, so it is fast, testable and safe on a Raspberry Pi; no model decides what to ask.

* extract(text)      free text -> {slot: True/False}. True = the patient has it, False = they explicitly said they do not
                     ("no chills"); a slot that is absent is simply unknown.
* differential(f)    which common conditions fit the slots so far, with the symptoms supporting each (NOT a diagnosis).
* red_flags(f)       answers that change urgency, whatever the rest of the story is.
"""
import re
from dataclasses import dataclass

from . import triage


@dataclass(frozen=True)
class Slot:
    key: str
    rx: str              # how the patient might say it
    phrase: str          # canonical wording appended to the symptom text when they confirm it (keeps medicines/specialist working)
    yes: str             # how the acknowledgement reads when True
    no: str              # ... and when False


S = Slot
SLOTS: tuple[Slot, ...] = (
    # --- general
    S("fever", r"fever|feverish|febrile|high temperature|\btemp(?:erature)?\b.{0,20}(?:high|raised|up)|\bshivering with", "fever", "fever", "no fever"),
    S("chills", r"chills?|shiver\w*|rigors?", "chills", "chills", "no chills"),
    S("body_ache", r"body ?ache|body pain|aching all over|muscle (?:ache|pain)s?|myalgia|joint pains?|aches? and pains?", "body ache", "body aches", "no body aches"),
    S("fatigue", r"fatigue|very tired|extremely tired|exhaust\w*|weakness|feel(?:ing)? weak|no energy|lethargi\w*", "fatigue", "tiredness", "no unusual tiredness"),
    S("mosquito_exposure", r"mosquito\w*|dengue (?:cases|outbreak|in (?:my|our))|malaria|stagnant water|monsoon|travel(?:led|ed)? to", "mosquito exposure", "mosquito/dengue exposure", "no mosquito exposure"),
    # --- nose
    S("nasal_block", r"nasal (?:block\w*|congest\w*)|blocked nose|nose (?:is )?(?:blocked|stuffy|congested)|(?:nose|nasal)\b.{0,20}(?:blockage|congestion)|blockage|stuffy|(?<!chest )(?<!lung )congest\w*", "blocked nose", "a blocked nose", "no nasal blockage"),
    S("runny_nose", r"runny|running nose|nose (?:is )?running|nasal discharge|dripping nose|rhinorrh\w*", "runny nose", "a runny nose", "no runny nose"),
    S("sneezing", r"sneez\w*", "sneezing", "sneezing", "no sneezing"),
    S("itchy_nose_eyes", r"itch\w* (?:nose|eyes?|throat|palate)|(?:nose|eyes?|throat) (?:is |are |feels? )?itch\w*|watery eyes|watering eyes", "itchy nose and eyes", "itchy nose/eyes", "no itching of the nose or eyes"),
    S("thick_discharge", r"(?:yellow|green|greenish|yellowish|thick)\s+(?:mucus|discharge|phlegm|snot)|(?:mucus|discharge|phlegm|snot)\b.{0,15}(?:yellow|green|thick)", "thick yellow-green mucus", "thick yellow/green mucus", "no thick coloured mucus"),
    S("facial_pain", r"facial (?:pain|pressure)|face (?:pain|pressure|hurts)|pain (?:around|behind|over) (?:the )?(?:eyes?|cheeks?|forehead|nose)|sinus (?:pain|pressure|headache)|pressure (?:in|around) (?:the )?(?:face|forehead|cheeks?|eyes?)", "facial pain", "facial pain or pressure", "no facial pain"),
    S("smell_loss", r"(?:loss|lost) (?:of )?(?:my )?(?:smell|taste)|can'?t smell|cannot smell|no sense of smell|anosmia", "loss of smell", "loss of smell", "normal sense of smell"),
    S("one_sided_nose", r"one nostril|single nostril|(?:nose|nostril|nasal|blocked|blockage)\b.{0,30}\bone side\b|\bone side\b.{0,20}(?:nose|nostril|blocked)|(?:blocked|blockage|stuffy) (?:on|in) (?:the )?(?:left|right)|only (?:the )?(?:left|right) (?:nostril|side of (?:my |the )?nose)", "one-sided blockage", "blockage on one side only", "blockage on both sides"),
    S("double_worsening", r"(?:got|getting) better (?:and )?(?:then|but) (?:got |became |is )?worse|better then worse|worse again|second wave|relapse\w*", "worse again after improving", "worse again after improving", "no relapse"),
    # --- throat
    S("sore_throat", r"sore throat|throat (?:pain|ache|hurts|is (?:sore|painful))|scratchy throat|painful throat|pharyngitis|tonsillitis", "sore throat", "a sore throat", "no sore throat"),
    S("painful_swallow", r"painful (?:to )?swallow\w*|pain (?:on|when|while|during) swallow\w*|swallow\w* (?:is |are )?(?:painful|hurts?|difficult)|difficult\w* swallow\w*|odynophagia", "painful swallowing", "painful swallowing", "no pain on swallowing"),
    S("swollen_glands", r"swollen (?:glands?|lymph\w*|nodes?)|(?:glands?|lymph nodes?) (?:are |is )?swollen|tender glands?|lump (?:in|on) (?:the |my )?neck", "swollen neck glands", "swollen neck glands", "no swollen glands"),
    S("white_patches", r"white (?:patches|spots|coating)|pus (?:spots|on)|exudate|patches on (?:the )?tonsils", "white patches on tonsils", "white patches on the tonsils", "no white patches"),
    S("hoarse", r"hoarse\w*|voice (?:change|has changed|is (?:gone|weak))|lost (?:my )?voice", "hoarse voice", "a hoarse voice", "no voice change"),
    S("airway_swelling", r"difficult\w* (?:opening|to open) (?:my |the )?mouth|trouble opening (?:my |the )?mouth|drooling|can'?t swallow (?:saliva|my spit)|muffled voice", "trouble opening mouth or drooling", "trouble opening the mouth / drooling", "no trouble opening the mouth"),
    # --- chest / breathing
    S("cough", r"cough\w*", "cough", "a cough", "no cough"),
    S("dry_cough", r"dry cough|cough (?:is )?dry|non-?productive|tickl\w* cough|cough without phlegm", "dry cough", "a dry cough", "not a dry cough"),
    S("wet_cough", r"phlegm|sputum|productive cough|wet cough|chesty cough|mucus (?:when|while) coughing|cough(?:ing)? up (?:mucus|phlegm)", "cough with phlegm", "a cough with phlegm", "no phlegm"),
    S("sputum_coloured", r"(?:yellow|green|greenish|brown|rust\w*|dark)\s+(?:phlegm|sputum|mucus)|(?:phlegm|sputum)\b.{0,15}(?:yellow|green|brown|rust)", "coloured phlegm", "yellow/green phlegm", "clear phlegm"),
    S("blood_sputum", r"blood in (?:my )?(?:sputum|phlegm|cough)|cough(?:ing|ed)? (?:up )?blood|blood[- ]stained (?:phlegm|sputum)|haemoptysis|hemoptysis", "coughing up blood", "coughing up blood", "no blood in the phlegm"),
    S("breath_short", r"short(?:ness)? of breath|breathless\w*|difficulty breathing|trouble breathing|can'?t catch (?:my )?breath|out of breath|breathing (?:is )?difficult|gasping", "shortness of breath", "shortness of breath", "no shortness of breath"),
    S("wheeze", r"wheez\w*|whistling (?:sound )?(?:when|while) breathing|noisy breathing", "wheezing", "wheezing", "no wheezing"),
    S("chest_pain", r"chest (?:pain|tightness|pressure|heaviness|hurts|is hurting|feels? (?:tight|heavy))|pain (?:in|across) (?:my |the )?chest|tight chest|heaviness in (?:my |the )?chest", "chest pain", "chest pain", "no chest pain"),
    S("tb_symptoms", r"weight loss|losing weight|night sweats|drenching sweats|loss of appetite", "weight loss or night sweats", "weight loss or night sweats", "no weight loss or night sweats"),
    # --- head / neuro
    S("headache", r"headache|migraine|head (?:pain|ache|hurts|is (?:pounding|throbbing|splitting))|pain in (?:my |the )?head", "headache", "a headache", "no headache"),
    S("worst_headache", r"worst headache|thunderclap|sudden (?:severe |terrible )?headache|worst (?:pain )?of my life|headache (?:came|started) suddenly", "sudden worst headache", "a sudden, worst-ever headache", "not a sudden worst headache"),
    S("head_one_sided", r"one side of (?:the |my )?head|(?:headache|pain|head)\b.{0,20}\bone side\b|\bone[- ]sided (?:headache|pain)|(?:left|right) side of (?:the |my )?head|(?:left|right)[- ]sided|unilateral|(?:behind|over) (?:the |my )?(?:left|right) (?:eye|temple)|\btemple\b", "one-sided headache", "one-sided pain", "pain on both sides / all over"),
    S("head_band", r"tight band|band[- ]like|like a (?:tight )?band|head ?band|(?:band|tight|squeez\w*|pressure) (?:around|like a band)|all over (?:the |my )?head|both sides of (?:the |my )?head|tension headache|pressure (?:on|around) (?:the |my )?head", "band-like headache", "band-like pressure", "no band-like pressure"),
    S("throbbing", r"throbb\w*|pulsat\w*|pounding|hammering", "throbbing pain", "throbbing pain", "not throbbing"),
    S("light_sensitive", r"(?:sensitiv\w*|bothers?|hurts?|worse with) (?:to |by |in )?(?:bright )?(?:light|lights|noise|sound)s?|photophobia|light (?:hurts|bothers)|bright lights?", "sensitivity to light", "sensitivity to light or sound", "no light sensitivity"),
    S("aura", r"\baura\b|zig-?zag|flashing lights?|flashes of light|blind spots?|visual (?:disturbance|changes)|seeing (?:spots|stars|lines)", "visual aura", "visual disturbance (aura)", "no visual aura"),
    S("stress_trigger", r"stress\w*|tension|lack of sleep|poor sleep|not sleeping|skipp\w+ meals|long hours|screen\w*|overwork\w*|anxious|anxiety", "stress or poor sleep", "stress or poor sleep", "no obvious stress trigger"),
    S("stiff_neck", r"stiff neck|neck (?:is )?stiff\w*|neck stiffness|can'?t (?:bend|touch) (?:my )?chin|pain (?:when|on) bending (?:my )?neck", "stiff neck", "a stiff neck", "no stiff neck"),
    S("neuro_deficit", r"(?:weakness|numbness|tingling) (?:in|on|of) (?:my |the )?(?:arm|leg|face|one side|hand|foot)|slurred speech|trouble speaking|face (?:droop\w*|is drooping)|sudden (?:loss of )?vision|vision (?:loss|loss in)|double vision|can'?t move (?:my )?(?:arm|leg)|one[- ]sided weakness", "weakness, numbness or speech trouble", "weakness, numbness or speech/vision trouble", "no weakness, numbness or speech/vision trouble"),
    S("retro_orbital", r"pain (?:behind|at the back of) (?:my |the )?eyes?|(?:behind|back of) (?:my |the )?eyes? (?:pain|ache|hurts)|retro-?orbital|eyes? (?:hurt|ache)s? (?:when|on) moving", "pain behind the eyes", "pain behind the eyes", "no pain behind the eyes"),
    S("dizziness", r"dizz\w*|vertigo|light-?headed\w*|spinning", "dizziness", "dizziness", "no dizziness"),
    # --- stomach / gut
    S("abdominal_pain", r"stomach (?:pain|ache|cramps?|hurts)|abdominal pain|belly (?:pain|ache|hurts)|tummy (?:pain|ache)|pain in (?:my |the )?(?:stomach|abdomen|belly)|(?:stomach|abdominal|tummy|belly) cramps?", "stomach pain", "stomach pain", "no stomach pain"),
    S("vomiting", r"vomit\w*|throwing up|threw up|puk\w*", "vomiting", "vomiting", "no vomiting"),
    S("nausea", r"nausea\w*|nauseous|feel(?:ing)? sick|queasy", "nausea", "nausea", "no nausea"),
    S("diarrhoea", r"diarrh\w*|loose (?:motions?|stools?)|watery (?:stools?|motions?)|runny stools?|loose bowels?", "diarrhoea", "loose motions", "no loose motions"),
    S("blood_stool", r"blood in (?:my |the )?(?:stool|motion|poop)|black,? (?:tarry )?(?:stools?|motions?)|bloody (?:stool|diarrh\w*)|tarry stools?|bright red blood (?:when|on)", "blood in stool", "blood or black stools", "no blood in the stool"),
    S("can_keep_fluids", r"(?:can|able to) (?:keep|hold) (?:down )?(?:fluids?|water|food)|drinking (?:well|fine)|tolerating fluids", "keeping fluids down", "able to keep fluids down", "unable to keep fluids down"),
    S("constant_pain", r"constant (?:pain|ache)|severe (?:stomach|abdominal|belly) pain|sharp (?:stomach |abdominal )?pain|pain (?:is )?(?:non-?stop|continuous)", "constant severe abdominal pain", "constant, severe pain", "pain that comes and goes"),
    S("rlq_pain", r"lower right (?:side|abdomen|belly)|right lower (?:side|abdomen|belly)|pain (?:near|around) (?:the )?(?:belly ?button|navel).{0,30}right|pain (?:moved|shifted) to (?:the )?(?:lower )?right", "lower right abdominal pain", "pain in the lower right abdomen", "no lower-right pain"),
    S("recent_food", r"street food|outside food|restaurant|hotel food|food poison\w*|stale food|undercooked|contaminated|ate (?:something|outside)|dinner outside|wedding|party", "unusual food recently", "unusual or outside food recently", "no unusual food"),
    S("heartburn", r"heartburn|acidity|acid reflux|reflux|burning (?:in|behind) (?:my |the )?(?:chest|stomach|food ?pipe)|sour (?:taste|belch\w*)|regurgitat\w*|gastric", "heartburn", "heartburn or acidity", "no heartburn"),
    S("after_meals", r"after (?:meals?|eating|food)|when lying down|at night when lying|bending (?:forward|over)|spicy food", "worse after meals or lying down", "worse after meals or lying down", "not related to meals"),
    S("bloating", r"bloat\w*|gas\b|flatulen\w*|distended|full stomach", "bloating", "bloating", "no bloating"),
    # --- urine
    S("burning_urine", r"burn\w* (?:while|when|during|on) (?:(?:i |you )?pass(?:ing)? (?:urine|water)|urinat\w*|peeing)|painful urination|dysuria|stinging (?:when|while) (?:passing urine|peeing)|burning urine|pain (?:when|while) (?:passing urine|peeing)", "burning on passing urine", "burning on passing urine", "no burning on passing urine"),
    S("frequency", r"frequent (?:urination|urine)|(?:pass|passing) urine (?:very )?(?:often|frequently)|(?:go|going) to the (?:toilet|bathroom) (?:very )?(?:often|frequently)|urinating (?:a lot|often)|urgency", "frequent urination", "frequent urination", "no increase in urination"),
    S("blood_urine", r"blood in (?:my |the )?urine|red urine|pink urine|hematuria|haematuria|cola[- ]coloured urine", "blood in urine", "blood in the urine", "no blood in the urine"),
    S("flank_pain", r"flank pain|pain (?:in|on) (?:my |the )?(?:side|loin|flank)\b|(?:side|back) pain (?:near|under|below) (?:the )?ribs|kidney (?:pain|area)|pain in (?:my )?(?:loin|flank)|pain (?:in|at) the (?:lower )?(?:back|side)", "flank pain", "pain in the side or lower back", "no flank pain"),
    S("bladder_bowel", r"(?:trouble|difficulty|unable|can'?t) (?:passing|pass) (?:urine|stool)|lost (?:control|bladder|bowel)|incontinen\w*|numb(?:ness)? (?:in|around) (?:the )?(?:groin|saddle|genitals)", "bladder or bowel trouble", "trouble passing urine or stool", "no bladder or bowel trouble"),
    # --- skin
    S("rash", r"\brash\w*|hives|welts|red (?:spots|patches|bumps)|spots on (?:my )?skin|skin eruptions?|measles|urticaria", "rash", "a rash", "no rash"),
    S("itchy_skin", r"itchy (?:skin|rash|body|patch\w*|arms?|legs?|back|hands?|feet|scalp)|skin (?:is |feels? )?itch\w*|itching (?:all over|of the skin)|pruritus|rash.{0,30}itch\w*|itch\w*.{0,20}rash", "itching", "itchy skin", "no itching"),
    S("rash_spreading", r"spread\w*|growing|getting bigger|more spots|new spots", "spreading rash", "a spreading rash", "not spreading"),
    S("blisters", r"blister\w*|peel\w*|oozing|weeping|skin (?:is )?coming off|sloughing", "blisters or peeling", "blisters or peeling", "no blisters or peeling"),
    S("new_product", r"new (?:soap|cream|detergent|lotion|shampoo|medicine|tablet|drug|food|perfume|cosmetic)s?|after eating (?:something )?(?:new|unusual)|started (?:a )?new|after (?:taking|starting|using|applying) (?:a |the )?(?:new )?(?:medicine|tablet|drug|cream|soap|antibiotic|painkiller|ointment)", "new product or medicine", "a new soap/cream/medicine/food before it started", "no new product or medicine"),
    S("angioedema", r"swelling (?:of|in) (?:the |my )?(?:lips?|face|tongue|eyelids?|throat)|(?:lips?|face|tongue|eyelids?|throat) (?:is |are )?swell\w*|swollen (?:lips?|face|tongue|eyelids?)|tight(?:ness)? in (?:my )?throat", "swelling of lips, face or tongue", "swelling of the lips, face or tongue", "no swelling of the lips, face or tongue"),
    # --- muscles / bones
    S("joint_pain", r"joint (?:pain|ache|swelling)s?|knee (?:pain|hurts)|shoulder (?:pain|hurts)|hip (?:pain|hurts)|elbow|wrist (?:pain|hurts)|ankle (?:pain|hurts)|arthritis|pain in (?:my |the )?(?:knee|shoulder|hip|ankle|wrist|elbow|joint)", "joint pain", "joint pain", "no joint pain"),
    S("back_pain", r"back ?pain|backache|lower back|upper back|lumbar|pain in (?:my |the )?back|(?:my |the )?back (?:hurts?|is (?:sore|painful|aching|stiff)|aches?)|(?:sore|aching|painful) back|spine|sciatica|slip(?:ped)? disc", "back pain", "back pain", "no back pain"),
    S("injury", r"injur\w*|\bfell\b|\bfall\b|fallen|twist\w*|sprain\w*|accident|lifted (?:something )?heavy|heavy lifting|hit (?:my|the)|sports?|strain\w*", "recent injury", "a recent injury, fall or strain", "no injury"),
    S("limb_swelling", r"(?:joint|knee|ankle|foot|feet|leg|hand|wrist|elbow|shoulder)s?\b.{0,20}(?:swell\w*|swollen|puffy)|(?:swell\w*|swollen|puffy) (?:joint|knee|ankle|foot|feet|leg|hand|wrist|elbow|shoulder)", "swelling", "swelling of the joint or limb", "no swelling"),
    S("morning_stiffness", r"morning stiffness|stiff (?:in the )?morning|stiffness (?:in the |every )?morning|stiff for (?:more than )?\d+ ?(?:minutes?|mins?|hours?)", "morning stiffness", "morning stiffness", "no morning stiffness"),
    S("leg_numbness", r"(?:numbness|tingling|pins and needles|weakness) (?:in|down) (?:my |the )?(?:leg|legs|foot|feet|arm|hand)|pain (?:shooting|radiating|going) down (?:my |the )?(?:leg|arm)", "numbness or weakness in a limb", "numbness, tingling or weakness in a limb", "no numbness or weakness"),
)
BY_SLOT = {s.key: s for s in SLOTS}
_COMPILED = {s.key: re.compile(s.rx, re.I) for s in SLOTS}

_YES = re.compile(r"^\s*(?:yes|yeah|yep|yup|yea|sure|i do|i have|i am|correct|definitely|certainly|a little|a bit|slightly|sometimes|somewhat|mild(?:ly)?|affirmative|ok yes)\b", re.I)
_NO = re.compile(r"^\s*(?:no|nope|nah|not really|none|never|i don'?t(?! know)|i do not(?! know)|i haven'?t|i have not|not at all|negative)\b", re.I)


def is_yes(text: str) -> bool:
    return bool(_YES.match(text)) and not _NO.match(text)


def is_no(text: str) -> bool:
    return bool(_NO.match(text))


def extract(text: str) -> dict[str, bool]:
    """Everything the text says about each symptom slot: True = has it, False = explicitly denies it, absent = not mentioned."""
    stripped = triage.strip_negated(text)
    out: dict[str, bool] = {}
    for slot in SLOTS:
        rx = _COMPILED[slot.key]
        if rx.search(stripped):
            out[slot.key] = True
        elif rx.search(text):
            out[slot.key] = False
    # "dry cough" settles whether there is phlegm; a band-like headache is not one-sided; a mild "itch" belongs to rashes only if a rash is mentioned
    if out.get("dry_cough") is True and "wet_cough" not in out:
        out["wet_cough"] = False
    if out.get("wet_cough") is True and out.get("dry_cough") is None:
        out["dry_cough"] = False
    if out.get("head_band") is True and "head_one_sided" not in out:
        out["head_one_sided"] = False
    if out.get("cough") is not True and (out.get("dry_cough") or out.get("wet_cough")):
        out["cough"] = True
    return out


def merge(known: dict[str, bool], new: dict[str, bool]) -> dict[str, bool]:
    """The latest statement wins ('no fever' earlier, 'now I have a fever' later)."""
    return {**known, **new}


def acknowledge(before: dict[str, bool], after: dict[str, bool], extra: list[str] | None = None, limit: int = 4) -> str:
    """'Noted: fever of 102°F, no chills.' - what was just understood, so the patient sees they were heard."""
    items = list(extra or [])
    have_fever_item = any(i.startswith("fever") for i in items)
    changed = [k for k, v in after.items() if before.get(k) != v]
    # positives first, then denials; red-flag screening denials are reassuring so they are kept
    order = {s.key: i for i, s in enumerate(SLOTS)}
    for k in sorted(changed, key=lambda k: (not after[k], order.get(k, 999))):
        if k == "fever" and after[k] and have_fever_item:
            continue
        items.append(BY_SLOT[k].yes if after[k] else BY_SLOT[k].no)
    items = [i for i in dict.fromkeys(items)][:limit]
    return ("Noted: " + ", ".join(items) + ".") if items else ""


def phrases(slots: dict[str, bool]) -> str:
    """Canonical symptom wording for the confirmed slots (appended to the symptom text so medicines / specialist see them)."""
    return "; ".join(BY_SLOT[k].phrase for k, v in slots.items() if v is True and k in BY_SLOT)


# ----------------------------------------------------------------------------------------------- derived facts
def derived(f: "triage.Facts") -> dict[str, bool]:
    """Slots computed from numbers rather than words."""
    d: dict[str, bool] = {}
    temp = f.temp_c if f.temp_c is not None else triage.parse_temperature_c(f.text)
    if temp is not None:
        d["fever"] = temp >= 38.0
        d["high_fever"] = temp >= 39.0
    if f.duration_days is not None:
        d["lasting_week"] = f.duration_days >= 7
        d["lasting_3weeks"] = f.duration_days >= 21
        d["lasting_few_days"] = 2 <= f.duration_days < 7
        d["sudden"] = f.duration_days <= 1
    if f.severity is not None:
        d["severe"] = f.severity >= 7
        d["mild"] = f.severity <= 3
    return d


def state(f: "triage.Facts") -> dict[str, bool]:
    """Everything known: what was said (words), explicit answers, and what numbers imply (numbers win for the fever)."""
    s = dict(f.slots)
    for k, v in derived(f).items():
        if k == "fever" and not v and s.get("fever") is True:
            continue                         # the patient says fever but the reading is borderline: keep what they said
        s[k] = v
    return s


# ----------------------------------------------------------------------------------------------- differential
@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    pos: dict               # slot -> weight when present
    neg: dict               # slot -> weight subtracted when present
    note: str               # what to tell the patient about it
    required_any: tuple = ()    # at least one of these slots must be True (the condition's defining symptom)
    min_hits: int = 2


CONDITIONS: tuple[Condition, ...] = (
    Condition("cold", "Common cold (viral upper respiratory infection)",
              {"runny_nose": 3, "sneezing": 2.5, "sore_throat": 2, "nasal_block": 2.5, "cough": 1.5, "mild": 1, "lasting_few_days": 1},
              {"high_fever": 3, "breath_short": 3, "chest_pain": 3, "severe": 1.5, "facial_pain": 1, "painful_swallow": 1},
              "Usually settles by itself in 7 to 10 days with rest and fluids.", required_any=("runny_nose", "nasal_block", "sore_throat", "sneezing")),
    Condition("flu", "Influenza-like illness (viral fever)",
              {"fever": 3, "high_fever": 1.5, "chills": 2.5, "body_ache": 3, "fatigue": 2, "cough": 1.5, "headache": 1.5, "sudden": 1.5, "sore_throat": 1},
              {"rash": 1.5, "retro_orbital": 1},
              "Usually improves in about a week; rest, fluids and fever control help.", required_any=("fever",)),
    Condition("allergic_rhinitis", "Allergic rhinitis (nasal allergy)",
              {"sneezing": 3, "itchy_nose_eyes": 3, "runny_nose": 2, "nasal_block": 1.5, "mild": 0.5},
              {"fever": 3.5, "high_fever": 2, "thick_discharge": 1.5, "facial_pain": 1, "painful_swallow": 1.5, "body_ache": 1.5},
              "Triggered by dust, pollen or pets; antihistamines and avoiding triggers help.", required_any=("sneezing", "itchy_nose_eyes", "runny_nose")),
    Condition("sinusitis", "Acute sinusitis",
              {"nasal_block": 2.5, "facial_pain": 3.5, "thick_discharge": 3, "double_worsening": 3, "lasting_week": 2, "fever": 1.5, "smell_loss": 1.5, "one_sided_nose": 0.5},
              {"itchy_nose_eyes": 1.5, "sneezing": 0.5},
              "Often viral; if it lasts over 10 days, or you get worse after improving, a doctor may treat it.", required_any=("nasal_block", "facial_pain", "thick_discharge")),
    Condition("strep", "Bacterial sore throat (e.g. strep throat)",
              {"sore_throat": 2.5, "painful_swallow": 2.5, "fever": 2, "swollen_glands": 3, "white_patches": 3.5, "high_fever": 1},
              {"cough": 2.5, "runny_nose": 2, "sneezing": 1.5, "nasal_block": 1},
              "Needs a throat examination (and sometimes a swab): only a doctor can decide on antibiotics.", required_any=("sore_throat",)),
    Condition("viral_pharyngitis", "Viral sore throat",
              {"sore_throat": 3, "cough": 2, "runny_nose": 2, "hoarse": 2, "mild": 1},
              {"white_patches": 2.5, "swollen_glands": 1.5, "high_fever": 1.5},
              "Usually settles in under a week; warm fluids and rest help.", required_any=("sore_throat",)),
    Condition("bronchitis", "Acute bronchitis (chest cold)",
              {"cough": 2.5, "wet_cough": 3, "sputum_coloured": 1.5, "wheeze": 1.5, "lasting_week": 1.5, "fever": 0.5},
              {"high_fever": 2, "breath_short": 1.5},
              "The cough can last 2 to 3 weeks; see a doctor if you become breathless or the fever is high.", required_any=("cough",)),
    Condition("pneumonia", "Chest infection / pneumonia (needs examination)",
              {"fever": 2.5, "high_fever": 1.5, "cough": 2, "wet_cough": 1.5, "sputum_coloured": 2, "breath_short": 4, "chest_pain": 2.5, "chills": 1.5, "fatigue": 1},
              {"runny_nose": 1.5, "sneezing": 1.5},
              "Fever with breathlessness or chest pain needs a doctor's examination, often with a chest X-ray.", required_any=("cough", "breath_short")),
    Condition("gastroenteritis", "Gastroenteritis (stomach infection / food-related)",
              {"vomiting": 3, "diarrhoea": 3, "abdominal_pain": 2, "nausea": 2, "fever": 1, "recent_food": 2.5},
              {"blood_stool": 1, "constant_pain": 1.5, "rlq_pain": 3},
              "Mostly settles in 2 to 3 days; the main task is to replace fluids (ORS).", required_any=("vomiting", "diarrhoea")),
    Condition("appendicitis", "Appendicitis (needs urgent examination)",
              {"abdominal_pain": 2, "rlq_pain": 5, "constant_pain": 2, "fever": 1.5, "vomiting": 1, "nausea": 1.5},
              {"diarrhoea": 1.5},
              "Pain moving to the lower right abdomen needs a doctor the same day.", required_any=("abdominal_pain",)),
    Condition("migraine", "Migraine",
              {"headache": 2, "head_one_sided": 3, "throbbing": 2.5, "light_sensitive": 3, "nausea": 2, "aura": 3, "vomiting": 1},
              {"fever": 2, "head_band": 1.5, "stiff_neck": 2},
              "Rest in a dark, quiet room; recurrent migraines are worth a doctor's review.", required_any=("headache",)),
    Condition("tension_headache", "Tension-type headache",
              {"headache": 2, "head_band": 3.5, "stress_trigger": 3, "mild": 1.5},
              {"nausea": 1.5, "throbbing": 1.5, "light_sensitive": 1.5, "fever": 2, "aura": 2},
              "Linked to stress, poor sleep or long screen time; rest, water and regular meals help.", required_any=("headache",)),
    Condition("uti", "Urinary tract infection",
              {"burning_urine": 4, "frequency": 3, "blood_urine": 1.5, "fever": 0.5},
              {"cough": 1, "runny_nose": 1},
              "A urine test confirms it; a doctor decides whether antibiotics are needed.", required_any=("burning_urine", "frequency")),
    Condition("kidney_infection", "Kidney infection (needs urgent examination)",
              {"burning_urine": 2, "frequency": 1.5, "flank_pain": 4, "fever": 3, "chills": 1.5, "vomiting": 1, "nausea": 1},
              {},
              "Fever with side/back pain and urinary symptoms needs a doctor the same day.", required_any=("flank_pain",)),
    Condition("dengue_like", "Dengue or other tropical viral fever",
              {"fever": 2.5, "high_fever": 2, "body_ache": 3, "retro_orbital": 3, "headache": 1.5, "rash": 2, "mosquito_exposure": 2.5, "fatigue": 1.5, "lasting_few_days": 1, "nausea": 1},
              {"runny_nose": 2, "sneezing": 1.5, "cough": 1.5, "sore_throat": 1},
              "A blood test (NS1 / platelets) is needed; avoid ibuprofen/aspirin and use paracetamol until a doctor has checked.", required_any=("fever",)),
    Condition("reflux", "Acid reflux / gastritis",
              {"heartburn": 4, "after_meals": 3, "abdominal_pain": 1.5, "bloating": 1.5, "nausea": 1},
              {"fever": 2, "diarrhoea": 1.5, "blood_stool": 1},
              "Smaller meals, avoiding spicy food and not lying down after meals help; antacids give short-term relief.", required_any=("heartburn",)),
    Condition("strain", "Muscle or joint strain",
              {"injury": 3.5, "back_pain": 2, "joint_pain": 2, "limb_swelling": 1.5},
              {"fever": 2.5, "morning_stiffness": 1.5, "leg_numbness": 1.5},
              "Rest, gentle movement and a short course of simple pain relief usually settle it.", required_any=("injury", "back_pain", "joint_pain")),
    Condition("arthritis", "Arthritis / inflammatory joint pain",
              {"joint_pain": 3, "morning_stiffness": 4, "limb_swelling": 2, "lasting_week": 1.5},
              {"injury": 2.5},
              "Joint pain with long morning stiffness is worth a doctor's review.", required_any=("joint_pain",)),
    Condition("contact_dermatitis", "Contact or allergic skin reaction",
              {"rash": 2.5, "itchy_skin": 3, "new_product": 3.5},
              {"fever": 2, "blisters": 0.5},
              "Stop the suspected product; a mild steroid cream and an antihistamine often help.", required_any=("rash",)),
    Condition("viral_rash", "Viral rash illness (needs examination if with fever)",
              {"rash": 3, "fever": 3, "rash_spreading": 2, "body_ache": 1, "fatigue": 1},
              {"itchy_skin": 1, "new_product": 2},
              "A rash with fever should be seen by a doctor.", required_any=("rash",)),
)


def differential(f: "triage.Facts", limit: int = 3, floor: float = 0.3) -> list[dict]:
    """Conditions that fit what we know, best first: [{name, key, score, because, note}]. Possibilities, not a diagnosis."""
    st = state(f)
    out = []
    for c in CONDITIONS:
        hits = [k for k in c.pos if st.get(k) is True]
        if c.required_any and not any(st.get(k) is True for k in c.required_any):
            continue
        if len([k for k in hits if k in BY_SLOT]) < c.min_hits:        # symptoms the patient reported, not just numbers like 'mild'
            continue
        raw = sum(c.pos[k] for k in hits) - sum(w for k, w in c.neg.items() if st.get(k) is True)
        raw -= 0.25 * sum(c.pos[k] for k in c.pos if st.get(k) is False)           # something they explicitly do not have
        top = sorted(c.pos.values(), reverse=True)
        denom = 0.75 * sum(top[:6])                                                  # a patient rarely has every feature
        score = max(0.0, min(1.0, raw / denom)) if denom else 0.0
        if score >= floor:
            out.append({"key": c.key, "name": c.name, "score": round(score, 2),
                        "because": [BY_SLOT[k].yes for k in hits if k in BY_SLOT][:5], "note": c.note})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:limit]


def scores(f: "triage.Facts") -> dict[str, float]:
    return {d["key"]: d["score"] for d in differential(f, limit=20, floor=0.0)}


# ----------------------------------------------------------------------------------------------- red flags from answers
RED_FLAGS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("chest_pain",), "emergency", "chest pain"),
    (("worst_headache",), "emergency", "a sudden, worst-ever headache"),
    (("neuro_deficit",), "emergency", "new weakness, numbness, or speech/vision trouble"),
    (("angioedema",), "emergency", "swelling of the lips, face or tongue"),
    (("airway_swelling",), "urgent", "trouble opening the mouth or drooling with a sore throat"),
    (("fever", "stiff_neck"), "urgent", "fever with a stiff neck"),
    (("blood_sputum",), "urgent", "coughing up blood"),
    (("blood_stool",), "urgent", "blood or black stools"),
    (("blood_urine",), "urgent", "blood in the urine"),
    (("rlq_pain",), "urgent", "pain in the lower right abdomen"),
    (("fever", "flank_pain"), "urgent", "fever with pain in the side or back (possible kidney infection)"),
    (("bladder_bowel",), "urgent", "trouble passing urine or stool with back pain"),
    (("breath_short", "fever"), "urgent", "fever with shortness of breath (possible chest infection)"),
    (("blisters", "rash", "fever"), "urgent", "a blistering rash with fever"),
)


def red_flags(f: "triage.Facts") -> list[tuple[str, str]]:
    """[(level, label)] for the answers that are warning signs. 'can_keep_fluids' being False is also one."""
    st = state(f)
    out = [(level, label) for need, level, label in RED_FLAGS if all(st.get(n) is True for n in need)]
    if st.get("can_keep_fluids") is False and (st.get("vomiting") or st.get("diarrhoea")):
        out.append(("urgent", "unable to keep fluids down"))
    if st.get("lasting_3weeks") and st.get("tb_symptoms"):
        out.append(("urgent", "a cough lasting over 3 weeks with weight loss or night sweats"))
    return out
