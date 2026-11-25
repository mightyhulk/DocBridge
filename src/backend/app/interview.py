"""The digital-doctor interview: decide, at every turn, the single most useful question to ask next.

How it decides (all transparent rules, no model, so it is instant on a Raspberry Pi):

1. What do we know?  Every symptom the patient mentioned becomes a slot: has it / explicitly does not have it / unknown
   (clinic.extract). "I have a fever of 102 degree fahrenheit" fills the fever AND the temperature, so neither is asked.
2. Which conditions fit so far?  clinic.differential scores a short list of common conditions from those slots.
3. Which question is worth most?  A question is a candidate only if (a) its symptom area is active, (b) its prerequisites hold
   (no cough question about phlegm unless there is a cough), and (c) it is still unknown. Its value is its clinical weight
   (red-flag screens are heavy) plus a bonus for separating the conditions currently in play (sinus pain is worth more when
   sinusitis and a cold are both likely). The best one is asked; the plan is recomputed after the answer, so the conversation
   branches on what the patient says.
4. When to stop?  Red flags are asked first; after about seven symptom questions, or when nothing left is worth asking, the
   interview moves on to age/conditions and the medicine-safety question, then the assessment.

Each question carries a one-line "why I am asking", like a doctor explaining themselves.
"""
import re
from dataclasses import dataclass

from . import clinic, triage
from .triage import Facts

MAX_SYMPTOM_QUESTIONS = 7
MIN_UTILITY = 5.0
HYP_BONUS = 3.0

Q_DURATION = "How long has this been going on, and is it getting better, worse, or staying the same?"
Q_SEVERITY = "On a scale of 0 (no problem) to 10 (worst imaginable), how bad is it right now?"
Q_PROFILE = "How old are you, and do you have any existing conditions (such as diabetes, heart or lung disease), or are you pregnant?"
Q_MEDS = ("Before I suggest any medicine: are you allergic to any medicines, and do you take any regular medicines "
          "(for example blood thinners, aspirin, or tablets for blood pressure, diabetes or depression)? Say 'none' if not.")
Q_OPEN = "Can you tell me a bit more? Where exactly do you feel it, and what other symptoms have you noticed?"

BASIC_WHY = {
    "duration": "How long it has lasted, and which way it is heading, tells me how worried to be.",
    "severity": "How bad it feels helps me decide how soon you should be seen.",
    "profile": "Your age and health conditions change how urgent this is and which medicines are safe.",
    "meds": "I need this before suggesting any medicine, to keep you safe.",
    "open": "I could not place your symptoms yet, so I want to understand them better.",
}


@dataclass(frozen=True)
class Q:
    id: str
    areas: tuple[str, ...]            # symptom areas in which this question makes sense ('*' = any active area)
    text: str
    why: str
    weight: float                     # clinical value: red-flag screens 9-10, key discriminators 6-8, details 4-5
    slot: str | None = None           # the clinic slot the answer fills (None: free-text topic, see `said`)
    needs: tuple[tuple[str, bool], ...] = ()      # slot conditions that must all hold
    any_of: tuple[str, ...] = ()      # at least one of these slots must be True
    hyps: tuple[str, ...] = ()        # conditions this question helps separate
    said: str | None = None           # for slot-less questions: regex over the raw text meaning "already told us"
    yn: bool = True                   # False for either/or questions where a bare "yes" says nothing ("dry, or with phlegm?")


T = True
CATALOG: tuple[Q, ...] = (
    # ------------------------------------------------------------------ fever
    Q("temp", ("fever",), "Have you measured your temperature? What was the highest reading (in °F or °C)?",
      "The actual reading tells me how significant the fever is.", 9.0, needs=(("fever", T),)),
    Q("stiff_neck", ("fever", "head"), "Is your neck stiff, or does it hurt to bend your head forward towards your chest?",
      "With a fever and headache I need to rule out an infection around the brain.", 10, "stiff_neck", needs=(("fever", T), ("headache", T))),
    Q("chills", ("fever",), "Do you have chills or shivering with the fever?", "Chills with fever point to an infection such as flu.",
      6.0, "chills", needs=(("fever", T),), hyps=("flu", "dengue_like", "pneumonia", "kidney_infection")),
    Q("body_ache", ("fever",), "Do you have body aches or joint pains?", "Marked body aches help tell flu or dengue from a simple cold.",
      6.5, "body_ache", needs=(("fever", T),), hyps=("flu", "dengue_like")),
    Q("fever_rash", ("fever",), "Have you noticed any rash on your skin?", "A rash with fever can point to dengue, measles or a drug reaction.",
      6.5, "rash", needs=(("fever", T),), hyps=("dengue_like", "viral_rash")),
    Q("retro_orbital", ("fever",), "Do you have pain behind your eyes, especially when you move them?", "Pain behind the eyes is typical of dengue.",
      6.0, "retro_orbital", needs=(("fever", T),), any_of=("body_ache", "headache", "rash", "high_fever"), hyps=("dengue_like",)),
    Q("mosquito", ("fever",), "Have there been dengue or malaria cases near you, or have you had many mosquito bites recently?",
      "This matters for dengue and malaria, which are common in some seasons and areas.", 5.0, "mosquito_exposure",
      needs=(("fever", T),), any_of=("body_ache", "retro_orbital", "rash", "high_fever"), hyps=("dengue_like",)),
    Q("fatigue", ("fever",), "Do you feel unusually tired or weak?", "Marked tiredness is common with flu and dengue.",
      4.5, "fatigue", needs=(("fever", T),), hyps=("flu", "dengue_like")),
    Q("fever_check", ("*",), "Do you have a fever?", "Fever changes how I think about almost every symptom.", 7.0, "fever"),
    # ------------------------------------------------------------------ nose
    Q("runny_nose", ("nose", "throat"), "Is your nose also running?", "A runny nose together with these symptoms usually points to a cold or allergy.",
      7.0, "runny_nose", hyps=("cold", "allergic_rhinitis", "sinusitis")),
    Q("sneezing", ("nose",), "Do you sneeze a lot, especially in the morning or around dust, pollen or pets?", "Repeated sneezing suggests an allergy.",
      6.5, "sneezing", hyps=("allergic_rhinitis", "cold")),
    Q("itchy_nose_eyes", ("nose",), "Do your nose, eyes or throat feel itchy, or do your eyes water?", "Itching is the hallmark of an allergy rather than an infection.",
      6.0, "itchy_nose_eyes", hyps=("allergic_rhinitis",)),
    Q("thick_discharge", ("nose",), "Is the mucus thick and yellow or green?", "Thick coloured mucus makes a sinus infection more likely.",
      6.5, "thick_discharge", any_of=("nasal_block", "runny_nose"), hyps=("sinusitis", "cold")),
    Q("facial_pain", ("nose",), "Do you feel pain or pressure in your face: around the eyes, cheeks or forehead, or worse when you bend forward?",
      "Facial pressure is the key sign of sinusitis.", 7.0, "facial_pain", needs=(("nasal_block", T),), hyps=("sinusitis",)),
    Q("double_worsening", ("nose",), "Did you start to get better and then become worse again?", "Getting worse after improving suggests a bacterial sinus infection.",
      5.0, "double_worsening", needs=(("nasal_block", T), ("lasting_week", T)), hyps=("sinusitis",)),
    Q("smell_loss", ("nose",), "Have you lost your sense of smell or taste?", "Loss of smell can accompany sinus problems and some viral infections.",
      5.0, "smell_loss", any_of=("nasal_block", "runny_nose"), hyps=("sinusitis", "cold")),
    Q("one_sided_nose", ("nose",), "Is the blockage only on one side?", "Blockage on one side only can mean a structural cause such as a deviated septum or a polyp.",
      5.0, "one_sided_nose", needs=(("nasal_block", T),)),
    # ------------------------------------------------------------------ throat
    Q("cough_check", ("nose", "throat", "fever"), "Do you have a cough?", "A cough with a sore throat suggests a viral cause; no cough makes a bacterial throat infection more likely.",
      5.5, "cough", any_of=("sore_throat", "runny_nose", "nasal_block", "fever"), hyps=("cold", "viral_pharyngitis", "strep", "bronchitis", "pneumonia")),
    Q("painful_swallow", ("throat",), "Is it painful to swallow?", "Pain on swallowing helps separate a bad throat infection from a scratchy throat.",
      7.5, "painful_swallow", needs=(("sore_throat", T),), hyps=("strep", "viral_pharyngitis")),
    Q("swollen_glands", ("throat",), "Do you feel swollen, tender glands in your neck?", "Tender neck glands make a bacterial throat infection more likely.",
      6.5, "swollen_glands", needs=(("sore_throat", T),), hyps=("strep",)),
    Q("white_patches", ("throat",), "Can you see white patches or spots on your tonsils? (Use a torch and a mirror.)", "White patches on the tonsils point to a bacterial infection.",
      6.5, "white_patches", needs=(("sore_throat", T),), hyps=("strep",)),
    Q("airway_swelling", ("throat",), "Do you have trouble opening your mouth, drooling, or a muffled voice?", "These can mean a deep throat infection that needs urgent care.",
      9.5, "airway_swelling", needs=(("sore_throat", T), ("painful_swallow", T))),
    Q("hoarse", ("throat",), "Has your voice become hoarse?", "A hoarse voice points to irritation of the voice box, usually from a virus.",
      4.5, "hoarse", needs=(("sore_throat", T),), hyps=("viral_pharyngitis",)),
    # ------------------------------------------------------------------ chest and breathing
    Q("cough_kind", ("chest",), "Is your cough dry, or are you bringing up phlegm?", "Dry and productive coughs have different causes.",
      8.2, "wet_cough", needs=(("cough", T),), hyps=("bronchitis", "pneumonia", "cold"), yn=False),
    Q("sputum_colour", ("chest",), "What colour is the phlegm: clear or white, yellow-green, or brownish?", "Coloured phlegm can mean a bacterial infection of the chest.",
      5.5, "sputum_coloured", needs=(("wet_cough", T),), hyps=("bronchitis", "pneumonia"), yn=False),
    Q("blood_sputum", ("chest",), "Have you coughed up any blood?", "Blood in the phlegm always needs a doctor's attention.", 8.0, "blood_sputum", needs=(("cough", T),)),
    Q("breath_short", ("chest",), "Are you short of breath, for example when walking a short distance, or even at rest?", "Breathlessness is the main sign that a chest infection may be serious.",
      8.5, "breath_short", any_of=("cough", "wheeze", "chest_pain"), hyps=("pneumonia", "bronchitis")),
    Q("wheeze", ("chest",), "Do you hear wheezing or a whistling sound when you breathe out?", "Wheezing suggests airway narrowing, as in bronchitis or asthma.",
      6.0, "wheeze", needs=(("cough", T),), hyps=("bronchitis",)),
    Q("chest_pain", ("chest",), "Do you have any chest pain, especially when breathing in or coughing?", "Chest pain needs to be ruled out as a warning sign.",
      10, "chest_pain", any_of=("cough", "breath_short")),
    Q("tb_symptoms", ("chest",), "Have you had weight loss, night sweats or loss of appetite?", "A cough lasting weeks with these needs tests for conditions like tuberculosis.",
      7.0, "tb_symptoms", needs=(("cough", T), ("lasting_3weeks", T))),
    # ------------------------------------------------------------------ headache
    Q("worst_headache", ("head",), "Did the headache start suddenly, and is it the worst headache you have ever had?", "A sudden, worst-ever headache can be an emergency.",
      10, "worst_headache", needs=(("headache", T),)),
    Q("neuro_deficit", ("head",), "Do you have any weakness, numbness, trouble speaking, or loss of vision?", "These would point to a problem with the brain's blood supply or nerves.",
      10, "neuro_deficit", needs=(("headache", T),)),
    Q("head_side", ("head",), "Is the pain on one side of your head, or all over or like a tight band?", "One-sided pain suggests migraine; a band-like ache suggests a tension headache.",
      6.5, "head_one_sided", needs=(("headache", T),), hyps=("migraine", "tension_headache"), yn=False),
    Q("throbbing", ("head",), "Is the pain throbbing or pulsating?", "Throbbing pain is typical of migraine.",
      5.5, "throbbing", needs=(("headache", T),), hyps=("migraine",)),
    Q("light_sensitive", ("head",), "Do bright lights or loud sounds make it worse?", "Sensitivity to light and sound is typical of migraine.",
      6.0, "light_sensitive", needs=(("headache", T),), hyps=("migraine",)),
    Q("head_nausea", ("head",), "Do you feel nauseous, or have you vomited with the headache?", "Nausea with headache points to migraine.",
      5.5, "nausea", needs=(("headache", T),), hyps=("migraine",)),
    Q("aura", ("head",), "Did you see flashing lights or zigzag lines before the headache began?", "Visual aura is specific to migraine.",
      5.0, "aura", needs=(("headache", T), ("head_one_sided", T)), hyps=("migraine",)),
    Q("stress_trigger", ("head",), "Have you been under stress, short of sleep, skipping meals or looking at screens for long hours?", "These are the usual triggers of tension headaches.",
      5.0, "stress_trigger", needs=(("headache", T),), hyps=("tension_headache",)),
    # ------------------------------------------------------------------ stomach and gut
    Q("vomiting", ("gut",), "Have you vomited?", "Vomiting together with these symptoms suggests a stomach infection.",
      6.5, "vomiting", any_of=("abdominal_pain", "nausea", "diarrhoea"), hyps=("gastroenteritis",)),
    Q("diarrhoea", ("gut",), "Do you have loose motions?", "Loose motions with vomiting or cramps suggest gastroenteritis.",
      6.5, "diarrhoea", any_of=("abdominal_pain", "nausea", "vomiting"), hyps=("gastroenteritis",)),
    Q("abdominal_pain", ("gut",), "Do you have stomach pain or cramps?", "Where the pain is and what it feels like narrows down the cause.",
      6.0, "abdominal_pain", any_of=("vomiting", "diarrhoea", "nausea"), hyps=("gastroenteritis", "appendicitis")),
    Q("blood_stool", ("gut",), "Have you seen blood in your stool, or black, tarry stools?", "Blood in the stool needs a doctor's attention.",
      9.5, "blood_stool", any_of=("abdominal_pain", "diarrhoea", "vomiting")),
    Q("can_keep_fluids", ("gut",), "Are you able to keep fluids down, and are you passing urine normally?", "Dehydration is the main danger with vomiting and loose motions.",
      9.0, "can_keep_fluids", any_of=("vomiting", "diarrhoea")),
    Q("rlq_pain", ("gut",), "Is the pain mainly in the lower right part of your belly, and does it get worse when you move or cough?", "That pattern can mean appendicitis, which needs urgent examination.",
      8.5, "rlq_pain", needs=(("abdominal_pain", T),), hyps=("appendicitis",)),
    Q("constant_pain", ("gut",), "Is the pain constant and severe, or does it come and go in waves of cramps?", "Constant severe pain is more worrying than cramps that come and go.",
      6.0, "constant_pain", needs=(("abdominal_pain", T),), hyps=("appendicitis", "gastroenteritis"), yn=False),
    Q("recent_food", ("gut",), "Did you eat outside food or anything unusual in the last day or two?", "Recent unusual food makes food poisoning more likely.",
      5.0, "recent_food", any_of=("vomiting", "diarrhoea"), hyps=("gastroenteritis",)),
    Q("after_meals", ("gut",), "Is it worse after meals, when lying down, or when bending forward?", "Pain that follows meals or lying down points to acid reflux.",
      6.5, "after_meals", needs=(("heartburn", T),), hyps=("reflux",)),
    # ------------------------------------------------------------------ urine
    Q("burning_urine", ("urine",), "Is there burning or pain when you pass urine?", "Burning on passing urine is the key sign of a urinary infection.",
      7.5, "burning_urine", any_of=("frequency", "flank_pain", "blood_urine"), hyps=("uti",)),
    Q("frequency", ("urine",), "Are you passing urine more often than usual?", "Frequent urination goes with a urinary infection.",
      6.5, "frequency", any_of=("burning_urine", "flank_pain"), hyps=("uti",)),
    Q("blood_urine", ("urine",), "Have you noticed blood in your urine?", "Blood in the urine needs a doctor's attention.",
      9.0, "blood_urine", any_of=("burning_urine", "frequency", "flank_pain")),
    Q("flank_pain", ("urine",), "Do you have pain in your side or lower back?", "Pain in the side with urinary symptoms can mean the kidney is involved.",
      7.5, "flank_pain", any_of=("burning_urine", "frequency"), hyps=("kidney_infection",)),
    # ------------------------------------------------------------------ skin
    Q("angioedema", ("skin",), "Is there any swelling of your lips, face or tongue, or any trouble breathing?", "Swelling of the face or breathing trouble with a rash can be a severe allergic reaction.",
      10, "angioedema", needs=(("rash", T),)),
    Q("itchy_skin", ("skin",), "Is the rash itchy?", "An itchy rash points more to an allergy or irritation.",
      5.5, "itchy_skin", needs=(("rash", T),), hyps=("contact_dermatitis",)),
    Q("rash_spreading", ("skin",), "Is the rash spreading?", "A spreading rash needs more attention than a stable one.",
      6.5, "rash_spreading", needs=(("rash", T),), hyps=("viral_rash",)),
    Q("blisters", ("skin",), "Are there blisters, peeling skin or open sores?", "Blistering or peeling can signal a more serious skin reaction.",
      7.0, "blisters", needs=(("rash", T),)),
    Q("new_product", ("skin",), "Did it start after a new soap, cream, medicine or food?", "A new product or medicine is the most common cause of a rash.",
      6.5, "new_product", needs=(("rash", T),), hyps=("contact_dermatitis",)),
    # ------------------------------------------------------------------ muscles and bones
    Q("bladder_bowel", ("msk",), "Any trouble passing urine or stool, or numbness around the groin?", "With back pain these are signs of nerve compression that need urgent care.",
      10, "bladder_bowel", needs=(("back_pain", T),)),
    Q("leg_numbness", ("msk",), "Do you have numbness, tingling or weakness in your legs or arms?", "Numbness or weakness suggests a nerve is affected.",
      9.0, "leg_numbness", needs=(("back_pain", T),)),
    Q("injury", ("msk",), "Did it start after an injury, a fall or heavy lifting?", "An injury makes a muscle or ligament strain, or a fracture, more likely.",
      7.5, "injury", any_of=("joint_pain", "back_pain"), hyps=("strain", "arthritis")),
    Q("limb_swelling", ("msk",), "Is the joint swollen or warm to the touch?", "A swollen, warm joint suggests inflammation or infection.",
      5.5, "limb_swelling", needs=(("joint_pain", T),), hyps=("arthritis", "strain")),
    Q("morning_stiffness", ("msk",), "Is it stiff for a long time in the morning (more than 30 minutes)?", "Long morning stiffness suggests inflammatory arthritis.",
      5.5, "morning_stiffness", needs=(("joint_pain", T),), hyps=("arthritis",)),
    # ------------------------------------------------------------------ areas without slots yet (one good question each)
    Q("ear_detail", ("ear",), "Is there ear discharge, reduced hearing or dizziness? Any recent swimming or ear cleaning?", "These separate an ear infection from blocked wax or a throat problem.",
      8.0, said=r"discharge|dizz\w*|swim\w*|cleaning|hearing (?:loss|is|has)|blocked ear"),
    Q("eye_detail", ("eye",), "Is there eye pain, discharge, a sudden change in vision, or sensitivity to light?", "Pain or vision change in the eye can need urgent care.",
      8.0, said=r"pain\w*|discharge|sudden|light|sticky|redness"),
    Q("dental_detail", ("dental",), "Is there swelling of the face or jaw, fever, pus, or trouble opening your mouth or swallowing?", "Facial swelling with a tooth problem can mean a spreading infection.",
      8.0, said=r"swell\w*|fever|swallow\w*|open\w*|\bpus\b|hot|cold|bite"),
    Q("mh_detail", ("mental",), "Have you had thoughts of harming yourself, and how much is this affecting your daily life, sleep or work?", "I ask everyone this: your safety comes first.",
      9.0, said=r"harm\w*|suicid\w*|end my life|daily|work|sleep\w*|appetite|hopeless"),
    Q("gyn_detail", ("gyn",), "Is there any chance you are pregnant? Any heavy bleeding, fever, discharge or severe pelvic pain?", "Pregnancy changes both the cause and which medicines are safe.",
      8.0, said=r"pregnan\w*|bleed\w*|heavy|fever|discharge|severe|missed"),
)
BY_ID = {q.id: q for q in CATALOG}

# Either/or questions ("dry, or with phlegm?"): a bare yes/no says nothing, but the words of the answer do.
EITHER_OR: dict[str, list[tuple[str, dict[str, bool]]]] = {
    "cough_kind": [(r"\bdry\b|tickl\w*|no phlegm|without phlegm", {"dry_cough": True, "wet_cough": False}),
                   (r"phlegm|wet|productive|mucus|sputum|chesty|bring(?:ing)? (?:it )?up", {"wet_cough": True, "dry_cough": False})],
    "sputum_colour": [(r"yellow|green|brown|rust|dark|thick", {"sputum_coloured": True}), (r"clear|white|transparent|colou?rless", {"sputum_coloured": False})],
    "head_side": [(r"one side|left|right|temple|unilateral|behind (?:the |my )?eye", {"head_one_sided": True}),
                  (r"all over|both|band|around|whole|everywhere|front|forehead|tight", {"head_one_sided": False})],
    "constant_pain": [(r"constant|severe|sharp|continuous|non-?stop|all the time", {"constant_pain": True}),
                      (r"comes? and goes?|cramp\w*|waves?|colicky|on and off|intermittent", {"constant_pain": False})],
}


def either_or(question_id: str, text: str) -> dict[str, bool]:
    """What the words of an answer to an either/or question settle: 'it is dry' -> no phlegm; 'all over my head' -> not one-sided."""
    for rx, update in EITHER_OR.get(question_id, []):
        if re.search(rx, text, re.I):
            return dict(update)
    return {}

# symptom areas -> slots that switch the area on; slot-less areas use a regex over the (non-negated) text
AREA_SLOTS = {
    "fever": ("fever",),
    "nose": ("nasal_block", "runny_nose", "sneezing", "itchy_nose_eyes", "thick_discharge", "facial_pain", "smell_loss"),
    "throat": ("sore_throat", "painful_swallow", "hoarse", "swollen_glands"),
    "chest": ("cough", "dry_cough", "wet_cough", "breath_short", "wheeze", "chest_pain"),
    "head": ("headache",),
    "gut": ("abdominal_pain", "vomiting", "diarrhoea", "nausea", "heartburn", "blood_stool", "bloating"),
    "urine": ("burning_urine", "frequency", "blood_urine", "flank_pain"),
    "skin": ("rash", "itchy_skin", "blisters"),
    "msk": ("joint_pain", "back_pain", "injury", "limb_swelling"),
}
AREA_REGEX = {
    "ear": r"\bears?\b|earache|ringing in",
    "eye": r"\beyes?\b|eyesight|red eye|vision",
    "dental": r"tooth|teeth|\bgums?\b|\bjaw\b|toothache",
    "mental": r"anxiety|anxious|depress\w*|panic|insomnia|can'?t sleep|low mood|hopeless|self[- ]harm",
    "gyn": r"period|menstru\w*|pregnan\w*|vagin\w*|pelvic|breast",
}


def active_areas(f: Facts) -> list[str]:
    st = clinic.state(f)
    areas = [a for a, slots in AREA_SLOTS.items() if any(st.get(s) is True for s in slots)]
    if "gut" in areas and st.get("headache") is True and not any(st.get(s) is True for s in ("abdominal_pain", "vomiting", "diarrhoea", "heartburn", "blood_stool")):
        areas.remove("gut")                  # nausea alongside a headache is part of the headache picture, not a stomach problem
    text = triage.strip_negated(f.text)
    areas += [a for a, rx in AREA_REGEX.items() if re.search(rx, text, re.I)]
    return areas


def _applicable(q: Q, st: dict, areas: list[str], f: Facts) -> bool:
    if "*" in q.areas:
        if not areas:
            return False
    elif not set(q.areas) & set(areas):
        return False
    if any(st.get(slot) is not want for slot, want in q.needs):
        return False
    if q.any_of and not any(st.get(s) is True for s in q.any_of):
        return False
    return True


def _known(q: Q, st: dict, f: Facts) -> bool:
    """Has the patient already told us what this question asks (a value, a yes or a no)?"""
    if q.id == "temp":
        return f.temp_c is not None or st.get("fever") is False
    if q.slot:
        return st.get(q.slot) is not None
    return bool(q.said and re.search(q.said, f.text, re.I))


def candidates(f: Facts, done: set[str]) -> list[tuple[float, Q]]:
    st = clinic.state(f)
    areas = active_areas(f)
    hyp = clinic.scores(f)
    out: list[tuple[float, Q]] = []
    for i, q in enumerate(CATALOG):
        if q.id in done or not _applicable(q, st, areas, f) or _known(q, st, f):
            continue
        bonus = HYP_BONUS * max((hyp.get(h, 0.0) for h in q.hyps), default=0.0)
        util = q.weight + bonus - i * 1e-4                  # catalog order breaks ties
        if util >= MIN_UTILITY:
            out.append((util, q))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def plan(f: Facts, done: set[str]) -> list[tuple[str, str, str]]:
    """The questions still worth asking, best first: [(id, question, why)]. The first is asked next, then the plan is rebuilt."""
    steps: list[tuple[str, str, str]] = []
    areas = active_areas(f)
    if not areas and "open" not in done:
        steps.append(("open", Q_OPEN, BASIC_WHY["open"]))
    if "duration" not in done and f.duration_days is None:
        steps.append(("duration", Q_DURATION, BASIC_WHY["duration"]))
    if "severity" not in done and f.severity is None:
        steps.append(("severity", Q_SEVERITY, BASIC_WHY["severity"]))
    asked = len([d for d in done if d in BY_ID])
    room = max(0, MAX_SYMPTOM_QUESTIONS - asked)
    steps += [(q.id, q.text, q.why) for _, q in candidates(f, done)[:min(room, 4)]]
    if "profile" not in done and f.age is None:
        steps.append(("profile", Q_PROFILE, BASIC_WHY["profile"]))
    if "meds" not in done and not f.allergies_asked:
        steps.append(("meds", Q_MEDS, BASIC_WHY["meds"]))
    return steps
