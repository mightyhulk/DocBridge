"""Over-the-counter medicine suggestions with a safety screen. Deterministic: doses come from this table, never from a model.

What this is: the standard adult OTC label doses of a handful of common medicines (paracetamol, ibuprofen, cetirizine, saline
drops, ORS, antacids, hydrocortisone 1%), chosen from the patient's symptoms and then SCREENED against what they told us
(age, pregnancy, conditions, allergies, regular medicines, fever duration). A medicine that is not safe for them is not listed;
it is shown in an "avoid" list with the reason.

What this is not: a prescription. Prescription-only medicines (antibiotics, steroid tablets, strong painkillers ...) are never
suggested: only a doctor who has examined the patient can decide on them. Children get no doses (they depend on weight).
Emergency cases get no medicines; urgent cases get paracetamol at most.

Dose sources: NHS medicines guides / US OTC drug labels for adult dosing.
"""
import re
from dataclasses import dataclass

from . import triage
from .triage import Assessment, Facts

DISCLAIMER = ("General over-the-counter information using standard label doses, screened against what you told me. It is not a "
              "prescription. Please confirm with a pharmacist or doctor, especially if you are pregnant, take other medicines, "
              "or have long-term conditions.")
RX_NOTE = ("Antibiotics, steroid tablets and other prescription medicines are not suggested here. Only a doctor who has examined you "
           "can decide whether you need them.")
CHILD_NOTE = ("Medicine doses for children depend on their weight and age, so I will not give doses. Please ask a pharmacist or a "
              "paediatrician, and never use adult doses.")


@dataclass(frozen=True)
class Drug:
    key: str
    name: str
    treats: str                 # regex over the patient's (non-negated) symptoms
    use: str
    dose: str
    maximum: str
    tips: tuple[str, ...] = ()
    needs_dose: bool = True     # False: no numeric dose to restrict (saline, ORS, salt-water gargles)


FORMULARY: tuple[Drug, ...] = (
    Drug("paracetamol", "Paracetamol (acetaminophen)",
         r"fever|temperature|chills|\bpain|ache|headache|migraine|sore throat|toothache|\bcold\b|\bflu\b|cramps?|muscle|joint|back pain|body ache",
         "fever and mild to moderate pain",
         "500 mg to 1 g (1-2 tablets of 500 mg) every 4 to 6 hours only if needed",
         "No more than 4 g (8 tablets of 500 mg) in 24 hours, with at least 4 hours between doses",
         ("Many cold-and-flu tablets already contain paracetamol: do not combine them.",
          "If you drink alcohol regularly or weigh under 50 kg, ask a pharmacist for a lower limit.",
          "See a doctor if the fever or pain lasts more than 3 days.")),
    Drug("ibuprofen", "Ibuprofen",
         r"\bpain|ache|headache|migraine|toothache|muscle|joint|back pain|cramps?|sore throat",
         "pain and inflammation (headache, muscle or joint pain, toothache, period cramps)",
         "200 mg to 400 mg every 6 to 8 hours, always with or after food",
         "No more than 1,200 mg in 24 hours without a doctor's advice; stop after 5 days of use",
         ("Take it with food and a glass of water.", "Stop and see a doctor if you get stomach pain, black stools or vomiting.")),
    Drug("cetirizine", "Cetirizine (antihistamine)",
         r"runny nose|sneez\w*|itch\w*|allerg\w*|hives|watering eyes|watery eyes|hay fever|pollen|dust",
         "allergy symptoms: sneezing, runny nose, itching, hives",
         "10 mg once a day, usually at night",
         "One 10 mg tablet in 24 hours",
         ("It can cause drowsiness: avoid driving and alcohol.", "If symptoms last more than 2 weeks, see a doctor.")),
    Drug("saline", "Saline (salt-water) nose drops or spray",
         r"nose|nasal|blocked|blockage|congest\w*|sinus|stuffy|runny",
         "a blocked or runny nose",
         "2 to 3 drops or 1 to 2 sprays in each nostril as often as needed",
         "No fixed limit; use plain saline only (not medicated sprays)",
         ("Steam inhalation for 5 to 10 minutes also helps.", "Do not use decongestant nose sprays for more than 3 days."), needs_dose=False),
    Drug("gargle", "Warm salt-water gargle and throat lozenges",
         r"sore throat|throat pain|scratchy throat|throat",
         "a sore throat",
         "Half a teaspoon of salt in a glass of warm water, gargle several times a day; lozenges as directed on the pack",
         "Lozenges: follow the pack (not for children under 6)",
         ("Warm fluids and honey also soothe the throat (honey: not under 1 year).",), needs_dose=False),
    Drug("ors", "Oral rehydration solution (ORS)",
         r"diarrh\w*|loose (?:motion|stool)s?|vomit\w*|dehydrat\w*",
         "fluid loss from diarrhoea or vomiting",
         "Sip ORS after every loose stool or vomit; mix one sachet in exactly the amount of clean water printed on the packet",
         "No fixed limit: keep sipping small amounts often",
         ("Go to a doctor if you cannot keep fluids down, have blood in the stool, or feel very drowsy.",), needs_dose=False),
    Drug("antacid", "Antacid (aluminium/magnesium hydroxide)",
         r"acidity|heartburn|indigestion|reflux|sour|\bgas\b|bloat\w*|burning in (?:the )?(?:chest|stomach)",
         "heartburn, acidity and indigestion",
         "1 to 2 tablets (or 10 to 20 mL) after meals and at bedtime, as printed on the pack",
         "Follow the pack's daily limit; do not use for more than 2 weeks without a doctor",
         ("Take other medicines at least 2 hours apart from an antacid.",)),
    Drug("hydrocortisone", "Hydrocortisone 1% cream",
         r"rash|itch\w*|insect bite|eczema|dermatitis|hives",
         "a mild itchy rash on a small area of skin",
         "A thin layer on the affected skin once or twice a day",
         "For up to 7 days; not on the face, broken or infected skin",
         ("Wash your hands after applying.", "See a doctor if the rash spreads, blisters or becomes painful.")),
)
BY_KEY = {d.key: d for d in FORMULARY}

SELF_CARE = [
    (r"fever|temperature|chills", "Rest, drink plenty of fluids, and wear light clothing. A lukewarm (not cold) sponge can help."),
    (r"cough", "Warm fluids and honey (not for under 1 year) ease a cough; avoid smoke and dust."),
    (r"nose|nasal|blocked|congest\w*|sinus|stuffy", "Steam inhalation and extra fluids loosen mucus; sleep with your head raised."),
    (r"sore throat|throat", "Warm fluids, rest your voice, and avoid very cold or spicy food."),
    (r"diarrh\w*|vomit\w*|loose (?:motion|stool)s?", "Eat light food (rice, banana, toast); avoid milk, oily or spicy food until you recover."),
    (r"headache|migraine", "Rest in a quiet, dim room and drink water; do not skip meals."),
    (r"acidity|heartburn|indigestion|reflux", "Eat smaller meals, avoid spicy food and lying down right after eating."),
]

# Names people use for medicines they ask about -> formulary key, or a class we deliberately do not prescribe
_ASK = [
    ("paracetamol", r"paracetamol|acetaminophen|crocin|dolo|calpol|tylenol|pcm"),
    ("ibuprofen", r"ibuprofen|brufen|advil|motrin|combiflam"),
    ("cetirizine", r"cetirizine|cetzine|okacet|zyrtec|antihistamine|allegra|levocet\w*"),
    ("saline", r"saline|nasal drops?|nose drops?|nasal spray|otrivin|nasivion"),
    ("ors", r"\bors\b|electral|oral rehydration"),
    ("antacid", r"antacid|digene|gelusil|eno\b|pantoprazole|omeprazole"),
    ("hydrocortisone", r"hydrocortisone|steroid cream"),
    ("rx", r"antibiotics?|amoxicillin|azithromycin|augmentin|cipro\w*|doxycycline|cefixime|prednisolone|steroids?|tramadol|morphine|codeine|"
           r"diazepam|alprazolam|sleeping pills?|sildenafil|insulin|antidepressants?"),
    ("other_otc", r"aspirin|ecosprin|disprin|diclofenac|naproxen|nimesulide|loperamide|imodium|cough syrup|decongestant|cold tablets?|vitamin\w*|zinc"),
]


@dataclass
class Ctx:
    age: int | None
    child: bool
    elderly: bool
    risks: set[str]
    allergies: set[str]
    meds: set[str]
    text: str
    fever: bool
    fever_days: float
    level: str
    screened: bool                      # the allergy / regular-medicine question has been answered


def _ctx(f: Facts, a: Assessment) -> Ctx:
    from . import clinic
    text = triage.strip_negated(f.symptom_text or f.text).lower() + " " + clinic.phrases(f.slots).lower()
    if any(d["key"] == "dengue_like" and d["score"] >= 0.45 for d in a.possible):
        text += " dengue"                                    # a likely dengue picture: anti-inflammatory painkillers are screened out
    temp = f.temp_c if f.temp_c is not None else triage.parse_temperature_c(f.text)
    fever = bool(re.search(r"fever|temperature|chills|feverish", text)) or (temp is not None and temp >= 38.0)
    return Ctx(age=f.age, child=f.age is not None and f.age < 12, elderly=f.age is not None and f.age >= 65,
               risks=set(f.risks), allergies=set(f.allergies), meds=set(f.medicines), text=text, fever=fever,
               fever_days=f.duration_days or 0.0, level=a.level, screened=f.allergies_asked)


def screen(d: Drug, c: Ctx) -> tuple[list[str], list[str]]:
    """(reasons NOT to use it, extra warnings to show with it) for this patient."""
    no: list[str] = []
    warn: list[str] = []
    if "many" in c.allergies:
        no.append("you reported allergies to many medicines: a pharmacist or doctor should choose for you")
    if d.key == "paracetamol":
        if "paracetamol" in c.allergies:
            no.append("you are allergic to it")
        if "liver disease" in c.risks:
            no.append("liver disease needs a doctor's advice on paracetamol")
        if "paracetamol" in c.meds:
            warn.append("You already take a paracetamol-containing medicine: count its paracetamol towards the 4 g daily limit.")
    elif d.key == "ibuprofen":
        if "nsaid" in c.allergies:
            no.append("you are allergic to aspirin/ibuprofen-type painkillers")
        if "lung disease" in c.risks:
            no.append("it can trigger asthma and breathing problems")
        for r, why in (("kidney disease", "it can harm weak kidneys"), ("heart disease", "it can raise blood pressure and strain the heart"),
                       ("liver disease", "liver disease needs a doctor's advice on painkillers"),
                       ("stomach ulcer or gastritis", "it can irritate the stomach and cause bleeding"),
                       ("bleeding disorder", "it raises the risk of bleeding"), ("pregnancy", "it is not advised in pregnancy")):
            if r in c.risks:
                no.append(why)
        if c.elderly:
            no.append("at 65 or older it carries a higher risk of stomach bleeding and kidney problems")
        for m, why in (("anticoagulant", "you take a blood thinner (bleeding risk)"), ("aspirin", "you take aspirin (bleeding risk)"),
                       ("nsaid", "you already take a similar painkiller"), ("bp_kidney", "it can interact with your blood-pressure / water tablets and strain the kidneys"),
                       ("lithium_methotrexate", "it can raise the level of lithium or methotrexate")):
            if m in c.meds:
                no.append(why)
        if c.fever and c.fever_days >= 3:
            no.append("with a fever lasting 3 or more days, anti-inflammatory painkillers are best avoided until a doctor has ruled out dengue")
        if re.search(r"dengue|vomit\w*|diarrh\w*|dehydrat\w*|platelet", c.text):
            no.append("it is best avoided with vomiting, diarrhoea, dehydration or possible dengue")
    elif d.key == "cetirizine":
        if "antihistamine" in c.allergies:
            no.append("you are allergic to antihistamines")
        if "kidney disease" in c.risks:
            no.append("kidney disease needs a lower dose chosen by a doctor or pharmacist")
        if "pregnancy" in c.risks:
            no.append("ask your doctor before taking it in pregnancy")
        if "antidepressant" in c.meds:
            warn.append("Together with your antidepressant it may make you more drowsy.")
    elif d.key == "antacid":
        if "kidney disease" in c.risks:
            no.append("aluminium/magnesium antacids are not advised in kidney disease")
    elif d.key == "hydrocortisone":
        if "pregnancy" in c.risks:
            no.append("ask your doctor before using steroid creams in pregnancy")
        if re.search(r"pus|infect\w*|blister\w*|spreading|fever|ooz\w*", c.text):
            no.append("a spreading, blistering or infected-looking rash needs a doctor to look at it first")
    if d.key in ("ibuprofen", "cetirizine", "paracetamol") and "pregnancy" in c.risks and d.key == "paracetamol":
        warn.append("In pregnancy use the lowest dose for the shortest time and tell your doctor.")
    return no, warn


def referral(a: Assessment) -> str:
    """The closing doctor referral, matched to how serious the case is. Every medicine answer ends with this."""
    who = a.specialty if a.specialty != "General Physician" else "doctor"
    if a.level == "emergency":
        return "Please get emergency care now instead of treating this at home."
    if a.level == "urgent":
        return f"Please see a {who} today: these medicines only ease symptoms and are not a substitute for being examined."
    if a.level == "see_doctor":
        return f"Please book a visit with a {who} within the next 2-3 days: these medicines only ease symptoms, they do not treat the cause."
    return f"If it is not clearly better in 3 days, or it gets worse, please see a {who}."


def severity_note(f: Facts, a: Assessment) -> str:
    """How firmly to use the doses, scaled to how severe the problem is."""
    sev = f.severity
    if sev is None or sev <= 3:
        return "Your symptoms sound mild: start with the lower dose, only if you need it, and keep the home care going."
    if sev <= 6:
        return "Your symptoms are moderate: the usual dose range below can be used as directed."
    return "Your symptoms are severe: use only the usual dose as directed (never more), and do not wait to be seen by a doctor."


def suggest(f: Facts, a: Assessment) -> dict:
    """Medicines to consider (screened), medicines to avoid (with the reason), and non-drug care, for this patient."""
    c = _ctx(f, a)
    out = {"suggestions": [], "avoid": [], "self_care": [], "not_prescribed": RX_NOTE, "child": c.child, "needs_screening": not c.screened,
           "note": "", "dose_guidance": severity_note(f, a), "referral": referral(a), "disclaimer": DISCLAIMER}
    if a.level == "emergency":
        out.update(not_prescribed=None, note="No medicines now: get emergency care first.")
        return out
    if c.child:
        out["note"] = CHILD_NOTE
    out["self_care"] = [tip for rx, tip in SELF_CARE if re.search(rx, c.text)][:3]
    if a.level == "urgent":
        out["note"] = "Do not delay seeing a doctor. Until then, only paracetamol for fever or pain is suggested."
    for d in FORMULARY:
        if not re.search(d.treats, c.text):
            continue
        if a.level == "urgent" and d.key != "paracetamol":
            continue
        if c.child and d.needs_dose:
            continue                                         # no numeric doses for children
        if d.needs_dose and not c.screened:
            continue                                         # never list a dosed medicine before the allergy/medicine check
        no, warn = screen(d, c)
        if no:
            out["avoid"].append({"key": d.key, "name": d.name, "reason": "; ".join(no)})
            continue
        out["suggestions"].append({"key": d.key, "name": d.name, "use": d.use, "dose": d.dose, "maximum": d.maximum,
                                   "tips": list(d.tips) + warn})
    if not c.screened and any(re.search(d.treats, c.text) for d in FORMULARY if d.needs_dose):
        out["note"] = ((out["note"] + " ") if out["note"] else "") + (
            "Tell me about any medicine allergies and regular medicines (or say 'none') and I will check which medicines are safe for you.")
    # ibuprofen is only worth showing as 'avoided' when paracetamol-or-ibuprofen was actually relevant
    return out


def which_drug(question: str) -> str | None:
    q = question.lower()
    for key, rx in _ASK:
        if re.search(rx, q):
            return key
    return None


def answer_about(question: str, f: Facts, a: Assessment) -> dict | None:
    """A verdict for 'can I take X?' / 'what is the dose of X?' from the formulary and the patient's screen; None if X is not named."""
    key = which_drug(question)
    if key is None:
        return None
    c = _ctx(f, a)
    if key == "rx":
        return {"verdict": "rx", "text": ("That is a prescription-only type of medicine, so I cannot suggest or prescribe it. A doctor needs to examine you and "
                                          "decide. " + (f"Please see a {a.specialty} " + ("today." if a.level == "urgent" else "soon.") if a.level != "self_care" else
                                                        "If you are worried, see a doctor or pharmacist."))}
    if key == "other_otc":
        return {"verdict": "ask", "text": "I do not have a verified dosing reference for that medicine. Please ask a pharmacist, who can check it against your other medicines and conditions."}
    d = BY_KEY[key]
    if a.level == "emergency":
        return {"verdict": "avoid", "text": "This looks like an emergency: please get emergency care now instead of taking medicines at home."}
    if c.child and d.needs_dose:
        return {"verdict": "ask", "text": CHILD_NOTE}
    if d.needs_dose and not c.screened:
        return {"verdict": "ask", "text": f"Before I can say whether {d.name.split(' (')[0]} is safe for you, tell me about any medicine allergies and regular medicines (or say 'none')."}
    no, warn = screen(d, c)
    name = d.name.split(" (")[0]
    if no:
        alt = ""
        if key != "paracetamol":
            p_no, _ = screen(BY_KEY["paracetamol"], c)
            if not p_no and re.search(BY_KEY["paracetamol"].treats, c.text):
                alt = " Paracetamol is the safer choice for you: " + BY_KEY["paracetamol"].dose + ". " + BY_KEY["paracetamol"].maximum + "."
        return {"verdict": "avoid", "text": f"I would not suggest {name} for you: {'; '.join(no)}.{alt} {referral(a)}"}
    if a.level == "urgent" and key != "paracetamol":
        return {"verdict": "ask", "text": f"Your symptoms need a doctor today. Please do not self-treat with {name}; ask the doctor."}
    extra = (" " + " ".join(warn)) if warn else ""
    return {"verdict": "ok", "text": f"{name} can be used for {d.use}: {d.dose}. {d.maximum}.{extra} {severity_note(f, a)} {referral(a)}"}
