"""Clinical decision support for the Clinical Assistant: the problems a clinician actually meets with a chart open.

  medication_check(patient, drug)   "Can I add ibuprofen?"  -> allergy / interaction / condition checks against THIS patient
  pair_check(drug_a, drug_b)        "warfarin with ibuprofen?" -> interaction between two drugs (no patient needed)
  review(patient)                   what is already wrong in the chart: prescribing problems + care gaps (guideline thresholds)
  summarize(patient)                structured snapshot (vitals with status, parsed medicines, alerts) for the UI

Deterministic and curated, like the dose table: a language model never decides whether a combination is safe. The rules encode
widely accepted prescribing references (e.g. BNF, FDA labels, ACC/AHA, ADA, KDIGO) in a compact form and every finding says what
to do. They are decision SUPPORT: the clinician verifies against the local formulary, and the list is deliberately limited to
common drugs, so "no issue found" means "no issue in these checks", never "safe".
"""
import re
from dataclasses import dataclass
from datetime import date

SEV_ORDER = {"contraindicated": 0, "major": 1, "moderate": 2, "info": 3}
AVOID = ("contraindicated", "major")


# ------------------------------------------------------------------------------------------------ drugs
@dataclass(frozen=True)
class Drug:
    name: str
    classes: tuple[str, ...]
    aliases: tuple[str, ...] = ()


def D(name, classes, aliases=()):
    return Drug(name, tuple(classes), tuple(aliases))


DRUGS: tuple[Drug, ...] = (
    # cardiovascular
    D("atorvastatin", ["statin"], ["lipitor"]), D("simvastatin", ["statin"], ["zocor"]), D("rosuvastatin", ["statin"], ["crestor"]),
    D("lisinopril", ["ace"], ["zestril"]), D("enalapril", ["ace"]), D("ramipril", ["ace"], ["altace"]),
    D("losartan", ["arb"], ["cozaar"]), D("telmisartan", ["arb"]), D("valsartan", ["arb"]), D("olmesartan", ["arb"]),
    D("amlodipine", ["ccb_dhp"], ["norvasc"]), D("nifedipine", ["ccb_dhp"]),
    D("diltiazem", ["ccb_nondhp"]), D("verapamil", ["ccb_nondhp"]),
    D("metoprolol", ["beta_blocker", "bb_selective"]), D("bisoprolol", ["beta_blocker", "bb_selective"]), D("atenolol", ["beta_blocker", "bb_selective"]),
    D("propranolol", ["beta_blocker", "bb_nonselective"]), D("carvedilol", ["beta_blocker", "bb_nonselective"]),
    D("furosemide", ["loop_diuretic", "diuretic", "sulfonamide_nonabx"], ["lasix", "frusemide"]),
    D("hydrochlorothiazide", ["thiazide", "diuretic", "sulfonamide_nonabx"], ["hctz"]), D("chlorthalidone", ["thiazide", "diuretic"]),
    D("spironolactone", ["k_sparing", "diuretic"]), D("potassium chloride", ["potassium"], ["kcl", "potassium"]),
    D("warfarin", ["anticoagulant", "vka"], ["coumadin"]), D("apixaban", ["anticoagulant", "doac"], ["eliquis"]),
    D("rivaroxaban", ["anticoagulant", "doac"], ["xarelto"]), D("dabigatran", ["anticoagulant", "doac"]),
    D("aspirin", ["antiplatelet", "nsaid", "salicylate"], ["ecosprin", "disprin"]), D("clopidogrel", ["antiplatelet"], ["plavix"]),
    D("digoxin", ["digoxin"]), D("amiodarone", ["amiodarone"]), D("isosorbide mononitrate", ["nitrate"], ["isosorbide", "nitroglycerin", "gtn"]),
    # diabetes / endocrine
    D("metformin", ["metformin", "biguanide"], ["glucophage"]), D("glimepiride", ["sulfonylurea", "sulfonamide_nonabx"], ["amaryl"]),
    D("gliclazide", ["sulfonylurea"]), D("insulin", ["insulin"]), D("pioglitazone", ["thiazolidinedione"]),
    D("levothyroxine", ["levothyroxine"], ["thyroxine", "eltroxin"]),
    # respiratory
    D("salbutamol", ["saba", "beta_agonist"], ["albuterol", "ventolin"]), D("budesonide", ["ics"], ["pulmicort"]),
    D("tiotropium", ["lama"], ["spiriva"]), D("theophylline", ["theophylline"]),
    # GI / pain / other
    D("omeprazole", ["ppi"], ["prilosec"]), D("pantoprazole", ["ppi"]), D("esomeprazole", ["ppi"]),
    D("paracetamol", ["paracetamol"], ["acetaminophen", "tylenol", "crocin", "dolo"]),
    D("ibuprofen", ["nsaid"], ["brufen", "advil", "motrin"]), D("diclofenac", ["nsaid"]), D("naproxen", ["nsaid"]), D("celecoxib", ["nsaid", "sulfonamide_nonabx"]),
    D("tramadol", ["opioid", "serotonergic"]), D("codeine", ["opioid"]), D("morphine", ["opioid"]),
    D("prednisolone", ["steroid"], ["prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone tablets"]),
    D("sertraline", ["ssri", "serotonergic"], ["zoloft"]), D("fluoxetine", ["ssri", "serotonergic"]), D("escitalopram", ["ssri", "serotonergic"]),
    D("citalopram", ["ssri", "serotonergic"]), D("amitriptyline", ["tca", "serotonergic"]),
    D("ferrous sulfate", ["iron"], ["iron", "ferrous"]), D("methotrexate", ["methotrexate"]), D("lithium", ["lithium"]),
    D("sildenafil", ["pde5"], ["viagra", "tadalafil"]), D("cetirizine", ["antihistamine"]),
    # anti-infectives
    D("amoxicillin", ["penicillin", "beta_lactam"], ["augmentin", "amoxyclav", "co-amoxiclav"]), D("ampicillin", ["penicillin", "beta_lactam"]),
    D("penicillin", ["penicillin", "beta_lactam"], ["phenoxymethylpenicillin"]), D("piperacillin", ["penicillin", "beta_lactam"]),
    D("cephalexin", ["cephalosporin", "beta_lactam"], ["cefalexin"]), D("ceftriaxone", ["cephalosporin", "beta_lactam"]), D("cefixime", ["cephalosporin", "beta_lactam"]),
    D("azithromycin", ["macrolide", "qt"], ["zithromax"]), D("clarithromycin", ["macrolide", "cyp3a4_inhibitor", "qt"]), D("erythromycin", ["macrolide", "cyp3a4_inhibitor", "qt"]),
    D("ciprofloxacin", ["fluoroquinolone", "qt"], ["cipro"]), D("levofloxacin", ["fluoroquinolone", "qt"]),
    D("metronidazole", ["metronidazole"], ["flagyl"]), D("fluconazole", ["azole", "cyp3a4_inhibitor"]),
    D("co-trimoxazole", ["sulfa_antibiotic", "trimethoprim"], ["cotrimoxazole", "trimethoprim-sulfamethoxazole", "bactrim", "septran", "sulfamethoxazole", "sulfadiazine"]),
    D("doxycycline", ["tetracycline"], ["tetracycline"]),
)
BY_NAME = {d.name: d for d in DRUGS}
_ALIAS = {a: d.name for d in DRUGS for a in (d.name, *d.aliases)}
_ALIAS_RX = re.compile(r"\b(" + "|".join(sorted((re.escape(a) for a in _ALIAS), key=len, reverse=True)) + r")s?\b", re.I)
CLASS_WORDS = {r"\bnsaids?\b": "ibuprofen", r"\bstatins?\b": "atorvastatin", r"\bace[- ]inhibitors?\b": "lisinopril",
               r"\bbeta[- ]?blockers?\b": "propranolol", r"\bblood thinners?\b": "warfarin", r"\banticoagulants?\b": "warfarin",
               r"\bssris?\b": "sertraline", r"\bppis?\b": "omeprazole"}


def classes_of(name: str) -> set[str]:
    d = BY_NAME.get(name)
    return {name, *d.classes} if d else {name}


def find_drugs(text: str) -> list[str]:
    """Canonical drug names mentioned, in order (brand names and a few class words are understood)."""
    seen: list[str] = []
    for m in _ALIAS_RX.finditer(text):
        n = _ALIAS[m.group(1).lower()]
        if n not in seen:
            seen.append(n)
    for rx, rep in CLASS_WORDS.items():
        if re.search(rx, text, re.I) and rep not in seen:
            seen.append(rep)
    return seen


# ------------------------------------------------------------------------------------------------ interactions
# (class/drug A, class/drug B, severity, what happens, what to do)
INTERACTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("anticoagulant", "nsaid", "major", "NSAIDs raise the risk of serious GI bleeding and impair platelet function on top of the anticoagulant.",
     "Avoid; use paracetamol for pain. If an NSAID is unavoidable, add a PPI and monitor closely."),
    ("anticoagulant", "antiplatelet", "major", "Combined anticoagulant and antiplatelet therapy markedly increases bleeding risk.",
     "Only if there is a clear indication (e.g. recent stent); otherwise avoid and review the need for each."),
    ("warfarin", "macrolide", "major", "Macrolides raise the INR (CYP inhibition / gut flora), with bleeding risk.", "Avoid if possible, or check INR within 3-5 days."),
    ("warfarin", "fluoroquinolone", "major", "Fluoroquinolones raise the INR.", "Avoid if possible, or check INR within 3-5 days."),
    ("warfarin", "metronidazole", "major", "Metronidazole strongly potentiates warfarin (INR rises, bleeding).", "Avoid, or reduce the warfarin dose and monitor INR closely."),
    ("warfarin", "sulfa_antibiotic", "major", "Co-trimoxazole strongly potentiates warfarin.", "Avoid; if essential, monitor INR closely."),
    ("warfarin", "azole", "major", "Fluconazole inhibits warfarin metabolism: INR rises.", "Avoid, or reduce the warfarin dose and monitor INR."),
    ("warfarin", "amiodarone", "major", "Amiodarone raises the INR (effect builds over weeks).", "Reduce the warfarin dose by about a third to a half and monitor INR."),
    ("warfarin", "ssri", "moderate", "SSRIs add to bleeding risk (platelet effect).", "Use with care; consider a PPI and monitor for bleeding."),
    ("warfarin", "paracetamol", "moderate", "Regular paracetamol (about 2 g/day for several days or more) can raise the INR.", "Occasional doses are fine; for regular use monitor INR."),
    ("warfarin", "steroid", "moderate", "Corticosteroids can alter the INR and add GI bleeding risk.", "Monitor INR; consider gastro-protection."),
    ("doac", "nsaid", "major", "NSAIDs increase the bleeding risk with direct oral anticoagulants.", "Avoid; use paracetamol."),
    ("nsaid", "ace", "moderate", "NSAIDs blunt the antihypertensive effect and can cause acute kidney injury with ACE inhibitors.", "Avoid regular use; monitor creatinine and BP."),
    ("nsaid", "arb", "moderate", "NSAIDs blunt the antihypertensive effect and can cause acute kidney injury with ARBs.", "Avoid regular use; monitor creatinine and BP."),
    ("nsaid", "diuretic", "moderate", "NSAIDs reduce the effect of diuretics and add kidney risk (the 'triple whammy' with an ACE inhibitor/ARB).", "Avoid or use the shortest course; monitor renal function."),
    ("nsaid", "ssri", "moderate", "SSRIs plus NSAIDs increase the risk of GI bleeding.", "Prefer paracetamol; if an NSAID is needed, add a PPI."),
    ("nsaid", "steroid", "moderate", "Corticosteroids plus NSAIDs increase GI ulcer and bleeding risk.", "Add a PPI or avoid the NSAID."),
    ("nsaid", "methotrexate", "major", "NSAIDs reduce methotrexate clearance: toxicity.", "Avoid, or monitor methotrexate levels and blood counts."),
    ("nsaid", "lithium", "major", "NSAIDs raise lithium levels: toxicity.", "Avoid; monitor lithium levels if unavoidable."),
    ("ibuprofen", "aspirin", "moderate", "Ibuprofen can block the antiplatelet effect of low-dose aspirin, and the combination adds GI risk.", "Take aspirin at least 30 min before (or 8 h after) ibuprofen, or use another analgesic."),
    ("ace", "arb", "major", "Dual renin-angiotensin blockade: hyperkalaemia, hypotension and acute kidney injury, without added benefit.", "Avoid the combination; keep one agent."),
    ("ace", "k_sparing", "major", "Hyperkalaemia risk.", "Check potassium and creatinine within a week; avoid if eGFR is low."),
    ("arb", "k_sparing", "major", "Hyperkalaemia risk.", "Check potassium and creatinine within a week; avoid if eGFR is low."),
    ("ace", "potassium", "major", "Hyperkalaemia risk.", "Avoid unless potassium is monitored."),
    ("arb", "potassium", "major", "Hyperkalaemia risk.", "Avoid unless potassium is monitored."),
    ("trimethoprim", "ace", "major", "Trimethoprim raises potassium: severe hyperkalaemia with ACE inhibitors.", "Avoid, or check potassium early."),
    ("trimethoprim", "arb", "major", "Trimethoprim raises potassium: severe hyperkalaemia with ARBs.", "Avoid, or check potassium early."),
    ("lithium", "ace", "major", "ACE inhibitors raise lithium levels.", "Monitor lithium levels."), ("lithium", "thiazide", "major", "Thiazides raise lithium levels.", "Avoid, or monitor lithium levels."),
    ("statin", "macrolide", "major", "Clarithromycin/erythromycin raise statin levels: myopathy and rhabdomyolysis.", "Pause the statin during the course, or use azithromycin."),
    ("statin", "azole", "major", "Azoles raise statin levels: myopathy and rhabdomyolysis.", "Pause the statin during the course."),
    ("serotonergic", "serotonergic", "major", "Two serotonergic drugs (e.g. SSRI + tramadol) risk serotonin syndrome and, with tramadol, seizures.", "Avoid; choose a non-serotonergic analgesic."),
    ("beta_blocker", "ccb_nondhp", "major", "Verapamil/diltiazem with a beta-blocker: bradycardia, AV block and heart failure.", "Avoid, or monitor ECG and heart rate closely."),
    ("bb_nonselective", "saba", "major", "Non-selective beta-blockers antagonise salbutamol and can trigger bronchospasm.", "Avoid in asthma/COPD; use a cardioselective agent if a beta-blocker is essential."),
    ("digoxin", "amiodarone", "major", "Amiodarone roughly doubles digoxin levels.", "Halve the digoxin dose and monitor levels."),
    ("digoxin", "ccb_nondhp", "major", "Verapamil/diltiazem raise digoxin levels and slow the heart.", "Reduce digoxin and monitor levels and heart rate."),
    ("digoxin", "macrolide", "major", "Macrolides raise digoxin levels.", "Monitor digoxin levels or avoid."),
    ("digoxin", "loop_diuretic", "moderate", "Diuretic-induced hypokalaemia increases digoxin toxicity.", "Monitor potassium."),
    ("sulfonylurea", "fluoroquinolone", "moderate", "Fluoroquinolones can cause hypo- or hyperglycaemia with sulfonylureas.", "Counsel on hypoglycaemia; monitor glucose."),
    ("sulfonylurea", "sulfa_antibiotic", "major", "Co-trimoxazole potentiates sulfonylureas: severe hypoglycaemia.", "Avoid, or reduce the sulfonylurea and monitor glucose."),
    ("sulfonylurea", "beta_blocker", "moderate", "Beta-blockers can mask the warning signs of hypoglycaemia.", "Counsel the patient; prefer a cardioselective agent."),
    ("diabetes_drug", "steroid", "moderate", "Corticosteroids raise blood glucose.", "Monitor glucose and adjust treatment."),
    ("methotrexate", "sulfa_antibiotic", "major", "Co-trimoxazole with methotrexate: severe bone-marrow suppression.", "Avoid."),
    ("nitrate", "pde5", "contraindicated", "Nitrates with sildenafil/tadalafil cause profound, potentially fatal hypotension.", "Contraindicated: never combine."),
    ("antiplatelet", "ppi", "info", "Omeprazole/esomeprazole can reduce the effect of clopidogrel.", "Prefer pantoprazole with clopidogrel."),
    ("levothyroxine", "iron", "moderate", "Iron binds levothyroxine in the gut and reduces its absorption: TSH rises.", "Separate the doses by at least 4 hours and recheck TSH."),
    ("levothyroxine", "ppi", "info", "PPIs can reduce levothyroxine absorption.", "Recheck TSH after starting a PPI."),
    ("iron", "fluoroquinolone", "moderate", "Iron chelates fluoroquinolones: poor antibiotic absorption.", "Give the antibiotic 2 h before or 6 h after iron."),
    ("iron", "tetracycline", "moderate", "Iron chelates tetracyclines: poor antibiotic absorption.", "Separate by at least 2-3 hours."),
    ("iron", "ppi", "info", "Acid suppression reduces oral iron absorption.", "Consider this if anaemia is not responding."),
    ("theophylline", "fluoroquinolone", "major", "Ciprofloxacin raises theophylline levels: toxicity.", "Avoid, or reduce the dose and monitor levels."),
    ("theophylline", "macrolide", "major", "Macrolides raise theophylline levels: toxicity.", "Monitor levels or avoid."),
    ("qt", "qt", "major", "Two QT-prolonging drugs (e.g. macrolide/fluoroquinolone with another) increase the risk of torsades de pointes.", "Avoid, or check ECG and electrolytes."),
    ("fluoroquinolone", "steroid", "moderate", "Fluoroquinolone plus corticosteroid increases the risk of tendon rupture.", "Avoid in older patients; warn about tendon pain."),
    ("loop_diuretic", "ace", "info", "First-dose hypotension and kidney effects when starting an ACE inhibitor in a patient on a loop diuretic.", "Start low and check renal function."),
    ("saba", "loop_diuretic", "info", "Both can lower potassium.", "Monitor potassium with frequent salbutamol use."),
    ("beta_agonist", "bb_selective", "info", "Cardioselective beta-blockers may slightly blunt salbutamol.", "Usually acceptable; monitor symptoms."),
    ("tramadol", "warfarin", "moderate", "Tramadol can raise the INR.", "Monitor INR."),
    ("metformin", "steroid", "info", "Corticosteroids raise glucose and can weaken metformin's effect.", "Monitor glucose."),
)
_DIABETES_DRUGS = {"metformin", "sulfonylurea", "insulin", "thiazolidinedione"}


def _matches(spec: str, cls: set[str]) -> bool:
    return spec in cls or (spec == "diabetes_drug" and bool(cls & _DIABETES_DRUGS))


def interaction(a: str, b: str) -> list[dict]:
    """Findings for the pair of drugs a and b (canonical names)."""
    ca, cb = classes_of(a), classes_of(b)
    out = []
    for x, y, sev, effect, action in INTERACTIONS:
        if x == y:
            hit = x in ca and x in cb and a != b                    # two different drugs of the same listed class
        else:
            hit = (_matches(x, ca) and _matches(y, cb)) or (_matches(x, cb) and _matches(y, ca))
        if hit:
            out.append({"severity": sev, "type": "interaction", "drugs": [a, b], "title": f"{a.title()} + {b.title()}", "detail": effect, "action": action})
    # same-class duplication (two NSAIDs, two ACE/ARBs, two statins, two SSRIs, two PPIs ...)
    dup = [c for c in ("nsaid", "statin", "ssri", "ppi", "ace", "arb", "beta_blocker", "anticoagulant") if c in ca and c in cb and a != b]
    if dup and not any(f["severity"] in AVOID for f in out):
        if not ({"ace", "arb"} & set(dup) and (("ace" in ca and "arb" in cb) or ("arb" in ca and "ace" in cb))):
            out.append({"severity": "major" if dup[0] in ("nsaid", "anticoagulant") else "moderate", "type": "duplication", "drugs": [a, b],
                        "title": f"{a.title()} + {b.title()}", "detail": f"Two drugs of the same class ({dup[0].replace('_', ' ')}): duplicate therapy adds risk without added benefit.",
                        "action": "Use one agent from the class."})
    return out


# ------------------------------------------------------------------------------------------------ conditions
CONDITION_TAGS = {
    "htn": r"hypertension|high blood pressure", "hld": r"hyperlipid|dyslipid|cholesterol", "dm": r"diabet", "gerd": r"gerd|reflux|ulcer|gastritis|dyspepsia",
    "asthma": r"asthma", "copd": r"copd|emphysema|chronic bronchitis", "ckd": r"kidney|renal|\bckd\b", "af": r"atrial fibrillation|\baf\b|afib",
    "hf": r"heart failure|\bchf\b|cardiomyopathy", "depression": r"depress", "hypothyroid": r"hypothyroid", "anemia": r"anaemia|anemia",
    "migraine": r"migraine", "oa": r"osteoarthritis|arthritis", "liver": r"liver|cirrhosis|hepatitis", "epilepsy": r"epilep|seizure",
    "cvd": r"coronary|myocardial|angina|stroke|\bcad\b|\bmi\b|peripheral arter",
}


def condition_tags(conditions: str) -> set[str]:
    return {tag for tag, rx in CONDITION_TAGS.items() if re.search(rx, conditions or "", re.I)}


# (drug class/name, condition tag, severity, what happens, what to do)
CONDITION_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    ("nsaid", "ckd", "major", "NSAIDs reduce renal blood flow and can accelerate kidney disease or cause acute kidney injury.", "Avoid; use paracetamol. Discuss with nephrology if an anti-inflammatory is essential."),
    ("nsaid", "hf", "major", "NSAIDs cause sodium/fluid retention and can precipitate decompensated heart failure.", "Avoid; use paracetamol."),
    ("nsaid", "htn", "moderate", "NSAIDs raise blood pressure and blunt antihypertensives.", "Use the lowest dose for the shortest time and monitor BP."),
    ("nsaid", "asthma", "moderate", "NSAIDs (especially aspirin) can trigger bronchospasm in sensitive asthmatics.", "Ask about previous reactions; prefer paracetamol."),
    ("nsaid", "gerd", "moderate", "NSAIDs irritate the stomach lining and can cause ulcers and bleeding.", "Prefer paracetamol; if an NSAID is needed add a PPI."),
    ("nsaid", "cvd", "major", "NSAIDs increase cardiovascular risk in established heart disease.", "Avoid; use paracetamol."),
    ("nsaid", "liver", "moderate", "NSAIDs raise the risk of bleeding and kidney injury in liver disease.", "Avoid; use paracetamol at a reduced dose."),
    ("metformin", "ckd", "major", "Metformin accumulates in kidney disease (lactic acidosis): contraindicated if eGFR is below 30; dose-reduce at 30-45.", "Check eGFR before prescribing or continuing."),
    ("metformin", "hf", "moderate", "Risk of lactic acidosis in unstable or acute heart failure.", "Avoid in acute/unstable HF; fine in stable HF."),
    ("metformin", "liver", "moderate", "Reduced lactate clearance in liver disease.", "Use with caution; review the need."),
    ("bb_nonselective", "asthma", "major", "Non-selective beta-blockers can cause severe bronchospasm.", "Avoid; use a cardioselective agent only if essential."),
    ("bb_nonselective", "copd", "major", "Non-selective beta-blockers worsen airflow obstruction.", "Avoid; a cardioselective beta-blocker is preferred if one is needed."),
    ("bb_selective", "asthma", "moderate", "Even cardioselective beta-blockers can occasionally provoke bronchospasm.", "Start low; avoid in active or severe asthma."),
    ("ace", "ckd", "moderate", "ACE inhibitors can raise creatinine and potassium in kidney disease (though often protective long-term).", "Check creatinine and potassium 1-2 weeks after starting or changing the dose."),
    ("arb", "ckd", "moderate", "ARBs can raise creatinine and potassium in kidney disease (though often protective long-term).", "Check creatinine and potassium 1-2 weeks after starting or changing the dose."),
    ("ccb_nondhp", "hf", "major", "Verapamil/diltiazem are negatively inotropic and can worsen heart failure with reduced ejection fraction.", "Avoid in HFrEF."),
    ("thiazolidinedione", "hf", "contraindicated", "Pioglitazone causes fluid retention and can precipitate heart failure.", "Contraindicated in heart failure."),
    ("thiazide", "dm", "moderate", "Thiazides can raise blood glucose.", "Monitor glucose; usually acceptable at low dose."),
    ("thiazide", "ckd", "moderate", "Thiazides lose effect when eGFR is low and can worsen electrolytes.", "Check eGFR and electrolytes."),
    ("steroid", "dm", "major", "Corticosteroids cause marked hyperglycaemia in diabetes.", "Monitor glucose closely and adjust treatment during the course."),
    ("steroid", "htn", "moderate", "Corticosteroids raise blood pressure and cause fluid retention.", "Monitor BP."),
    ("steroid", "gerd", "moderate", "Corticosteroids add to the risk of GI irritation and bleeding.", "Consider a PPI."),
    ("steroid", "hf", "moderate", "Fluid retention may worsen heart failure.", "Monitor weight and symptoms."),
    ("sulfonylurea", "ckd", "major", "Sulfonylureas accumulate in kidney disease: prolonged, severe hypoglycaemia.", "Avoid or choose a safer agent (e.g. gliclazide, linagliptin) and check eGFR."),
    ("tramadol", "epilepsy", "major", "Tramadol lowers the seizure threshold.", "Avoid."),
    ("tramadol", "ckd", "moderate", "Metabolites accumulate in kidney disease.", "Reduce the dose and interval."),
    ("fluoroquinolone", "af", "moderate", "QT prolongation and arrhythmia risk.", "Check ECG and electrolytes or choose another antibiotic."),
    ("macrolide", "af", "moderate", "QT prolongation and arrhythmia risk; many interactions with anticoagulants.", "Check ECG and interactions, or choose another antibiotic."),
    ("saba", "af", "moderate", "Beta-agonists can provoke tachyarrhythmias.", "Use the lowest effective dose."),
    ("levothyroxine", "af", "moderate", "Over-replacement can trigger or worsen atrial fibrillation.", "Check TSH; avoid over-replacement."),
    ("levothyroxine", "cvd", "moderate", "Start low and go slow in coronary disease.", "Check TSH and symptoms."),
    ("digoxin", "ckd", "major", "Digoxin is renally cleared: toxicity in kidney disease.", "Reduce the dose; monitor levels and eGFR."),
    ("sulfa_antibiotic", "ckd", "moderate", "Co-trimoxazole raises creatinine and potassium.", "Use a reduced dose and check potassium."),
    ("antiplatelet", "gerd", "moderate", "Aspirin/antiplatelets increase upper-GI bleeding risk.", "Consider a PPI."),
    ("paracetamol", "liver", "moderate", "Reduced hepatic reserve.", "Use a lower maximum daily dose (e.g. 2 g)."),
    ("potassium", "ckd", "major", "Hyperkalaemia risk in kidney disease.", "Avoid unless potassium is monitored."),
    ("amiodarone", "hypothyroid", "moderate", "Amiodarone commonly disturbs thyroid function.", "Monitor TSH."),
)

# ------------------------------------------------------------------------------------------------ allergies
# (allergy keyword regex, drug class/name, severity, text)
ALLERGY_RULES: tuple[tuple[str, str, str, str], ...] = (
    (r"penicillin|amoxicillin|ampicillin", "penicillin", "contraindicated", "Documented penicillin allergy: risk of anaphylaxis."),
    (r"penicillin|amoxicillin|ampicillin", "cephalosporin", "moderate", "Penicillin allergy: about 1-2% cross-reactivity with cephalosporins; use only if the reaction was not severe, or choose a non-beta-lactam."),
    (r"sulfa|sulpha|sulfonamide", "sulfa_antibiotic", "contraindicated", "Documented sulfonamide-antibiotic allergy: risk of severe skin reactions."),
    (r"sulfa|sulpha|sulfonamide", "sulfonamide_nonabx", "info", "Sulfa allergy is listed; cross-reactivity with non-antibiotic sulfonamides is low, but use clinical judgement."),
    (r"ibuprofen|brufen", "ibuprofen", "contraindicated", "Documented ibuprofen allergy."),
    (r"ibuprofen|brufen|aspirin|nsaid", "nsaid", "major", "NSAID/aspirin allergy: cross-reactivity between NSAIDs is common (and can cause bronchospasm or angioedema)."),
    (r"aspirin|salicylate", "aspirin", "contraindicated", "Documented aspirin allergy."),
    (r"codeine|morphine|opioid", "opioid", "major", "Documented opioid allergy."),
    (r"macrolide|erythromycin|azithromycin", "macrolide", "contraindicated", "Documented macrolide allergy."),
    (r"fluoroquinolone|ciprofloxacin|levofloxacin", "fluoroquinolone", "contraindicated", "Documented fluoroquinolone allergy."),
    (r"tetracycline|doxycycline", "tetracycline", "contraindicated", "Documented tetracycline allergy."),
    (r"paracetamol|acetaminophen", "paracetamol", "contraindicated", "Documented paracetamol allergy."),
)


def allergy_findings(allergies: str, drug: str) -> list[dict]:
    cls = classes_of(drug)
    out, seen = [], set()
    for rx, target, sev, text in ALLERGY_RULES:
        if re.search(rx, allergies or "", re.I) and _matches(target, cls) and (target, text) not in seen:
            if target == "nsaid" and drug == "ibuprofen" and re.search("ibuprofen|brufen", allergies or "", re.I):
                continue                                              # the specific ibuprofen rule already says it
            if target == "nsaid" and drug == "aspirin":
                continue                                              # covered by the aspirin rule
            seen.add((target, text))
            out.append({"severity": sev, "type": "allergy", "drugs": [drug], "title": f"Allergy: {drug.title()}", "detail": text,
                        "action": "Do not prescribe." if sev in AVOID else "Prescribe only after reviewing the original reaction."})
    return out


# ------------------------------------------------------------------------------------------------ the patient's medicines
def parse_medications(text: str) -> list[dict]:
    """'Atorvastatin 40 mg nightly; Salbutamol inhaler as needed' -> [{'name': 'atorvastatin', 'text': 'Atorvastatin 40 mg nightly', ...}]"""
    out = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        found = find_drugs(part)
        name = found[0] if found else re.sub(r"\s+\d.*$", "", part).lower()
        out.append({"name": name, "text": part, "classes": sorted(classes_of(name) - {name}), "known": bool(found)})
    return out


def _age_findings(drug: str, p: dict) -> list[dict]:
    age = p.get("age") or 0
    cls = classes_of(drug)
    out = []
    if age >= 65 and "nsaid" in cls:
        out.append({"severity": "moderate", "type": "condition", "drugs": [drug], "title": f"{drug.title()} at age {age}", "detail": "Age 65 or over: higher risk of GI bleeding and kidney injury with NSAIDs.", "action": "Prefer paracetamol; if needed use the lowest dose with a PPI."})
    if age >= 60 and "fluoroquinolone" in cls:
        out.append({"severity": "moderate", "type": "condition", "drugs": [drug], "title": f"{drug.title()} at age {age}", "detail": "Age 60 or over: increased risk of tendinopathy and tendon rupture.", "action": "Consider an alternative antibiotic; warn about tendon pain."})
    if age >= 65 and "sulfonylurea" in cls:
        out.append({"severity": "moderate", "type": "condition", "drugs": [drug], "title": f"{drug.title()} at age {age}", "detail": "Older adults are prone to severe, prolonged hypoglycaemia with sulfonylureas.", "action": "Prefer a lower-risk agent; relax the HbA1c target."})
    if p.get("sex") == "F" and 12 <= age <= 50 and ({"ace", "arb", "statin", "vka"} & cls):
        out.append({"severity": "moderate", "type": "condition", "drugs": [drug], "title": f"{drug.title()} in a woman of child-bearing age", "detail": "Harmful in pregnancy (fetal toxicity or teratogenic).", "action": "Confirm pregnancy status and contraception before prescribing."})
    return out


def condition_findings(drug: str, tags: set[str]) -> list[dict]:
    cls = classes_of(drug)
    return [{"severity": sev, "type": "condition", "drugs": [drug], "title": f"{drug.title()} with {tag.upper() if len(tag) <= 3 else tag}", "detail": text, "action": action}
            for spec, tag, sev, text, action in CONDITION_RULES if tag in tags and _matches(spec, cls)]


def _sorted(findings: list[dict]) -> list[dict]:
    uniq = {(f["type"], f["title"], f["detail"]): f for f in findings}
    return sorted(uniq.values(), key=lambda f: SEV_ORDER[f["severity"]])


def verdict(findings: list[dict]) -> str:
    if any(f["severity"] in AVOID for f in findings):
        return "avoid"
    return "caution" if any(f["severity"] == "moderate" for f in findings) else "ok"


def medication_check(p: dict, drug: str) -> dict:
    """Is it safe to add `drug` for this patient? Checks allergies, every current medicine, and the patient's conditions and age."""
    meds = parse_medications(p.get("medications", ""))
    tags = condition_tags(p.get("conditions", ""))
    findings = allergy_findings(p.get("allergies", ""), drug) + condition_findings(drug, tags) + _age_findings(drug, p)
    for m in meds:
        if m["known"] and m["name"] != drug:
            findings += interaction(drug, m["name"])
        elif m["name"] == drug:
            findings.append({"severity": "info", "type": "duplication", "drugs": [drug], "title": f"Already on {drug.title()}", "detail": f"The record already lists: {m['text']}.", "action": "Review the existing prescription instead of adding another."})
    findings = _sorted(findings)
    return {"drug": drug, "verdict": verdict(findings), "findings": findings, "checked": {"allergies": p.get("allergies") or "none recorded",
            "medications": [m["text"] for m in meds], "conditions": p.get("conditions") or "none recorded", "age": p.get("age")}}


def pair_check(a: str, b: str) -> dict:
    findings = _sorted(interaction(a, b))
    return {"drugs": [a, b], "verdict": verdict(findings), "findings": findings}


# ------------------------------------------------------------------------------------------------ vitals and care gaps
def bp_status(sys_: int | None, dia: int | None) -> tuple[str, str]:
    if not sys_ or not dia:
        return "unknown", "no reading"
    if sys_ >= 180 or dia >= 120:
        return "critical", "hypertensive crisis range (>=180/120)"
    if sys_ >= 140 or dia >= 90:
        return "high", "stage 2 hypertension range (>=140/90)"
    if sys_ >= 130 or dia >= 80:
        return "elevated", "stage 1 range (130-139 / 80-89)"
    if sys_ >= 120:
        return "borderline", "elevated (120-129 / <80)"
    return "normal", "normal (<120/80)"


def hr_status(hr: int | None) -> tuple[str, str]:
    if not hr:
        return "unknown", "no reading"
    return ("low", "bradycardia (<60)") if hr < 60 else ("high", "tachycardia (>100)") if hr > 100 else ("normal", "normal (60-100)")


def bmi_status(b: float | None) -> tuple[str, str]:
    if not b:
        return "unknown", "no reading"
    if b < 18.5:
        return "low", "underweight (<18.5)"
    if b >= 35:
        return "high", "class 2-3 obesity (>=35)"
    if b >= 30:
        return "elevated", "obesity (>=30)"
    return ("borderline", "overweight (25-29.9)") if b >= 25 else ("normal", "normal (18.5-24.9)")


def a1c_status(v: float | None) -> tuple[str, str]:
    if not v:
        return "unknown", "no result recorded"
    if v >= 9.0:
        return "high", "poor control / well above target (>=9%)"
    if v >= 6.5:
        return "elevated", "diabetes range (>=6.5%)"
    return ("borderline", "prediabetes range (5.7-6.4%)") if v >= 5.7 else ("normal", "normal (<5.7%)")


def ldl_status(v: int | None) -> tuple[str, str]:
    if not v:
        return "unknown", "no result recorded"
    if v >= 190:
        return "high", "very high (>=190 mg/dL)"
    if v >= 160:
        return "elevated", "high (160-189)"
    if v >= 130:
        return "borderline", "borderline high (130-159)"
    return ("borderline", "near optimal (100-129)") if v >= 100 else ("normal", "optimal (<100)")


def _alert(sev, cat, title, detail, action=""):
    return {"severity": sev, "category": cat, "title": title, "detail": detail, "action": action}


def care_gaps(p: dict, today: date | None = None) -> list[dict]:
    """Guideline-based gaps and abnormal results in the chart (ACC/AHA BP, ADA diabetes, lipid and screening thresholds)."""
    today = today or date.today()
    tags = condition_tags(p.get("conditions", ""))
    meds = parse_medications(p.get("medications", ""))
    classes = {c for m in meds for c in m["classes"]} | {m["name"] for m in meds}
    out: list[dict] = []
    sys_, dia, a1c, ldl, bmi = p.get("bp_sys"), p.get("bp_dia"), p.get("hba1c"), p.get("ldl"), p.get("bmi")

    st, why = bp_status(sys_, dia)
    if st == "critical":
        out.append(_alert("critical", "blood pressure", f"BP {sys_}/{dia}: {why}", "Severe elevation.", "Repeat now; assess for end-organ symptoms and treat urgently."))
    elif st == "high":
        out.append(_alert("warning", "blood pressure", f"BP {sys_}/{dia}: {why}", "Hypertension on treatment is not at target." if "htn" in tags else "Raised BP with no hypertension diagnosis recorded.",
                          "Review adherence and intensify treatment (ACC/AHA target <130/80)." if "htn" in tags else "Confirm with repeat or home readings and evaluate for hypertension."))
    elif st == "elevated" and "htn" in tags:
        out.append(_alert("info", "blood pressure", f"BP {sys_}/{dia}: above the <130/80 target", "Treated hypertension, not yet at ACC/AHA target.", "Consider lifestyle measures and treatment review."))

    if "dm" in tags:
        if not a1c:
            out.append(_alert("warning", "diabetes", "No HbA1c recorded", "Diabetes without a recent HbA1c.", "Order HbA1c (at least every 6-12 months)."))
        elif a1c >= 9.0:
            out.append(_alert("warning", "diabetes", f"HbA1c {a1c}%: poorly controlled", "Well above the usual <7% target (ADA).", "Intensify therapy; check adherence, diet and complications."))
        elif a1c >= 7.0:
            out.append(_alert("info", "diabetes", f"HbA1c {a1c}%: above the general <7% target", "Individualise the target (age, comorbidity, hypoglycaemia risk).", "Review therapy at the next visit."))
        if "metformin" not in classes and "insulin" not in classes and "sulfonylurea" not in classes and "thiazolidinedione" not in classes:
            out.append(_alert("info", "diabetes", "No glucose-lowering medicine recorded", "Type 2 diabetes with no antidiabetic drug on the list.", "Confirm whether lifestyle-only management is intended."))
    elif a1c and a1c >= 6.5:
        out.append(_alert("warning", "diabetes", f"HbA1c {a1c}%: diabetes range, no diagnosis recorded", "Meets the diagnostic threshold (>=6.5%).", "Confirm with a repeat test and document the diagnosis."))
    elif a1c and a1c >= 5.7:
        out.append(_alert("info", "diabetes", f"HbA1c {a1c}%: prediabetes range", "Raised risk of progression to diabetes.", "Lifestyle advice and annual testing."))

    if ldl:
        if ldl >= 190:
            out.append(_alert("warning", "lipids", f"LDL {ldl} mg/dL: very high", "LDL of 190 or more warrants high-intensity statin therapy and a check for familial hypercholesterolaemia.", "Treat; consider a secondary-cause and family screen."))
        elif "statin" not in classes and ldl >= 130 and 40 <= (p.get("age") or 0) <= 75 and ("dm" in tags or "cvd" in tags or "hld" in tags):
            out.append(_alert("info", "lipids", f"LDL {ldl} mg/dL with no statin recorded", "Statin therapy is indicated for diabetes or hyperlipidaemia at this age.", "Consider starting a statin (shared decision)."))
        elif "statin" in classes and ldl >= 100 and ("dm" in tags or "cvd" in tags):
            out.append(_alert("info", "lipids", f"LDL {ldl} mg/dL on a statin: above goal", "Diabetes/CVD goals are typically <100 (<70 with vascular disease).", "Check adherence; consider intensifying."))

    if "af" in tags and "anticoagulant" not in classes:
        out.append(_alert("warning", "anticoagulation", "Atrial fibrillation with no anticoagulant recorded", "Stroke prevention is indicated for most patients (assess CHA2DS2-VASc).", "Assess stroke risk and bleeding risk; consider anticoagulation."))
    if "hf" in tags and not ({"ace", "arb", "beta_blocker"} & classes):
        out.append(_alert("info", "heart failure", "Heart failure without ACE inhibitor/ARB or beta-blocker recorded", "These are cornerstone therapies for HFrEF.", "Review guideline-directed medical therapy."))
    if bmi and bmi >= 30:
        out.append(_alert("info", "weight", f"BMI {bmi}: {bmi_status(bmi)[1]}", "Obesity is a risk factor for diabetes, hypertension and OA.", "Offer weight-management support."))
    if p.get("smoker"):
        out.append(_alert("warning" if ({"copd", "asthma", "cvd"} & tags) else "info", "smoking", "Current smoker", "Smoking cessation is the most effective single intervention.", "Offer cessation advice and pharmacotherapy."))
    if len(meds) >= 5:
        out.append(_alert("info", "polypharmacy", f"{len(meds)} regular medicines", "Polypharmacy raises interaction and adherence risk.", "Consider a structured medication review."))
    lv = p.get("last_visit")
    try:
        months = (today - date.fromisoformat(lv)).days / 30.4 if lv else None
    except ValueError:
        months = None
    if months and months > 12 and tags:
        out.append(_alert("info", "follow-up", f"Last visit {round(months)} months ago", "Chronic conditions are due for review (usually at least yearly).", "Offer a review appointment."))
    return out


def review(p: dict, today: date | None = None) -> list[dict]:
    """Everything worth a clinician's attention in the chart: prescribing problems first, then care gaps; most severe first."""
    meds = [m for m in parse_medications(p.get("medications", "")) if m["known"]]
    tags = condition_tags(p.get("conditions", ""))
    alerts: list[dict] = []
    for m in meds:
        for f in allergy_findings(p.get("allergies", ""), m["name"]):
            if f["severity"] != "info":
                alerts.append(_alert("critical" if f["severity"] in AVOID else "warning", "allergy", f"On {m['name'].title()} despite a recorded allergy", f["detail"], "Review urgently and stop or confirm the allergy status."))
        for f in condition_findings(m["name"], tags) + _age_findings(m["name"], p):
            why = re.sub(rf"^{re.escape(m['name'].title())} (?:with|at|in) ", "", f["title"])           # 'Metformin with CKD' -> 'CKD'
            if f["severity"] in AVOID:
                alerts.append(_alert("critical" if f["severity"] == "contraindicated" else "warning", "drug-condition", f"{m['name'].title()} in this patient: {why}", f["detail"], f["action"]))
            elif f["severity"] == "moderate":
                alerts.append(_alert("info", "drug-condition", f"{m['name'].title()}: {why}", f["detail"], f["action"]))
    for i, a in enumerate(meds):
        for b in meds[i + 1:]:
            if a["name"] == b["name"]:
                continue
            for f in interaction(a["name"], b["name"]):
                if f["severity"] in AVOID:
                    alerts.append(_alert("critical" if f["severity"] == "contraindicated" else "warning", "interaction", f"{f['title']}: {f['severity']} interaction", f["detail"], f["action"]))
                elif f["severity"] == "moderate":
                    alerts.append(_alert("info", "interaction", f"{f['title']}: moderate interaction", f["detail"], f["action"]))
    alerts += care_gaps(p, today)
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted({(a["category"], a["title"]): a for a in alerts}.values(), key=lambda a: order[a["severity"]])


def summarize(p: dict, today: date | None = None) -> dict:
    meds = parse_medications(p.get("medications", ""))
    vit = []
    for key, label, value, unit, status in (
        ("bp", "Blood pressure", f"{p['bp_sys']}/{p['bp_dia']}" if p.get("bp_sys") else None, "mmHg", bp_status(p.get("bp_sys"), p.get("bp_dia"))),
        ("hr", "Heart rate", p.get("heart_rate"), "bpm", hr_status(p.get("heart_rate"))),
        ("bmi", "BMI", p.get("bmi"), "", bmi_status(p.get("bmi"))),
        ("hba1c", "HbA1c", p.get("hba1c") or None, "%", a1c_status(p.get("hba1c"))),
        ("ldl", "LDL", p.get("ldl"), "mg/dL", ldl_status(p.get("ldl")))):
        if value:
            vit.append({"key": key, "label": label, "value": value, "unit": unit, "status": status[0], "note": status[1]})
    return {
        "patient": {"patient_id": p["patient_id"], "full_name": p["full_name"], "age": p["age"], "sex": p["sex"], "city": p.get("city")},
        "conditions": [c.strip() for c in (p.get("conditions") or "").split(",") if c.strip()],
        "medications": [{"name": m["name"].title(), "text": m["text"], "classes": m["classes"]} for m in meds],
        "allergies": [a.strip() for a in (p.get("allergies") or "").split(",") if a.strip()],
        "smoker": bool(p.get("smoker")), "last_visit": p.get("last_visit"),
        "vitals": vit, "alerts": review(p, today),
        "recent_encounters": p.get("recent_encounters", [])[:3],
    }


# ------------------------------------------------------------------------------------------------ chat intents
_CHECK_CUE = re.compile(r"\b(can i|could i|should i|may i|can we|should we|is it (?:ok|okay|safe|fine|alright)|safe to|ok to|okay to|any (?:issues?|problems?|concerns?)|"
                        r"prescrib\w*|start|add|give|put (?:him|her|them)|initiate|commence|co-?prescribe|contraindicat\w*|suitable|appropriate)\b", re.I)
_PAIR_CUE = re.compile(r"\b(interact\w*|together|combin\w*|with|and|plus|alongside|concurrent\w*|co-?administ\w*|mix)\b", re.I)
_REVIEW_CUE = re.compile(r"\b(summar\w*|overview|snapshot|alerts?|care gaps?|gaps?|red flags? (?:in|for) (?:this|the) patient|what (?:should|do) i (?:know|worry|watch)|"
                         r"medication review|medicines? review|drug review|review\b[^.?]{0,30}\b(?:medications?|medicines?|meds|drugs?|chart|record|patient)\b|"
                         r"check\b[^.?]{0,25}\b(?:medications?|medicines?|meds|chart)\b|any (?:problems?|issues?|concerns?|interactions?) "
                         r"(?:with|in|among) (?:his|her|their|the|this)|prescribing (?:problems?|issues?)|anything (?:i )?(?:should|need to) (?:know|worry|flag))", re.I)
_TARGET_CUE = re.compile(r"\b(at target|on target|at goal|controlled|well controlled|in range|acceptable|good enough|ok|okay)\b", re.I)
_DOSE_CUE = re.compile(r"\b(dose|dosage|dosing|how (?:much|many)|\bmg\b|maximum|max|usual)\b", re.I)

SEV_WORD = {"contraindicated": "CONTRAINDICATED", "major": "MAJOR", "moderate": "moderate", "info": "note"}
VERDICT_HEAD = {"avoid": "Not recommended", "caution": "Use with caution", "ok": "No problems found in these checks"}


def _finding_lines(findings: list[dict], limit: int = 8) -> str:
    return "\n".join(f"• [{SEV_WORD[f['severity']]}] {f['title']}: {f['detail']} {f['action']}".strip() for f in findings[:limit])


def render_medcheck(p: dict, res: dict) -> str:
    d = res["drug"].title()
    head = f"{VERDICT_HEAD[res['verdict']]}: {d} for {p['full_name']} ({p['patient_id']})."
    body = _finding_lines(res["findings"])
    chk = res["checked"]
    scope = (f"Checked against: allergies ({chk['allergies']}); {len(chk['medications'])} current medicine(s); conditions ({chk['conditions']}); age {chk['age']}.")
    foot = "Decision support from curated references: verify against your formulary, renal function (eGFR) and the patient's history." if res["findings"] else \
        "This covers common interactions and cautions only; it is not a guarantee of safety. Check eGFR, pregnancy status and local guidance."
    return "\n".join(x for x in (head, body, scope, foot) if x)


def render_pair(res: dict) -> str:
    a, b = (x.title() for x in res["drugs"])
    if not res["findings"]:
        return f"No interaction between {a} and {b} found in the curated checks. This is not a guarantee: check a full interaction reference for your patient."
    return f"{VERDICT_HEAD[res['verdict']]}: {a} with {b}.\n{_finding_lines(res['findings'])}\nBased on standard prescribing references; verify against your formulary."


def render_review(s: dict) -> str:
    pt = s["patient"]
    lines = [f"{pt['full_name']} ({pt['patient_id']}), {pt['age']}y {pt['sex']}. Conditions: {', '.join(s['conditions']) or 'none recorded'}. "
             f"Medicines: {len(s['medications'])}. Allergies: {', '.join(s['allergies']) or 'none known'}."]
    if s["alerts"]:
        lines.append(f"{len(s['alerts'])} item(s) worth attention:")
        for a in s["alerts"][:10]:
            tag = {"critical": "CRITICAL", "warning": "warning", "info": "note"}[a["severity"]]
            lines.append(f"• [{tag}] {a['title']}. {a['detail']}" + (f" {a['action']}" if a["action"] else ""))
    else:
        lines.append("No prescribing problems or care gaps found in the available checks.")
    return "\n".join(lines)


def answer(question: str, p: dict | None) -> dict | None:
    """A deterministic answer for clinical-safety questions, or None to let the normal pipeline handle it."""
    drugs = find_drugs(question)
    q = question
    if len(drugs) >= 2 and _PAIR_CUE.search(q) and not _DOSE_CUE.search(q):
        res = pair_check(drugs[0], drugs[1])
        return _wrap(render_pair(res), "pair", res)
    if p:
        # "Is ibuprofen safe with kidney disease?" names a condition THIS patient does not have: that is a general question, so it must
        # not be answered as a check of this patient ("no problems found").
        foreign = {t for t, rx in CONDITION_TAGS.items() if re.search(rx, q, re.I)} - condition_tags(p.get("conditions", ""))
        if len(drugs) == 1 and _CHECK_CUE.search(q) and not _DOSE_CUE.search(q) and not foreign:
            res = medication_check(p, drugs[0])
            return _wrap(render_medcheck(p, res), "medcheck", res)
        if _REVIEW_CUE.search(q):
            s = summarize(p)
            return _wrap(render_review(s), "review", {"alerts": s["alerts"]})
        if _TARGET_CUE.search(q):
            m = re.search(r"\b(bp|blood pressure|hba1c|a1c|ldl|cholesterol|bmi|heart rate|pulse)\b", q, re.I)
            if m:
                return _wrap(_target_answer(p, m.group(1).lower()), "target", {})
    return None


def _wrap(text: str, kind: str, data: dict) -> dict:
    return {"text": text, "kind": kind, "data": data}


def _target_answer(p: dict, what: str) -> str:
    name = f"{p['full_name']} ({p['patient_id']})"
    tags = condition_tags(p.get("conditions", ""))
    if what in ("bp", "blood pressure"):
        st, why = bp_status(p.get("bp_sys"), p.get("bp_dia"))
        goal = "below 130/80 mmHg (ACC/AHA" + ("; also the ADA goal in diabetes)" if "dm" in tags else ")")
        at = p.get("bp_sys") and p["bp_sys"] < 130 and p["bp_dia"] < 80
        return f"{name}: BP {p.get('bp_sys')}/{p.get('bp_dia')} mmHg is {'at target' if at else 'above target'}. Goal: {goal}. Category: {why}."
    if what in ("hba1c", "a1c"):
        v = p.get("hba1c")
        if not v:
            return f"{name}: no HbA1c is recorded."
        goal = "below 7% for most adults with diabetes (ADA), individualised for age and hypoglycaemia risk" if "dm" in tags else "below 5.7% (normal)"
        return f"{name}: HbA1c {v}% is {a1c_status(v)[1]}. Goal: {goal}."
    if what in ("ldl", "cholesterol"):
        v = p.get("ldl")
        goal = "below 100 mg/dL (below 70 with established vascular disease)" if ({"dm", "cvd"} & tags) else "below 130 mg/dL for most adults, lower with risk factors"
        return f"{name}: LDL {v} mg/dL is {ldl_status(v)[1]}. Goal: {goal}." if v else f"{name}: no LDL is recorded."
    if what == "bmi":
        return f"{name}: BMI {p.get('bmi')} is {bmi_status(p.get('bmi'))[1]}."
    st, why = hr_status(p.get("heart_rate"))
    return f"{name}: heart rate {p.get('heart_rate')} bpm is {why}."
