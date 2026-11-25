// Titles and grouping for the History list.

// Replies to the welcome prompt ("Do you want to change any of these?"), greetings and one-word answers say nothing about
// the problem, so they must never become the title of a saved consultation.
const FILLER = /^(?:(?:no|nope|nah|yes|yeah|yep|ok|okay|sure|hello|hi|hey|thanks?|thank you|never|none|nothing|uhm+|um+|hmm+|good (?:morning|day|evening)|have a good day|no changes?|no change needed|(?:i )?(?:do not|don'?t) (?:want|need) to change(?: anything)?|(?:no,? )?(?:i )?(?:do not|don'?t) want to change(?: anything)?|(?:yes,? )?i have your details|i have to change .*|i (?:do not|don'?t) know(?: how to change)?)\W*)+$/i;

export const isFiller = (t) => {
  const s = (t || "").replace(/\s+/g, " ").trim();
  return s.split(" ").filter(Boolean).length < 2 && s.length < 4 || FILLER.test(s);
};

const clip = (s, n = 48) => {
  s = s.replace(/\s+/g, " ").trim().replace(/[.!?…]+$/, "");
  if (s.length <= n) return s;
  const cut = s.slice(0, n);
  return cut.slice(0, Math.max(cut.lastIndexOf(" "), 20)).trim() + "…";
};

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

// The first thing the patient said that actually describes a problem.
export function pickTitle(log, fallback = "Consultation") {
  const m = (log || []).find((x) => x.role === "user" && !isFiller(x.content));
  return m ? cap(clip(m.content)) : fallback;
}

// Stored titles from older versions can be filler ("No", "No changes") - show those as a plain label instead.
export const showTitle = (h) => (h.kind === "assistant" ? cap(clip(h.title || "Chat")) : isFiller(h.title) ? "Consultation" : cap(clip(h.title)));

export const LEVEL_LABEL = { self_care: "Self-care", see_doctor: "See a doctor", urgent: "Urgent", emergency: "Emergency" };

const day = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();

// "Today" / "Yesterday" / "Earlier" bucket for the group headers.
export function bucket(iso, now = new Date()) {
  const diff = Math.round((day(now) - day(new Date(iso))) / 864e5);
  return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : diff < 7 ? "This week" : "Earlier";
}

// "3:18 PM" for today and yesterday, "Oct 22" otherwise.
export function when(iso, now = new Date()) {
  const d = new Date(iso), diff = Math.round((day(now) - day(d)) / 864e5);
  return diff <= 1 ? d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : d.toLocaleDateString([], { month: "short", day: "numeric" });
}
