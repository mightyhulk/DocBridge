"""Nearby doctors / hospitals for the patient consult, from OpenStreetMap (Overpass + Nominatim): free, no API key.

Privacy: only rounded coordinates (about 1 km) and the *type* of specialist are sent; never symptoms or identity.

Honest limits: OpenStreetMap lists facilities, not live availability, and its `healthcare:speciality` tag is only filled in
for part of the world. Results are therefore labelled "verified" (the listing names the speciality), "hospital" (a hospital,
which usually has the department) or "unverified" (a clinic/doctor with no speciality listed: call ahead).
"""
import asyncio
import math
import re
import time

import httpx

from .config import get_settings

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
NOMINATIM = "https://nominatim.openstreetmap.org"
RADII_M = [5000, 15000, 40000]

_cache: dict[tuple, tuple[float, list[dict]]] = {}   # raw Overpass results by (lat, lon, radius)


def round_coord(v: float) -> float:
    return round(float(v), get_settings().geo_precision_digits)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def build_query(lat: float, lon: float, radius_m: int) -> str:
    return (f'[out:json][timeout:25];(nwr["amenity"~"^(doctors|clinic|hospital|dentist)$"](around:{radius_m},{lat},{lon});'
            f'nwr["healthcare"~"^(doctor|clinic|hospital|dentist)$"](around:{radius_m},{lat},{lon}););out center tags 200;')


def _address(t: dict) -> str:
    street = " ".join(x for x in (t.get("addr:housenumber"), t.get("addr:street")) if x)
    parts = [street, t.get("addr:suburb") or t.get("addr:neighbourhood"), t.get("addr:city") or t.get("addr:town") or t.get("addr:village"),
             t.get("addr:postcode")]
    return ", ".join(p for p in parts if p) or t.get("addr:full", "")


def parse_elements(data: dict, lat: float, lon: float) -> list[dict]:
    """Overpass JSON -> list of facilities (named only) with distance from the patient."""
    out, seen = [], set()
    for el in data.get("elements", []):
        t = el.get("tags") or {}
        name = (t.get("name") or t.get("name:en") or "").strip()
        clat = el.get("lat") or (el.get("center") or {}).get("lat")
        clon = el.get("lon") or (el.get("center") or {}).get("lon")
        if not name or clat is None or clon is None:
            continue
        kind = t.get("amenity") or t.get("healthcare") or "clinic"
        kind = {"doctor": "doctors"}.get(kind, kind)
        key = (name.lower(), round(clat, 3), round(clon, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "name": name, "kind": kind,
            "speciality": (t.get("healthcare:speciality") or t.get("healthcare:specialty") or "").replace("_", " ").lower(),
            "emergency": t.get("emergency") == "yes",
            "distance_km": round(haversine_km(lat, lon, clat, clon), 1),
            "address": _address(t),
            "phone": t.get("phone") or t.get("contact:phone") or "",
            "website": t.get("website") or t.get("contact:website") or "",
            "opening_hours": t.get("opening_hours", ""),
            "lat": clat, "lon": clon,
            "map_url": f"https://www.openstreetmap.org/?mlat={clat}&mlon={clon}#map=17/{clat}/{clon}",
            "directions_url": f"https://www.google.com/maps/dir/?api=1&destination={clat},{clon}",
        })
    return out


# Facilities that must never be offered as the doctor for a medical complaint.
_NOT_ALLOPATHIC = re.compile(r"ayurved|ayurvedic|homoeo|homeo|unani|siddha|naturopath|veterinar|animal|\bpet\b|yoga|spa\b", re.I)
# What a facility's *name* tells us when OpenStreetMap has no speciality tag: (name regex, speciality tokens it implies)
_NAME_HINTS = [
    (r"cardi|heart|vascular", ["cardio"]), (r"neuro|brain|spine", ["neuro"]), (r"ortho|bone|joint|trauma", ["orthop", "trauma"]),
    (r"gastro|liver|hepat|laparoscop", ["gastro"]), (r"\beye\b|ophthal|netralaya|vision", ["ophthal", "eye"]),
    (r"dental|dentist|teeth", ["dent"]), (r"maternity|gynae?c|women|mother|\bivf\b|fertility|obstet", ["gyn", "obstet"]),
    (r"child|paediatr|pediatr|\bkids?\b|baby|newborn|shishu", ["paediatr", "pediatr", "child"]),
    (r"urolog|kidney|nephro|renal|dialysis", ["urolog", "nephro", "kidney"]), (r"cancer|oncolog|\bonco", ["onco"]),
    (r"\bskin\b|derma|cosmetic|hair", ["dermat"]), (r"\bent\b|ear nose", ["otolaryng", "ent", "ear_nose"]),
    (r"mental|psychiat|de-?addiction|neuropsych", ["psychiat", "psycholog"]), (r"diabet|endocrin|thyroid", ["endocrin", "diabet"]),
    (r"chest|lung|pulmon|respirat|tb\b", ["pulmon", "respirat", "chest"]),
]
_NO_EMERGENCY_FOR = ("ophthal", "dent", "dermat", "otolaryng")        # an eye/dental/skin clinic is not where chest pain goes


def name_specialties(name: str) -> list[list[str]]:
    return [toks for rx, toks in _NAME_HINTS if re.search(rx, name, re.I)]


def _overlaps(hint_tokens: list[str], wanted: list[str]) -> bool:
    return any(h in w or w in h for h in hint_tokens for w in wanted)


def _match(f: dict, tokens: list[str]) -> str:
    spec = f["speciality"]
    if any(tok in spec for tok in tokens) or (f["kind"] == "dentist" and "dent" in tokens):
        return "verified"
    if f["kind"] == "hospital":
        return "hospital"
    return "unverified"


def rank(facilities: list[dict], tokens: list[str], *, emergency: bool = False, limit: int = 8) -> list[dict]:
    """Verified specialists first, then hospitals, then other clinics; nearest first inside each group.
    For emergencies only hospitals are returned (emergency departments first)."""
    general = any(t in ("general", "internal", "family", "primary") for t in tokens)
    res = []
    for f in facilities:
        if _NOT_ALLOPATHIC.search(f["name"]):
            continue                                             # Ayurvedic / homeopathic / veterinary: not who to send a patient to
        hints = name_specialties(f["name"])
        m = _match(f, tokens)
        if emergency:
            if f["kind"] != "hospital" or any(_overlaps(h, list(_NO_EMERGENCY_FOR)) for h in hints):
                continue
            m = "hospital"
        elif f["kind"] == "dentist" and "dent" not in tokens:
            continue                                             # never send a headache patient to a dentist
        else:
            if hints and m != "verified" and f["kind"] != "dentist":
                # No speciality tag, but the name says what it is: 'Heart Care Hospital' suits a cardiologist, 'Eye Hospital' does not
                m = "verified" if any(_overlaps(h, tokens) for h in hints) else "unverified"
            elif general and m == "unverified" and f["kind"] in ("doctors", "clinic"):
                m = "verified"                                   # a GP / clinic is exactly what a general complaint needs
        res.append({**f, "match": m})
    order = {"verified": 0, "hospital": 1, "unverified": 2}
    res.sort(key=lambda f: ((0 if f["emergency"] else 1) if emergency else order[f["match"]], f["distance_km"]))
    return res[:limit]


async def _overpass(client: httpx.AsyncClient, query: str) -> dict:
    """Race the public Overpass mirrors and take the first good answer (any one of them is often down or overloaded)."""
    async def one(url: str) -> dict:
        r = await client.post(url, data={"data": query}, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    tasks = {asyncio.create_task(one(u)): u for u in OVERPASS}
    errors: list[str] = []
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
                errors.append(f"{tasks[t].split('/')[2]}: {type(t.exception()).__name__} {t.exception()}".strip())
    finally:
        for t in tasks:
            t.cancel()
    raise RuntimeError("Overpass unavailable (" + "; ".join(errors) + ")")


def parse_nominatim(rows: list[dict], lat: float, lon: float) -> list[dict]:
    """Nominatim search results -> the same facility dicts as parse_elements (used when Overpass is down)."""
    elements = []
    for r in rows:
        tags = {**(r.get("extratags") or {}), "name": r.get("name") or (r.get("namedetails") or {}).get("name", ""),
                "amenity": r.get("type") if r.get("type") in ("hospital", "clinic", "doctors", "dentist") else ""}
        a = r.get("address") or {}
        tags.update({"addr:street": a.get("road", ""), "addr:suburb": a.get("suburb", ""), "addr:city": a.get("city") or a.get("town") or a.get("village", ""),
                     "addr:postcode": a.get("postcode", "")})
        if tags["amenity"] and tags["name"]:
            elements.append({"type": "node", "lat": float(r["lat"]), "lon": float(r["lon"]), "tags": tags})
    return parse_elements({"elements": elements}, lat, lon)


async def _nominatim_facilities(client: httpx.AsyncClient, lat: float, lon: float, radius: int) -> list[dict]:
    d_lat = radius / 111_000
    d_lon = radius / (111_000 * max(math.cos(math.radians(lat)), 0.2))
    box = f"{lon - d_lon},{lat + d_lat},{lon + d_lon},{lat - d_lat}"           # left,top,right,bottom
    rows: list[dict] = []
    for i, term in enumerate(("hospital", "clinic", "doctor")):
        if i:
            await asyncio.sleep(1.1)                                              # Nominatim usage policy: max 1 request/second
        r = await client.get(f"{NOMINATIM}/search", params={"q": term, "format": "jsonv2", "limit": 40, "viewbox": box, "bounded": 1,
                                                            "extratags": 1, "addressdetails": 1}, timeout=15)
        if r.status_code == 200:
            rows += r.json()
    return parse_nominatim(rows, lat, lon)


async def _facilities(client: httpx.AsyncClient, lat: float, lon: float, radius: int) -> list[dict]:
    """All named doctors/clinics/hospitals/dentists around a point. Cached: one Overpass call serves every speciality."""
    key = ("raw", lat, lon, radius)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]
    try:
        fac = parse_elements(await _overpass(client, build_query(lat, lon, radius)), lat, lon)
    except Exception as e:
        try:                                                  # Overpass down: Nominatim's search still knows hospitals and clinics
            fac = await _nominatim_facilities(client, lat, lon, radius)
        except Exception as e2:
            raise RuntimeError(f"{e}; Nominatim fallback failed: {type(e2).__name__} {e2}".strip()) from e2
        if not fac:
            raise
    _cache[key] = (time.time(), fac)
    return fac


async def nearby(lat: float, lon: float, tokens: list[str], *, emergency: bool = False) -> dict:
    """Best matches around the patient. Returns {"results": [...], "radius_km": n, "error": str|None}."""
    s = get_settings()
    lat, lon = round_coord(lat), round_coord(lon)
    results, radius, error = [], 0, None
    enough = 1 if emergency else 3
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": s.geo_user_agent}) as client:
        for radius in RADII_M:
            try:
                fac = await _facilities(client, lat, lon, radius)
            except Exception as e:
                error = str(e)
                break
            results = rank(fac, tokens, emergency=emergency, limit=s.doctors_max_results)
            if sum(r["match"] in ("verified", "hospital") for r in results) >= enough:
                break
    return {"results": results, "radius_km": radius // 1000, "error": error if not results else None}


async def geocode(place: str) -> dict | None:
    """'Bhubaneswar' / '751024' / 'Connaught Place, Delhi' -> {lat, lon, label} (Nominatim). Place names only, no symptoms."""
    s = get_settings()
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": s.geo_user_agent}) as client:
        r = await client.get(f"{NOMINATIM}/search", params={"q": place[:120], "format": "jsonv2", "limit": 1})
        r.raise_for_status()
        rows = r.json()
    if not rows:
        return None
    return {"lat": float(rows[0]["lat"]), "lon": float(rows[0]["lon"]), "label": rows[0].get("display_name", place)[:120]}


async def reverse_geocode(lat: float, lon: float) -> str:
    """Friendly place name for the coordinates, or '' on any failure (cosmetic only)."""
    s = get_settings()
    try:
        async with httpx.AsyncClient(timeout=6, headers={"User-Agent": s.geo_user_agent}) as client:
            r = await client.get(f"{NOMINATIM}/reverse", params={"lat": round_coord(lat), "lon": round_coord(lon),
                                                                  "format": "jsonv2", "zoom": 12})
            a = r.json().get("address", {})
        return ", ".join(x for x in (a.get("suburb") or a.get("city_district"), a.get("city") or a.get("town") or a.get("village"),
                                     a.get("state")) if x)
    except Exception:
        return ""
