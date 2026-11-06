"""Ochil Dasturxon AI — Uzbek dish classifier (Streamlit web app).

Design inspired by the tilework of the Registan in Samarkand: lapis and turquoise
tiles, eight-pointed girih stars, gold accents and a pishtaq (arched portal) header.
"""

import base64
import io
from urllib.parse import quote

import streamlit as st
from PIL import Image

from classifier import CLASSES, load_model, predict

# ----- Display names & project stats -----
LABELS = {"lagman": "Lagman", "manti": "Manti", "plov": "Plov", "somsa": "Somsa"}
EMOJI = {"lagman": "🍜", "manti": "🥟", "plov": "🍚", "somsa": "🥐"}
LOW_CONFIDENCE = 0.60
STATS = [("99.3%", "test accuracy"), ("0.993", "macro F1"), ("405", "test images")]

# ----- Palette (Registan tiles) -----
LAPIS, LAPIS_DEEP, TURQ, GOLD, CREAM = "#14398a", "#0a1d4d", "#2fb3c0", "#d9a441", "#f3e7cf"

st.set_page_config(page_title="Ochil Dasturxon AI · Uzbek Dish Classifier", page_icon="🕌", layout="centered")


def svg_uri(svg):
    return "data:image/svg+xml;utf8," + quote(svg)


# eight-pointed girih star (two overlapping squares) used across the page
def star(cx, cy, r, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    s1 = f"{cx-r},{cy-r} {cx+r},{cy-r} {cx+r},{cy+r} {cx-r},{cy+r}"
    d = r * 1.414
    s2 = f"{cx},{cy-d} {cx+d},{cy} {cx},{cy+d} {cx-d},{cy}"
    return f'<polygon points="{s1}" {a}/><polygon points="{s2}" {a}/>'


# page background: sandy brick wall with a faint carved star lattice
BRICK = "#8a5a2b"
BG = svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84">'
    + star(42, 42, 13, fill="#fff8e6", fill_opacity=".22", stroke=BRICK, stroke_opacity=".22", stroke_width="1")
    + star(0, 0, 6, fill=BRICK, fill_opacity=".14") + star(84, 0, 6, fill=BRICK, fill_opacity=".14")
    + star(0, 84, 6, fill=BRICK, fill_opacity=".14") + star(84, 84, 6, fill=BRICK, fill_opacity=".14")
    + "</svg>"
)

# tile frieze band (used as borders and dividers)
BAND = svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="18">'
    f'<rect width="36" height="18" fill="{LAPIS}"/>'
    f'<polygon points="18,1 34,9 18,17 2,9" fill="{TURQ}"/>'
    f'<polygon points="18,5 26,9 18,13 10,9" fill="{CREAM}"/>'
    f'<circle cx="18" cy="9" r="1.6" fill="{LAPIS}"/>'
    f'<circle cx="0" cy="9" r="2" fill="{GOLD}"/><circle cx="36" cy="9" r="2" fill="{GOLD}"/>'
    "</svg>"
)

# pishtaq (arched portal) header with tiled spandrels
TILE = (
    '<pattern id="tile" width="44" height="44" patternUnits="userSpaceOnUse">'
    f'<rect width="44" height="44" fill="{LAPIS}"/>'
    + star(22, 22, 9, fill=TURQ, stroke=CREAM, stroke_width="1")
    + f'<circle cx="22" cy="22" r="3.2" fill="{CREAM}"/>'
    + star(0, 0, 4, fill=GOLD) + star(44, 0, 4, fill=GOLD) + star(0, 44, 4, fill=GOLD) + star(44, 44, 4, fill=GOLD)
    + "</pattern>"
)
ARCH = "M150 372 L150 190 Q150 82 360 34 Q570 82 570 190 L570 372 Z"
INNER = "M168 372 L168 194 Q168 100 360 56 Q552 100 552 194 L552 372"
HERO_SVG = f"""
<svg class="pishtaq" viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Which Uzbek dish is this?">
  <defs>{TILE}
    <linearGradient id="niche" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0d2a6b"/><stop offset="1" stop-color="{LAPIS_DEEP}"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="720" height="380" rx="8" fill="url(#tile)"/>
  <rect x="10" y="10" width="700" height="370" rx="4" fill="none" stroke="{GOLD}" stroke-width="3"/>
  <rect x="22" y="22" width="676" height="358" fill="none" stroke="{CREAM}" stroke-opacity=".55" stroke-width="1"/>
  <path d="{ARCH}" fill="url(#niche)" stroke="{GOLD}" stroke-width="4"/>
  <path d="{INNER}" fill="none" stroke="{TURQ}" stroke-width="1.5" stroke-dasharray="6 5"/>
  <text x="360" y="134" text-anchor="middle" class="t-eyebrow">OCHIL DASTURXON AI</text>
  <text x="360" y="196" text-anchor="middle" class="t-title">Which <tspan class="t-gold">Uzbek dish</tspan></text>
  <text x="360" y="246" text-anchor="middle" class="t-title">is this?</text>
  <text x="360" y="292" text-anchor="middle" class="t-sub">A ResNet18 fine-tuned on Central Asian dishes</text>
  <text x="360" y="316" text-anchor="middle" class="t-sub">will name your photo</text>
  {star(360, 352, 8, fill=GOLD)}
</svg>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Marcellus&display=swap');
:root { --lapis:__LAPIS__; --deep:__DEEP__; --turq:__TURQ__; --gold:__GOLD__; --cream:__CREAM__;
        --muted:#b8c6e0; --card:rgba(10,29,77,.94); }

.stApp {
  background:
    url("__BG__"),
    repeating-linear-gradient(0deg, rgba(138,90,43,.07) 0 1px, transparent 1px 28px),
    radial-gradient(circle at 50% 0%, #f3dca8 0%, #e2bf7f 45%, #c99a55 100%);
  background-attachment: fixed; color: var(--cream);
}
header[data-testid="stHeader"], #MainMenu, footer { visibility:hidden; height:0; }
.block-container { padding-top: 2.2rem; max-width: 780px; }

.band { height: 18px; background: url("__BAND__") repeat-x center; border-top:2px solid var(--gold); border-bottom:2px solid var(--gold); }

/* header portal */
.pishtaq { width:100%; height:auto; display:block; filter: drop-shadow(0 12px 24px rgba(90,58,23,.45)); }
.t-eyebrow { font: 600 15px 'Marcellus', serif; letter-spacing: 5px; fill: var(--gold); }
.t-title { font: 44px 'Marcellus', serif; fill: var(--cream); }
.t-gold { fill: var(--gold); }
.t-sub { font: 15px 'Source Sans Pro', sans-serif; fill: var(--muted); }

.chips { display:flex; justify-content:center; gap:.55rem; flex-wrap:wrap; margin: 1.3rem 0 1.4rem; }
.chip { font-family:'Marcellus',serif; border:1px solid var(--gold); background: var(--lapis); color: var(--cream);
        box-shadow: 0 3px 8px rgba(90,58,23,.25);
        padding:.32rem .95rem; border-radius: 6px; font-size:.95rem; }

/* upload box framed like a tile panel */
[data-testid="stFileUploader"] label { display:none; }
[data-testid="stFileUploaderDropzone"] {
  background: var(--card); border: 2px solid var(--gold); border-radius: 10px; min-height: 200px;
  outline: 1px dashed var(--turq); outline-offset: -10px;
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.9rem; text-align:center; padding:1.6rem;
  transition: background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(20,57,138,.9); }
[data-testid="stFileUploaderDropzone"] * { color: var(--cream) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { order:-1; display:flex; flex-direction:column; align-items:center; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content:"✦"; color: var(--gold); font-size: 2.2rem; line-height:1; margin-bottom:.4rem; }
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
  content:"Drop a photo of your dish"; display:block; font-family:'Marcellus',serif; font-size:1.3rem; margin-bottom:.25rem; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--muted) !important; font-size:.82rem; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
  background: var(--gold) !important; border:none !important; border-radius: 6px !important; font-weight:700; padding:.4rem 1.3rem !important; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] * { color: var(--deep) !important; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-minimal"],
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-borderlessIcon"] * { color: var(--gold) !important; }
[data-testid="stFileChip"] { background: rgba(47,179,192,.12) !important; border:1px solid var(--turq) !important; }
[data-testid="stFileChip"] svg { color: var(--gold) !important; }

/* result card: arch-shaped photo, tile band on top */
.result { background: var(--card); border:2px solid var(--gold); border-radius: 10px; margin: 1.4rem 0; overflow:hidden;
          box-shadow: 0 10px 26px rgba(90,58,23,.35); }
.result .body { display:flex; gap:1.5rem; align-items:center; padding: 1.3rem; }
.result img { width: 44%; aspect-ratio: 4/5; object-fit: cover; border: 3px solid var(--gold);
              border-radius: 50% 50% 8px 8px / 32% 32% 8px 8px; }
.result .info { flex:1; min-width:0; }
.fname { color: var(--muted); font-size:.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.dish { font-family:'Marcellus',serif; font-size: 2.3rem; color: var(--cream); margin:.15rem 0 .35rem; }
.badge { display:inline-block; padding:.25rem .75rem; border-radius:6px; font-weight:700; font-size:.85rem; margin-bottom:.9rem; }
.badge.ok { background: rgba(47,179,192,.18); color:#7fe3ea; border:1px solid var(--turq); }
.badge.low { background: rgba(217,164,65,.16); color: var(--gold); border:1px solid var(--gold); }
.note { color: var(--gold); font-size:.85rem; margin:-.4rem 0 .8rem; }
.bar { margin:.5rem 0; }
.bar .row { display:flex; justify-content:space-between; font-size:.9rem; color: var(--muted); }
.bar .track { height:9px; border-radius:2px; background: rgba(255,255,255,.08); overflow:hidden; margin-top:.22rem; }
.bar .fill { height:100%; background: linear-gradient(90deg, var(--turq), var(--gold)); }
.bar.top .row { color: var(--cream); font-weight:700; }

/* stats as three small arches */
.stats { display:flex; gap:.8rem; margin: 2rem 0 1rem; }
.stat { flex:1; text-align:center; padding: 1.4rem .4rem .9rem; background: var(--card); border:2px solid var(--gold);
        border-radius: 50% 50% 8px 8px / 26% 26% 8px 8px; box-shadow: 0 6px 14px rgba(90,58,23,.3); }
.stat b { display:block; font-family:'Marcellus',serif; font-size: 1.7rem; color: var(--gold); font-weight:400; }
.stat span { color: var(--muted); font-size:.85rem; }
.foot { text-align:center; color: #5a3a17; font-size:.85rem; margin: 1.1rem 0 1.4rem; }
.foot a { color: var(--lapis); font-weight:600; }

@media (max-width: 640px) {
  .result .body { flex-direction: column; }
  .result img { width: 80%; }
  .stats { gap:.4rem; }
  .stat b { font-size: 1.3rem; }
}
</style>
"""
for key, val in {"__BG__": BG, "__BAND__": BAND, "__LAPIS__": LAPIS, "__DEEP__": LAPIS_DEEP,
                 "__TURQ__": TURQ, "__GOLD__": GOLD, "__CREAM__": CREAM}.items():
    CSS = CSS.replace(key, val)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


def to_base64(img, size=560):
    """Small JPEG of the upload, embedded directly in the result card."""
    thumb = img.convert("RGB")
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def result_card(name, img, probs):
    order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])
    best, conf = CLASSES[order[0]], float(probs[order[0]])
    if conf >= LOW_CONFIDENCE:
        badge = f'<div class="badge ok">{conf:.1%} confident</div>'
    else:
        badge = (f'<div class="badge low">{conf:.1%} · not sure</div>'
                 '<div class="note">This may not be one of the 4 dishes, or the photo is unusual.</div>')
    bars = "".join(
        f'<div class="bar{" top" if i == order[0] else ""}"><div class="row">'
        f'<span>{EMOJI[CLASSES[i]]} {LABELS[CLASSES[i]]}</span><span>{probs[i]:.1%}</span></div>'
        f'<div class="track"><div class="fill" style="width:{probs[i] * 100:.1f}%"></div></div></div>'
        for i in order
    )
    return (
        f'<div class="result"><div class="band"></div><div class="body">'
        f'<img src="data:image/jpeg;base64,{to_base64(img)}" alt="uploaded dish">'
        f'<div class="info"><div class="fname">{name}</div>'
        f'<div class="dish">{EMOJI[best]} {LABELS[best]}</div>{badge}{bars}</div></div></div>'
    )


# ----- Page -----
st.markdown(
    HERO_SVG.replace("\n", "")
    + '<div class="chips">' + "".join(f'<div class="chip">{EMOJI[c]} {LABELS[c]}</div>' for c in CLASSES) + "</div>",
    unsafe_allow_html=True,
)

files = st.file_uploader("Upload food photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if files:
    model = get_model()
    for f in files:
        try:
            img = Image.open(f)
            img.load()
        except Exception:
            st.error(f"Could not open {f.name}. Please upload a JPG, PNG or WEBP image.")
            continue
        st.markdown(result_card(f.name, img, predict(model, img)), unsafe_allow_html=True)

st.markdown(
    '<div class="stats">' + "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in STATS) + "</div>"
    '<div class="band"></div>'
    '<div class="foot">PyTorch · ResNet18 fine-tuned on the '
    ' Central Asian Food Dataset (ISSAI)'
    " · the model always picks one of the 4 dishes<br>Design inspired by the tilework of the Registan, Samarkand</div>",
    unsafe_allow_html=True,
)
