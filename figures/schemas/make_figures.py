"""Génère les schémas de l'article (SVG) en mesurant le texte avec la vraie police (DejaVu Sans).

    python3 figures/schemas/make_figures.py          # écrit architecture.svg, protocol.svg, decomposition.svg
    bash figures/schemas/render.sh                   # puis PDF (article) et PNG (README) avec Chromium

Chaque ligne est mesurée avec Pillow (police DejaVu, celle du rendu) : un texte trop long est coupé en
plusieurs lignes, et un cadre trop petit pour son contenu lève une erreur au lieu de déborder.
Outil de figure seulement (Pillow du système), hors des dépendances du projet.
"""
from __future__ import annotations

import html
from pathlib import Path

from PIL import ImageFont

OUT = Path(__file__).resolve().parent
FONTS = Path("/usr/share/fonts/truetype/dejavu")
STYLE = {   # classe -> (fichier de police, taille px, couleur, interligne)
    "h": ("DejaVuSans-Bold.ttf", 14.5, "#1f2328", 19),
    "b": ("DejaVuSans.ttf", 12.5, "#1f2328", 17),
    "bb": ("DejaVuSans-Bold.ttf", 12.5, "#1f2328", 17),
    "s": ("DejaVuSans.ttf", 11.5, "#59636e", 15),
    "m": ("DejaVuSansMono.ttf", 12, "#1f2328", 17),
    "ms": ("DejaVuSansMono.ttf", 11, "#59636e", 15),
    "k": ("DejaVuSans-Bold.ttf", 10.5, "#59636e", 15),
}
FILL = {   # style de cadre -> (fond, bord, pointillés)
    "panel": ("#f6f8fa", "#d1d9e0", None), "box": ("#ffffff", "#d1d9e0", None),
    "gen": ("#ddf4ff", "#54aeff", None), "orc": ("#ffffff", "#8c959f", "5 3"),
    "chip": ("#ddf4ff", "#54aeff", None), "none": ("#ffffff", "#8c959f", "4 3"),
    "jud": ("#fbefff", "#8250df", None), "llm": ("#fff1e5", "#bc4c00", "5 3"),
    "data": ("#fff8c5", "#d4a72c", None), "real": ("#ddf4ff", "#54aeff", None),
    "c0": ("#ffffff", "#1a7f37", None), "lay": ("#fbefff", "#8250df", None),
    "llm4": ("#fff1e5", "#bc4c00", None), "disc": ("#ffffff", "#1f2328", None),
    "ok": ("#dafbe1", "#2da44e", None), "err": ("#ffebe9", "#cf222e", None),
    "prop": ("#fff8c5", "#d4a72c", None), "root": ("#f6f8fa", "#8c959f", None),
}
_font_cache: dict = {}


def width(text: str, cls: str) -> float:
    f, size, _, _ = STYLE[cls]
    key = (f, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(FONTS / f), size=round(size * 4))
    w = _font_cache[key].getlength(text) / 4
    return w + (len(text) * 1.0 if cls == "k" else 0.0)   # letter-spacing des étiquettes


def wrap(text: str, cls: str, maxw: float) -> list[str]:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if width(cand, cls) <= maxw or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    for ln in lines:
        if width(ln, cls) > maxw:
            raise ValueError(f"mot trop long pour {maxw:.0f}px : {ln!r}")
    return lines


class SVG:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.parts = w, h, []
        self.parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                          f'role="img" aria-label="{html.escape(title)}">')
        self.parts.append(f"<title>{html.escape(title)}</title>")
        self.parts.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                          'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#59636e"/></marker></defs>')
        self.parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')

    def rect(self, x, y, w, h, style, r=8, sw=1.3):
        fill, stroke, dash = FILL[style]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, segs, anchor="start"):
        """segs : chaîne (classe b) ou liste de (texte, classe) sur une ligne."""
        if isinstance(segs, str):
            segs = [(segs, "b")]
        out = []
        for t, cls in segs:
            f, size, color, _ = STYLE[cls]
            fam = "DejaVu Sans Mono, Menlo, Consolas, monospace" if "Mono" in f else \
                  "DejaVu Sans, Helvetica, Arial, sans-serif"
            weight = "700" if "Bold" in f else "400"
            ls = ' letter-spacing="1"' if cls == "k" else ""
            out.append(f'<tspan font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{color}"{ls}>'
                       f'{html.escape(t)}</tspan>')
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.parts.append(f'<text x="{x:g}" y="{y:g}"{a} xml:space="preserve">{"".join(out)}</text>')

    def seg_width(self, segs) -> float:
        return sum(width(t, c) for t, c in ([(segs, "b")] if isinstance(segs, str) else segs))

    def arrow(self, d):
        self.parts.append(f'<path d="{d}" fill="none" stroke="#59636e" stroke-width="1.7" marker-end="url(#a)"/>')

    def dot(self, cx, cy, color):
        self.parts.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="6" fill="{color}"/>')

    def save(self, name):
        self.parts.append("</svg>")
        (OUT / name).write_text("\n".join(self.parts) + "\n", encoding="utf-8")
        print("écrit", OUT / name)


class Column:
    """Contenu empilé dans un panneau : cadres et notes, chaque ligne mesurée et coupée."""
    PAD = 12

    def __init__(self, svg: SVG, x, y, w, h, label, style="panel"):
        self.svg, self.x, self.y0, self.w, self.h = svg, x, y, w, h
        svg.rect(x, y, w, h, style, r=12, sw=1.2)
        svg.text(x + 14, y + 26, [(label, "k")])
        self.cy = y + 40
        Column.bottom = max(getattr(Column, "bottom", 0), 0)

    def _lines(self, items, maxw):
        """items : (texte, classe) coupés à maxw, ou listes de segments (une ligne, vérifiée)."""
        out = []
        for it in items:
            if isinstance(it, list):
                if self.svg.seg_width(it) > maxw:
                    raise ValueError(f"ligne trop longue ({self.svg.seg_width(it):.0f} > {maxw:.0f}) : {it}")
                out.append((it, STYLE[it[0][1]][3]))
            else:
                t, cls = it
                out += [([(ln, cls)], STYLE[cls][3]) for ln in wrap(t, cls, maxw)]
        return out

    def card(self, style, items, inset=10, gap=8, after=None):
        x, w = self.x + inset, self.w - 2 * inset
        lines = self._lines(items, w - 2 * self.PAD)
        h = 10 + sum(lh for _, lh in lines) + 8
        self.svg.rect(x, self.cy, w, h, style)
        y = self.cy + 6
        for segs, lh in lines:
            y += lh
            self.svg.text(x + self.PAD, y - 4, segs)
        top = self.cy
        self.cy += h + gap
        return top, h

    def note(self, items, gap=8, inset=14):
        lines = self._lines(items, self.w - 2 * inset)
        y = self.cy
        for segs, lh in lines:
            y += lh
            self.svg.text(self.x + inset, y - 3, segs)
        self.cy = y + gap

    def space(self, dy):
        self.cy += dy

    def check(self):
        Column.bottom = max(Column.bottom, self.cy)
        if self.cy > self.y0 + self.h + 1:
            raise ValueError(f"panneau à x={self.x} : contenu jusqu'à {self.cy:.0f} > {self.y0 + self.h}")


# ============================================================ architecture
def architecture(H=1000):
    W, T = 1130, 16
    svg = SVG(W, H, "Proposer-judge architecture: propose candidates, judge in one forward pass, decide with a calibrated threshold")
    cols = [(16, 214), (258, 236), (522, 196), (746, 216), (990, 124)]
    ph = H - 2 * T
    mid = T + ph / 2

    c = Column(svg, cols[0][0], T, cols[0][1], ph, "INPUT")
    c.card("box", [("Current turn", "h"), ("“a flight from Lyon to Paris on Thursday”", "b")])
    c.card("box", [("Dialogue context", "h"), ("history and state (multi-turn)", "s")])
    c.card("box", [("Intent (gold)", "h"), ([("find_flight", "m")])])
    c.space(8)
    c.note([("ONE REQUEST PER SLOT", "k")], gap=2)
    c.note([("q = (turn, slot σ) for every slot σ of the intent:", "b")], gap=2)
    c.note([("from_city, to_city, date, time, airline, class …", "ms")], gap=10)
    c.note([("Most slots are not expressed: abstaining is the common case.", "s")])
    c.check()

    c = Column(svg, cols[1][0], T, cols[1][1], ph, "1 · PROPOSE")
    c.card("gen", [("Lexicon", "h"), ("values seen in training", "s")], gap=6)
    c.card("gen", [("GLiNER", "h"), ("zero-shot span extraction", "s")], gap=6)
    c.card("gen", [("Ontology", "h"), ("closed values found in the text", "s")], gap=6)
    c.card("gen", [("History", "h"), ("values of earlier turns", "s")], gap=6)
    c.note([("Real lists: union of the four, deduplicated, shuffled, at most 20.", "s")], gap=10)
    c.card("orc", [("Oracle (analysis)", "h"), ("gold value + k − 1 distractors, gold removed with probability p_absent: measures the judge alone", "s"), ([("grid: k=5, p_absent=0", "ms")])])
    c.check()

    c = Column(svg, cols[2][0], T, cols[2][1], ph, "OPTIONS FOR σ")
    c.note([[("slot: ", "b"), ("from_city", "m")]], gap=6)
    for txt, st in (("A  Lyon", "chip"), ("B  Paris", "chip"), ("C  Thursday", "chip"), ("∅  not specified", "none")):
        x, w = c.x + 12, c.w - 24
        svg.rect(x, c.cy, w, 26, st, r=13)
        svg.text(x + 12, c.cy + 18, [(txt, "m" if st == "chip" else "ms")])
        c.cy += 32
    c.space(4)
    c.note([("Neutral letter keys. “Not specified” is always the last option: the judge can abstain and can never invent a value.", "s")], gap=12)
    c.note([("C0 BASELINE", "k")], gap=2)
    c.note([("A fine-tuned BIO tagger (mDeBERTa-v3) reads the value directly in the turn, without proposers or judge.", "s")])
    c.check()

    c = Column(svg, cols[3][0], T, cols[3][1], ph, "2 · JUDGE")
    c.card("jud", [("Laya", "h"), ("non-autoregressive, < 0.5 B", "s"),
                   ("input:", "bb"), ("state (message, intent, filled slots, history) + question about σ + options A … ∅", "s"),
                   ("model:", "bb"), ("bidirectional encoder + option head, one forward pass per turn", "s"),
                   ("output:", "bb"), ([("p(A) = 0.97", "m")])])
    c.note([("C1 zero-shot · C2 fine-tuned with the log score · C3 with list augmentation", "s")], gap=10)
    c.card("llm", [("C4 · LLM judges", "h"), ("27–180 B, same prompt and options, first-token log-probabilities", "s")])
    c.check()

    c = Column(svg, cols[4][0], T, cols[4][1], ph, "3 · DECIDE")
    c.card("box", [("calibrate", "bb"), ("temperature T fitted on dev", "s")], inset=8)
    c.card("box", [("threshold τ", "bb"), ("for a target risk (2 %, 5 %)", "s")], inset=8)
    c.space(6)
    for lab, rule, color in (("act", "value, p ≥ τ", "#1a7f37"), ("ignore", "∅, p ≥ τ", "#59636e"),
                             ("ask", "p < τ: defer to the user", "#9a6700")):
        svg.dot(c.x + 20, c.cy + 6, color)
        svg.text(c.x + 32, c.cy + 11, [(lab, "bb")])
        c.cy += 18
        c.note([(rule, "s")], gap=10)
    c.note([("evaluated by risk–coverage curves (AURC)", "s")])
    c.check()

    for (x1, w1), (x2, _) in zip(cols, cols[1:]):
        svg.arrow(f"M{x1 + w1 + 2},{mid} L{x2 - 3},{mid}")
    return svg


# ============================================================ protocol
def row(svg, y, h, label, notes):
    svg.rect(16, y, 1098, h, "panel", r=12, sw=1.2)
    svg.text(30, y + 26, [(label, "k")])
    yy = y + 46
    for t in notes:
        for ln in wrap(t, "s", 116):
            svg.text(30, yy, [(ln, "s")])
            yy += 15


def box_lines(svg, w, items, pad=12):
    out = []
    for it in items:
        if isinstance(it, list):
            if svg.seg_width(it) > w - 2 * pad:
                raise ValueError(f"ligne trop longue : {it}")
            out.append((it, STYLE[it[0][1]][3]))
        else:
            t, cls = it
            out += [([(ln, cls)], STYLE[cls][3]) for ln in wrap(t, cls, w - 2 * pad)]
    return out


def need(svg, w, items, pad=12):
    return 8 + sum(lh for _, lh in box_lines(svg, w, items, pad)) + 8


def boxed(svg, x, y, w, h, style, items, pad=12, center=False):
    """Cadre ; h=None : hauteur du contenu. center : contenu centré verticalement."""
    lines = box_lines(svg, w, items, pad)
    n = 8 + sum(lh for _, lh in lines) + 8
    h = n if h is None else h
    if n > h + 0.5:
        raise ValueError(f"cadre trop petit ({n:.0f} > {h}) : {items[0]}")
    svg.rect(x, y, w, h, style)
    yy = y + 8 + ((h - n) / 2 if center else 0)
    for segs, lh in lines:
        yy += lh
        svg.text(x + pad, yy - 4, segs)
    return h


def free(svg, x, y, segs, maxw):
    """Texte libre sur une ligne, vérifié."""
    if svg.seg_width(segs) > maxw:
        raise ValueError(f"texte libre trop long ({svg.seg_width(segs):.0f} > {maxw}) : {segs}")
    svg.text(x, y, segs)


def protocol():
    W, X0, XR = 1130, 160, 1102
    svg = SVG(W, 1000, "Experimental protocol: data, candidates, conditions, measures and development/test discipline")
    y = 16

    def band(label, notes, boxes, arrow_before_last=False):
        """boxes : (x, w, style, items) ; hauteur commune = le plus haut contenu."""
        nonlocal y
        hb = max(need(svg, w, items) for _, w, _, items in boxes)
        hn = 46 + 15 * sum(len(wrap(t, "s", X0 - 44)) for t in notes)
        h = max(hb + 28, hn + 10)
        svg.rect(16, y, XR + 12 - 16, h, "panel", r=12, sw=1.2)
        svg.text(30, y + 26, [(label, "k")])
        yy = y + 46
        for t in notes:
            for ln in wrap(t, "s", X0 - 44):
                svg.text(30, yy, [(ln, "s")])
                yy += 15
        for i, (x, w, st, items) in enumerate(boxes):
            boxed(svg, x, y + 14, w, hb, st, items)
            if arrow_before_last and i == len(boxes) - 2:
                svg.arrow(f"M{x + w + 4},{y + 14 + hb / 2} L{x + w + 22},{y + 14 + hb / 2}")
        y += h + 16

    gw = 200
    band("1 · DATA", ["3 benchmarks × English and French"], [
        (X0, gw, "data", [("MultiATIS++", "h"), ("flights, one domain, single turn; LDC licence", "s")]),
        (X0 + 212, gw, "data", [("MTOP", "h"), ("11 domains, many intents, single turn", "s")]),
        (X0 + 424, gw, "data", [("Multi3WOZ", "h"), ("multi-domain dialogues, multi-turn: history and state", "s")]),
        (X0 + 662, XR - (X0 + 662), "box", [("Unified format", "h"), ("turn, intent, slots, ontology; normalised values; one request per slot", "s")]),
    ], arrow_before_last=True)
    half = (XR - X0 - 12) / 2
    band("2 · CANDIDATES", ["same lists and option order for every judge"], [
        (X0, half, "orc", [("Oracle lists — the judge alone", "h"),
                           ("gold value + k − 1 hard distractors; gold removed with probability p_absent", "s"),
                           ([("grid:   k=5, p_absent=0, hard_no_overlap", "ms")]),
                           ([("sweeps: k ∈ {2,5,10,20} × p_absent ∈ {0,0.2,0.5}", "ms")])]),
        (X0 + half + 12, half, "real", [("Real lists — the whole pipeline", "h"),
                                        ("union of lexicon, GLiNER, ontology and history proposers; deduplicated by normalised value, shuffled, at most 20 options", "s"),
                                        ("errors split into candidate recall, choice and abstention", "s")]),
    ])
    cw = (XR - X0 - 4 * 10) / 5
    conds = [("c0", "C0 · tagger", "mDeBERTa-v3 with BIO labels; no proposers, no judge; real setting only"),
             ("lay", "C1 · zero-shot", "Laya as released: the floor of the judge"),
             ("lay", "C2 · fine-tuned", "Laya, log score on oracle lists (k=5); checkpoint chosen on dev F1"),
             ("lay", "C3 · augmented", "C2 + random list sizes, missing gold and slot confusions (v1 generic: ablation)"),
             ("llm4", "C4 · LLM judges", "Qwen3.8-27B, Gemma-4-31B-it, Qwen3.8-Flash-Next; zero-shot")]
    band("3 · CONDITIONS", ["seeds: 3 (C0–C2, LLMs), 5 (C3)", "trained: C0 C2 C3", "zero-shot: C1 C4"],
         [(X0 + i * (cw + 10), cw, st, [(t, "h"), (d, "s")]) for i, (st, t, d) in enumerate(conds)])
    mw = (XR - X0 - 3 * 10) / 4
    meas = [("Accuracy", "slot F1, precision, recall, frame accuracy"),
            ("Error decomposition", "candidate recall, choice accuracy, false positives, abstention when the gold value is missing"),
            ("Calibration, abstention", "temperature, ECE, AURC, coverage at 2 % and 5 % risk"),
            ("Cost", "latency per request (batch of 1), throughput, GPUs, memory")]
    band("4 · MEASURES", [], [(X0 + i * (mw + 10), mw, "box", [(t, "h"), (d, "s")]) for i, (t, d) in enumerate(meas)])
    svg.rect(16, y, XR + 12 - 16, 54, "disc", r=12, sw=1.2)
    svg.text(30, y + 25, [("DISCIPLINE", "k")])
    x = X0
    for segs in ([("dev sets", "bb"), (": every protocol decision", "b")], [("frozen protocol", "bb"), (" (31 Oct 2026)", "b")],
                 [("test sets", "bb"), (": opened once", "b")]):
        svg.text(x, y + 25, segs)
        x += svg.seg_width(segs) + 14
        if segs[0][0] != "test sets":
            svg.arrow(f"M{x},{y + 21} L{x + 26},{y + 21}")
            x += 40
    free(svg, X0, y + 43, [("cluster bootstrap 95 % intervals · paired permutation tests with Holm correction · code fingerprint in every run", "s")], XR - X0)
    y += 54 + 16
    return svg, y


# ============================================================ decomposition
def decomposition():
    W, H = 1130, 410
    svg = SVG(W, H, "Error decomposition: each request ends in exactly one outcome")
    svg.text(16, 26, [("SLOT EXPRESSED BY THE USER (A GOLD VALUE EXISTS)", "k")])
    boxed(svg, 16, 82, 170, 56, "root", [("Gold pair", "h"), ("(slot, value) to find", "s")], center=True)
    svg.arrow("M188,110 L230,110")
    boxed(svg, 234, 82, 170, 56, "box", [("Gold value in the list?", "h")], center=True)
    svg.arrow("M319,140 L319,176")
    svg.text(328, 164, [("no", "bb")])
    boxed(svg, 234, 180, 250, None, "prop", [("Recall error", "h"), ("caused by the proposers; the best possible answer is then ∅", "s")])
    svg.arrow("M406,110 L448,110")
    svg.text(414, 102, [("yes", "bb")])
    boxed(svg, 452, 82, 170, 56, "box", [("Judge answers", "h"), ("one option or ∅", "s")], center=True)
    for y, st, segs in ((40, "ok", [("gold value", "bb"), (" → correct", "b")]),
                        (91, "err", [("other value", "bb"), (" → choice error", "b")]),
                        (142, "err", [("∅", "bb"), (" → wrong abstention", "b")])):
        boxed(svg, 690, y, 210, 38, st, [segs], center=True)
    svg.arrow("M624,96 C656,96 656,59 686,59")
    svg.arrow("M624,110 L686,110")
    svg.arrow("M624,124 C656,124 656,161 686,161")
    boxed(svg, 916, 40, 198, None, "box", [("MEASURED BY", "k"), ("candidate recall", "bb"),
          ("gold pairs whose value is in the list", "s"), ("choice accuracy", "bb"),
          ("correct answers when the value is in the list", "s"),
          ("abstention when absent", "bb"), ("∅ answered when the gold value is not in the list", "s")])
    svg.parts.append('<line x1="16" y1="280" x2="1114" y2="280" stroke="#d1d9e0" stroke-width="1.2"/>')
    svg.text(16, 306, [("SLOT NOT EXPRESSED (THE MAJORITY OF REQUESTS)", "k")])
    boxed(svg, 16, 330, 170, 56, "root", [("Unexpressed", "h"), ("slot of the intent", "s")], center=True)
    svg.arrow("M188,358 L448,358")
    free(svg, 200, 380, [("empty list: ∅ without calling the judge", "s")], 240)
    boxed(svg, 452, 330, 170, 56, "box", [("Judge answers", "h"), ("one option or ∅", "s")], center=True)
    boxed(svg, 690, 318, 210, 38, "ok", [[("∅", "bb"), (" → correct abstention", "b")]], center=True)
    boxed(svg, 690, 362, 210, 38, "err", [[("a value", "bb"), (" → false positive", "b")]], center=True)
    svg.arrow("M624,348 C656,348 656,337 686,337")
    svg.arrow("M624,368 C656,368 656,381 686,381")
    boxed(svg, 916, 318, 198, None, "box", [("false-positive rate", "bb"),
          ("unexpressed requests that receive a value: the main error of LLM judges", "s")])
    svg.save("decomposition.svg")


if __name__ == "__main__":
    Column.bottom = 0
    architecture(1000)                                   # passe à blanc : hauteur du contenu
    architecture(int(Column.bottom) + 30).save("architecture.svg")
    svg, y = protocol()
    svg.parts[0] = svg.parts[0].replace('height="1000" viewBox="0 0 1130 1000"', f'height="{y}" viewBox="0 0 1130 {y}"')
    svg.parts[3] = svg.parts[3].replace('height="1000"', f'height="{y}"')
    svg.save("protocol.svg")
    decomposition()
