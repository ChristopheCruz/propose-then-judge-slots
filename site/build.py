"""Génère le site multilingue : site/index.html (anglais) et site/<langue>/index.html.

    python3 site/build.py

Gabarit : site/src/template.html ({{clé}} = texte traduit, [[0.941]] = nombre au format de la langue).
Traductions : site/src/i18n/<langue>.py (dictionnaire T). Bibliothèque standard seulement.
"""
from __future__ import annotations

import json
import re
import runpy
from pathlib import Path

ICI = Path(__file__).resolve().parent
LANGUES = ["en", "fr", "es", "de", "zh", "ar"]          # la première est servie à la racine
NOMS = {"en": "English", "fr": "Français", "es": "Español", "de": "Deutsch", "zh": "中文", "ar": "العربية"}
COURTS = {"en": "EN", "fr": "FR", "es": "ES", "de": "DE", "zh": "中文", "ar": "عربي"}
LOCALES = {"en": "en_GB", "fr": "fr_FR", "es": "es_ES", "de": "de_DE", "zh": "zh_CN", "ar": "ar_AR"}
DECIMALE = {"en": ".", "fr": ",", "es": ",", "de": ",", "zh": ".", "ar": "."}
POLICES = {"zh": ("'Noto Sans SC',Inter", "&family=Noto+Sans+SC:wght@400;500;700"),
           "ar": ("'Noto Sans Arabic',Inter", "&family=Noto+Sans+Arabic:wght@400;500;700")}
BASE = "https://christophecruz.github.io/mypage.github.io/slotjudge/"   # adresse publique prévue
LANGUES_MODELE = ("af ar az bn ca da de el en es fa fi fr he hi hu hy id it ja jv ka kn ko lv ml mn ms my nb nl "
                  "pl pt ro ru sl sq sv ta te th tl tr ur vi zh").split()


# Écriture de droite à gauche : plages de nombres, unités et noms latins à ponctuation finale
# (« 0.38–0.84 », « 15–21 % », « 25–50 ms », « MultiATIS++ ») isolés de gauche à droite, sinon inversés.
ATTRIBUTS = {"meta_title", "meta_desc", "lang_label", "theme_label", "logo_alt", "q_aria", "fig_alt"}
LTR = re.compile(r"(?<![\w.\-])(?:≈\s?)?\d+(?:[.,]\d+)*(?:\s?[–-]\s?\d+(?:[.,]\d+)*)?(?:\s?(?:%|ms|×|GiB))?(?![\w\-])|MultiATIS\+\+")


# Page interactive : (libellé du menu, phrase sous « How it works »).
LIEN_EXPLAIN = {"en": ("Explore", "Move the sliders in the interactive guide"),
                "fr": ("Explorer", "Manipulez les réglages dans le guide interactif"),
                "es": ("Explorar", "Mueva los controles en la guía interactiva"),
                "de": ("Erkunden", "Bewegen Sie die Regler im interaktiven Leitfaden"),
                "zh": ("探索", "在交互式指南中拖动控件"),
                "ar": ("استكشف", "حرّك أدوات التحكم في الدليل التفاعلي")}


def isoler_ltr(textes: dict[str, str]) -> dict[str, str]:
    def un(cle: str, v: str) -> str:
        if cle in ATTRIBUTS:
            return v
        morceaux = re.split(r"(<[^>]+>)", v)          # ne jamais toucher aux balises
        return "".join(m if m.startswith("<") else LTR.sub(lambda x: f'<bdi dir="ltr">{x.group(0)}</bdi>', m)
                       for m in morceaux)
    return {k: un(k, v) for k, v in textes.items()}


def chemin(lang: str) -> str:
    return "" if lang == LANGUES[0] else f"{lang}/"


def page(lang: str, gabarit: str, textes: dict[str, str]) -> str:
    racine = "" if lang == LANGUES[0] else "../"
    menu = "\n".join(
        f'        <li><a href="{racine}{chemin(l)}" hreflang="{l}" lang="{l}"'
        f'{" aria-current=\"page\"" if l == lang else ""}>{NOMS[l]}</a></li>' for l in LANGUES)
    hreflang = "\n".join(f'<link rel="alternate" hreflang="{l}" href="{BASE}{chemin(l)}">' for l in LANGUES)
    hreflang += f'\n<link rel="alternate" hreflang="x-default" href="{BASE}">'
    police, extra = POLICES.get(lang, ("Inter", ""))
    valeurs = {
        **(isoler_ltr(textes) if lang == "ar" else textes), "lang": lang, "dir": "rtl" if lang == "ar" else "ltr", "root": racine,
        "og_locale": LOCALES[lang], "hreflang": hreflang, "lang_menu": menu, "lang_short": COURTS[lang],
        "font": police, "font_extra": extra,
        "nav_explain": f'<a href="explain/">{LIEN_EXPLAIN[lang][0]}</a>',
        "explain_link": f'<p><a href="explain/">{LIEN_EXPLAIN[lang][1]} {"←" if lang == "ar" else "→"}</a></p>',
        "chips": "".join(f'<span class="chip">{c}</span>' for c in LANGUES_MODELE),
    }
    manquantes = sorted(set(re.findall(r"\{\{(\w+)\}\}", gabarit)) - set(valeurs))
    if manquantes:
        raise SystemExit(f"{lang} : clés manquantes {manquantes}")
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: valeurs[m.group(1)], gabarit)
    return re.sub(r"\[\[([0-9.]+)\]\]", lambda m: m.group(1).replace(".", DECIMALE[lang]), html)


def page_explain(lang: str, gabarit: str, textes: dict) -> str:
    """Guide interactif : site/explain/ (anglais) et site/<langue>/explain/ ; les textes de l'interface passent au script."""
    racine = "../" if lang == LANGUES[0] else "../../"
    valeurs = {k: v for k, v in textes.items() if isinstance(v, str)}
    site = runpy.run_path(str(ICI / "src" / "i18n" / f"{lang}.py"))["T"]
    valeurs.update({k: site[k] for k in ("nav_how", "nav_results", "nav_model", "nav_start", "nav_uses", "nav_cite", "lang_label")})
    if lang == "ar":
        titres = {k: valeurs[k] for k in ("title", "desc")}
        nombres = {k: re.sub(r"\[\[([0-9.]+)\]\]", r"\1", v) for k, v in valeurs.items()}   # avant l'isolement
        valeurs = {**isoler_ltr(nombres), **titres}
    police, extra = POLICES.get(lang, ("Inter", ""))
    courant = ' aria-current="page"'
    menu = "\n".join(
        f'        <li><a href="{racine}{chemin(x)}explain/" hreflang="{x}" lang="{x}"{courant if x == lang else ""}>{NOMS[x]}</a></li>'
        for x in LANGUES)
    valeurs.update(lang=lang, dir="rtl" if lang == "ar" else "ltr", root=racine, home="../", font=police, font_extra=extra,
                   js=json.dumps(textes, ensure_ascii=False).replace("</", "<\\/"), nav_here=LIEN_EXPLAIN[lang][0],
                   lang_short=COURTS[lang], lang_menu=menu)
    manquantes = sorted(set(re.findall(r"\{\{(\w+)\}\}", gabarit)) - set(valeurs))
    if manquantes:
        raise SystemExit(f"explain {lang} : clés manquantes {manquantes}")
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: valeurs[m.group(1)], gabarit)
    return re.sub(r"\[\[([0-9.]+)\]\]", lambda m: m.group(1).replace(".", DECIMALE[lang]), html)


def main() -> None:
    gabarit = (ICI / "src" / "template.html").read_text(encoding="utf-8")
    ref = set(runpy.run_path(str(ICI / "src" / "i18n" / "en.py"))["T"])
    for lang in LANGUES:
        textes = runpy.run_path(str(ICI / "src" / "i18n" / f"{lang}.py"))["T"]
        if set(textes) != ref:
            raise SystemExit(f"{lang} : clés différentes de en.py : {sorted(set(textes) ^ ref)}")
        sortie = ICI / chemin(lang) / "index.html"
        sortie.parent.mkdir(parents=True, exist_ok=True)
        sortie.write_text(page(lang, gabarit, textes), encoding="utf-8")
        print(f"{lang}: {sortie.relative_to(ICI)}")
    gabarit = (ICI / "src" / "explain.html").read_text(encoding="utf-8")
    pages = runpy.run_path(str(ICI / "src" / "i18n_explain.py"))["T"]
    for lang in LANGUES:
        textes = pages[lang]
        if set(textes) != set(pages["en"]):
            raise SystemExit(f"explain {lang} : clés différentes : {sorted(set(textes) ^ set(pages['en']))}")
        sortie = ICI / chemin(lang) / "explain" / "index.html"
        sortie.parent.mkdir(parents=True, exist_ok=True)
        sortie.write_text(page_explain(lang, gabarit, textes), encoding="utf-8")
        print(f"{lang}: {sortie.relative_to(ICI)}")


if __name__ == "__main__":
    main()
