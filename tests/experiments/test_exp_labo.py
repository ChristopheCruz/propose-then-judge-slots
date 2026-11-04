"""Banc d'essai de scripts/labo.sh sans serveur ni GitHub.

- « GitHub » : un dépôt nu local ; « serveur » : un clone dans un dossier temporaire ;
- ssh : un faux ssh qui exécute la commande localement dans un environnement minimal
  (PATH=/usr/bin:/bin, comme une session ssh non interactive, sans identité git) ;
- uv : un faux uv placé dans ~/.local/bin (HOME temporaire), qui note ses arguments ;
- l'expérience : un petit script bash qui lit les données liées et intercepte SIGTERM.

LABO_SH=<chemin> permet de rejouer le banc sur une autre version du script.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LABO_SH = Path(os.environ.get("LABO_SH", REPO / "scripts" / "labo.sh"))

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not all(shutil.which(c) for c in ("git", "flock", "setsid", "bash")),
    reason="banc labo.sh : Linux avec git, flock et setsid")

FAKE_SSH = r"""#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
  case "$1" in -p|-o|-i|-l) shift 2 ;; -A|-t|-T|-q) shift ;; *) break ;; esac
done
shift   # hôte
exec env -i HOME="$HOME" PATH="$LABO_TEST_BIN:/usr/bin:/bin" LANG=C.UTF-8 \
  TMUX_TMPDIR="$TMUX_TMPDIR" LABO_TEST_BIN="$LABO_TEST_BIN" bash -c "$*"
"""

FAKE_UV = r"""#!/usr/bin/env bash
echo "faux-uv $* UV_NO_SYNC=${UV_NO_SYNC:-}"
[[ "$1" == run ]] || exit 0
shift; while [[ "$1" == --* ]]; do shift; done
exec "$@"
"""

FAKE_EXP = r"""#!/usr/bin/env bash
cfg="$1"; out="$2"; duree="$(cat "$cfg")"
echo "exp pwd=$PWD GPU=$CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
cat data/processed/marqueur.txt
trap 'echo interrompu > "$out/partiel.txt"; exit 143' TERM
for ((i = 0; i < duree * 10; i++)); do sleep 0.1; done
echo '{"slot_f1": 1.0}' > "$out/metrics.json"
"""


class Banc:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.origin, self.poste, self.serveur = tmp / "origin.git", tmp / "poste", tmp / "serveur"
        self.home, self.bin = tmp / "home", tmp / "bin"
        for d in (self.home / ".local" / "bin", self.bin, tmp / "tmux", self.poste / "configs"):
            d.mkdir(parents=True)
        self._exe(self.bin / "ssh", FAKE_SSH)
        self._exe(self.home / ".local" / "bin" / "uv", FAKE_UV)
        self.env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.home), "LANG": "C.UTF-8",
                    "TMUX_TMPDIR": str(tmp / "tmux"), "LABO_TEST_BIN": str(self.bin),
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t"}
        self.git("init", "-q", "--bare", "-b", "main", str(self.origin), cwd=tmp)
        self.git("init", "-q", "-b", "main", cwd=self.poste)
        (self.poste / "scripts").mkdir()
        shutil.copy(LABO_SH, self.poste / "scripts" / "labo.sh")
        self._exe(self.poste / "exp.sh", FAKE_EXP)
        (self.poste / "configs" / "court.yaml").write_text("0\n")
        (self.poste / ".gitignore").write_text("labo.env\ndata/\n")
        (self.poste / "labo.env").write_text(
            f'SERVEUR="faux"\nREMOTE_DIR="{self.serveur}"\nDEPOT_GIT="{self.origin}"\nBRANCHE=main\n'
            'EXP_CMD=\'uv run --no-sync bash exp.sh "$CONFIG" "$RUN_DIR"\'\n'
            'GIT_NOM="Serveur test"\nGIT_EMAIL="serveur@test"\n')
        self.commit("initial")
        self.git("remote", "add", "origin", str(self.origin), cwd=self.poste)
        self.git("push", "-q", "origin", "main", cwd=self.poste)

    @staticmethod
    def _exe(path: Path, text: str):
        path.write_text(text)
        path.chmod(0o755)

    def git(self, *args, cwd):
        subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True, capture_output=True)

    def commit(self, msg):
        self.git("add", "-A", cwd=self.poste)
        self.git("commit", "-q", "-m", msg, cwd=self.poste)

    def labo(self, *args, stdin=None, check=True):
        p = subprocess.run(["bash", "scripts/labo.sh", *args], cwd=self.poste, env=self.env, text=True,
                           capture_output=True, input=stdin, timeout=120, check=False)
        if check and p.returncode != 0:
            raise AssertionError(f"labo.sh {args} : code {p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def meta(self, run_id):
        f = self.serveur / "results" / run_id / "meta.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def attendre(self, run_id, timeout=60):
        fin = time.time() + timeout
        while time.time() < fin:
            if self.meta(run_id).get("statut") not in (None, "en_cours"):
                return self.meta(run_id)
            time.sleep(0.2)
        raise AssertionError(f"run {run_id} non terminé : {self.meta(run_id)}")

    def runs(self):
        return sorted(p.name for p in (self.serveur / "results").iterdir() if (p / "meta.json").exists())

    def journal_origin(self):
        return subprocess.run(["git", "log", "--format=%s", "main"], cwd=self.origin, env=self.env,
                              capture_output=True, text=True, check=False).stdout

    def fermer(self):
        if shutil.which("tmux"):
            subprocess.run(["tmux", "kill-server"], env=self.env, capture_output=True, check=False)


@pytest.fixture
def banc(tmp_path):
    b = Banc(tmp_path)
    yield b
    b.fermer()


def _id(sortie: str) -> str:
    return next(line.split("Run lancé : ")[1].split()[0] for line in sortie.splitlines() if "Run lancé" in line)


def test_labo_cycle_complet(banc):
    banc.labo("setup")
    (banc.serveur / "data" / "processed").mkdir(parents=True)
    (banc.serveur / "data" / "processed" / "marqueur.txt").write_text("marqueur-ok\n")
    assert banc.labo("statut").returncode == 0

    # 1. un run court : données liées, uv sans resynchronisation, résultats poussés
    rid = _id(banc.labo("run", "configs/court.yaml", "--gpus", "0").stdout)
    meta = banc.attendre(rid)
    log = (banc.serveur / "results" / rid / "log.txt").read_text()
    assert meta["statut"] == "termine", log
    assert "marqueur-ok" in log and "UV_NO_SYNC=1" in log and "PYTHONUNBUFFERED=1" in log
    assert (banc.serveur / "results" / rid / "metrics.json").exists()
    assert not (banc.serveur / ".labo" / "arbres" / rid).exists()
    assert (banc.serveur / "data" / "processed" / "marqueur.txt").exists()   # la suppression de l'instantané ne suit pas le lien
    fin = time.time() + 30
    while f"résultats : {rid}" not in banc.journal_origin() and time.time() < fin:
        time.sleep(0.2)
    assert f"résultats : {rid}" in banc.journal_origin()

    # 2. nouveau commit local alors que le serveur a poussé des résultats : sync les intègre
    (banc.poste / "configs" / "long.yaml").write_text("30\n")
    banc.commit("config longue")
    rid2 = _id(banc.labo("run", "configs/long.yaml", "--gpus", "1", "--nom", "long").stdout)
    pid = banc.serveur / "results" / rid2 / ".pid"
    fin = time.time() + 30
    while not pid.exists() and time.time() < fin:
        time.sleep(0.1)
    time.sleep(0.5)
    # pendant le run, un autre poste pousse sur « GitHub » : la sauvegarde devra rebaser
    autre = banc.tmp / "autre"
    banc.git("clone", "-q", str(banc.origin), str(autre), cwd=banc.tmp)
    (autre / "note.txt").write_text("x\n")
    banc.git("add", "note.txt", cwd=autre)
    banc.git("commit", "-q", "-m", "autre poste", cwd=autre)
    banc.git("push", "-q", "origin", "main", cwd=autre)
    banc.labo("stop", rid2)
    meta2 = banc.attendre(rid2)
    assert meta2["statut"] == "arrete" and meta2["code_sortie"] == 143
    assert (banc.serveur / "results" / rid2 / "partiel.txt").read_text().strip() == "interrompu"
    fin = time.time() + 30
    while f"résultats : {rid2}" not in banc.journal_origin() and time.time() < fin:
        time.sleep(0.2)
    assert f"résultats : {rid2}" in banc.journal_origin(), (banc.serveur / "results" / rid2 / "log.txt").read_text()

    # 3. file d'attente éditée sous Windows (CRLF), deux GPU lancés dans la même seconde,
    #    même config sans nom : identifiants distincts, config sans \r
    avant = set(banc.runs())
    (banc.poste / "file.txt").write_text("# essai\r\n0 configs/court.yaml\r\n1 configs/court.yaml\r\n\r\n")
    sortie = banc.labo("file", "file.txt").stdout
    assert "GPU 0" in sortie and "GPU 1" in sortie
    fin = time.time() + 60
    while len(set(banc.runs()) - avant) < 2 and time.time() < fin:
        time.sleep(0.2)
    nouveaux = sorted(set(banc.runs()) - avant)
    assert len(nouveaux) == 2, nouveaux
    for r in nouveaux:
        assert banc.attendre(r)["statut"] == "termine", (banc.serveur / "results" / r / "log.txt").read_text()
        assert banc.meta(r)["config"] == "configs/court.yaml"

    # 4. une config non commitée est refusée avec un message clair
    (banc.poste / "configs" / "oubli.yaml").write_text("0\n")
    p = banc.labo("run", "configs/oubli.yaml", "--sans-sync", check=False)
    assert p.returncode != 0 and "Config absente du commit" in p.stderr

    # 5. tous les résultats finissent sur « GitHub », sans les marqueurs internes
    banc.labo("sauver", "fin du test")
    suivis = subprocess.run(["git", "ls-tree", "-r", "--name-only", "main"], cwd=banc.origin,
                            capture_output=True, text=True, check=True).stdout
    assert f"results/{rid2}/meta.json" in suivis and ".pid" not in suivis
