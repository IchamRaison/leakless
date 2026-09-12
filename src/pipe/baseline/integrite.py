"""Audit de fichiers uniquement, jamais des features ou des étiquettes du modèle."""
import csv
import hashlib
import io
import json
from pathlib import Path

from pipe.tslm.prepare import MANIFEST_HASHES
from .donnees import exiger, verifier_hash


def hashes_audio(manifestes):
    chemin = Path(manifestes) / "split_v2_audit.csv"
    contenu = chemin.read_bytes()
    exiger(hashlib.sha256(contenu).hexdigest() == MANIFEST_HASHES[chemin.name], "Audit source modifié")
    # MD5 historique de Nevil sert uniquement à détecter un mauvais WAV.
    # La confiance dans cette table repose sur son SHA256 gelé, pas sur MD5 seul.
    lignes = list(csv.DictReader(io.StringIO(contenu.decode("utf-8"))))
    resultat = {ligne["clip_id"]: ligne["md5"] for ligne in lignes}
    exiger(len(resultat) == len(lignes) == 1000, "Couverture des empreintes audio incorrecte")
    return resultat


def verifier_audio(contenu, identifiant, hashes):
    exiger(identifiant in hashes, f"Empreinte audio absente : {identifiant}")
    exiger(hashlib.md5(contenu).hexdigest() == hashes[identifiant],
           f"Audio différent de la source gelée : {identifiant}")


def lire_provenance(chemin, sha256_attendu):
    verifier_hash(sha256_attendu)
    contenu = Path(chemin).read_bytes()
    exiger(hashlib.sha256(contenu).hexdigest() == sha256_attendu, "Provenance modifiée après gel")
    return json.loads(contenu)
