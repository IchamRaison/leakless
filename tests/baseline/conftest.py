import pytest

from pipe.baseline.donnees import lire_json
from pipe.baseline.fixture import creer_fixture


@pytest.fixture
def jeu(tmp_path):
    chemin = creer_fixture(tmp_path / "fixture")
    return chemin, lire_json(chemin), lire_json(chemin.parent / "features.json"), lire_json(chemin.parent / "split.json")
