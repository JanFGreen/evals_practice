from tests.helpers import load

m = load("exercises.d3_security", "evalkit.security")

CANARY = "ORBITA-CANARY-5521-ZETA"


def test_find_canaries():
    assert m.find_canaries(f"id: {CANARY.lower()} in the text", [CANARY]) == [CANARY]
    assert m.find_canaries("clean answer", [CANARY]) == []


def test_retrieval_leaks():
    retrieved = [{"id": "a#0", "workspace": "nordwind"}, {"id": "b#1", "workspace": "orbita"}]
    assert m.retrieval_leaks(retrieved, "nordwind") == ["b#1"]
    assert m.retrieval_leaks(retrieved[:1], "nordwind") == []
