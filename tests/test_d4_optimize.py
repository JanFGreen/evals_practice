from evalkit.dataset import Case
from tests.helpers import load

m = load("exercises.d4_optimize", "evalkit.dataset")

ITEMS = [{"id": i, "tag": "a" if i < 10 else "b"} for i in range(16)]


def test_stratified_split_keeps_proportions_and_all_items():
    train, test = m.stratified_split(ITEMS, key=lambda x: x["tag"], test_frac=0.5, seed=3)
    assert sorted(x["id"] for x in train + test) == list(range(16))
    assert not {x["id"] for x in train} & {x["id"] for x in test}
    assert sum(x["tag"] == "a" for x in test) == 5
    assert sum(x["tag"] == "b" for x in test) == 3


def test_stratified_split_is_seeded():
    first = m.stratified_split(ITEMS, key=lambda x: x["tag"], test_frac=0.5, seed=3)
    assert first == m.stratified_split(ITEMS, key=lambda x: x["tag"], test_frac=0.5, seed=3)
    assert first != m.stratified_split(ITEMS, key=lambda x: x["tag"], test_frac=0.5, seed=4)


def test_find_leaked_cases():
    cases = [Case(id="g1", question="Do I have to report a near miss?"), Case(id="g2", question="What is the daily allowance?")]
    prompt = "Examples:\nQ: Do I have to report a near miss?\nA: Yes, within 72 hours."
    assert m.find_leaked_cases(prompt, cases) == ["g1"]
