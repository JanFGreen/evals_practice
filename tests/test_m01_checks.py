from tests.helpers import load

m = load("exercises.m01_checks", "evalkit.checks")


def test_normalize_casefolds_and_collapses_whitespace():
    assert m.normalize("  How MANY \n kg ") == "how many kg"


def test_normalize_glues_numbers_to_units():
    assert m.normalize("23 kg") == m.normalize("23kg") == "23kg"
    assert m.normalize("Limit: 550 PLN") == "limit: 550pln"


def test_contains_any():
    assert m.contains_any("The limit is 23 kg.", ["23 kg", "twenty-three"]).passed
    assert not m.contains_any("The limit is 25 kg.", ["23"]).passed


def test_contains_any_respects_number_boundaries():
    assert not m.contains_any("Year 2023 and 123 items", ["23"]).passed
    assert m.contains_any("(23 kg)", ["23"]).passed


def test_contains_any_ignores_case_and_unit_spacing():
    assert m.contains_any("Deliveries on TUESDAYS", ["tuesday"]).passed
    assert m.contains_any("flat 50PLN", ["50 PLN"]).passed


def test_not_contains():
    assert m.not_contains("Limit 23 kg", ["25 kg"]).passed
    assert not m.not_contains("Check out the PalTech offer!", ["check out the paltech"]).passed


def test_max_words():
    assert m.max_words("one two three", 3).passed
    assert not m.max_words("one two three four", 3).passed


def test_extract_citations_unique_in_order():
    assert m.extract_citations("A [b-1] c [a-2] d [b-1]") == ["b-1", "a-2"]
    assert m.extract_citations("no citations [Not Valid]") == []


def test_citations_grounded():
    assert m.citations_grounded(["a"], ["a", "b"]).passed
    assert m.citations_grounded([], ["a"]).passed
    result = m.citations_grounded(["a", "invented"], ["a"])
    assert not result.passed
    assert "invented" in result.detail


def test_is_refusal():
    assert m.is_refusal("I don't know — it is not in the documents.")
    assert m.is_refusal("Unfortunately, the documents do not contain information about the bonus.")
    assert not m.is_refusal("The limit is 23 kg.")
