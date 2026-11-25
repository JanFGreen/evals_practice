from tests.helpers import load

m = load("exercises.d7_pareto", "evalkit.pareto")


def test_pareto_front():
    points = [
        {"name": "duzy-high", "quality": 0.90, "cost": 0.020, "latency": 4000},
        {"name": "duzy-low", "quality": 0.88, "cost": 0.010, "latency": 2000},
        {"name": "maly", "quality": 0.70, "cost": 0.001, "latency": 800},
        {"name": "dominated", "quality": 0.70, "cost": 0.010, "latency": 2500},
    ]
    assert {p["name"] for p in m.pareto_front(points)} == {"duzy-high", "duzy-low", "maly"}


def test_identical_points_do_not_dominate_each_other():
    p = {"quality": 0.8, "cost": 0.01, "latency": 1000}
    assert len(m.pareto_front([dict(p), dict(p)])) == 2
