from tests.helpers import load

m = load("exercises.d2_multiturn", "evalkit.simulator")

TRANSCRIPT = [
    {"user": "What is the hotel limit?", "assistant": "The hotel limit is 400 PLN."},
    {"user": "And in Warsaw?", "assistant": "In Warsaw 550 PLN."},
]


def test_goal_reached_on_second_turn():
    assert m.conversation_metrics(TRANSCRIPT, ["550"]) == {
        "turns": 2, "goal_reached": True, "turns_to_goal": 2, "refusals": 0}


def test_keywords_can_be_spread_across_turns():
    assert m.conversation_metrics(TRANSCRIPT, ["400 PLN", "550"])["turns_to_goal"] == 2
    assert m.conversation_metrics(TRANSCRIPT, ["400"])["turns_to_goal"] == 1


def test_goal_not_reached_and_refusals_counted():
    transcript = TRANSCRIPT + [{"user": "And the bonus?", "assistant": "I don't know — it is not in the documents."}]
    r = m.conversation_metrics(transcript, ["999"])
    assert r["goal_reached"] is False
    assert r["turns_to_goal"] is None
    assert r["refusals"] == 1
