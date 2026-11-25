"""Generates data/prod_traces.jsonl: 14 days of synthetic production traffic with a drift in week two."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261120

TOPICS = {
    # topic: (questions, covered by docs?, typical retrieval top score)
    "lifting": (["How much can I lift alone?", "Parcel is 30 kg, what do I do?", "Kg limit per person?"], True, 4.0),
    "breaks": (["How long is my break?", "Is there a break when it's hot?", "How often is rotation?"], True, 4.5),
    "incidents": (["How do I report an accident?", "By when do I report an injury?", "Safety desk number?"], True, 5.0),
    "remote_work": (["How many days remote?", "Internet allowance?", "Can I work from home?"], True, 4.2),
    "travel": (["How much is the daily allowance?", "Hotel limit Warsaw?", "By when do I settle a trip?"], True, 4.8),
    "overtime": (["How is overtime paid?", "Do I have to stay after hours?", "Night shift bonus?"], False, 1.2),
}
WEEK1 = {"lifting": 0.25, "breaks": 0.2, "incidents": 0.15, "remote_work": 0.2, "travel": 0.18, "overtime": 0.02}
WEEK2 = {"lifting": 0.18, "breaks": 0.15, "incidents": 0.12, "remote_work": 0.15, "travel": 0.15, "overtime": 0.25}


def main() -> None:
    rng = random.Random(SEED)
    start = date(2026, 11, 2)
    rows = []
    for d in range(14):
        day = start + timedelta(days=d)
        mix = WEEK1 if d < 7 else WEEK2
        prompt_version = "v2" if d < 9 else "v2.1"
        for i in range(rng.randint(35, 50)):
            topic = rng.choices(list(mix), weights=list(mix.values()))[0]
            questions, covered, score = TOPICS[topic]
            refused = (not covered and rng.random() < 0.8) or (covered and rng.random() < 0.04)
            p_down = 0.45 if not covered else 0.06
            r = rng.random()
            feedback = "down" if r < p_down else "up" if r < p_down + 0.25 else None
            base_latency = 1800 if prompt_version == "v2" else 2600
            rows.append({
                "trace_id": f"{rng.getrandbits(128):032x}",
                "day": day.isoformat(),
                "question": rng.choice(questions),
                "topic": topic,
                "workspace": "nordwind",
                "prompt_version": prompt_version,
                "retrieval_top_score": round(max(0.0, rng.gauss(score, 1.0)), 2),
                "refused": refused,
                "feedback": feedback,
                "latency_ms": round(rng.lognormvariate(0, 0.35) * base_latency),
                "output_tokens": rng.randint(40, 220),
                "status": "ok" if rng.random() > 0.01 else "error",
            })
    out = ROOT / "data/prod_traces.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"saved {len(rows)} traces to {out}")


if __name__ == "__main__":
    main()
