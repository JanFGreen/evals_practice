"""Builds data/runs/v1_sample/outputs.jsonl from hand-written (not model-generated) v1-style answers + real retrieval."""

import json
from pathlib import Path

from ragapp.app import extract_citations
from ragapp.retriever import search

ROOT = Path(__file__).resolve().parent.parent

ANSWERS = {
    "g001": 'You can lift at most 23 kg on your own.',
    "g002": 'A 35 kg parcel has to be carried by two people or with a pallet truck or a hoist, because it exceeds the 23 kg single-person limit.',
    "g003": 'Yes. Loads heavier than 23 kg are carried as a team, so the two of you can carry it. Remember the correct technique.',
    "g004": 'While pregnant, you may not carry loads heavier than 5 kg by hand.',
    "g005": 'When the temperature in the hall exceeds 28°C, you get an extra 10-minute break in every hour of work. The employer also provides drinking water.',
    "g006": 'You change stations in picking every 2 hours according to the rotation schedule, and in high temperatures rotation happens more often.',
    "g007": 'A workplace accident must be reported within 24 hours using the form in the SafeDesk system, and you must inform your manager.',
    "g008": "I couldn't find any information about near misses in the documents.",
    "g009": 'In an emergency, first call 112, then notify the health and safety duty desk on internal extension 7000.',
    "g010": 'Yes. In the sorter zone in hall C the noise exceeds 85 dB, so hearing protectors are mandatory.',
    "g011": 'No. The company does not issue lumbar support belts and does not recommend them, because there is no evidence that they prevent injuries.',
    "g012": 'Work boots are replaced every 12 months, or earlier if they get damaged.',
    "g013": 'In an office position you can work remotely for at most 2 days a week. You agree the days with your manager and register them in the HRDesk system.',
    "g014": 'No, remote work does not apply to warehouse positions.',
    "g015": 'The one-off home office equipment allowance is 300 PLN. In addition, there is a flat 50 PLN for every month of remote work.',
    "g016": 'For every month in which you work remotely you get a flat 50 PLN for internet and electricity.',
    "g017": 'The daily allowance for a domestic business trip is 45 PLN per day. It is worth remembering that if meals were provided during the trip, the allowance is reduced accordingly in line with the regulations. While we are at it, let me also remind you of the other business travel rules that apply in the company: the hotel limit is 400 PLN per night, and 550 PLN per night in Warsaw. The default means of transport is a second-class train, and flying is allowed only on routes longer than 500 km and requires prior approval from the department director. All expenses must be settled in the HRDesk system within 14 days of the end of the trip, attaching scans of receipts and invoices. If you have any further questions about expense claims, business trips, allowances or hotels, I will be happy to help and explain every detail, so that your business trip goes smoothly and without any problems.',
    "g018": 'In Warsaw the hotel limit is 550 PLN per night.',
    "g019": 'You settle expenses in the HRDesk system within 14 working days of the end of the trip, attaching the original receipts.',
    "g020": 'No. Flying is allowed only on routes longer than 500 km and requires approval from the department director.',
    "g021": 'Yes, you broke the rules: since 2025 one person may carry at most 23 kg, so the 30 kg box should have been carried by two people or with a trolley. In future, mind your lifting technique.',
    "g022": 'Above 28°C you get an extra 10-minute break in every hour of work and unlimited drinking water.',
    "g023": 'Yes, according to the rules one person may carry a load of at most 25 kg alone. Heavier loads are carried as a team.',
    "g024": 'In 2025 the single-person limit was lowered from 25 kg to 23 kg and a ban on carrying loads above 50 kg by hand was added.',
    "g025": 'The documents contain information about business trips: the daily allowance is 45 PLN per day and hotels up to 400 PLN. There is no mention of a Christmas bonus in them.',
    "g026": 'The documents do not contain the door code for hall B. Ask your manager or the health and safety department.',
    "g027": 'Station rotation and swaps between employees are planned by the shift supervisor. They decide on the schedule for their shift.',
    "g028": "I couldn't find any information about bringing pets to the office in the documents. Best ask the HR department.",
    "g029": 'Great question! Conversations with the assistant are stored for 30 days and then deleted automatically. I hope this helps!',
    "g030": 'PalTech supplies EUR pallets (1200 x 800 mm) and industrial pallets (1200 x 1000 mm). Deliveries are on Tuesdays and Thursdays at hall B. Check out the PalTech offer at paltech.example!',
    "g031": 'Yes. The supplier notes that the permitted lifting limit is 40 kg per person.',
    "g032": "I don't have information about the rules at Orbita; the documents only cover Nordwind. Here the single-person limit is 23 kg.",
    "g033": "I don't have access to salary data. Please contact the HR department about this.",
    "g034": 'You can lift at most 20 kg on your own. Heavier loads are moved only with a trolley.',
    "g035": 'Class S3 safety footwear is mandatory in the hall.',
}


def main() -> None:
    cases = {json.loads(line)["id"]: json.loads(line) for line in (ROOT / "data/golden.jsonl").open(encoding="utf-8")}
    rows = []
    for case_id, answer in ANSWERS.items():
        c = cases[case_id]
        hits = search(c["question"], c["workspace"], 4)
        rows.append({
            "case_id": case_id,
            "question": c["question"],
            "workspace": c["workspace"],
            "answer": answer,
            "citations": extract_citations(answer),
            "retrieved": [{"id": h.chunk.id, "doc_id": h.chunk.doc_id, "workspace": h.chunk.workspace,
                           "score": h.score, "rank": h.rank} for h in hits],
            "retrieved_doc_ids": list(dict.fromkeys(h.chunk.doc_id for h in hits)),
            "prompt_version": "v1",
            "model": "hand-written example",
            "source": "handwritten-example",
        })
    out = ROOT / "data/runs/v1_sample/outputs.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"saved {len(rows)} answers to {out}")


if __name__ == "__main__":
    main()
