import argparse
import json

from ragapp.app import answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Nordwind Assist a question")
    parser.add_argument("question")
    parser.add_argument("--workspace", default="nordwind")
    parser.add_argument("--prompt", default="v2", help="prompt version: v1, v2, ...")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    result = answer(args.question, args.workspace, prompt_version=args.prompt, k=args.k)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print(result.answer)
    print(f"\n— sources in context: {', '.join(result.retrieved_doc_ids) or 'none'}")
    print(f"— model: {result.model} | status: {result.status} | {result.latency_ms} ms"
          + (f" | ${result.cost_usd:.4f}" if result.cost_usd is not None else ""))


if __name__ == "__main__":
    main()
