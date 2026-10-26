"""Command line: `tevro predict request.json`."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tevro", description="Tevro decision model")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="answer one question from a JSON file")
    p.add_argument("input", help='JSON file with "state" and "question" (use - to read stdin)')
    p.add_argument("--model", default=None, help="Hugging Face repo id or local folder (default: kartikg1/tevro-decider)")
    p.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    args = parser.parse_args(argv)

    from tevro.predict import DEFAULT_MODEL, load  # imported here so --help does not load the model stack

    request = json.load(sys.stdin if args.input == "-" else open(args.input, encoding="utf-8"))
    model = load(args.model or DEFAULT_MODEL, args.device)
    print(json.dumps(model.answer(request["state"], request["question"]), indent=2))


if __name__ == "__main__":
    main()
