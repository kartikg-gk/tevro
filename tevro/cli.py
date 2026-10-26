"""Command line: `tevro predict request.json` and `tevro serve`."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tevro", description="Tevro decision model")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="answer one question from a JSON file")
    p.add_argument("input", help='JSON file with "state" and "question" (use - to read stdin)')
    s = sub.add_parser("serve", help="run a local /v1/systemone server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8008)
    for cmd in (p, s):
        cmd.add_argument("--model", default=None, help="Hugging Face repo id or local folder (default: kartikg1/tevro-decider)")
        cmd.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    args = parser.parse_args(argv)

    if args.command == "predict" and args.input != "-":
        try:
            request = json.load(open(args.input, encoding="utf-8-sig"))
        except FileNotFoundError:
            sys.exit(f"tevro: no such file: {args.input}")
    elif args.command == "predict":
        request = json.load(sys.stdin)
    elif args.command == "serve":
        try:
            import uvicorn

            from tevro.serve import create_app
        except ImportError:
            sys.exit('the server needs the serve extra: pip install "tevro[serve]"')

    from tevro.predict import DEFAULT_MODEL, load  # imported here so --help does not load the model stack

    model = load(args.model or DEFAULT_MODEL, args.device)
    if args.command == "predict":
        print(json.dumps(model.answer(request["state"], request["question"]), indent=2))
    else:
        uvicorn.run(create_app(model, args.model or DEFAULT_MODEL), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
