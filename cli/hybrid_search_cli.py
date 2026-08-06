import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize documents")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                scores = args.scores
                min_score = min(scores)
                max_score = max(scores)
                if min_score == max_score:
                    for _ in scores:
                        print(f"* 1.0")
                else:
                    for raw_score in scores:
                        score = (raw_score - min_score) / (max_score - min_score)
                        print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()