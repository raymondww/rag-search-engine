import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal (image/text) search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate and print the shape of an image embedding"
    )
    verify_parser.add_argument("image_path", help="Path to the image file.")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image query"
    )
    image_search_parser.add_argument("image_path", help="Path to the image file.")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            from rag_engine.multimodal_search import verify_image_embedding
            verify_image_embedding(args.image_path)

        case "image_search":
            from rag_engine.multimodal_search import image_search_command
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (similarity: {result['score']:.3f})")
                print(f"   {result['description'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
