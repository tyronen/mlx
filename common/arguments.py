import argparse


def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--entity", help="W and B entity", default="tyronenicholas")
    parser.add_argument("--project", help="W and B project")
    parser.add_argument(
        "--custom", help="Whether to use custom decoder", action="store_true"
    )
    parser.add_argument("--sweep", help="Run a sweep", action="store_true")
    parser.add_argument("--check", help="Make sure it works", action="store_true")
    parser.add_argument("--model_path", help="Path to the model")
    parser.add_argument("--epochs", help="Number of epochs", default=5)
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "flickr"],
        help="Dataset to use: 'coco' or 'flickr' (default: coco)",
    )
    parser.add_argument(
        "--official_captions",
        action="store_true",
        help="Use official COCO captions instead of synthetic ones",
    )
    return parser
