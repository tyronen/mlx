import argparse

def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--entity", help="W and B entity", default="tyronenicholas")
    parser.add_argument("--project", help="W and B project", required=True)
    parser.add_argument("--base", help="Whether to use base decoder", action="store_true")
    parser.add_argument("--sweep", help="Run a sweep", action="store_true")
    parser.add_argument("--check", help="Make sure it works", action="store_true")
    parser.add_argument("--model_path", help="Path to the model", required=True)
    return parser.parse_args()