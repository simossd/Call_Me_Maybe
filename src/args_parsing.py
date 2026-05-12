from argparse import ArgumentParser as Ar
from argparse import Namespace


def parse() -> Namespace:
    args = Ar()
    args.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    args.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    args.add_argument(
        "--output",
        default="data/output/functions_result.json"
    )
    args = args.parse_args()
    return args
