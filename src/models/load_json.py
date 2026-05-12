import json
from typing import List
from .valid import FunctionsDef, Prompt


def LoadPrompt(path: str) -> List[Prompt]:
    with open(path, "r") as p_file:
        prompts = json.load(p_file)
    return [Prompt(**item) for item in prompts]

def LoadFunctions(path: str) -> List[FunctionsDef]:
    with open(path, "r") as p_file:
        functions = json.load(p_file)
    return [FunctionsDef(**item) for item in functions]
