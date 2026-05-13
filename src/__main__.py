from .args_parsing import parse
from argparse import Namespace
from src.models.load_json import LoadPrompt, LoadFunctions
from .model_utils import (sys_prompt, sys_vocab,
                          clean_vocab)
from llm_sdk import Small_LLM_Model as Model
import json
from pathlib import Path
from .generate_results import generate
import os
import time




def main() -> None:

    args: Namespace = parse()

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # "Loading Prompts..."
    prompts = LoadPrompt(args.input)
    
    # "Loading Functions..."
    functions = LoadFunctions(args.functions_definition)
    
    # "Generating system prompt..."
    system_prompt = sys_prompt(functions)
    
    model = Model()

    # "Loading Model's Vocabs
    uncleaned_vocab = sys_vocab(model)
    vocab = clean_vocab(uncleaned_vocab)

    # "Generating Results..."
    os.system('clear')
    start = time.time()
    resutls = generate(prompts, system_prompt, model, vocab)

    print("\nResults are getting saved ...")
    with open(output_file, 'w') as f:
        json.dump(resutls, f, indent=3)
    print(f"{time.time() - start:.2f}s" )




if __name__ == "__main__":
    main()