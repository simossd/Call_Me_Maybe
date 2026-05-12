from .args_parsing import parse
from argparse import Namespace
from src.models.load_json import LoadPrompt, LoadFunctions
from .model_utils import sys_prompt, sys_vocab
from llm_sdk import Small_LLM_Model as Model






def main() -> None:

    args: Namespace = parse()

    print("Loading Prompts...")
    prompts = LoadPrompt(args.input)
    
    print("Loading Functions...")
    functions = LoadFunctions(args.functions_definition)
    
    print("Generating system prompt...")
    system_prompt = sys_prompt(functions)
    
    model = Model()
    sys_vocab(model)











if __name__ == "__main__":
    main()