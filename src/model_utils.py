from .models.valid import FunctionsDef
from llm_sdk import Small_LLM_Model as Model
import json



def sys_vocab(model: Model):
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r") as vocab_file:
        vocab = json.load(vocab_file)
    raw_vocab = vocab.get("model", {}).get("vocab", {})
    return raw_vocab


def sys_prompt(functions) -> str:
    prompt = [
            "STRICT SYSTEM RULES: use ONLY a matching function "
            "from the list below",
            "If No function matches the user's intent (even if "
            "types match), set name:\"none\".",
            "Never use an unrelated function for a differnet task.",
            "",
            "Available functions:"
        ]
    for f in functions:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in f.parameters.items()
        )
        func = f"   -{f.name}({params}): {f.description}"
        prompt.append(func)
    prompt.append(
        '\nOutput ONLY valid JSON: '
        '{"name": "<fn>", "args": "{<args>}"}'
    )
    final_prompt = "\n".join(prompt)
    return final_prompt
