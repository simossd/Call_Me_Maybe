from .models.valid import FunctionsDef
from llm_sdk import Small_LLM_Model as Model
import json


def final_answer(answer: str):
    start = answer.find("{")
    if start == -1:
        return None
    bracket_count = 0
    for i in range(start, len(answer)):
        if answer[i] == "{":
            bracket_count += 1
        if answer[i] == "}":
            bracket_count -= 1
        if bracket_count == 0:
            return answer[start:i+1]
    return None

def best_next_id(promtps_logits, cleaned_vocab):
    return max(cleaned_vocab, key=lambda x: promtps_logits[x])

def clean_vocab(vocab: dict):
    choosed_toks = set(
        'BCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        '0123456789*.,_:-+/\'!?()[]{}"ĠĊ'
    )
    cleaned_vocab = set()
    for tok_str, tok_id in vocab.items():
        if not tok_str:
            continue
        is_valid = True
        for c in tok_str:
            if c not in choosed_toks:
                is_valid = False
                break
        if is_valid:
            cleaned_vocab.add(tok_id)
    return cleaned_vocab

def sys_vocab(model: Model):
    vocab_path = model.get_path_to_tokenizer_file()
    print(vocab_path)
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
    '\nOutput ONLY valid JSON:\n'
    '{"name":"<fn>","parameters":{"param":"value"}}'
)
    final_prompt = "\n".join(prompt)
    return final_prompt
