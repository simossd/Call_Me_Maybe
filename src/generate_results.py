from .model_utils import best_next_id, final_answer
import json


def generate(prompts, system_prompt, model, vocab):
    results = []
    i = 1
    print("--------------- json generation start ------------")
    for prompt in prompts:
        prompt = prompt.prompt
        full_prompt = f"{system_prompt}\n\nUser prompt: {prompt}\nAssistant:"
        full_prompt_tokens = model.encode(full_prompt)[0].tolist()
        generated_toks_ids = []
        generated_toks_ids.extend(model.encode('{"name": "')[0].tolist())


        for _ in range(50):
            prompt_logits = model.get_logits_from_input_ids(full_prompt_tokens + generated_toks_ids)
            next_tok = best_next_id(prompt_logits, vocab)
            generated_toks_ids.append(next_tok)
            answer = model.decode(generated_toks_ids)
            if final_answer(answer):
                break
        if answer:
            answer = json.loads(answer)
        elif not answer:
            answer = {"name": "none", "parameters": {}}
        
        results.append(
            {
                "prompt": prompt,
                "name": answer.get("name", "none"),
                "parameters": answer.get("parameters", {})
            }
        )
        print(f"\n\n-- json output prompt {i} --\n")
        print(json.dumps(results[i - 1], indent=3))
        i+=1
    return results