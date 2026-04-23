import numpy as np
from llm_sdk import Small_LLM_Model
import json
from sys import exit, stderr
from pathlib import Path
from src.models import Functions


class VocabManager:
    def __init__(self, model: Small_LLM_Model, functions: Functions):
        self.model = model
        self.functions = functions

        project_root = Path(__file__).parent.parent
        v_path = project_root / "vocab.json"
        print(v_path)
        try:
            with open(v_path, "r") as f:
                self.tok2id = json.load(f)
                self.id2tok = {v: k for k, v in self.tok2id.items()}
        except Exception as e:
            print(f"CMM: problem loading llm vocab: {e}", file=stderr)
            exit(1)

        self.vocab_size = len(self.tok2id)

        dummy_ids = model.encode("hello").tolist()[0]
        dummy_logits = model.get_logits_from_input_ids(dummy_ids)
        self.mvs = len(dummy_logits)

        self.M_numbers = np.full(self.mvs, -np.inf, dtype=np.float32)
        self.M_chars = np.full(self.mvs, -np.inf, dtype=np.float32)
        self.M_bool = np.full(self.mvs, -np.inf, dtype=np.float32)
        self.M_fun_name = np.full(self.mvs, -np.inf, dtype=np.float32)

        ids = set()
        names = [self.model.encode(f.name) for f in self.functions]
        [ids.update(f[0].tolist()) for f in names]
        self.M_fun_name[list(ids)] = 0.0

        for id, text in self.id2tok.items():
            clean_chars: str = text.replace("Ġ", " ").replace("Ċ", "\n")
            clean = text.replace("Ġ", "").replace("Ċ", "")

            if not clean_chars:
                continue

            if "\n" not in clean_chars:
                self.M_chars[id] = 0.0

            if not clean:
                continue

            if clean == '"':
                self.M_fun_name[id] = 0.0

            if all(c in "0123456789.-," for c in clean):
                if sum([1 for c in clean if c in ",.-"]) > 1:
                    continue
                self.M_numbers[id] = 0.0

            if clean in [
                "true",
                "false",
                "t",
                "r",
                "u",
                "e",
                "f",
                "a",
                "l",
                "s",
                ",",
                "}",
            ]:
                self.M_bool[id] = 0.0
