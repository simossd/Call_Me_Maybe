import numpy as np
from llm_sdk import Small_LLM_Model
import json
from sys import exit, stderr
from pathlib import Path
from src.models import Functions


class Vocabs:
    # loading the vocabs and mask the uneccecaty tokens
