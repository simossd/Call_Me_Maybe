from pydantic import BaseModel
from typing import Any

class Prompt(BaseModel):
    prompt: str

class Parameters(BaseModel):
    type: str

class FunctionsDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameters]
    returns: dict[str, Any]
