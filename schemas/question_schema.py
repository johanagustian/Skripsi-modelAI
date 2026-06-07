from typing import Literal

from pydantic import BaseModel

class GenerateRequest(BaseModel):
    difficulty: Literal["middle", "high"]
    context: str
    mode_kreatif: bool = True