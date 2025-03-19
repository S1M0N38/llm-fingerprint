from pydantic import BaseModel


class Prompt(BaseModel):
    """Model for LLM prompts."""

    id: str
    prompt: str


class Sample(BaseModel):
    """Model for LLM completion samples."""

    id: str
    model: str = ""
    prompt_id: str
    completion: str


class Result(BaseModel):
    """Model for Database result."""

    model: str
    score: float
