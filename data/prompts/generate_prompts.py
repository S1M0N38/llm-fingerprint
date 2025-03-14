import uuid

from llm_fingerprint.models import Prompt

texts = [
    # Add here the text prompts you want to generate...
]

for text in texts:
    prompt = Prompt(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, text)), prompt=text)
    print(prompt.model_dump_json())
