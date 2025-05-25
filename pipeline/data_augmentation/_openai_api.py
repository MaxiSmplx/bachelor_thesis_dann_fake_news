from openai import OpenAI
from dotenv import load_dotenv
import os

def call_openai(context: str, prompt: str, model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.7) -> str:
    """Send a chat completion request to OpenAI and return the generated text.

    Args:
        context (str): System-level instructions to shape the assistants behavior.
        prompt (str): The users message to be answered by the model.
        model (str, optional): The name of the OpenAI model to use.
            Defaults to `"gpt-4.1-nano-2025-04-14"`.

    Returns:
        str: The text output produced by the model in response to the prompt.
    """
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_SECRET_KEY"))

    response = client.responses.create(
        model=model,
        input = [
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature = temperature,
    )
    return response.output_text