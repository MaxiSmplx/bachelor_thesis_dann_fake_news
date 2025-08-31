from openai import OpenAI
from dotenv import load_dotenv
import os

def call_openai(context: str, prompt: str, model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.7) -> str:
    """Send a chat completion request to the OpenAI API and return the generated text.

    Parameters
    ----------
    context : str
        System-level instructions to guide the assistant's behavior.
    prompt : str
        User input or query to be answered by the model.
    model : str, default "gpt-4.1-nano-2025-04-14"
        OpenAI model to use for text generation.
    temperature : float, default 0.7
        Sampling temperature; higher values increase randomness.

    Returns
    -------
    str
        Text output produced by the model in response to the prompt.

    Notes
    -----
    - Requires `OPENAI_SECRET_KEY` to be set in the environment.
    - Uses the `responses.create` endpoint to generate text completions.
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