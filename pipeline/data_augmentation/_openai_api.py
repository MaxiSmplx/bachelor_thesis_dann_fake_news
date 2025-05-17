from openai import OpenAI
from dotenv import load_dotenv
import os

def call_openai(context: str, prompt: str) -> str:
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_SECRET_KEY"))

    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
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
        temperature = 0.7,
    )
    return response.output_text