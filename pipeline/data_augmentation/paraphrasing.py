from data_augmentation._openai_api import call_openai

def paraphrase_llm(text:str) -> str:
    context = f"You are a paraphrasing expert: a concise, context-aware assistant that rewrites text with improved clarity, style, and fluency, preserving the original intent and tone."
    prompt = f"Paraphrase the following text. Return only the new version, with no extra commentary or formatting: {text}"

    return call_openai(context, prompt)