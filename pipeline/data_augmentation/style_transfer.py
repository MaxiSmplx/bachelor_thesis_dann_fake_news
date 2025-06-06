from data_augmentation._openai_api import call_openai

def transfer_style(text: str, target_style: str) -> str:
    context = f"You are a style transfer assistant: skilled at rewriting text to match a specified style while preserving the original meaning. You ensure that the output aligns with the tone, vocabulary, and sentence structure typical of the target style."
    prompt = f"Rewrite the text below in the style of '{target_style}'. Preserve the meaning, but change the tone, vocabulary, and structure to match the target style. Return only the revised text, with no explanations or formatting: {text}"

    return call_openai(context, prompt)