from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from data_loader import get_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os
import pandas as pd
import argparse

def get_data(cross_domain: bool = False, augmented: bool = False, balanced: bool = False):
    return get_dataset("test", cross_domain=cross_domain, augmented=augmented, balanced=balanced)

def prompt_gpt(data: pd.DataFrame, model: str) -> list[dict[str:str]]:
    prompts = [
        {
            "text": row["text"],
            "label_orig": "real" if row["label"] == 0 else "fake"
        }
        for _, row in data.iterrows()
    ]

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_SECRET_KEY"))

    responses = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        response = client.responses.create(
            model=model,
            input = [
                {
                    "role": "system",
                    "content": "You are a fake news detector. Respond with only 'real' or 'fake'."
                },
                {
                    "role": "user",
                    "content": prompt["text"]
                },
            ],
            temperature = 0.2,
        )

        label_pred = response.output_text.lower()
        responses.append(
            {
                "prompt": prompt["text"], 
                "label_original": prompt["label_orig"], 
                "label_predicted": label_pred
            }
        )

    return responses


def process_results(responses: list[dict[str:str]], no_samples: int) -> tuple[list[str], list[str]]:
    y_true = [response["label_original"] for response in responses]
    y_pred = [response["label_predicted"] for response in responses]


    # Sanity check: Ensure the LLM output contains exactly one label per sample, and only valid labels ('real' or 'fake').
    if all(item in {"real", "fake"} for item in y_pred) and len(y_pred) == no_samples:
        print("Successfully labeled all samples provided")
    else:
        print(f"ERROR: Only labeled {no_samples - sum(item not in {'real', 'fake'} for item in y_pred)} / {no_samples} samples correctly")

    return y_true, y_pred


def run(cross_domain: bool = False, balanced: bool = False, augmented: bool = False, no_samples: int = 1000, model: str = "gpt-4.1-nano-2025-04-14"):
    data = get_data(cross_domain=cross_domain, balanced=balanced, augmented=augmented).sample(n=no_samples)

    print(f"Using gpt-model: {model}")

    responses = prompt_gpt(data, model)

    y_true, y_pred = process_results(responses, no_samples)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="real")
    recall = recall_score(y_true, y_pred, pos_label="real")
    f1 = f1_score(y_true, y_pred, pos_label="real")

    print(
        f"\nEvaluation Metrics:\n"
        f"---------------------------\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1 Score : {f1:.4f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt LLM model")

    parser.add_argument("--cross_domain", action="store_true", help="Train in cross-domain setting")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano-2025-04-14", help="GPT Model name")
    parser.add_argument("--samples", type=int, help="Number of samples to use")

    args = parser.parse_args()

    run(
        cross_domain=args.cross_domain,
        augmented=args.augmented,
        balanced=args.balanced,
        no_samples=args.samples,
        model=args.model
    )