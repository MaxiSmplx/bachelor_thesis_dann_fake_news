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
from datetime import datetime
from time import perf_counter

def get_data(cross_domain: bool = False, augmented: bool = False, balanced: bool = False) -> pd.DataFrame:
    return get_dataset("test", cross_domain=cross_domain, augmented=augmented, balanced=balanced)

def prompt_gpt(data: pd.DataFrame, model: str) -> list[dict[str, str]]:
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


def process_results(responses: list[dict[str, str]], no_samples: int) -> tuple[list[str], list[str]]:
    y_true = [response["label_original"] for response in responses]
    y_pred = [response["label_predicted"] for response in responses]


    # Sanity check: Ensure the LLM output contains exactly one label per sample, and only valid labels ('real' or 'fake').
    if all(item in {"real", "fake"} for item in y_pred) and len(y_pred) == no_samples:
        print("Successfully labeled all samples provided")
    else:
        print(f"ERROR: Only labeled {no_samples - sum(item not in {'real', 'fake'} for item in y_pred)} / {no_samples} samples correctly")
        wrong_indices = [i for i, item in enumerate(y_pred) if item not in ("real", "fake")]

        y_pred = [val for i, val in enumerate(y_pred) if i not in wrong_indices]
        y_true = [val for i, val in enumerate(y_true) if i not in wrong_indices]

    return y_true, y_pred


def run(cross_domain: bool = False, balanced: bool = False, augmented: bool = False, no_samples: int = 1000, model: str = "gpt-4.1-nano-2025-04-14") -> None:
    """Run evaluation with an LLM model on a sampled dataset and log results.

    Parameters
    ----------
    cross_domain : bool, default False
        If True, sample data from cross-domain setup; else in-domain.
    balanced : bool, default False
        Use balanced data if available.
    augmented : bool, default False
        Use augmented data if available.
    no_samples : int, default 1000
        Number of samples to evaluate.
    model : str, default "gpt-4.1-nano-2025-04-14"
        Model identifier (must exist in `inference_costs` for cost estimation).

    Returns
    -------
    None
        Prints evaluation metrics, estimated costs (if model is known),
        and logs results to `models/LLM/output/{model}/training_summary_*.txt`.

    Notes
    -----
    - Approximates total inference cost based on average token length and model pricing.
    - Computes Accuracy, Precision, Recall, and F1 score.
    - Saves evaluation summary and metrics to disk.
    """

    data_raw = get_data(cross_domain=cross_domain, balanced=balanced, augmented=augmented)

    data = data_raw.sample(n=no_samples)

    print(f"Using gpt-model: {model}")

    inference_costs = {
        "gpt-4.1-nano-2025-04-14": (0.1, 0.4),
        "gpt-4.1-2025-04-14": (2, 8),
        "gpt-4o-mini-2024-07-18": (0.15, 0.6),
        "gpt-4o-2024-11-20": (2.5, 10),
        "gpt-5-2025-08-07": (1.25, 10),
        "gpt-5-nano-2025-08-07": (0.05, 0.4)
    }

    if not os.path.isdir(f"models/LLM/output/{model}"):
        os.makedirs(f"models/LLM/output/{model}")

    log_output_file = f"models/LLM/output/{model}/training_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    with open(log_output_file,"a") as f:
        f.write(
            f"Training Summary for {model} \n"
            f"Settings: \n"
            f"  {'cross-domain' if cross_domain else 'in-domain'} \n"
            f"  {'balanced' if balanced else 'unbalanced'} \n"
            f"  {'augmented' if augmented else 'not-augmented'} \n"
            f"  original test dataset length: {len(data_raw)} \n"
        )

    if model in inference_costs.keys():
        avg_tokens = data['text'].apply(len).mean() * 0.75
        cost_test_data = inference_costs[model][0] / 1_000_000 * avg_tokens
        cost_prompt = inference_costs[model][0] / 1_000_000 * (65 * 0.75)
        output_cost = inference_costs[model][1] / 1_000_000 * (4 * 0.75)
        with open(log_output_file,"a") as f:
            f.write(
                f"  approx total cost: ${(total_costs := (cost_test_data + cost_prompt + output_cost) * no_samples):.4f} \n"
            )
        print(f"Approximated total costs: ${total_costs:.4f}")

    start_time = perf_counter()
    responses = prompt_gpt(data, model)
    inference_time = perf_counter() - start_time

    y_true, y_pred = process_results(responses, no_samples)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="real", zero_division=0)
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

    with open(log_output_file ,"a") as f:
        f.write(
            f"  samples configured: {no_samples} -> samples classified: {len(y_pred)} \n"
            f"Inference time: {inference_time:.2f} seconds \n\n"
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
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to use")

    args = parser.parse_args()

    run(
        cross_domain=args.cross_domain,
        augmented=args.augmented,
        balanced=args.balanced,
        no_samples=args.samples,
        model=args.model
    )