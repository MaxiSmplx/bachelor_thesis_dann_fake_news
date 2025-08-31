from train import train
from test import test

import os
from datetime import datetime
import argparse
import yaml
import pandas as pd


def find_best_model(
        batch_size: int,
        epochs: int,
        cross_domain: bool,
        augmented: bool,
        balanced: bool,
        iterations: int
):
    """Train and evaluate multiple models to find the best-performing one.

    Parameters
    ----------
    batch_size : int
        Batch size used during training.
    epochs : int
        Number of epochs per training run.
    cross_domain : bool
        If True, use cross-domain setup; else in-domain.
    augmented : bool
        Use augmented data if available.
    balanced : bool
        Use balanced data if available.
    iterations : int
        Number of train/test cycles to run.

    Notes
    -----
    - Writes logs and results to `models/Domain Adversarial Neural Network/logs/find_best_model/`.
    - Tracks accuracy, precision, recall, and F1 score.
    - Reports the highest accuracy achieved across iterations.
    """

    output_folder_path = f"models/Domain Adversarial Neural Network/logs/find_best_model"
    output_file_path = f"{output_folder_path}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
    
    with open("pipeline/config.yml", "r") as f:
        config = yaml.safe_load(f)
    data_folder_name = {
        (False, False): 'raw',
        (True,  False): 'balanced',
        (False, True):  'augmented',
        (True,  True):  'balanced_augmented'
    }[(balanced, augmented)]
    data_folder_attribute = "cross_domain" if cross_domain else "in_domain"
    data_folder_path = os.path.join(f"pipeline/{config['output']}", data_folder_attribute, data_folder_name)
    test_data = pd.read_parquet(f"{data_folder_path}/preprocessed_data_test.parquet")
    with open(output_file_path, "a") as f:
        f.write(
            f"Training in a {'cross-domain' if cross_domain else 'in-domain'} setting \n"
            f"  Testing on domain(s): {test_data['domain'].unique().tolist()} \n\n\n"
        )

    best_acc = 0.0

    for iteration in range(iterations):
        model_path = train(
            batch_size=batch_size,
            num_epochs=epochs,
            cross_domain=cross_domain,
            balanced=balanced,
            augmented=augmented,
            logging=False
        )

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        perf_metrics = test(
            model_checkpoint=model_name,
            logging=False,
            cross_domain=cross_domain,
            balanced=balanced,
            augmented=augmented
        )

        if perf_metrics["Accuracy"] > best_acc:
            best_acc = perf_metrics["Accuracy"]

        with open(output_file_path, "a") as f:
            f.write(
                f"===Iteration {iteration+1}/{iterations}=== \n"
                f"  Model evaluated: {model_name} \n"
                f"  Performance Metrics \n"
                f"      Accuracy: {perf_metrics['Accuracy']:.4f}\n"
                f"      Precision: {perf_metrics['Precision']:.4f}\n"
                f"      Recall: {perf_metrics['Recall']:.4f}\n"
                f"      F1 Score: {perf_metrics['F1 Score']:.4f}\n\n"
            )
    
    with open(output_file_path, "a") as f:
        f.write(f"Best Accuracy found: {best_acc}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best DANN model")

    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--cross_domain", action="store_true", help="Train in cross-domain setting")
    parser.add_argument("--augmented", action="store_true", help="Use augmented data")
    parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--iters", type=int, default=5, help="Training and Testing iterations")

    args = parser.parse_args()

    find_best_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        cross_domain=args.cross_domain,
        augmented=args.augmented,
        balanced=args.balanced,
        iterations=args.iters
    )