from train import train
from test import test

import os
from datetime import datetime
import argparse


def find_best_model(
        batch_size: int,
        epochs: int,
        cross_domain: bool,
        augmented: bool,
        balanced: bool,
        iterations: int
):
    output_folder_path = f"models/Domain Adversarial Neural Network/logs/find_best_model"
    output_file_path = f"{output_folder_path}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}.txt"
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)

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