import json
import matplotlib.pyplot as plt
import os


def visualize_metrics(json_file="training_metrics.json", output_dir="plots"):
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract metrics
    epochs = [entry["epoch"] for entry in data]
    perplexities = [entry["val_perplexity"] for entry in data]
    losses = [entry["avg_train_loss"] for entry in data]
    lrs = [entry["learning_rate"] for entry in data]
    grad_norms = [entry["avg_grad_norm"] for entry in data]

    # Set larger font size
    plt.rcParams.update({"font.size": 14})

    # Plot Validation Perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, perplexities, marker="o", linestyle="-", color="b")
    plt.title("Перплексия върху validation set")
    plt.xlabel("Епоха")
    plt.ylabel("Перплексия")
    plt.grid(True)
    plt.xlim(left=1 - 0.15, right=len(epochs) + 0.15)
    plt.savefig(os.path.join(output_dir, "validation_perplexity.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker="o", linestyle="-", color="r")
    plt.title("Средна загуба при обучение")
    plt.xlabel("Епоха")
    plt.ylabel("Загуба")
    plt.grid(True)
    plt.xlim(left=1 - 0.15, right=len(epochs) + 0.15)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, marker="o", linestyle="-", color="g")
    plt.title("Скорост на обучение (Learning Rate)")
    plt.xlabel("Епоха")
    plt.ylabel("LR")
    plt.grid(True)
    plt.xlim(left=1 - 0.15, right=len(epochs) + 0.15)
    plt.savefig(os.path.join(output_dir, "learning_rate.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, marker="o", linestyle="-", color="m")
    plt.title("Средна норма на градиента по епохи")
    plt.xlabel("Епоха")
    plt.ylabel("Норма на градиента")
    plt.grid(True)
    plt.xlim(left=1 - 0.15, right=len(epochs) + 0.15)
    plt.savefig(os.path.join(output_dir, "gradient_norm.png"))
    plt.close()

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    visualize_metrics()
