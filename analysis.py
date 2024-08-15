plt.switch_backend("agg")


def plot_training_logs(logs: list[TrainingLog], depth, width, results_dir):
    # Convert logs to DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "step": [log.step for log in logs],
            "train_loss": [log.train_loss for log in logs],
            "val_loss": [
                log.val_loss if log.val_loss is not None else float("nan")
                for log in logs
            ],
            "val_accuracy": [
                log.val_accuracy if log.val_accuracy is not None else float("nan")
                for log in logs
            ],
        }
    )

    os.makedirs(results_dir, exist_ok=True)
    # Set the style and color palette
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses on the left y-axis
    sns.lineplot(
        data=df, x="step", y="train_loss", ax=ax1, label="Train Loss", linewidth=3
    )

    sns.lineplot(
        data=df, x="step", y="val_loss", ax=ax1, label="Validation Loss", linewidth=3
    )
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax1.grid(False)

    # Plot accuracy on the right y-axis
    sns.lineplot(
        data=df,
        x="step",
        y="val_accuracy",
        ax=ax2,
        color="red",
        label="Validation Accuracy",
    )
    ax2.set_ylabel("Accuracy")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax2.set_ylim(0, 1)
    ax2.legend_.remove()

    # Set title and adjust layout
    plt.title(f"Training Progress (layers={depth}, width={width})", fontsize=14)
    plt.tight_layout()

    plt.savefig(f"{results_dir}/mnist_d{depth}w{width}")
    # plt.show()
    plt.close()


plot_training_logs(logs, depth, width, results_dir)
