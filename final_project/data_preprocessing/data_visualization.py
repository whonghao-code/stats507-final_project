import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
os.makedirs("./visualizations", exist_ok=True)

# initialization
dataset = load_dataset("dair-ai/emotion")
train_df = dataset["train"].to_pandas().copy()
label2emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
emotion_order = ["sadness", "joy", "love", "anger", "fear", "surprise"]
colors = sns.color_palette("viridis", len(emotion_order))


# Split by spaces and count the number of original words.
def count_original_words(text):
    return len(text.split())


# Plot a bar chart of category distribution.
def plot_class_distribution():
    train_df["emotion"] = train_df["label"].map(label2emotion)
    emotion_counts = train_df["emotion"].value_counts().reindex(emotion_order, fill_value=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors, zorder=2)
    plt.title("Class Distribution of dair-ai/emotion Dataset", fontsize=16, pad=15)
    plt.xlabel("Emotion Category", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="-", alpha=0.2, zorder=1)
    ax = plt.gca()
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2., height + 20,
            f"{int(height)}", ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig("./visualizations/class_distribution_original_style.png", dpi=300, bbox_inches="tight")


# Plot a box plot of word count distribution.
def plot_text_word_count_by_emotion():
    train_df["word_count"] = train_df["text"].apply(count_original_words)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="emotion", y="word_count", data=train_df,
        palette=colors, order=emotion_order, zorder=2,
        linewidth=1.2
    )
    plt.title("Text Word Count Distribution by Emotion", fontsize=16, pad=15)
    plt.xlabel("Emotion Category", fontsize=12)
    plt.ylabel("Number of Words (Original Text)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="-", alpha=0.2, zorder=1)
    ax = plt.gca()
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    plt.ylim(0, 70)
    plt.tight_layout()
    plt.savefig("./visualizations/text_word_count_original_style.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_class_distribution()
    plot_text_word_count_by_emotion()