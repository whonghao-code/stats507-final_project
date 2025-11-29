import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings("ignore")

# Initialization
label_emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
class_names = [label_emotion_map[i] for i in sorted(label_emotion_map.keys())]


model_path = "../LLM_fine_tuning/bert-emotion-final-cpu"
input_csv_path = "original_data/original_test_dataset.csv"
output_csv_path = "prediction_results/results_of_original_test_dataset.csv"
max_length = 128
device = torch.device("cpu")

# load model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()


# load and preprocess csv
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Successfully read the input CSV with {len (df)} data entries in total")
    return df


df = load_csv_data(input_csv_path)
texts = df["text"].tolist()
true_labels = df["label"].tolist()

inputs = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
).to(device)

# Model prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_labels = torch.argmax(probabilities, dim=-1).tolist()
    confidences = [probabilities[i][pred_idx].item() for i, pred_idx in enumerate(predicted_labels)]

# evaluation function
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, weighted_f1, _ = precision_recall_fscore_support(
    true_labels, predicted_labels, average="weighted"
)

# Generate a new CSV containing prediction results
df["predicted_label"] = predicted_labels
df["confidence"] = [round(c, 4) for c in confidences]
df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"\nCSV containing prediction results has been saved to:{output_csv_path}")

# ---------------------- 7. 输出评估指标 ----------------------
print(f"Accuracy：{accuracy:.4f}")
print(f"Weighted F1：{weighted_f1:.4f}")
print("=" * 60)

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
cm_decimal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10, 8))

im = sns.heatmap(
    cm_decimal,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        "shrink": 0.8,
        "label": "Fraction",
        "pad": 0.05
    },
    ax=ax,
    vmin=0, vmax=1.0,
    annot_kws={"size": 20}
)

ax.set_xlabel("Predict label", fontsize=20, fontweight='bold', labelpad=20)
ax.set_ylabel("Truth label", fontsize=20, fontweight='bold', labelpad=0)
ax.tick_params(axis='x', rotation=0, labelsize=20)
ax.tick_params(axis='y', rotation=0, labelsize=20)
ax.set_title("Confusion Matrix using Bert_large_ft_emotion", fontsize=20, fontweight='bold', pad=20)

cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel(
    "Fraction",
    fontsize=20,
    fontweight='bold',
    labelpad=20
)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
os.makedirs("./visualizations", exist_ok=True)
plt.savefig("./visualizations/bert_large_ft_emotion_confusion_matrix.png", dpi=300, bbox_inches="tight")
