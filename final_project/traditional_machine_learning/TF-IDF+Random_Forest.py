import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

label_emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
class_names = [label_emotion_map[i] for i in sorted(label_emotion_map.keys())]

dataset = load_dataset("dair-ai/emotion")
train_data = dataset["train"]
test_data = dataset["test"]
X_train_text = train_data["text"]
y_train = train_data["label"]
X_test_text = test_data["text"]
y_test = test_data["label"]


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True
)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# Train a random forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_tfidf, y_train)

y_pred = rf_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(y_test, y_pred, average="weighted")

print("dair-ai/emotion Text multi-class classification results（random-forest+TF-IDF）")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")



plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
cm = confusion_matrix(y_test, y_pred)
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
ax.set_title("Confusion Matrix using TF-IDF+Random Forest", fontsize=20, fontweight='bold', pad=20)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel(
    "Fraction",
    fontsize=20,
    fontweight='bold',
    labelpad=20
)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("./visualizations/original_data_confusion_matrix_rf.png", dpi=300, bbox_inches="tight")