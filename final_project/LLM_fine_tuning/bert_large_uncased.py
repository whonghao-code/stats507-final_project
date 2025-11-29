import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import warnings

warnings.filterwarnings("ignore")

# Initialization
device = torch.device("cpu")
max_length = 128
batch_size = 16
lr = 2e-5
num_epochs = 4
model_save_path = "./bert-large-emotion-final-cpu"
best_f1 = 0.0

# Load + preprocess the dataset
dataset = load_dataset("dair-ai/emotion")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, shuffle=False)

# build model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=6, ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()


# evaluation function
def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate(model, loader, split):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds = torch.argmax(outputs.logits, dim=-1)
            all_labels.extend(batch["label"].numpy())
            all_preds.extend(preds.numpy())
    metrics = compute_metrics(all_labels, all_preds)
    print(f"\n{split} Metric: Accuracy={metrics['accuracy']:.4f}，F1={metrics['f1']:.4f}\n")
    return metrics


# training loop
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        # forward propagation
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = criterion(outputs.logits, batch["label"])

        # backward propagation + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print logs every 20 steps
        if (step + 1) % 20 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"epoch={epoch + 1}，step={step + 1}，loss={avg_loss:.4f}")

    return total_loss / len(loader)


# Start training
print(f"Start training（batch_size={batch_size}，max_length={max_length}）")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    print(f"Training loss of epoch {epoch + 1}：{train_loss:.4f}")

    # Save the optimal model
    val_metrics = evaluate(model, val_loader, "validation set")
    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Save the optimal model（F1={best_f1:.4f}）")

# Test set evaluation
print("Test set evaluation：")
best_model = BertForSequenceClassification.from_pretrained(model_save_path).to(device)
evaluate(best_model, test_loader, "Test set")
print(f"Training completed! The model is saved in：{model_save_path}")