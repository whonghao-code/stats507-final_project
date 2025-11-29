import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import random
import warnings

warnings.filterwarnings('ignore')

# initialization
LABEL_MAPPING = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

dataset = load_dataset("dair-ai/emotion")
train_df = dataset["train"].to_pandas()[["text", "label"]].copy()
test_df = dataset["test"].to_pandas()[["text", "label"]].copy()

# Calculate the number of samples in each category of the training set.
label_counts = train_df["label"].value_counts().sort_index()

# Determine the maximum number of samples and the target sample count (80% of the maximum number of samples).
max_count = label_counts.max()
target_per_class = int(max_count * 0.8)


# Load the multilingual back-translation model.
def load_multilingual_backtrans_models():
    language_pairs = [
        ("en", "de"),
        ("en", "es"), ("en", "fr"), ("en", "zh"),("en", "ru"), ("en", "ar"), ("en", "it"), ("en", "nl")
    ]
    models = {}
    for src, tgt in language_pairs:
        # Forward translation (English → Intermediate Language)
        forward_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        forward_tok = MarianTokenizer.from_pretrained(forward_name)
        forward_mod = MarianMTModel.from_pretrained(forward_name)
        # Reverse translation (Intermediate Language → English)
        reverse_name = f"Helsinki-NLP/opus-mt-{tgt}-{src}"
        reverse_tok = MarianTokenizer.from_pretrained(reverse_name)
        reverse_mod = MarianMTModel.from_pretrained(reverse_name)
        models[tgt] = (forward_tok, forward_mod, reverse_tok, reverse_mod)
    return models


# Load the multilingual back-translation model (will be downloaded on the first run)...
backtrans_models = load_multilingual_backtrans_models()
available_langs = list(backtrans_models.keys())


# Back-translation
def back_translate(text, target_lang):
    forward_tok, forward_mod, reverse_tok, reverse_mod = backtrans_models[target_lang]
    inputs = forward_tok(text, return_tensors="pt", truncation=True, max_length=100)
    forward_out = forward_mod.generate(**inputs, max_length=100)
    mid_text = forward_tok.decode(forward_out[0], skip_special_tokens=True)
    inputs = reverse_tok(mid_text, return_tensors="pt", truncation=True, max_length=100)
    reverse_out = reverse_mod.generate(**inputs, max_length=100)
    aug_text = reverse_tok.decode(reverse_out[0], skip_special_tokens=True)
    # Deduplication
    return aug_text.strip() if aug_text.strip() != text.strip() else None

# Basic filtering conditions: 1. Length: 3-70 2. Not empty
def filter_valid_sample(augmented):
    if not augmented:
        return False
    word_count = len(augmented.split())
    return 3 <= word_count <= 70

# For convenience, traverse all categories and save a separate file for each category.
for label in range(0,6):
    label_name = LABEL_MAPPING[label]
    original_texts = train_df[train_df["label"] == label]["text"].tolist()
    current_count = len(original_texts)
    need_generate = max(0, target_per_class - current_count)

    print(f"\n【Process categories】{label_name} ({label})")
    print(f"Original sample count：{current_count}，number of samples to be generated：{need_generate}")

    if need_generate <= 0:
        # No need to generate; save the original samples directly.
        class_df = pd.DataFrame([{"text": text, "label": label} for text in original_texts])
        output_path = f"augmented_data/augmented_{label_name}_label_{label}.csv"
        class_df[["text", "label"]].to_csv(output_path, index=False, encoding="utf-8")
        continue

    new_samples = []
    used_texts = set(original_texts)
    while len(new_samples) < need_generate:
        original_text = random.choice(original_texts)
        target_lang = random.choice(available_langs)
        aug_text = back_translate(original_text, target_lang=target_lang)
        if aug_text and aug_text not in used_texts and filter_valid_sample(aug_text):
            new_samples.append({"text": aug_text, "label": label})
            used_texts.add(aug_text)
            if len(new_samples) % 100 == 0:
               print(f"Generated {len(new_samples)}/{need_generate}")

    # Merge all samples
    all_class_samples = [{"text": text, "label": label} for text in original_texts] + new_samples
    class_df = pd.DataFrame(all_class_samples)[["text", "label"]]
    output_path = f"augmented_data/augmented_{label_name}_label_{label}.csv"
    class_df.to_csv(output_path, index=False, encoding="utf-8")


test_output_path = "augmented_data/original_test_dataset.csv"
final_test_df = test_df[["text", "label"]].copy()
final_test_df.to_csv(test_output_path, index=False, encoding="utf-8")




