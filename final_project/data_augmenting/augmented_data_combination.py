import pandas as pd
import re

# Merge six files
files = [
    "augmented_data/augmented_anger_label_3.csv",
    "augmented_data/augmented_fear_label_4.csv",
    "augmented_data/augmented_joy_label_1.csv",
    "augmented_data/augmented_love_label_2.csv",
    "augmented_data/augmented_sadness_label_0.csv",
    "augmented_data/augmented_surprise_label_5.csv"
]

df_list = []
for file in files:
    df = pd.read_csv(file)
    df_list.append(df)
combined_df = pd.concat(df_list, ignore_index=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    lower_text = text.lower()
    return re.sub(r'[^a-zA-Z\s]', "", lower_text)

combined_df["text"] = combined_df["text"].apply(clean_text)
combined_df.to_csv("augmented_data/augmented_train_dataset.csv", index=False)