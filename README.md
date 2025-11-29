# stats507-final_project

## Project Overview
Sentiment analysis is a key task in Natural Language Processing (NLP) with wide applications, but binary classification benchmarks cannot meet industrial multi-class needs. This study focuses on multi-class sentiment classification using the dair-ai/emotion dataset (6 emotion categories, notable class imbalance), exploring three core objectives: the impact of multilingual back-translation augmentation, performance differences between fine-tuned BERT models and traditional machine learning (TF-IDF+SVM/XGBoost/Random Forest), and effects of BERT parameter scales. Experimental results show that back-translation effectively balances class distribution and improves minority category recognition; fine-tuned BERT outperforms traditional methods via contextual semantics capture; larger-parameter BERT yields limited performance gains with higher training costs. This study validates data augmentation and pre-trained model advantages for imbalanced multi-class sentiment tasks.

For full details on the background, methodology, experimental setup, and result analysis, please consult the official report 'Final_project -Honghao Wang.pdf' in this directory.

## Project Structure
```plaintext
final_project/
├── data_augmenting/               
│   ├── augmented_data_combination.py  # Merges multiple CSV files of augmented data and generates a combined augmented training dataset for subsequent model training.
│   ├── augmented_data_visualization.py # Visualizes the category distribution and text word count distribution of the augmented dataset.
│   ├── data_augmentation.py       # Performs data augmentation on the emotion dataset via multilingual back-translation, generates additional samples for each emotion category, and saves the augmented dataset by category.
│   └── TF-IDF+SVM.py              # Extracts text features using TF-IDF, trains a SVM for emotion classification, and generates accuracy, weighted F1, confusion matrix to evaluate model performance.
│
├── data_preprocessing/           
│   └── data_visualization.py      # Visualizes the emotion category distribution and text word count distribution of the original dair-ai/emotion dataset.
│
├── industrial_program/           
│   └── prediction_program.py      # Loads the pre-trained BERT model to perform emotion prediction, outputs prediction results and confidence scores, and generates accuracy, weighted F1, and confusion matrix.
│
├── LLM_fine_tuning/               
│   ├── bert_base_finetune.py      # Fine-tunes the BERT-base (uncased) model for emotion classification and saves the optimal model weights.
│   └── bert_large_finetune.py     # Fine-tunes the BERT-large (uncased) model for emotion classification and saves the optimal model weights.
│
├── traditional_ML/                
│   ├── TF-IDF+XGBoost.py          # Extracts text features using TF-IDF, trains a XGboost model for emotion classification, and generates accuracy, weighted F1, confusion matrix to evaluate model performance.
│   └── TF-IDF+RandomForest.py     # Extracts text features using TF-IDF, trains a Random Forest model for emotion classification, and generates accuracy, weighted F1, confusion matrix to evaluate model performance.
```

## Environment Setup
The project requires Python 3.8.19. All dependencies are listed in requirements.txt. You can install them using:
```bash
pip install -r requirements.txt
```

## Experimental Workflow
### 1. Exploring Of Original Dataset
Generate visualizations of emotion category distribution and text word count distribution for the original dair-ai/emotion dataset to intuitively show class imbalance and text features.
```bash
python final_project/data_preprocessing/data_visualization.py
```

### 2. Multilingual Back-Translation Data Augmentation
Generate augmented samples via back-translation with 8 English-foreign language pairs to balance sample counts across emotion categories, then merge and visualize the augmented data to create a usable training dataset.
```bash
python final_project/data_augmenting/data_augmentation.py
python final_project/data_augmenting/augmented_data_combination.py
python final_project/data_augmenting/augmented_data_visualization.py
```

### 3. TF-IDF+SVM Model Comparison Experiment
Train TF-IDF+SVM models based on the original dataset and the augmented dataset respectively, compare accuracy, weighted F1-score and confusion matrices of the model under the two datasets, and analyze the impact of data augmentation on model performance.
```bash
python final_project/data_augmenting/TF-IDF+SVM.py
```

### 4. TF-IDF+SVM Model Comparison Experiment
Train TF-IDF+XGBoost and TF-IDF+Random Forest models, calculate model evaluation metrics and generate confusion matrices to complete performance verification of traditional machine learning models
```bash
python final_project/traditional_ML/TF-IDF+XGBoost.py
python final_project/traditional_ML/TF-IDF+RandomForest.py
```

### 5. BERT Model Fine-Tuning and Prediction
Fine-tune BERT-base (110M parameters) and BERT-large (340M parameters) models respectively, save the optimal model based on validation set performance; load the pre-trained BERT model to perform emotion prediction, output prediction labels and confidence scores, and calculate accuracy, weighted F1-score and generate confusion matrix to complete evaluation.
Note: Failure to run bert_base_uncased.py and bert_large_uncased.py will prevent running industrial_program.py due to the lack of fine-tuned model files.
```bash
python final_project/LLM_fine_tuning/bert_base_uncased.py
python final_project/LLM_fine_tuning/bert_large_uncased.py
python final_project/industrial_program/industrial_program.py
```


## Contact Information
Honghao Wang

Dept. of Statistics, University of Michigan, Ann Arbor

Email: whonghao@umich.edu
