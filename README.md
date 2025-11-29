# stats507-final_project

## Project Overview
Sentiment analysis is a key task in Natural Language Processing (NLP) with wide applications, but binary classification benchmarks cannot meet industrial multi-class needs. This study focuses on multi-class sentiment classification using the dair-ai/emotion dataset (6 emotion categories, notable class imbalance), exploring three core objectives: the impact of multilingual back-translation augmentation, performance differences between fine-tuned BERT models and traditional machine learning (TF-IDF+SVM/XGBoost/Random Forest), and effects of BERT parameter scales. Experimental results show that back-translation effectively balances class distribution and improves minority category recognition; fine-tuned BERT outperforms traditional methods via contextual semantics capture; larger-parameter BERT yields limited performance gains with higher training costs. This study validates data augmentation and pre-trained model advantages for imbalanced multi-class sentiment tasks.

For full details on the background, methodology, experimental setup, and result analysis, please consult the official report 'Final_project -Honghao Wang.pdf' in this directory.

## 目录结构
```plaintext
final_project/
├── data_augmenting/               # 数据增强模块
│   ├── augmented_data/            # 增强后的数据集
│   ├── visualizations/            # 数据增强效果可视化（如分布对比图）
│   ├── augmented_data_combination.py  # 原始数据与增强数据合并脚本
│   ├── augmented_data_visualization.py # 增强数据可视化脚本
│   ├── data_augmentation.py       # 核心回译增强逻辑（基于翻译API实现）
│   └── TF-IDF+SVM.py              # 基于增强数据的TF-IDF+SVM模型训练/预测
...
