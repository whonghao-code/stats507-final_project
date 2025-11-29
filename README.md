# stats507-final_project

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
