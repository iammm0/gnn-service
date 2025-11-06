# 训练数据统计

已成功生成包含 **100条** 训练数据的文件：`docs/training_data_100.json`

## 数据特点

训练数据涵盖了以下场景类型：

1. **心理活动/情感变化类** - 包含情绪低落、情绪转折、自我建议等语义类型
2. **工作场景类** - 包含问题描述、目标设定、解决方案等语义类型
3. **人际关系类** - 包含关系困境、担忧、内心冲突等语义类型
4. **学习成长类** - 包含学习困难、成就感、未来规划等语义类型
5. **生活感悟类** - 包含时间感慨、人生感悟、成长认知等语义类型

## 数据格式

每条训练数据包含：
- `text`: 完整的长文本
- `chunks`: 按照语义/心理活动变化分割的文本块
  - `start`: 起始位置
  - `end`: 结束位置
  - `content`: 文本内容
  - `semantic_type`: 语义类型标注

## 使用方法

### 训练模型

```bash
python train_text_splitter.py --train_data docs/training_data_100.json --epochs 3
```

### 或使用API

```python
import requests

url = "http://localhost:8000/train_text_splitter/"
data = {
    "train_data_path": "docs/training_data_100.json",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5
}
response = requests.post(url, json=data)
```

## 数据扩展

如果需要更多训练数据，可以：
1. 使用 `scripts/generate_training_data.py` 脚本生成更多数据
2. 手动添加更多标注好的训练样本
3. 根据实际应用场景定制训练数据

## 注意事项

- 确保训练数据的标注质量，分割点应该准确反映语义/心理活动变化
- 建议准备验证数据集用于评估模型性能
- 训练数据越多，模型效果越好，建议至少100条以上

