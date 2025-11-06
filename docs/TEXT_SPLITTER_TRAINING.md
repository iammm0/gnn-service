# 文本分割模型训练指南

## 概述

文本分割模型训练功能允许您根据提供的训练数据，训练一个能够根据语义/心理活动变化进行文本分割的模型。与传统的基于固定规则（如标点符号、固定长度）的分割方法不同，这个模型可以学习识别语义变化点，例如心理活动的转换、话题的切换等。

## 训练数据格式

训练数据需要是JSON格式，包含以下结构：

```json
[
    {
        "text": "完整的长文本",
        "chunks": [
            {
                "start": 0,
                "end": 50,
                "content": "第一段文本",
                "semantic_type": "语义类型（可选）"
            },
            {
                "start": 50,
                "end": 100,
                "content": "第二段文本",
                "semantic_type": "语义类型（可选）"
            }
        ]
    }
]
```

### 字段说明

- `text`: 完整的长文本内容
- `chunks`: 分割后的文本块列表
  - `start`: 文本块在原文中的起始位置（字符位置）
  - `end`: 文本块在原文中的结束位置（字符位置）
  - `content`: 文本块的内容
  - `semantic_type`: 语义类型（可选），用于描述该文本块的语义特征

### 示例

参考 `docs/training_data_example.json` 文件，其中包含了一个完整的示例。

## 训练方法

### 方法1：使用命令行脚本

```bash
python train_text_splitter.py \
    --train_data docs/training_data_example.json \
    --val_data docs/val_data.json \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --output_dir models/text_splitter \
    --base_model bert-base-chinese
```

### 方法2：使用API接口

```python
import requests

url = "http://localhost:8000/train_text_splitter/"

data = {
    "train_data_path": "docs/training_data_example.json",
    "val_data_path": "docs/val_data.json",  # 可选
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5
}

response = requests.post(url, json=data)
print(response.json())
```

### 方法3：直接使用Python代码

```python
from src.text_splitter_trainer import TextSplitterTrainer

trainer = TextSplitterTrainer(
    model_name="bert-base-chinese",
    output_dir="models/text_splitter"
)

trainer.train(
    train_data_path="docs/training_data_example.json",
    val_data_path="docs/val_data.json",  # 可选
    epochs=3,
    batch_size=8,
    learning_rate=2e-5
)
```

## 训练参数说明

- `train_data_path`: 训练数据文件路径（必需）
- `val_data_path`: 验证数据文件路径（可选，用于评估模型性能）
- `epochs`: 训练轮数（默认：3）
- `batch_size`: 批次大小（默认：8，根据GPU内存调整）
- `learning_rate`: 学习率（默认：2e-5）
- `output_dir`: 模型保存目录（默认：models/text_splitter）
- `base_model`: 基础预训练模型（默认：bert-base-chinese）

## 模型使用

训练完成后，模型会自动保存到指定的输出目录。在文本处理时，系统会自动检测并使用训练好的模型进行文本分割。

### 自动使用

当模型训练完成后，`_split_long_text` 函数会自动尝试加载并使用训练好的模型。如果模型不存在或加载失败，会回退到规则分割方法。

### 手动指定模型路径

```python
from src.text_splitter import SemanticTextSplitter

splitter = SemanticTextSplitter(model_path="models/text_splitter")
chunks = splitter.split_by_semantic_chunks(text)
```

## 训练数据准备建议

1. **数据量**: 建议至少准备100-200条训练样本，数据量越大，模型效果越好
2. **数据质量**: 确保分割点标注准确，符合语义/心理活动变化
3. **数据多样性**: 包含不同类型的文本（对话、叙述、心理活动等）
4. **分割标准**: 统一分割标准，例如：
   - 心理活动变化
   - 话题切换
   - 情感转换
   - 场景变化

## 示例：心理活动分割

对于您提供的示例文本：

```
她今天看了我一眼，那眼神有什么特别的含义吗？是不是她对我也有一点好感？为什么她会选择坐在我旁边，是巧合还是有意为之？她的背影真的是太迷人了，每次看到都让我的心跳加速。我是不是应该鼓起勇气去跟她说话，哪怕只是一个简单的问候，也许能拉近我们的距离。我该怎么开口才不显得尴尬？如果她回应得很友好，那是不是意味着她也愿意和我多交流？这些问题在我脑海中不断盘旋，我真的好希望能找到答案。
```

可以按照以下方式分割：

1. **观察与疑问** (0-19): "她今天看了我一眼，那眼神有什么特别的含义吗？"
2. **推测与期待** (19-40): "是不是她对我也有一点好感？"
3. **分析行为动机** (40-60): "为什么她会选择坐在我旁边，是巧合还是有意为之？"
4. **情感反应** (60-85): "她的背影真的是太迷人了，每次看到都让我的心跳加速。"
5. **自我鼓励与计划** (85-115): "我是不是应该鼓起勇气去跟她说话，哪怕只是一个简单的问候，也许能拉近我们的距离。"
6. **方法思考** (115-135): "我该怎么开口才不显得尴尬？"
7. **结果预期** (135-160): "如果她回应得很友好，那是不是意味着她也愿意和我多交流？"
8. **内心独白总结** (160-180): "这些问题在我脑海中不断盘旋，我真的好希望能找到答案。"

## 注意事项

1. **GPU内存**: 训练需要GPU支持，确保有足够的GPU内存
2. **训练时间**: 根据数据量，训练可能需要几分钟到几小时
3. **模型大小**: 训练好的模型会占用一定的磁盘空间（约500MB-1GB）
4. **版本兼容**: 确保transformers和torch版本兼容

## 故障排查

1. **模型加载失败**: 检查模型路径是否正确，模型文件是否完整
2. **训练失败**: 检查训练数据格式是否正确，GPU内存是否足够
3. **分割效果不佳**: 增加训练数据量，调整训练参数，或检查数据质量

## 相关文件

- `src/text_splitter_trainer.py`: 训练模块
- `src/text_splitter.py`: 文本分割器模块
- `train_text_splitter.py`: 训练脚本
- `docs/training_data_example.json`: 训练数据示例

