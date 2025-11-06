# 项目结构说明

## 目录结构

```
gnn-service/
├── main.py                    # 应用入口文件（根目录）
├── requirements.txt           # Python依赖包列表
├── Dockerfile                 # Docker构建文件
│
├── src/                       # 源代码目录
│   ├── __init__.py           # 包初始化文件
│   ├── config.py             # 配置文件（模型列表和系统参数）
│   ├── model_manager.py      # 模型管理器（多模型加载和策略管理）
│   ├── ner_re.py             # 命名实体识别和关系抽取模块
│   └── graph.py              # 知识图谱构建模块
│
├── models/                    # 模型文件目录
│   └── bert-base-chinese/    # BERT中文模型文件
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       └── vocab.txt
│
└── docs/                      # 文档目录
    ├── README.md             # 项目主要文档
    ├── CUDA_STATUS.md        # CUDA状态诊断文档
    ├── GPU_SETUP_COMPLETE.md # GPU配置完成文档
    └── models_config_example.json  # 模型配置示例
```

## 文件说明

### 根目录文件
- **main.py**: FastAPI应用主入口，包含所有API路由定义
- **requirements.txt**: 项目依赖包列表
- **Dockerfile**: Docker镜像构建配置文件

### src/ 源代码目录
- **config.py**: 系统配置，包括模型列表、长文本处理配置、系统参数
- **model_manager.py**: 模型管理器，负责加载、管理和切换多个NER模型
- **ner_re.py**: 命名实体识别和关系抽取功能实现（支持多模型和长文本）
- **graph.py**: 知识图谱构建和图操作功能

### models/ 模型目录
- 存放所有预训练的模型文件
- 当前包含 `bert-base-chinese` 模型
- 其他模型（如RoBERTa、MacBERT）首次使用时从HuggingFace自动下载

### docs/ 文档目录
- **README.md**: 项目主要文档，包含安装、使用、API文档等
- **CUDA_STATUS.md**: CUDA状态诊断和问题排查文档
- **GPU_SETUP_COMPLETE.md**: GPU配置完成总结文档
- **models_config_example.json**: 模型配置示例文件

## 导入路径说明

### 从main.py导入
```python
from src.ner_re import ner, extract_relations
from src.graph import build_graph, graph_info
from src.model_manager import ModelManager, ModelStrategy
from src.config import MODEL_CONFIG
```

### src目录内的相对导入
```python
# 在ner_re.py中
from .model_manager import ModelManager, ModelStrategy
from .config import MODEL_CONFIG, LONG_TEXT_CONFIG
```

## 模型路径配置

在 `src/config.py` 中配置模型路径：

```python
"models": [
    {
        "name": "bert-base-chinese",
        "path": "models/bert-base-chinese",  # 本地模型路径
        "enabled": True,
    },
    {
        "name": "roberta-base-chinese",
        "path": "hfl/chinese-roberta-wwm-ext",  # HuggingFace模型
        "enabled": True,
    },
]
```

## 运行方式

### 本地运行
```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 运行服务
python main.py
# 或
uvicorn main:app --reload
```

### Docker运行
```bash
docker build -t gnn-service .
docker run -p 8000:8000 gnn-service
```

## 注意事项

1. **模型路径**: 本地模型路径相对于项目根目录
2. **导入路径**: 使用 `src.xxx` 格式从根目录导入
3. **相对导入**: 在src目录内使用 `.xxx` 格式的相对导入
4. **文档位置**: 所有文档文件都在 `docs/` 目录下

