"""
GNN Service - 基于BERT的知识图谱构建服务

提供中文文本的命名实体识别和关系抽取功能，并构建知识图谱。
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.ner_re import ner, extract_relations
from src.graph import build_graph, graph_info
from src.model_manager import ModelManager, ModelStrategy
from src.config import MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局模型管理器
_model_manager = None

# 启动时检查CUDA状态
try:
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA可用 - GPU: {torch.cuda.get_device_name(0)}, CUDA版本: {torch.version.cuda}")
    else:
        logger.warning("CUDA不可用，将使用CPU运行")
except Exception as e:
    logger.warning(f"检查CUDA时出错: {e}")

# 启动时加载所有启用的模型
def initialize_models():
    """在启动时加载所有启用的模型"""
    global _model_manager
    try:
        logger.info("=" * 60)
        logger.info("开始初始化模型管理器...")
        logger.info("=" * 60)
        
        _model_manager = ModelManager(MODEL_CONFIG)
        
        # 只加载启用的模型
        enabled_models = [m for m in MODEL_CONFIG["models"] if m.get("enabled", True)]
        _model_manager.model_configs = enabled_models
        
        logger.info(f"配置了 {len(enabled_models)} 个启用的模型:")
        for model in enabled_models:
            logger.info(f"  - {model.get('name')}: {model.get('path')} ({model.get('description', '')})")
        
        # 加载所有启用的模型
        loaded_count = _model_manager.load_all_models()
        
        logger.info("=" * 60)
        logger.info(f"模型初始化完成，成功加载 {loaded_count} 个模型")
        logger.info("=" * 60)
        
        return _model_manager
        
    except Exception as e:
        logger.error(f"初始化模型失败: {str(e)}", exc_info=True)
        raise

# 在应用启动时初始化模型
initialize_models()

# 初始化文本分割器（如果模型存在）
_text_splitter = None
try:
    from src.text_splitter import get_text_splitter
    _text_splitter = get_text_splitter()
    if _text_splitter.use_model:
        logger.info("文本分割模型已加载")
    else:
        logger.info("使用规则文本分割（未找到训练好的模型）")
except Exception as e:
    logger.warning(f"初始化文本分割器失败: {e}")

app = FastAPI(
    title="GNN Service",
    description="基于BERT的中文命名实体识别和知识图谱构建服务（支持智能文本分割）",
    version="2.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求数据模型
class TextInput(BaseModel):
    """文本输入模型"""
    text: str = Field(..., min_length=1, max_length=100000, description="待处理的文本内容（支持长文本）")
    model_name: Optional[str] = Field(None, description="指定使用的模型名称（可选，默认使用配置的默认模型）")
    strategy: Optional[str] = Field(None, description="模型使用策略：single（单模型）、vote（投票）、union（并集）、intersection（交集）、weighted（加权）")


class GraphResponse(BaseModel):
    """图响应模型"""
    nodes: List[Dict[str, Any]]
    edges: List[Any]


class TrainTextSplitterRequest(BaseModel):
    """训练文本分割模型请求"""
    train_data_path: str = Field(..., description="训练数据文件路径（JSON格式）")
    val_data_path: Optional[str] = Field(None, description="验证数据文件路径（可选）")
    epochs: int = Field(3, description="训练轮数")
    batch_size: int = Field(8, description="批次大小")
    learning_rate: float = Field(2e-5, description="学习率")


@app.get("/")
def root():
    """根路径，返回服务信息"""
    try:
        if _model_manager is None:
            loaded_models = []
        else:
            loaded_models = list(_model_manager.models.keys())
    except:
        loaded_models = []
    
    # 检查文本分割模型状态
    text_splitter_status = "not_loaded"
    if _text_splitter:
        text_splitter_status = "loaded" if _text_splitter.use_model else "rule_based"
    
    return {
        "service": "GNN Service",
        "version": "2.1.0",
        "description": "基于多模型的中文命名实体识别和知识图谱构建服务（支持智能文本分割和长文本）",
        "endpoints": {
            "/process_text/": "POST - 处理文本并构建知识图谱",
            "/health": "GET - 健康检查",
            "/models": "GET - 获取NER模型列表",
            "/models/{model_name}": "GET - 获取指定NER模型信息",
            "/text_splitter": "GET - 获取文本分割模型信息",
            "/train_text_splitter/": "POST - 训练文本分割模型"
        },
        "loaded_models": loaded_models,
        "text_splitter_status": text_splitter_status,
        "available_strategies": ["single", "vote", "union", "intersection", "weighted"]
    }


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "gnn-service"}


@app.get("/models")
def list_models():
    """获取所有可用模型列表"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        models_info = []
        for name, info in _model_manager.models.items():
            models_info.append({
                "name": name,
                "type": info.get("type", "unknown"),
                "path": info.get("path", ""),
                "device": info.get("device", "cpu")
            })
        
        # 安全地获取策略值
        strategy_value = _model_manager.strategy
        if isinstance(strategy_value, str):
            default_strategy = strategy_value
        else:
            # 如果是枚举对象，获取其value
            default_strategy = strategy_value.value if hasattr(strategy_value, 'value') else str(strategy_value)
        
        return {
            "models": models_info,
            "default_model": _model_manager.default_model,
            "default_strategy": default_strategy,
            "total_count": len(models_info)
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """获取指定NER模型的详细信息"""
    try:
        if _model_manager is None:
            raise HTTPException(status_code=500, detail="模型管理器未初始化")
        
        if model_name not in _model_manager.models:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        model_info = _model_manager.models[model_name]
        return {
            "name": model_name,
            "type": model_info.get("type", "unknown"),
            "path": model_info.get("path", ""),
            "original_path": model_info.get("original_path", ""),
            "device": model_info.get("device", "cpu"),
            "status": "loaded"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@app.get("/text_splitter")
def get_text_splitter_info():
    """获取文本分割模型信息"""
    try:
        if _text_splitter is None:
            return {
                "status": "not_initialized",
                "message": "文本分割器未初始化",
                "model_path": None,
                "use_model": False
            }
        
        return {
            "status": "loaded" if _text_splitter.use_model else "rule_based",
            "message": "使用训练好的模型" if _text_splitter.use_model else "使用规则分割",
            "model_path": _text_splitter.model_path,
            "use_model": _text_splitter.use_model,
            "description": "基于BERT的语义文本分割模型，能够根据心理活动变化进行文本分割"
        }
    except Exception as e:
        logger.error(f"获取文本分割模型信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文本分割模型信息失败: {str(e)}")


@app.post("/train_text_splitter/")
def train_text_splitter(data: TrainTextSplitterRequest):
    """
    训练文本分割模型
    
    根据提供的训练数据，训练一个能够根据语义/心理活动变化进行文本分割的模型。
    
    Args:
        train_data_path: 训练数据文件路径（JSON格式）
        val_data_path: 验证数据文件路径（可选）
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        
    Returns:
        训练结果信息
    """
    try:
        from src.text_splitter_trainer import TextSplitterTrainer
        
        logger.info(f"开始训练文本分割模型，训练数据: {data.train_data_path}")
        
        trainer = TextSplitterTrainer()
        trainer.train(
            train_data_path=data.train_data_path,
            val_data_path=data.val_data_path,
            epochs=data.epochs,
            batch_size=data.batch_size,
            learning_rate=data.learning_rate
        )
        
        return {
            "status": "success",
            "message": "模型训练完成",
            "model_path": trainer.output_dir
        }
        
    except Exception as e:
        logger.error(f"训练文本分割模型失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")


@app.post("/process_text/", response_model=GraphResponse)
def process_text(data: TextInput) -> Dict[str, Any]:
    """
    处理文本并构建知识图谱（支持多模型和长文本）
    
    Args:
        data: 包含待处理文本的输入数据
        - text: 待处理文本（支持长文本）
        - model_name: 指定使用的模型（可选）
        - strategy: 模型使用策略（可选）
        
    Returns:
        包含节点和边的知识图谱数据
        
    Raises:
        HTTPException: 当处理过程中出现错误时
    """
    try:
        logger.info(f"收到文本处理请求，文本长度: {len(data.text)}, 模型: {data.model_name}, 策略: {data.strategy}")
        
        # 提取实体数据（支持多模型和策略）
        logger.debug("开始执行命名实体识别...")
        entities = ner(data.text, model_name=data.model_name, strategy=data.strategy)
        logger.info(f"识别到 {len(entities)} 个实体")
        
        # 提取关系（传入原始文本以提高准确率）
        logger.debug("开始提取关系...")
        relations = extract_relations(entities, text=data.text)
        logger.info(f"提取到 {len(relations)} 个关系")
        
        # 构建图
        logger.debug("开始构建知识图谱...")
        G = build_graph(entities, relations)
        logger.info(f"图谱构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        
        # 返回图的基本信息
        result = graph_info(G)
        logger.info("文本处理完成")
        return result
        
    except Exception as e:
        logger.error(f"处理文本时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理文本时发生错误: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)