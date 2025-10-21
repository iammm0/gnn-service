from fastapi import FastAPI
from pydantic import BaseModel
from ner_re import ner, extract_relations
from graph import build_graph, graph_info

app = FastAPI()


# 请求数据模型
class TextInput(BaseModel):
    text: str


@app.post("/process_text/")
def process_text(data: TextInput):
    # 提取实体数据
    entities = ner(data.text)

    # 提取关系
    relations = extract_relations(entities)

    # 构建图
    G = build_graph(entities, relations)

    # 返回图的基本信息
    return graph_info(G)