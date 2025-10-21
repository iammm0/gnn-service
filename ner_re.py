from transformers import BertTokenizer, BertForTokenClassification, pipeline
from modelscope import pipeline as modelscope_pipeline

# 加载BERT模型进行命名实体识别
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForTokenClassification.from_pretrained('bert-base-chinese')
nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)

# 加载阿里社区的文档分割模型
document_segmenter = modelscope_pipeline('document-segmentation',
                                         model='iic/nlp_bert_document-segmentation_chinese-base')


# 命名实体识别
def ner(text):
    # 使用阿里模型进行文档分割
    segments = document_segmenter(text)
    print(f"分段结果: {segments}")

    # 使用BERT进行命名实体识别
    bert_entities = nlp_bert(text)

    entities = []
    for entity in bert_entities:
        # 默认的节点属性
        node_data = {
            "text": entity['word'],  # 实体文本
            "label": entity['entity'],  # 实体类型
            "highlight": False,  # 初始不高亮
            "magnified": False,  # 初始不放大
        }

        # 根据实体类型和文本内容设置高亮和放大
        if node_data["label"] == "PERSON":  # 示例：人物实体高亮
            node_data["highlight"] = True
        if "北京" in node_data["text"]:  # 示例：包含“北京”的实体放大
            node_data["magnified"] = True

        entities.append((node_data["text"], node_data["label"], node_data["highlight"], node_data["magnified"]))

    return {"segments": segments, "entities": entities}


# 简单的关系抽取（基于实体顺序的简单关系）
def extract_relations(entities):
    relations = []

    # 根据实体顺序添加简单的“关系” (例如：实体1 和 实体2 有某种关系)
    for i in range(len(entities) - 1):
        relation = (entities[i][0], "与", entities[i + 1][0])  # 假设实体之间的关系是“与”
        relations.append(relation)

    return relations