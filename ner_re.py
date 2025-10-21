from transformers import BertTokenizer, BertForTokenClassification, pipeline

# 加载BERT模型进行命名实体识别
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForTokenClassification.from_pretrained('bert-base-chinese')
nlp_bert = pipeline("ner", model=model, tokenizer=tokenizer)


# 命名实体识别
def ner(text):
    bert_entities = nlp_bert(text)

    # 假设我们根据一些条件来设置 highlight 和 magnified
    entities = []
    for entity in bert_entities:
        # 默认的节点属性
        node_data = {
            "text": entity['word'],  # 实体文本
            "label": entity['entity'],  # 实体类型
            "highlight": False,  # 初始不高亮
            "magnified": False,  # 初始不放大
        }

        # 你可以在这里根据实体类型或其他规则动态修改 highlight 和 magnified
        if node_data["label"] == "PERSON":  # 示例：人物实体高亮
            node_data["highlight"] = True
        if "北京" in node_data["text"]:  # 示例：包含“北京”的实体放大
            node_data["magnified"] = True

        entities.append((node_data["text"], node_data["label"], node_data["highlight"], node_data["magnified"]))

    return entities


# 简单的关系抽取（示例：基于规则抽取简单的"关系"）
def extract_relations(entities):
    relations = []

    # 示例：创建一个基于实体顺序的简单关系
    for i in range(len(entities) - 1):
        relation = (entities[i][0], "与", entities[i + 1][0])  # 假设实体之间的关系是“与”
        relations.append(relation)

    # 可以根据实体类型或顺序添加更多的自定义规则
    return relations