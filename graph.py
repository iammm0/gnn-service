import networkx as nx


# 构建节点的数据格式
def build_graph(entities, relations):
    G = nx.Graph()

    # 遍历所有实体，构建节点数据
    for entity in entities:
        # 每个节点的数据可以是一个字典，包括 title, description, tags, highlight, magnified 等属性
        node_data = {
            "title": entity[0],  # 实体文本作为标题
            "description": f"描述：{entity[0]}",  # 简单描述为实体文本
            "tags": [entity[1]],  # 假设实体类型作为标签
            "highlight": entity[2],  # 是否高亮
            "magnified": entity[3],  # 是否放大
        }

        G.add_node(entity[0], **node_data)

    # 添加关系（边）
    for rel in relations:
        G.add_edge(rel[0], rel[2], relationship=rel[1])  # 关系依旧是基于实体文本

    return G


# 输出图的基本信息
def graph_info(G):
    # 将图节点信息转换成前端所需的格式
    node_info = [
        {
            "id": node,
            "type": "thought",
            "data": {
                "title": G.nodes[node]["title"],
                "description": G.nodes[node]["description"],
                "tags": G.nodes[node]["tags"],
                "highlight": G.nodes[node]["highlight"],
                "magnified": G.nodes[node]["magnified"],
            },
            "position": {"x": 0, "y": 0},  # 位置可以根据实际需求动态计算
        }
        for node in G.nodes
    ]

    # 关系数据（边）的格式可以根据需要进一步处理
    return {"nodes": node_info, "edges": list(G.edges(data=True))}
