import matplotlib.pyplot as plt
import networkx as nx

def plot_nn():
    # 创建一个图形对象
    G = nx.DiGraph()

    # 输入层（3个节点）
    input_nodes = ['x1', 'x2', 'x3']
    # 隐藏层（大量节点示例，使用 "..." 表示）
    hidden_nodes = ['h1', 'h2', '...', 'h3', 'h4']
    hidden_nodes_true = hidden_nodes.copy()
    hidden_nodes_true.remove('...')
    # 输出层（2个节点）
    output_nodes = ['y1', 'y2']

    # 添加节点
    for node in input_nodes + hidden_nodes + output_nodes:
        G.add_node(node)

    # 创建输入层与隐藏层之间的连接
    for i in input_nodes:
        for h in hidden_nodes_true:  # 排除 "..." 节点
            G.add_edge(i, h)
    
    # 创建隐藏层与输出层之间的连接
    for h in hidden_nodes_true:  # 排除 "..." 节点
        for o in output_nodes:
            G.add_edge(h, o)

    # 设置图形位置
    pos = {
        # 输入层的位置
        'x1': (0, 0), 'x2': (0, 1), 'x3': (0, 2),
        # 隐藏层的位置
        'h1': (1, 0), 'h2': (1, 1), 'h3': (1, 3), 'h4': (1, 4), '...': (1, 2),
        # 输出层的位置
        'y1': (2, 0), 'y2': (2, 1)
    }

    # 绘制图形
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)
    plt.title("Neural Network Architecture")
    plt.show()

# 调用函数绘制神经网络示意图
plot_nn()
