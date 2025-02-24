import matplotlib.pyplot as plt
import networkx as nx

# 创建一个图形对象
G = nx.DiGraph()

input_foot = ['1', '2', 'i', '1023', '1024']
hidden_foot1 = ['1', '2', 'i', '2047', '2048']
hidden_foot2 = ['1', '2', 'i', '511', '512']
hidden_foot3 = ['1', '2', 'i', '255', '256']
hidden_foot4 = ['1', '2', 'i', '63', '64']

# 输入层（3个节点）
pe_nodes = ['$PE^R_{'+f'{f}'+'}$' for f in input_foot] + ['$PE^I_{'+f'{f}'+'}$' for f in input_foot]
input_nodes = ['$X^R_{'+f'{f}'+'}$' for f in input_foot] + ['$X^I_{'+f'{f}'+'}$' for f in input_foot]
# 隐藏层（大量节点示例，使用 "..." 表示）
hidden_nodes1 = ['$H^1_{'+f'{f}'+'}$' for f in hidden_foot1]
hidden_nodes2 = ['$H^2_{'+f'{f}'+'}$' for f in hidden_foot2]
hidden_nodes3 = ['$H^3_{'+f'{f}'+'}$' for f in hidden_foot3]
hidden_nodes4 = ['$H^4_{'+f'{f}'+'}$' for f in hidden_foot4]
hidden_nodes = hidden_nodes1 + hidden_nodes2 + hidden_nodes3 + hidden_nodes4
# 输出层（2个节点）
output_nodes = ['$Y$']

# 添加节点
for node in pe_nodes + input_nodes + hidden_nodes + output_nodes:
    G.add_node(node)
print(G)

# 创建输入层与隐藏层之间的连接
for i in input_nodes:
    for h in hidden_nodes1:
        G.add_edge(i, h)

# 创建隐藏层与输出层之间的连接
for h in hidden_nodes1:
    for _ in hidden_nodes2:
        G.add_edge(h, _)
for h in hidden_nodes2:
    for _ in hidden_nodes3:
        G.add_edge(h, _)
for h in hidden_nodes3:
    for _ in hidden_nodes4:
        G.add_edge(h, _)
for h in hidden_nodes4:
    G.add_edge(h, output_nodes[0])

# 设置图形位置
pos = {}
pos_omit = []
def generate_position(nodes, x, pos, shift=-1):
    for i, node in enumerate(nodes):
        if 'i' in node:
            pos_omit.append((x, i-len(nodes)/2+shift))
            shift += 1
        pos[node] = (x, i-len(nodes)/2+shift)
        if 'i' in node:
            shift += 1
            pos_omit.append((x, i-len(nodes)/2+shift))

generate_position(pe_nodes, -0.2, pos, shift=-2)
generate_position(input_nodes, 0, pos, shift=-2)
generate_position(hidden_nodes1, 0.5, pos)
generate_position(hidden_nodes2, 1, pos)
generate_position(hidden_nodes3, 1.5, pos)
generate_position(hidden_nodes4, 2, pos)
generate_position(output_nodes, 2.5, pos, shift=0)

# 绘制图形
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_size=1250, node_color='skyblue', font_size=11, font_weight='bold', arrows=True)
for x, y in pos_omit:
    plt.text(x, y, '...', fontsize=11, fontweight='bold', ha='center', va='center')
for i, node in enumerate(pe_nodes):
    plt.text(pos[node][0]+0.1, pos[node][1], '+', fontsize=11, fontweight='bold', ha='center', va='center')
plt.title("Neural Network Architecture")
plt.show()
