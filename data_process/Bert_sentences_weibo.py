import networkx as nx
import numpy as np
import torch
import math
import scipy
import pickle
import os
import random
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

# MODELNAME ='bert-base-chinese'
#
# tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
# model = BertModel.from_pretrained(MODELNAME)  # 模型
# model.eval()
#
# url =r'D:\数据集\2020-5-18\AllRumorDetection-master-data'
#
# if __name__ == '__main__':
#     max_len = 0
#     for _, _, filenames in os.walk(os.path.join(url, 'pro_content')):
#         for file in filenames:
#             f = open(os.path.join(url, 'pro_content', file), 'r', encoding='utf-8')
#             f_w = open(os.path.join(url, 'pro_content_emb', str(file)), 'a')
#             for line in f.readlines():
#                 line = line.strip('\n').split('\t')
#
#                 with torch.no_grad():
#                     input_ids = tokenizer.encode(
#                         line[2],
#                         add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
#                         # max_length=114,  # 设定最大文本长度
#                         # padding = 'max_length',   # pad到最大的长度
#                         return_tensors='pt'  # 返回的类型为pytorch tensor
#                     )
#                     encoded_layers, _ = model(input_ids)
#                 sentence_vec = torch.mean(encoded_layers[11], 1).squeeze()
#                 sentence_vec = sentence_vec.numpy().tolist()
#
#                 input_ids = [str(x) for x in sentence_vec]
#                 if len(input_ids) > max_len:
#                     max_len = len(input_ids)
#                 f_w.write(line[0] + '\t' + line[1] + '\t' + ' '.join(input_ids) + '\n')
#     print( '%d', max_len)
#     max_len = 0


# #一个数据一个文件,下采样，获取样本数量
# TIME_WINDOWS =6
# NUM_NODES =50
# MAX_TXT_LEN =768
# url = r'D:\数据集\2020-5-18\pheme-rnr-dataset'
# datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']  #
# types = ['rumours','non-rumours']

# for dataset in datasets:
#     for type in types:
#         data_count = 0
#
#         f = open(os.path.join(url, 'pro_cascades', dataset + '_' + type + '_cascade.txt'), 'r', encoding='utf-8')
#         for line in f.readlines():
#             g = nx.DiGraph()
#             node_time = {}
#             cascade = {}
#             line = line.strip('\n').split('\t')
#             wid = line[1]
#             paths = line[5].split(' ')
#             if len(paths) <10:
#                 continue
#             data_count +=1
#         print('%s %s %d' %(dataset,type,data_count))
#
# #

#一个数据一个文件
TIME_WINDOWS =6
NUM_NODES =50
MAX_TXT_LEN =768
url = r'D:\数据集\2020-5-18\AllRumorDetection-master-data'
dataset ='Weibo'

newwid = [i for i in range(4664)]
len_data =0
random.shuffle(newwid )

batch_x_len={}

# 保留长度靠前的数据
f = open(os.path.join(url, 'pro_cascades', "Weibo_cascade.txt"), 'r', encoding='utf-8')
for line in f.readlines():
    g = nx.DiGraph()
    node_time = {}
    cascade = {}
    line = line.strip('\n').split('\t')
    wid = line[1]
    paths = line[5].split(' ')
    label = int(line[6])

    if len(paths)<10:
        continue


    # 记录该信息的最后转发时间，并将所有时间划分为6个时间间隔
    max_time = 0
    # 节点的对应新id
    node_new_id = {}
    n = 0
    cascade[wid] = {}

    for path in paths:
        nodes = path.split(':')[0].split('/')
        t = int(path.split(":")[1])

        # 控制节点数量
        if n >= NUM_NODES:
            for i in range(len(nodes) - 1):
                if nodes[i + 1] not in node_new_id.keys():
                    continue
                else:
                    if nodes[i] in node_new_id.keys():
                        g.add_edge(nodes[i], nodes[i + 1])
                    node_time[nodes[i + 1]].append(t)
                    if t > max_time:
                        max_time = t
        else:
            if len(nodes) == 1:
                if nodes[0] not in node_time.keys():
                    node_time[nodes[0]] = [0]
                    node_new_id[nodes[0]] = n
                    n += 1
            else:
                for i in range(len(nodes) - 1):
                    g.add_edge(nodes[i], nodes[i + 1])
                    if nodes[i] not in node_time.keys():
                        node_time[nodes[i]] = [1]
                        node_new_id[nodes[i]] = n
                        n += 1
                    if nodes[i + 1] not in node_time.keys():
                        node_time[nodes[i + 1]] = []
                        node_new_id[nodes[i + 1]] = n
                        n += 1
                    node_time[nodes[i + 1]].append(t)

                    if t > max_time:
                        max_time = t

    bert_f = open(os.path.join(url, 'pro_content_emb', wid + '.txt'), 'r', encoding='utf-8')
    user_content = np.zeros((TIME_WINDOWS, NUM_NODES, MAX_TXT_LEN), dtype=float)

    # user_time_content darry
    node_app = {}
    for line in bert_f.readlines():
        line = line.strip('\n').split('\t')
        node = line[1]
        txt = line[2].split(' ')
        if node not in node_new_id.keys():
            continue
        if node not in node_app.keys():
            node_app[node] = 0
        node_app[node] += 1

        if txt == ['']:
            t = math.floor(node_time[node][node_app[node] - 1] / (max_time / TIME_WINDOWS))
            user_content[t:, node_new_id[node], :] = user_content[t:, 0, :]
        else:
            txt = list(map(float, txt))
            txt_len = len(txt)
            id = node_new_id[node]
            t = math.floor(node_time[node][node_app[node] - 1] / (max_time / TIME_WINDOWS))
            user_content[t:, node_new_id[node], :txt_len] = np.array(txt)

    # user without content = source content
    for node in list(nx.nodes(g)):
        if node not in node_app.keys():
            t = math.floor(node_time[node][0] / (max_time / TIME_WINDOWS))

            user_content[t:, node_new_id[node], :] = user_content[t:, 0, :]

    g_new = nx.DiGraph()
    for (s, t) in list(nx.edges(g)):
        g_new.add_edge(node_new_id[s], node_new_id[t])
    g_new.remove_edges_from(list(g_new.selfloop_edges()))
    g_adj = nx.adj_matrix(g_new).todense()
    N = nx.number_of_nodes(g_new)
    if N < NUM_NODES:
        col_padding_L = np.zeros(shape=(N, NUM_NODES - N))
        L_col_padding = np.column_stack((g_adj, col_padding_L))
        row_padding = np.zeros(shape=(NUM_NODES - N, NUM_NODES))
        L_col_row_padding = np.row_stack((L_col_padding, row_padding))
        A = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.int)
    else:
        A = scipy.sparse.coo_matrix(g_adj, dtype=np.int)

    file = open(
        os.path.join(r'E:\WDK_workshop\USAT-RD\data\weibo_all_emb', str(newwid[len_data]) + '_' + str(label) + '_' + dataset + '_N50T6.pkl'),
        'wb')
    pickle.dump((user_content.tolist(), A, label), file)
    len_data += 1
