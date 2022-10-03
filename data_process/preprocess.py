# #encoding= 'utf-8'
# #pheme-rnr-dataset
# import json
# import pprint
# import os
# import glob
# import pickle
# from datetime import datetime
# # from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# import re
# import sys
# sys.setrecursionlimit(10000)
#
# def add(structure,content_dict, replies, s):
#     for t in replies[s]:
#         if t in structure.keys():
#             structure = add(structure, replies, t)
#             structure[s]['cascade'].extend(structure[t]['cascade'])
#             content_dict[s].extend(content_dict[t])
#             del structure[t]
#             del content_dict[t]
#     return structure, content_dict
#
#
# if __name__  == '__main__':
#
#     url = r'D:\数据集\2020-5-18\pheme-rnr-dataset'
#     datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']  #
#     types = ['rumours', 'non-rumours']
#
#     for dataset in datasets:
#         for type in types:
#             i = 1
#             if type == 'rumours':
#                 label = 1
#             else:
#                 label = 0
#
#             global structure
#             structure = {}
#             content_dict = {}
#             global replies
#             replies = {}
#             for tweets in glob.glob(os.path.join(url, dataset, type, '*')):
#
#                 source = tweets + "\\source-tweet\\*.json"
#                 sourceData = json.load(open(glob.glob(source)[0]))
#                 sourceWId = sourceData['id_str']
#                 sourceUId = sourceData['user']['id_str']
#                 structure[sourceWId] = {}
#                 content_dict[sourceWId] = []
#
#                 # content
#                 txt = sourceData['text']
#                 # Remove '@name'
#                 txt = re.sub(r'(@.*?)[\s]', ' ', txt)
#                 # Replace '&amp;' with '&'
#                 txt = re.sub(r'&amp;', '&', txt)
#                 txt = re.sub(r'\n+', ' ', txt)
#                 # Remove trailing whitespace 删除空格
#                 txt = re.sub(r'\s+', ' ', txt).strip()
#                 # remove http
#                 p = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.S)
#                 txt = re.sub(p, '', txt)
#                 sourcetxt = txt
#                 content_dict[sourceWId].append(str(sourceWId) + '\t' + str(sourceUId) + '\t' + sourcetxt)
#
#                 # publication time
#                 time_pub = sourceData['created_at']
#                 t = time_pub.split(' ')
#                 t = t[1] + " " + t[2] + " " + t[5] + " " + t[3]
#                 time_pub = datetime.strptime(t, '%b %d %Y %H:%M:%S')
#                 time_pub = int(datetime.timestamp(time_pub))
#                 structure[sourceWId]['time'] = time_pub
#                 structure[sourceWId]['userId'] = sourceUId
#                 structure[sourceWId]['cascade'] = []
#                 structure[sourceWId]['label'] = label
#
#                 reactions = tweets + "\\reactions\\*.json"
#                 for reaction in glob.glob(reactions):
#                     reactionData = json.load(open(reaction))
#                     reactionWId = reactionData["id_str"]
#                     if reactionWId == sourceWId:
#                         continue
#                     replyTo = reactionData["in_reply_to_user_id_str"]
#                     reactionUId = reactionData['user']['id_str']
#                     # content process
#                     txt = reactionData['text']
#                     txt = re.sub(r'(@.*?)[\s]+', ' ', txt)
#                     # Replace '&amp;' with '&'
#                     txt = re.sub(r'&amp;', '&', txt)
#                     txt = re.sub(r'\
#                     n+', ' ', txt)
#                     # Remove trailing whitespace 删除空格
#                     txt = re.sub(r'\s+', ' ', txt).strip()
#                     txt = re.sub(p, '', txt)
#                     reactiontxt = txt
#                     # public time
#                     time = reactionData['created_at']  # Sun Nov 06 21:21:26 +0000 2011
#                     t = time.split(' ')
#                     t = t[1] + " " + t[2] + " " + t[5] + " " + t[3]
#                     time = datetime.strptime(t, '%b %d %Y %H:%M:%S')
#                     time = int(datetime.timestamp(time))
#                     if sourceWId not in replies.keys():
#                         replies[sourceWId] = []
#                     replies[sourceWId].append(reactionWId)
#                     if replyTo == None:
#                         replyTo = sourceUId
#                     structure[sourceWId]['cascade'].append([replyTo, reactionUId, time])
#                     content_dict[sourceWId].append(str(reactionWId) + '\t' + str(reactionUId) + '\t' + reactiontxt)
#
#             for s in replies.keys():
#                 for t in replies[s]:
#                     if t in structure.keys():
#                         structure, content_dict = add(structure, content_dict, replies, t)
#
#             n = 1
#             result_dir = os.path.join(url, 'pro_cascades', dataset + '_' + type + "_cascade.txt")
#
#             with open(result_dir, 'a', encoding='utf-8') as fp:
#                 for key, value in structure.items():
#                     c = []
#
#                     if len(structure[key]['cascade']) == 0:
#                         continue
#                     for i in range(len(structure[key]['cascade'])):
#                         c.append(structure[key]['cascade'][i][0] + '/' + structure[key]['cascade'][i][1] + ':' + str(
#                             structure[key]['cascade'][i][2] - structure[key]['time']))
#
#                     cascade = str(n) + '\t' + str(key) + '\t' + str(structure[key]['userId']) + '\t' + str(
#                         structure[key]['time']) + '\t' + str(
#                         len(c) + 1) + '\t' + str(structure[key]['userId']) + ':' + str(0) + ' ' + ' '.join(
#                         c) + '\t' + structure[key]['label'] + '\n'  # str(node_time.get(node)
#                     fp.write(cascade)
#                     content_dir = os.path.join(url, 'pro_content', dataset, type, str(key) + ".txt")
#                     con_f = open(content_dir, 'a', encoding='utf-8')
#                     con_f.write('\n'.join(content_dict[key]))
#                     n += 1
#
#         #pprint.pprint(structure)
#
#
#
##encoding= 'utf-8'
#weibo
import json
import pprint
import os
import glob
import pickle
import random
from datetime import datetime
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import re
import sys
sys.setrecursionlimit(10000)
import json

def add(structure,content_dict, replies, s):
    for t in replies[s]:
        if t in structure.keys():
            structure = add(structure, replies, t)
            structure[s]['cascade'].extend(structure[t]['cascade'])
            content_dict[s].extend(content_dict[t])
            del structure[t]
            del content_dict[t]
    return structure, content_dict

if __name__  == '__main__':

    url = r'D:\数据集\2020-5-18\AllRumorDetection-master-data'
    cas_data = open(os.path.join(url,'Weibo.txt'))
    structure = {}
    content_dict = {}

    for line in cas_data.readlines():
        line = line.strip('\n').split('\t')
        wid = line[0].split(':')[1]
        label = line[1].split(':')[1]

        cont_data = open(os.path.join(url,'Weibo', wid +'.json'),'r',encoding='utf-8')
        users = json.load(cont_data)

        content_dict_one ={}

        for user in users:

            if user == users[0] :
                sourceWId = user['id']
                sourceUId = user['mid']
                structure[sourceWId] = {}
                content_dict[sourceWId] = []
                # content
                txt = user['text']
                # Remove '@name'
                txt = re.sub(r'(@.*?)[\s]', ' ', txt)
                # Replace '&amp;' with '&'
                txt = re.sub(r'&amp;', '&', txt)
                txt = re.sub(r'\n+', ' ', txt)
                # Remove trailing whitespace 删除空格
                txt = re.sub(r'\s+', ' ', txt).strip()
                # remove http
                p = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.S)
                txt = re.sub(p, '', txt)
                sourcetxt = txt
                content_dict[sourceWId].append(str(sourceWId) + '\t' + str(sourceUId) + '\t' + sourcetxt)
                content_dict_one[sourceUId]= txt

                # publication time
                time_pub = int(user['t'])

                structure[sourceWId]['time'] = time_pub
                structure[sourceWId]['userId'] = sourceUId
                structure[sourceWId]['cascade'] = []
                structure[sourceWId]['label'] = label
            else:
                reactionWId = user["id"]
                if reactionWId == sourceWId:
                    continue
                replyTo = user["parent"]
                reactionUId = user['mid']
                # content process
                txt = user['text']
                if txt == '"转发微博"' or txt == "轉發微博。" or txt== "转发微博。" or txt =='转' or txt=='转！':
                    if replyTo not in content_dict_one.keys():
                        txt = content_dict_one[sourceUId]
                    else:
                        txt = content_dict_one[replyTo]

                content_dict_one[reactionUId]=txt
                txt = re.sub(r'(@.*?)[\s]+', ' ', txt)
                # Replace '&amp;' with '&'
                txt = re.sub(r'&amp;', '&', txt)
                txt = re.sub(r'\n+', ' ', txt)
                # Remove trailing whitespace 删除空格
                txt = re.sub(r'\s+', ' ', txt).strip()
                txt = re.sub(p, '', txt)
                reactiontxt = txt

                # public time
                time = user['t']  # Sun Nov 06 21:21:26 +0000 2011

                if replyTo == None or replyTo == 'null':
                    replyTo = sourceUId
                structure[sourceWId]['cascade'].append([replyTo, reactionUId, time])
                content_dict[sourceWId].append(str(reactionWId) + '\t' + str(reactionUId) + '\t' + reactiontxt)

    result_dir = os.path.join(url, 'pro_cascades', "Weibo_cascade.txt")

    n=0

    with open(result_dir, 'a', encoding='utf-8') as fp:
        for key, value in structure.items():
            c = []
            if len(structure[key]['cascade']) == 0:
                continue
            for i in range(len(structure[key]['cascade'])):
                c.append(structure[key]['cascade'][i][0] + '/' + structure[key]['cascade'][i][1] + ':' + str(
                    structure[key]['cascade'][i][2] - structure[key]['time']))

            cascade = str(n) + '\t' + str(key) + '\t' + str(structure[key]['userId']) + '\t' + str(
                structure[key]['time']) + '\t' + str(
                len(c) + 1) + '\t' + str(structure[key]['userId']) + ':' + str(0) + ' ' + ' '.join(
                c) + '\t' + structure[key]['label'] + '\n'  # str(node_time.get(node)
            fp.write(cascade)
            # content_dir = os.path.join(url, 'pro_content', str(key) + ".txt")
            # con_f = open(content_dir, 'a', encoding='utf-8')
            # con_f.write('\n'.join(content_dict[key]))
            n += 1


