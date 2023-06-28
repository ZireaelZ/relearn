import json
import re
from os.path import join
from codecs import open

import numpy as np
from sklearn.utils import resample

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

def load_bert_vocab(vocabpath):
    vocab=dict()
    index=0
    with open(vocabpath,'r',encoding='utf-8')as f:
        for line in f.readlines():
            token=line.strip()
            vocab[token]=index
            index+=1
    return vocab

def Text2Wordlists(textNames):
    wordlists=[]
    for textName in textNames:
        with open(textName,encoding='utf-8') as f:
            for t in f.readlines():
                texts=t.split('。')
                for text in texts:
                    text1 = re.sub(r'[(|（].*?[)|）]', '', text)  # 去除括号内内容
                    text2=re.sub(r'&[gl]t;','',text1)
                    text2 = re.sub(r'[\r\n]', '', text2)
                    text2 = re.sub(r' ', '', text2)
                    sentencelenth=400
                    times=int(len(text2)/sentencelenth)+1
                    for i in range(times):
                        if i==times-1:
                            wordlists.append([text2[j] for j in range(sentencelenth*i,len(text2))])
                        else:wordlists.append([text2[j] for j in range(sentencelenth*i,sentencelenth*(i+1))])
    return wordlists

def expanddata(wordlists,taglists,n):#贝叶斯自举法扩充数据
    # 定义每个样本的错误概率（这里为了示例，假设每个样本的错误概率为0.1）
    error_probs = np.full(len(wordlists), 0.001)
    # 定义扩展后的数据集
    new_wordlists = []
    new_taglists = []
    for i in range(n):
        # 从样本中进行Bootstrap采样
        sampled_indices = resample(range(len(wordlists)), replace=True, n_samples=len(wordlists), random_state=i)
        sampled_wordlists = [wordlists[j] for j in sampled_indices]
        sampled_taglists = [taglists[j] for j in sampled_indices]

        # # 对每个样本进行Bernoulli抽样，生成新样本
        # new_wordlist = []
        # new_taglist = []
        # for j in range(len(wordlists)):
        #     if np.random.binomial(1, error_probs[j]) == 1:
        #         new_wordlist.append(resample(sampled_wordlists[j], replace=True, n_samples=1, random_state=i)[0])
        #         new_taglist.append(resample(sampled_taglists[j], replace=True, n_samples=1, random_state=i)[0])
        #     else:
        #         new_wordlist.append(wordlists[j])
        #         new_taglist.append(taglists[j])

        # 将新样本添加到扩展后的数据集中
        new_wordlists.append(sampled_wordlists)
        new_taglists.append(sampled_taglists)

    # 将扩展后的数据集合并为一个数据集
    merged_wordlists = [word for sublist in new_wordlists for word in sublist]
    merged_taglists = [tag for sublist in new_taglists for tag in sublist]
    return  merged_wordlists,merged_taglists

def exportWordAndText(wordlists,taglists):
    pass
#删除json中少于两个实体的数据
def deletejsonless2(jsonpaths,outname):
    datas = []
    for jp in jsonpaths:
        with open(jp, encoding='utf-8') as f:
            for line in f.readlines():
                jsons = json.loads(line)
                if len(jsons['label']) >2:
                    datas.append([jsons['text'], jsons['label']])
    with open(outname,'w',encoding='utf-8') as  f:
        for dt in datas:
            jsondata={"text":dt[0],"label":dt[1]}
            json.dump(jsondata, f, ensure_ascii=False)
            f.write('\r\n')

#将关系标注转换为json格式的数据
def Text2JSON(textpath,jsonpath):
    jsons=[]
    with open(textpath,'r',encoding='utf-8')as f:
        for line in f.readlines():
            line=line.split('	')
            ajson={}
            ajson['text']=line[3]
            #实体数据
            entities={}
            entities[0]={
                'start':line[3].index(line[0]),
                'offset':len(line[0]),
                'type':'n',
                'content':line[0]
            }
            entities[1] = {
                'start': line[3].index(line[1]),
                'offset': len(line[1]),
                'type': 'n',
                'content': line[1]
            }
            #关系数据
            relations=[]
            relations.append({
                'subject':0,
                'object':1,
                'relation':line[2]
            })
            ajson['entities']=entities
            ajson['relations'] = relations
            jsons.append(ajson)
    with open(jsonpath,'w',encoding='utf-8')as f:
        for js in jsons:
            json.dump(js, f, ensure_ascii=False)
            f.write('\r\n')




def readJson(jsonpaths,relpath):
    #初始化句子列表，关系列表，关系词典
    sentencelist=[]
    rellist=[]
    reldict={}
    #读取关系词典
    with open(relpath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            reldict[line.split(' ')[0]]=line.strip().split(' ')[1]
    for jsonpath in jsonpaths:
        with open(jsonpath, encoding='utf-8') as f:
            for line in f.readlines():
                ajson = json.loads(line)
                for label in ajson['relations']:
                    sentence=ajson['text'].strip()+ajson['entities'][str(label['subject'])]['content']+ajson['entities'][str(label['object'])]['content']
                    relation=[0 for i in range(len(reldict.keys()))]
                    relation[int(reldict[label['relation']])]=1
                    sentencelist.append(sentence)
                    rellist.append(relation)
    return sentencelist,rellist,reldict



