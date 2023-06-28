from transformers import AutoModel
import torch.nn as nn
import torch
from config import Trainconfig
#经典分类模型
bert_path='./bertModels/bert_pretrained'
class BertClassifier(nn.Module):
    def __init__(self,relation_num):
        super().__init__()
        self.bertEncoder=AutoModel.from_pretrained(bert_path)
        self.dropout=nn.Dropout(Trainconfig.dropout,inplace=False)
        self.linear=nn.Linear(768,relation_num)
        # self.lossFun=nn.CrossEntropyLoss
    def forward(self,sent_in,attention_Mask,token):
        bertout=self.bertEncoder(sent_in,attention_mask=attention_Mask,token_type_ids=token)
        #选择使用Pooled Output用于文本分类
        # bertout=bertout[1:]
        bertout=bertout['pooler_output']
        dropout=self.dropout(bertout)
        linearout=self.linear(dropout)
        # linearout=nn.Softmax(linearout)
        return  linearout



class BiRNNClassifier(nn.Module):
    def __int__(self,relation_num):
        super().__init__()