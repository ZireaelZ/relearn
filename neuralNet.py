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
        # self.hidd=nn.Linear(768,200)
        self.linear=nn.Linear(768,relation_num)
        # self.lossFun=nn.CrossEntropyLoss
    def forward(self,sent_in,attention_Mask,token):
        bertout=self.bertEncoder(sent_in,attention_mask=attention_Mask,token_type_ids=token)
        #选择使用Pooled Output用于文本分类
        # bertout=bertout[1:]
        bertout=bertout['pooler_output']
        dropout=self.dropout(bertout)
        # hidout=self.hidd(dropout)
        linearout=self.linear(dropout)
        # linearout=nn.Softmax(linearout)
        return  linearout



class BiRNNClassifier(nn.Module):
    def __init__(self, relation_num):
        super().__init__()
        self.bertEncoder = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(Trainconfig.dropout, inplace=False)

        self.rnn= nn.LSTM(768, 256, batch_first=True, bidirectional=True,num_layers=2)
        # self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(768, relation_num)
        self.fc=nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, relation_num)
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self, sent_in, attention_Mask, token):
        bertout = self.bertEncoder(sent_in, attention_mask=attention_Mask, token_type_ids=token)
        # 选择使用Pooled Output用于文本分类
        # bertout=bertout[1:]
        self.rnn.flatten_parameters()
        _, (rnnout,_) = self.rnn(bertout[0])
        # output = self.linear(dropout)
        rnnout=torch.cat((rnnout[-2], rnnout[-1]),dim=1)
        # bertout = bertout['pooler_output']
        # dropout = self.dropout(bertout)
        # hidout=self.hidd(dropout)
        # linearout = self.linear(dropout)
        # linearout=nn.Softmax(linearout)
        score=self.fc(rnnout)
        score=self.sigmoid(score)
        return score