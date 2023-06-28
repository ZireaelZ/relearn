from copy import deepcopy
from  util import  sort_by_lengths,bert_tokenizer_tensor
import torch
import torch.nn as nn
from neuralNet import BertClassifier
from config import Trainconfig

class BertClassifierModel(object):
    def __init__(self,labelNum):
        self.batchsize=Trainconfig.batchsize
        self.lr=Trainconfig.lr
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoches = Trainconfig.epoch
        self.model=BertClassifier(labelNum)
        self.model.to(self.device)
        self.lossfun=nn.CrossEntropyLoss()
        self.printstep=Trainconfig.printstep
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.lr)
        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None
    def train(self, train_wordlists, train_taglists, dev_wordlists, dev_taglists,tag2id):
        #分成batch，然后按epoch训练
        B=self.batchsize
        word_lists, tag_lists, _ = sort_by_lengths(train_wordlists, train_taglists)
        for e in range(1, self.epoches + 1):
            self.step = 0
            losses = 0
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind + B]
                batch_tags = tag_lists[ind:ind + B]
                losses += self.trainstep(batch_sents, batch_tags,tag2id)
                if self.step % Trainconfig.printstep == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.printstep
                    ))
                    losses = 0.
                # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.eval(dev_wordlists, dev_taglists,tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))
    def eval(self,dev_wordlists, dev_taglists, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_wordlists), self.batchsize):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_wordlists[ind:ind + self.batchsize]
                batch_tags = dev_taglists[ind:ind + self.batchsize]
                tensorized_sents, mask,tokentypeid = bert_tokenizer_tensor(batch_sents)
                tensorized_sents = tensorized_sents.to(self.device)
                mask = mask.to(self.device)
                tokentypeid=tokentypeid.to(self.device)
                targets =torch.tensor(batch_tags)
                targets = targets.to(self.device)
                targets = torch.argmax(targets, dim=1)
                scores = self.model(tensorized_sents, mask,tokentypeid)
                # 计算损失
                loss = self.lossfun(scores, targets).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step
            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
            return val_loss
    def test(self,word_lists, tag_lists, tag2id):
        # word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, mask,tokenid = bert_tokenizer_tensor(word_lists)
        tensorized_sents = tensorized_sents.to(self.device)
        mask=mask.to(self.device)
        tokenid=tokenid.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model(
                tensorized_sents,mask,tokenid)
            batch_tagids1 = torch.argmax(batch_tagids,dim=1)
        # 将id转化为标注
        return batch_tagids1.tolist()
        # pred_tag_lists = []
        # id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        # for i, ids in enumerate(batch_tagids1):
        #     tag_list = []
        #     for j in range(lengths[i]):
        #         if self.crf:
        #             tag_list.append(id2tag[ids[j]])
        #         else:
        #             tag_list.append(id2tag[ids[j].item()])
        #     pred_tag_lists.append(tag_list)
        #
        # # indices存有根据长度排序后的索引映射的信息
        # # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # # 索引为2的元素映射到新的索引是1...
        # # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        # ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        # indices, _ = list(zip(*ind_maps))
        # pred_tag_lists = [pred_tag_lists[i] for i in indices]
        # tag_lists = [tag_lists[i] for i in indices]

        # return pred_tag_lists, tag_lists
    def trainstep(self,batch_words,batch_tags,  tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        #转换为张量，是否可以直接用berttokenizer,本处直接使用berttokenizer
        tensorized_sents, mask,tokentypeid = bert_tokenizer_tensor(batch_words)
        tensorized_sents = tensorized_sents.to(self.device)
        mask = mask.to(self.device)
        tokentypeid=tokentypeid.to(self.device)
        targets=torch.tensor(batch_tags)
        targets = targets.to(self.device)
        # # forward
        # if self.crf:
        #     mask = (tensorized_sents != self.model.PAD)
        #     mask = mask.to(self.device)
        #     # self.model.lstm.flatten_parameters()
        #     # forward
        #     loss, scores = self.model(tensorized_sents, lengths, targets, mask, batch_BMES, self.lexifmodel)
        # # forward
        # else:
        #     scores = self.model(tensorized_sents)
        scores=self.model(tensorized_sents,mask,tokentypeid)
        # scores=
        # 计算损失 更新参数
        self.optimizer.zero_grad()
        targets=torch.argmax(targets,dim=1)
        loss = self.lossfun(scores, targets).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class TemplateModel(object):
    def __int__(self,labelNum):
        pass
    def trainstep(self):
        pass
    def train(self):
        pass
    def eval(self):
        pass
    def test(self):
        pass