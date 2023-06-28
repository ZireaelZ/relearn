# from transformers import  BertTokenizer
# bertpath="./bertModels/bert_pretrained"
# tokenizer = BertTokenizer.from_pretrained(bertpath)
# train_word_lists=['[PAD][PAD][PAD][PAD][PAD][PAD]在这个例子中[PAD]，首先加载模型，然后构建分类器。接着型和分类器组合起来，形成一个完整的模型。然后定义优化器，使用优化器对整个模型的参数进行微调。在训练过程中，模型的所有参数都会参与更新']
# print(len(train_word_lists[0]))
# tokenized_texts = [tokenizer(sent) for sent in train_word_lists]
# print(len(tokenizer(train_word_lists[0])['input_ids']))
# # a=[0 for i in range(5)]
# # b={'a':1,"b":0}
# print(len(b))
#数据转tensor测试
#
# dev_jsonpath='data/datatyoe1/train.json'
#
# from util import bert_tokenizer_tensor
# from  data import  readJson
#
# relpath='data/datatyoe1/relation2id.txt'
# dev_sentencelist, dev_rellist, _= readJson([dev_jsonpath], relpath)
# bat,mask,token=bert_tokenizer_tensor(dev_sentencelist)
# print('1')
# maxlen=10
# sentence=[0,1,2,3,4,5]
# print(sentence+[0]*(maxlen-len(sentence)))