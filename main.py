from  data import Text2JSON,readJson,readJson2
from sklearn.model_selection import train_test_split
from  TrainAndEval import bertClassifier_train_and_eval
def main():
    print("读取数据...")
    # textpath='data/datatyoe1/test.txt'
    # relpath='data/datatyoe1/relation2id.txt'
    # test_jsonpath='data/datatyoe1/test.json'
    # train_jsonpath='data/datatyoe1/train.json'
    # dev_jsonpath='data/datatyoe1/dev.json'
    relpath = 'data/DuIE/duie_schema.txt'
    test_jsonpath = 'data/DuIE/standard_duie_sample.json'
    train_jsonpath = 'data/DuIE/standard_duie_train.json'
    dev_jsonpath = 'data/DuIE/standard_duie_dev.json'
    #将文本转为标准JSON格式
    #Text2JSON(textpath,jsonpath)
    train_sentencelist,train_rellist,reldict=readJson([dev_jsonpath],relpath)
    # dev_sentencelist, dev_rellist, _= readJson([dev_jsonpath], relpath)
    train_sentencelist, dev_sentencelist, train_rellist,dev_rellist=train_test_split(train_sentencelist,train_rellist,test_size=0.1,random_state=42)
    dev_sentencelist, test_sentencelist, dev_rellist, test_rellist = train_test_split(dev_sentencelist,
                                                                                        dev_rellist, test_size=0.5,
                                                                                        random_state=42)
    # test_sentencelist, test_rellist, __ = readJson([test_jsonpath], relpath)
    bertClassifier_train_and_eval((train_sentencelist,train_rellist),(dev_sentencelist, dev_rellist),(test_sentencelist, test_rellist),reldict)

def main2():
    print("读取数据...")
    # textpath='data/datatyoe1/test.txt'
    # relpath='data/datatyoe1/relation2id.txt'
    # test_jsonpath='data/datatyoe1/test.json'
    # train_jsonpath='data/datatyoe1/train.json'
    # dev_jsonpath='data/datatyoe1/dev.json'
    relpath = 'data/DuIE/duie_schema.txt'
    test_jsonpath = 'data/DuIE/standard_duie_sample.json'
    train_jsonpath = 'data/DuIE/standard_duie_train.json'
    dev_jsonpath = 'data/DuIE/standard_duie_dev.json'
    # 将文本转为标准JSON格式
    # Text2JSON(textpath,jsonpath)
    train_sentencelist, train_rellist, reldict = readJson3([dev_jsonpath], relpath)
    # dev_sentencelist, dev_rellist, _= readJson([dev_jsonpath], relpath)
    train_sentencelist, dev_sentencelist, train_rellist, dev_rellist = train_test_split(train_sentencelist,
                                                                                        train_rellist, test_size=0.1,
                                                                                        random_state=42)
    dev_sentencelist, test_sentencelist, dev_rellist, test_rellist = train_test_split(dev_sentencelist,
                                                                                      dev_rellist, test_size=0.5,
                                                                                      random_state=42)
    # test_sentencelist, test_rellist, __ = readJson([test_jsonpath], relpath)
    bertClassifier_train_and_eval((train_sentencelist, train_rellist), (dev_sentencelist, dev_rellist),
                                  (test_sentencelist, test_rellist), reldict)


if __name__ == '__main__':
    main()