import pickle
import  time
from utils import save_model
from util import getKeyByvalue
from evaluating import Metrics
from model import BertClassifierModel

modelpath='./Trainmodel/'
#单层bert模型训练
def bertClassifier_train_and_eval(train_data, dev_data, test_data,rel2id):
    train_sentence, train_relations = train_data
    dev_sentence, dev_relations = dev_data
    test_sentence, test_relations = test_data
    start = time.time()
    relsize=len(rel2id)
    model_name = "bertClassifier"
    bertClassiModel=BertClassifierModel(relsize)
    bertClassiModel.train(train_sentence,train_relations,dev_sentence,dev_relations,rel2id)
    save_model(bertClassiModel, modelpath + model_name + ".pkl")
    # with open(modelpath + model_name + ".pkl",'rb')as f:
    #     bertClassiModel=pickle.load(f)
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    pred_relation,test_relations= bertClassiModel.test(
        test_sentence, test_relations, rel2id)
    test_relations=[getKeyByvalue(rel2id,relid) for  relid in test_relations]
    pred_relation = [getKeyByvalue(rel2id, relid) for relid in pred_relation]
    metrics = Metrics(test_relations, pred_relation, remove_O=False)
    metrics.report_scores()
    return  pred_relation