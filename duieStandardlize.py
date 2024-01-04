import json
jsonpaths=[
'duie_dev.json',
'duie_sample.json',
'duie_train.json']

datapath='./data/DuIE/'


def duieStanlize(jsonpaths):
    for jp in jsonpaths:
        with open(datapath+jp,'r',encoding='utf-8')as f:
            alljs=[]
            for line in f.readlines():
                ajson=json.loads(line.strip())
                bjson={}
                bjson['text']=ajson['text']
                entities={}
                relations=[]
                for relation in ajson['spo_list']:
                    try:
                        subject={
                            'start':ajson['text'].index(relation['subject']),
                            'offset':len(relation['subject']),
                            'type':relation['subject_type'],
                            'content':relation['subject']
                        }
                        object={'start':ajson['text'].index(relation['object']['@value']),
                            'offset':len(relation['object']['@value']),
                            'type':relation['object_type']['@value'],
                            'content':relation['object']['@value']}
                        index = len(entities)
                        entities[index] = subject
                        entities[index + 1] = object
                        rel = {'subject': index,
                               'object': index + 1,
                               'relation': relation['predicate']}
                        relations.append(rel)
                    except:
                        print('出现错误')
                bjson['entities']=entities
                bjson['relations']=relations
                alljs.append(bjson)
        with open(datapath+'standard_'+jp,'w',encoding='utf-8')as f:
            for jsonI in alljs:
                json.dump(jsonI, f, ensure_ascii=False)
                f.write('\n')
            print('写入'+datapath+'standard_'+jp)

def duieRelation2id(relationpath):
    with open(relationpath,'r',encoding='utf-8')as f:
        predicate=[]
        for line in f.readlines():
            linejson=json.loads(line)
            predicate.append(linejson['predicate'])
    with open(relationpath[:-4]+'txt', 'w', encoding='utf-8') as f:
        for i in range(len(predicate)):
            f.write(predicate[i]+' '+str(i)+'\n')
# duieStanlize(jsonpaths)
duieRelation2id('./data/DuIE/duie_schema.json')