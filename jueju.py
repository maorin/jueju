# -*- coding: utf-8 -*-
import os
import jieba
from sklearn.externals import joblib
from click.types import Path

path = os.path.split(os.path.realpath(__file__))[0]

#转换为Y
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def what_kind(title):
    clf = joblib.load(path + '/data/jueju_clf.pkl') 
    mlb = joblib.load(path + '/data/jueju_mlb.pkl') 
    
    title_list = []
    seg_list = jieba.cut(title, cut_all=False)

    for a in seg_list:
        if a.strip():
            title_list.append(a)
    
    test_new = setOfWords2Vec(list(mlb.classes_), title_list)
    
    #打印预测结果
    result =  clf.predict(test_new)
    return result
        



