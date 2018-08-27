#coding:utf8

"""
metric.py
Answer quality score calculation method
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score

def evaluate(actual, pred):
    m_precision = precision_score(actual, pred, pos_label=1)
    m_recall = recall_score(actual, pred, pos_label=1)
    fpr, tpr, thresholds = roc_curve(actual, pred, pos_label=1) # If the results are abnormal, please adjust the pos_label (1/-1(2))
    m_f1 = f1_score(actual, pred, pos_label=1)
    m_acc = accuracy_score(actual, pred)
    a = np.asscalar(m_acc)*100.0
    p = np.asscalar(m_precision)*100.0
    r = np.asscalar(m_recall)*100.0
    f = np.asscalar(m_f1)*100.0
    auc1 = float(auc(np.asarray(fpr), np.asarray(tpr)) * 100)
    print 'Accuracy:%.2f%%' % a
    print 'Precision:%.2f%%' % p
    print 'Recall:%.2f%%' % r
    print 'F1-Measure:%.2f%%' % f
    print 'AUC:%.2f%%' % auc1
    return p, r, f, auc1

if __name__ == "__main__":
      result = '.\\svm\\temp\\demo_test.result'
      
      gold = '.\\data\\demo_test'

      gold_data = [int(line.strip().split()[0]) for line in file(gold)]
      
      classifier_data = [int(line.strip().split()[0]) for line in open(result).readlines()[1:]]
      
      p, r, f, auc1 = evaluate(gold_data, classifier_data)

      print p, r, f, auc1
      















      
      
      
