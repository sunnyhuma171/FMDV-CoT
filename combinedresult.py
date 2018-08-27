#coding:utf8

from metric import evaluate
import numpy as np

"""
classify_result.py
Ensemble learning rules for combining results
"""

def combined_result(gold_file, classifier_result1, classifier_result2, method = 'sum'):
      infile = open(classifier_result1)
      data = infile.readlines()
      label1 = data[0].strip().split()
      result1 = [line.strip().split() for line in data[1:]]
      infile.close()

      infile = open(classifier_result2)
      data = infile.readlines()
      label2 = data[0].strip().split()
      result2 = [line.strip().split() for line in data[1:]]
      infile.close()

      infile = open(gold_file)
      data = infile.readlines()
      gold_data = [int(line.strip().split()[0]) for line in data]
      infile.close()
      

      if len(result1) != len(result2):
            raise ValueError('two list should be equal!')

      if label1[1] != label2[1]:
            raise ValueError('train data not consistent')
      
      result = []
      if method == 'max' or method == None:            
            for i in range(len(result1)):
                  temp = [float(result1[i][1]), float(result1[i][2]), float(result2[i][1]), float(result2[i][2])]
                  index = temp.index(max(temp))

                  if label1[1] == '-1':
                        if index == 0 or index == 2:
                              result.append(-1)
                        else:
                              result.append(1)
                  else:
                        if index == 0 or index == 2:
                              result.append(1)
                        else:
                              result.append(-1)
                        
      elif method == 'product':
            for i in range(len(result1)):
                  if float(result1[i][1])*float(result2[i][1]) >= float(result1[i][2])*float(result2[i][2]):
                        result.append(1)
                  else:
                        result.append(-1)

      elif method == 'min':
            for i in range(len(result1)):
                  temp = [min([float(result1[i][1]), float(result2[i][1])]), min([float(result1[i][2]), float(result2[i][2])])]
                  index = temp.index(max(temp))
                  if index == 0:
                        result.append(1)
                  else:
                        result.append(-1)
      
      elif method == 'sum':
            for i in range(len(result1)):
                  temp = [float(result1[i][1]) + float(result2[i][1]), float(result1[i][2]) + float(result2[i][2])]
                  index = temp.index(max(temp))
                  if index == 0:
                        result.append(1)
                  else:
                        result.append(-1)
      p, r, f, auc = evaluate(np.asarray(gold_data), np.asarray(result))

      return (p, r, f, auc)
      
if __name__ == "__main__":
      gold_file = './data/label'
      classifier_result1 = './fm/temp/view1.predict'
      classifier_result2 = './fm/temp/view2.predict'
      p, r, f, auc = combined_result(gold_file, classifier_result1, classifier_result2, method = 'sum')
      print p, r, f, auc







