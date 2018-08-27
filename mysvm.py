#coding:utf-8

from subprocess import *
import os
import sys
from metric import evaluate
import numpy as np

class MySvm():
      temp_dir = '.\\svm\\temp'
      
      def __init__(self, train_file, test_file, test_file2 = None):
            
            if not os.path.exists(MySvm.temp_dir):
                  os.mkdir(MySvm.temp_dir)            
            
            self.train_file = train_file
            self.test_file = test_file 
            self.test_file2 = test_file2
            
            self.train_tail = os.path.split(self.train_file)[1]
            self.test_tail = os.path.split(self.test_file)[1]
            
            if self.test_file2:
                  self.test_tail2 = os.path.split(self.test_file2)[1]

            self.svmtrain_exe = r'.\\svm\\svm-train.exe'
            self.svmpredict_exe = r'.\\svm\\svm-predict.exe'
            
            assert os.path.exists(self.svmtrain_exe), 'svm-train executable not found'
            assert os.path.exists(self.svmpredict_exe), 'svm-predict executable not found'
          
            self.model_file = os.path.join(MySvm.temp_dir, self.train_tail + '.model')            
            
            self.result_file = os.path.join(MySvm.temp_dir, self.test_tail + '.result')
            
            if self.test_file2:
                  self.result_file2 = os.path.join(MySvm.temp_dir, self.test_tail2 + '.result')

            if os.path.exists(self.model_file):
                  os.remove(self.model_file)

            if os.path.exists(self.result_file):
                  os.remove(self.result_file)

            if self.test_file2 and os.path.exists(self.result_file2):
                  os.remove(self.result_file2)
            
            
      def train(self, arg = None):
            if arg == None:
                  command_train = '{0} -h 0 -b 1 -c 2 {1} {2}'.format(self.svmtrain_exe, self.train_file, self.model_file)
            else:
                  command_train = '{0} -h 0 -b 1 -c 2 {1} {2} {3}'.format(self.svmtrain_exe, arg, self.train_file, self.model_file)
            
            Popen(command_train, shell = True, stdout = PIPE).communicate()


      def predict(self):
            command_predict = '{0} -b 1 {1} {2} {3}'.format(self.svmpredict_exe, self.test_file, self.model_file, self.result_file)
            Popen(command_predict, shell = True, stdout = PIPE).communicate()

            if self.test_file2:
                  command_predict2 = '{0} -b 1 {1} {2} {3}'.format(self.svmpredict_exe, self.test_file2, self.model_file, self.result_file2)
                  Popen(command_predict2, shell = True, stdout = PIPE).communicate()

      def evaluat(self):

            if self.test_file2:                  
                  in_file = open(self.test_file2)
                  gold_data = [int(line.strip().split()[0]) for line in in_file]
                  in_file.close()

                  in_file = open(self.result_file2)            
                  classifier_data = [int(line.strip().split()[0]) for line in in_file.readlines()[1:]]
                  in_file.close()

                  p, r, f, auc = evaluate(np.asarray(gold_data), np.asarray(classifier_data))

                  return (p, r, f, auc)
            else:
                  in_file = open(self.test_file)
                  gold_data = [int(line.strip().split()[0]) for line in in_file]
                  in_file.close()

                  in_file = open(self.result_file)            
                  classifier_data = [int(line.strip().split()[0]) for line in in_file.readlines()[1:]]
                  in_file.close()

                  p, r, f, auc = evaluate(np.asarray(gold_data), np.asarray(classifier_data))

                  return (p, r, f, auc)
                  
            

if __name__ == '__main__':
      pass
      
      
  
