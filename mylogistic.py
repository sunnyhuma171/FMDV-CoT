#coding:utf-8

import sys
import os
from subprocess import *
from metric import evaluate
import numpy as np

class MyLogistic():

      temp_dir = '.\\logistic\\temp'

      def __init__(self, train_file, test_file, test_file2 = None):
            
            if not os.path.exists(MyLogistic.temp_dir):
                  os.mkdir(MyLogistic.temp_dir)

            self.train_file = train_file
            self.test_file = test_file
            # self.test_file2 = None
            self.test_file2 = test_file2

            self.train_tail = os.path.split(self.train_file)[1]
            self.test_tail = os.path.split(self.test_file)[1]

            self.logistic_train_exe = r'.\\logistic\\train.exe'
            self.logistic_predict_exe = r'.\\logistic\\predict.exe'

            assert os.path.exists(self.logistic_train_exe), 'train executable not found'
            assert os.path.exists(self.logistic_predict_exe), 'predict exwcutable not found'

            self.model_file = os.path.join(MyLogistic.temp_dir, self.train_tail + '.model')
            self.result_file = os.path.join(MyLogistic.temp_dir, self.test_tail + '.result')

            if test_file2:
                  self.test_file2 = test_file2
                  self.test_tail2 = os.path.split(self.test_file2)[1]
                  self.result_file2 = os.path.join(MyLogistic.temp_dir, self.test_tail2 + '.result')

            if os.path.exists(self.model_file):
                  os.remove(self.model_file)
            if os.path.exists(self.result_file):
                  os.remove(self.result_file)
            if self.test_file2 and os.path.exists(self.result_file2):
                  os.remove(self.result_file2)
                  

      def data_process(self, test_file):
            infile = open(test_file)
            data = infile.readlines()
            infile.close()
            
            test = data[0].strip().split()[0]
            if test == '1' or test == '-1':
                  pass
            else:
                  outfile = open(test_file, 'w')
                  for d in data:
                        outfile.write('1 ' + d)
                  outfile.close()
                  
      
      def train(self):
            command_train = '{0} -s 0 -c 1 {1} {2}'.format(self.logistic_train_exe, self.train_file, self.model_file)
            Popen(command_train, shell = True, stdout = PIPE).communicate()
            

      def predict(self):
            self.data_process(self.test_file)
            command_predict = '{0} -b 1 {1} {2} {3}'.format(self.logistic_predict_exe, self.test_file, self.model_file, self.result_file)
            Popen(command_predict, shell = True, stdout = PIPE).communicate()

            if self.test_file2:
                  self.data_process(self.test_file2)
                  command_predict2 = '{0} -b 1 {1} {2} {3}'.format(self.logistic_predict_exe, self.test_file2, self.model_file, self.result_file2)
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
            
            













