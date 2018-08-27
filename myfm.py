#coding:utf-8

import sys
import os
from subprocess import *
from metric import evaluate
import numpy as np

class MyFM():

      temp_dir = '.\\fm\\temp'

      def __init__(self, train_file, test_file, test_file2 = None):
            
            if not os.path.exists(MyFM.temp_dir):
                  os.mkdir(MyFM.temp_dir)

            self.train_file = train_file
            self.test_file = test_file
            self.test_file2 = test_file2

            self.train_tail = os.path.split(self.train_file)[1]
            self.test_tail = os.path.split(self.test_file)[1]

            self.fm_train_exe = r'.\\fm\\libfm.exe'

            assert os.path.exists(self.fm_train_exe), 'train executable not found'
            self.predict_file = os.path.join(MyFM.temp_dir, self.test_tail + '.predict')
            self.result_file = os.path.join(MyFM.temp_dir, self.test_tail + '.result')

            if test_file2:
                  self.test_file2 = test_file2
                  self.test_tail2 = os.path.split(self.test_file2)[1]
                  self.predict_file2 = os.path.join(MyFM.temp_dir, self.test_tail2 + '.predict')
                  self.result_file2 = os.path.join(MyFM.temp_dir, self.test_tail2 + '.result')
            if os.path.exists(self.predict_file):
                  os.remove(self.predict_file)
            if os.path.exists(self.result_file):
                  os.remove(self.result_file)
            if self.test_file2 and os.path.exists(self.predict_file2):
                  os.remove(self.predict_file2)
            if self.test_file2 and os.path.exists(self.result_file2):
                  os.remove(self.result_file2)
                  

      def data_process(self, infile, outfile):
            infile = open(infile)
            data = infile.readlines()
            infile.close()
            
            outfile = open(outfile, 'w')
            outfile.write('labels 1 -1' + '\n')
            for d in data:
                  if float(d.strip()) > 0.5:
                        outfile.write('1 ' + d.strip() + ' ' + str(1 - float(d.strip())) + '\n')
                  else:
                        outfile.write('-1 ' + d.strip() + ' ' + str(1 - float(d.strip())) + '\n')
            outfile.close()

                  
      
      def train(self):
#=========================================================================MCMC=============================================================================
            command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method mcmc -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            # print command_train
            Popen(command_train, shell = True, stdout = PIPE).communicate()

            if self.test_file2:
                  command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method mcmc -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
                  Popen(command_train, shell = True, stdout = PIPE).communicate()
#===========================================================================SGD============================================================================
            # command_train = "{0} -task c -verbosity 0 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgd -learn_rate 0.01 -regular '0,0,0.01' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 0 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgd -learn_rate 0.01 -regular '0,0,0.01' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
#============================================================================ALS===========================================================================
            # command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method als -regular '0,0,10' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method als -regular '0,0,10' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
#==============================================================================SGDA========================================================================
            # command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation {4}".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file, self.train_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation {4}".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2, self.train_file)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
            

      def predict(self):
            self.data_process(self.predict_file, self.result_file)

            if self.test_file2:
                  self.data_process(self.predict_file2, self.result_file2)


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
            
            













