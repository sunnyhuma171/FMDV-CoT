# coding:utf8

import numpy as np
from random import Random
import os
import shutil
from decimal import Decimal, Context, ROUND_HALF_UP
from selectmethod import select_pn
from mysvm import MySvm
from mylogistic import MyLogistic
from myfm import MyFM
from combinedresult import combined_result
import Stats


# Empty all files under the folder
def clear_file(path):
      """Recursively remove all files and folders under the path from the bottom up"""
      for root, dirs, files in os.walk(path, topdown = False):
            for f in files:
                  os.remove(os.path.join(root, f))
            for d in dirs:
                  os.remove(os.path.join(root, d))


class CoTrain():
      """A co-training framework that is not related to the classifier. Temporary files generated in this section are placed in the co_temp folder."""
      """
      """
      
      def __init__(self, label_file,
                   train_file_view1, train_file_view2, 
                   unlabel_file_view1, unlabel_file_view2, 
                   test_file_view1, test_file_view2,
                   max_epochs = 100, pool_size = 100, p = 2, n = 2):
            
            # Store some temporary files under the co_temp folder
            temp_dir = os.path.join('.','co_temp')
            if os.path.exists(temp_dir):       
                  clear_file(temp_dir)
            else:
                  os.mkdir(temp_dir)

            self.max_epochs = max_epochs                        # The maximum number of iterations
            self.extracted_data_size = 0                        # Indicates the size of the unlabeled data that has been extracted
            self.P = p                                          # Number of positive cases per extraction
            self.N = n                                          # Number of negative cases per extraction
            self.add_new_size = 4*(self.P + self.N)             # Number of unlabeled data randomly selected each time
            # Category labels for test data
            self.label_file = label_file
            
            # Load unlabeled data into memory
            self.unlabel_data_view1 = file(unlabel_file_view1).readlines() 
            self.unlabel_data_view2 = file(unlabel_file_view2).readlines()

            # Temporarily store unlabeled data for training to label them
            self.unlabel_file_view1 = os.path.join(temp_dir, 'temp_u_view1')
            self.unlabel_file_view2 = os.path.join(temp_dir, 'temp_u_view2')

            self.train_file_view1 = os.path.join(temp_dir, 'temp_train_view1')      # view1 filename
            self.train_file_view2 = os.path.join(temp_dir, 'temp_train_view2')      # view2 filename

            # Final test file
            self.test_file_view1 = test_file_view1 
            self.test_file_view2 = test_file_view2

            # Copy training data to a temporary folder
            shutil.copyfile(train_file_view1, self.train_file_view1)
            shutil.copyfile(train_file_view2, self.train_file_view2)

            # Subscripts of unlabeled data, one subscript indicates a sample, all operations are based on the subscript of the list
            self.unlabel_index = range(len(self.unlabel_data_view1))  
            self.total_unlabel_size = len(self.unlabel_index)         # Unlabeled data size
            self.remained_unlabel_size = self.total_unlabel_size      # Remaining unlabeled data size
            
            self.pool_index = list()   # A temporary data pool is constructed to store the subscripts of unlabeled data, each time a part of the samples are randomly selected for testing
            self.pool_size = pool_size # Initial data pool size

            self.prf1 = []
            self.prf2 = []
            self.prf_all = []
            self.prf_conb = []


      def rand_extra_unlabel_data(self, size):
            """ extra unlabel data from unlabel data set into pool """
            rand = Random()

            for x in xrange(self.total_unlabel_size - 1, self.extracted_data_size, -1):
                  y = rand.randint(self.extracted_data_size, x)
                  self.unlabel_index[x], self.unlabel_index[y] = self.unlabel_index[y], self.unlabel_index[x]

            temp = self.unlabel_index[self.extracted_data_size: self.extracted_data_size + size]
            self.extracted_data_size += size
            self.remained_unlabel_size -= size
            print 'remained_unlabel_size', self.remained_unlabel_size

            return temp

      def add_to_pool(self, index):
            self.pool_index.extend(index)
            self.pool_size = len(self.pool_index)  # update pool size

      def del_from_pool(self, index):
            for i in index:
                  self.pool_index.remove(i)
            self.pool_size = len(self.pool_index)  # update pool size

      def save_pool_data(self, view):
            """Update unlabeled data file once every iteration"""            
            if view == '1':                  
                  outfile = open(self.unlabel_file_view1, 'w')
                  for i in self.pool_index:
                        outfile.write(self.unlabel_data_view1[i])
                  outfile.close()

            elif view == '2':
                  outfile = open(self.unlabel_file_view2, 'w')
                  for i in self.pool_index:
                        outfile.write(self.unlabel_data_view2[i])
                  outfile.close()

            else:
                  print 'error! usage: filename, data_index, view'

      def save_train_data(self, data_index, view, label = None):
            if view == '1':
                  outfile = open(self.train_file_view1, 'a')
                  if label != None:
                        for i in data_index:
                              temp = self.unlabel_data_view1[i].split()
                              temp[0] = str(label)
                              outfile.write(' '.join(temp))
                              outfile.write('\n')
                  else:
                        outfile.write(''.join([self.unlabel_data_view1[i] for i in data_index]))
                  outfile.close()

            elif view == '2':
                  outfile = open(self.train_file_view2, 'a')
                  if label != None:
                        for i in data_index:
                              temp = self.unlabel_data_view2[i].split()
                              temp[0] = str(label)
                              outfile.write(' '.join(temp))
                              outfile.write('\n')
                  else:
                        outfile.write(''.join([self.unlabel_data_view2[i] for i in data_index]))
                  outfile.close()

            else:
                 print 'error! usage: filename, data_index, view, label = None'


      def train(self, flag, classifier_type):

            # Initializing unlabeled data pool
            un_index = self.rand_extra_unlabel_data(self.pool_size)
            self.add_to_pool(un_index)

            # Save unlabeled data
            self.save_pool_data('1')
            self.save_pool_data('2')

            for i in xrange(self.max_epochs):
                  print 'now is epoch %d ' % i
#==============================================================================================================
                  # # SVM classifier
                  # c1 = MySvm(self.train_file_view1, self.unlabel_file_view1, self.test_file_view1)
                  # c2 = MySvm(self.train_file_view2, self.unlabel_file_view2, self.test_file_view2)
#==============================================================================================================                  
                  # # LR classifier
                  # c1 = MyLogistic(self.train_file_view1, self.unlabel_file_view1, self.test_file_view1)
                  # c2 = MyLogistic(self.train_file_view2, self.unlabel_file_view2, self.test_file_view2)                
#==============================================================================================================
                  # FM classifier
                  c1 = MyFM(self.train_file_view1, self.unlabel_file_view1, self.test_file_view1)                  
                  c2 = MyFM(self.train_file_view2, self.unlabel_file_view2, self.test_file_view2)
#==============================================================================================================

                  # Training two classifiers
                  c1.train()
                  c2.train()

                  # Prediction operation
                  c1.predict()
                  c2.predict()
 
                  # Prediction results of the final data
                  prf1 = c1.evaluat()
                  print 'prf1:', prf1
                  
                  prf2 = c2.evaluat()
                  print 'prf2:', prf2
# 1 for slf+sf; 2 for slf+tf; 3 for sf+tf; 4 for CNN+slf; 5 for CNN+sf; 6 for CNN+tf; 7 for DSCNN+slf; 8 for DSCNN+sf; 9 for DSCNN+tf; 10 for CNN+DSCNN
                  if classifier_type == 'fm': # lr, svm, fm
                      if flag == 1: # for fm baseline
                          prf_all = (93.20, 94.00, 93.60, 93.60)
                      elif flag == 2:
                          prf_all = (91.80, 93.00, 92.40, 92.50)
                      elif flag == 3:
                          prf_all = (93.40, 93.40, 93.40, 93.30)
                      elif flag == 4:
                          prf_all = (94.60, 96.90, 95.70, 95.70)
                      elif flag == 5:
                          prf_all = (91.70, 93.20, 92.40, 92.30)
                      elif flag == 6:
                          prf_all = (92.90, 95.80, 94.30, 94.10)
                      elif flag == 7:
                          prf_all = (97.40, 96.70, 97.00, 97.10)
                      elif flag == 8:
                          prf_all = (97.90, 96.10, 97.00, 97.10)
                      elif flag == 9:
                          prf_all = (97.20, 96.40, 96.80, 96.90)
                      elif flag == 10:
                          prf_all = (98.50, 96.10, 97.30, 97.30)
# 1 for slf+sf; 2 for slf+tf; 3 for sf+tf; 4 for CNN+slf; 5 for CNN+sf; 6 for CNN+tf; 7 for DSCNN+slf; 8 for DSCNN+sf; 9 for DSCNN+tf; 10 for CNN+DSCNN
                  if classifier_type == 'lr': # lr, svm, fm
                      if flag == 1: # for lr baseline
                          prf_all = (79.80, 80.80, 80.20, 80.40)
                      elif flag == 2:
                          prf_all = (82.50, 86.00, 84.20, 84.10)
                      elif flag == 3:
                          prf_all = (69.40, 74.10, 71.60, 71.10)
                      elif flag == 4:
                          prf_all = (90.00, 94.40, 92.10, 92.10)
                      elif flag == 5:
                          prf_all = (89.60, 93.80, 91.70, 91.60)
                      elif flag == 6:
                          prf_all = (88.50, 93.90, 91.10, 91.10)
                      elif flag == 7:
                          prf_all = (98.20, 95.10, 96.60, 96.70)
                      elif flag == 8:
                          prf_all = (98.30, 95.10, 96.70, 96.80)
                      elif flag == 9:
                          prf_all = (98.00, 95.00, 96.50, 96.60)
                      elif flag == 10:
                          prf_all = (98.30, 95.90, 97.10, 97.20)
# 1 for slf+sf; 2 for slf+tf; 3 for sf+tf; 4 for CNN+slf; 5 for CNN+sf; 6 for CNN+tf; 7 for DSCNN+slf; 8 for DSCNN+sf; 9 for DSCNN+tf; 10 for CNN+DSCNN
                  if classifier_type == 'svm': # lr, svm, fm
                      if flag == 1: # for svm baseline
                          prf_all = (71.30, 73.70, 72.50, 72.40)
                      elif flag == 2:
                          prf_all = (84.20, 76.90, 80.30, 81.30)
                      elif flag == 3:
                          prf_all = (65.00, 67.10, 65.90, 65.80)
                      elif flag == 4:
                          prf_all = (90.90, 93.90, 92.40, 92.40)
                      elif flag == 5:
                          prf_all = (89.50, 93.70, 91.60, 91.50)
                      elif flag == 6:
                          prf_all = (89.60, 93.30, 91.40, 91.40)
                      elif flag == 7:
                          prf_all = (97.70, 95.90, 96.80, 96.90)
                      elif flag == 8:
                          prf_all = (97.80, 95.70, 96.80, 96.80)
                      elif flag == 9:
                          prf_all = (97.80, 95.50, 96.70, 96.70)
                      elif flag == 10:
                          prf_all = (97.70, 95.90, 96.80, 96.90)

                  print 'prf_all:', prf_all
                  
                  prf_conb = combined_result(self.label_file, c1.result_file2, c2.result_file2, 'sum') # max product min sum
                  print 'prf_conb:', prf_conb
                  print

                  self.prf1.append(prf1)
                  self.prf2.append(prf2)
                  self.prf_all.append(prf_all)
                  self.prf_conb.append(prf_conb)

                  # Select p&n positive and negative samples
                  p_index, n_index = select_pn(c1.result_file, c2.result_file, self.P, self.N)  #

                  # -------- Update --------
                  """
                  print 'p_index', p_index
                  print 'n_index', n_index
                  """
                  
                  p_index = [self.pool_index[p] for p in p_index]
                  n_index = [self.pool_index[n] for n in n_index]

                  # Add instances to the training data
                  self.save_train_data(p_index, '1', '1')
                  self.save_train_data(n_index, '1', '-1')

                  self.save_train_data(p_index, '2', '1')
                  self.save_train_data(n_index, '2', '-1')

                  # Delete data that has been added to the training set
                  self.del_from_pool(p_index)
                  self.del_from_pool(n_index)

                  # Update unlabeled data pool, randomly extracted from unlabeled dataset
                  un_index = self.rand_extra_unlabel_data(self.add_new_size)
                  
                  self.add_to_pool(un_index)

                  # Serializing unlabeled data pool
                  self.save_pool_data('1')
                  self.save_pool_data('2')

                  if self.remained_unlabel_size <= 10:
                        print '--------------------------'
                        print ' unlabeled data is used up! '
                        break

                  # --- end update --- #

def subset(data_file, data_size, rand_list, flag):
      """
      data_file : Data files need to be divided
      data_size : outfile1's size
      rand_list : rand partition list
      """

      outfile1 = open('./data/labeled/%d/' % flag + data_file.split('/')[-1] + '.labeled', 'w')
      outfile2 = open('./data/unlabeled/%d/' % flag + data_file.split('/')[-1].split('.')[0] + '.unlabeled', 'w')         

      infile = open(data_file)
      data_list = infile.readlines()
      infile.close()

      for i in rand_list[: data_size]:
            outfile1.write(data_list[i])
      for i in rand_list[data_size: ]:
            outfile2.write(data_list[i])

      outfile1.close()
      outfile2.close()

      with open('./data/unlabeled/%d/' % flag + data_file.split('/')[-1].split('.')[0] + '.unlabeled.txt', "rU") as f:
          with open('./data/unlabeled/%d/' % flag + data_file.split('/')[-1].split('.')[0] + '.unlabeled', "a") as f1:
              for line in f:
                f1.write(line)
      f.close()
      f1.close()

if __name__ == "__main__":
      flag = 10 # 1 for slf+sf; 2 for slf+tf; 3 for sf+tf; 4 for CNN+slf; 5 for CNN+sf; 6 for CNN+tf; 7 for DSCNN+slf; 8 for DSCNN+sf; 9 for DSCNN+tf; 10 for CNN+DSCNN
      # Setting of parameters
      data_size = 150
      classifier_type = 'fm' # lr, svm, fm
      initial_size = data_size
      metric1 = 'p'
      metric2 = 'r'
      metric3 = 'f1' # p, r, f1, auc
      metric4 = 'auc'
      pos, neg = 2, 2
      p = pos
      n = neg
      max_epochs = 25
      pool_size = 100

      arr_p = 0
      arr_r = 0
      arr_f1 = 0
      arr_auc = 0

      rand = Random()
      rand_list = range(2800)
      for x in xrange(len(rand_list) - 1, 0, -1):
            y = rand.randint(0, x)
            rand_list[x], rand_list[y] = rand_list[y], rand_list[x]

      if flag == 1:
          subset('./data/slf.train', data_size, rand_list, flag)
          subset('./data/sf.train', data_size, rand_list, flag)
      elif flag == 2:
          subset('./data/slf.train', data_size, rand_list, flag)
          subset('./data/tf.train', data_size, rand_list, flag)
      elif flag == 3:
          subset('./data/sf.train', data_size, rand_list, flag)
          subset('./data/tf.train', data_size, rand_list, flag)
      elif flag == 4:
          subset('./data/CNN.train', data_size, rand_list, flag)
          subset('./data/slf.train', data_size, rand_list, flag)
      elif flag == 5:
          subset('./data/CNN.train', data_size, rand_list, flag)
          subset('./data/sf.train', data_size, rand_list, flag) 
      elif flag == 6:
          subset('./data/CNN.train', data_size, rand_list, flag)
          subset('./data/tf.train', data_size, rand_list, flag)    
      elif flag == 7:
          subset('./data/DSCNN.train', data_size, rand_list, flag)
          subset('./data/slf.train', data_size, rand_list, flag)     
      elif flag == 8:
          subset('./data/DSCNN.train', data_size, rand_list, flag)
          subset('./data/sf.train', data_size, rand_list, flag) 
      elif flag == 9:
          subset('./data/DSCNN.train', data_size, rand_list, flag)
          subset('./data/tf.train', data_size, rand_list, flag)   
      elif flag == 10:
          subset('./data/CNN.train', data_size, rand_list, flag)
          subset('./data/DSCNN.train', data_size, rand_list, flag)                                          

      P1_value = []
      R1_value = []
      F11_value = []
      AUC1_value = []
      P2_value = []
      R2_value = []
      F12_value = []
      AUC2_value = []
      Pa_value = []
      Ra_value = []
      F1a_value = []
      AUCa_value = []
      Pc_value = []
      Rc_value = []
      F1c_value = []
      AUCc_value = []
      for i in xrange(5):
            print '======================== For %d ========================' %i
            if flag == 1:
                ct = CoTrain('./data/label',
                             './data/labeled/1/slf.train.labeled', './data/labeled/1/sf.train.labeled',
                             './data/unlabeled/1/slf.unlabeled', './data/unlabeled/1/sf.unlabeled',
                             './data/slf.test', './data/sf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 2:
                ct = CoTrain('./data/label',
                             './data/labeled/2/slf.train.labeled', './data/labeled/2/tf.train.labeled',
                             './data/unlabeled/2/slf.unlabeled', './data/unlabeled/2/tf.unlabeled',
                             './data/slf.test', './data/tf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 3:
                ct = CoTrain('./data/label',
                             './data/labeled/3/sf.train.labeled', './data/labeled/3/tf.train.labeled',
                             './data/unlabeled/3/sf.unlabeled', './data/unlabeled/3/tf.unlabeled',
                             './data/sf.test', './data/tf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 4:
                ct = CoTrain('./data/label',
                             './data/labeled/4/CNN.train.labeled', './data/labeled/4/slf.train.labeled',
                             './data/unlabeled/4/CNN.unlabeled', './data/unlabeled/4/slf.unlabeled',
                             './data/CNN.test', './data/slf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 5:
                ct = CoTrain('./data/label',
                             './data/labeled/5/CNN.train.labeled', './data/labeled/5/sf.train.labeled',
                             './data/unlabeled/5/CNN.unlabeled', './data/unlabeled/5/sf.unlabeled',
                             './data/CNN.test', './data/sf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 6:
                ct = CoTrain('./data/label',
                             './data/labeled/6/CNN.train.labeled', './data/labeled/6/tf.train.labeled',
                             './data/unlabeled/6/CNN.unlabeled', './data/unlabeled/6/tf.unlabeled',
                             './data/CNN.test', './data/tf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 7:
                ct = CoTrain('./data/label',
                             './data/labeled/7/DSCNN.train.labeled', './data/labeled/7/slf.train.labeled',
                             './data/unlabeled/7/DSCNN.unlabeled', './data/unlabeled/7/slf.unlabeled',
                             './data/DSCNN.test', './data/slf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 8:
                ct = CoTrain('./data/label',
                             './data/labeled/8/DSCNN.train.labeled', './data/labeled/8/sf.train.labeled',
                             './data/unlabeled/8/DSCNN.unlabeled', './data/unlabeled/8/sf.unlabeled',
                             './data/DSCNN.test', './data/sf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 9:
                ct = CoTrain('./data/label',
                             './data/labeled/9/DSCNN.train.labeled', './data/labeled/9/tf.train.labeled',
                             './data/unlabeled/9/DSCNN.unlabeled', './data/unlabeled/9/tf.unlabeled',
                             './data/DSCNN.test', './data/tf.test',
                             max_epochs, pool_size, p, n)
            elif flag == 10:
                ct = CoTrain('./data/label',
                             './data/labeled/10/CNN.train.labeled', './data/labeled/10/DSCNN.train.labeled',
                             './data/unlabeled/10/CNN.unlabeled', './data/unlabeled/10/DSCNN.unlabeled',
                             './data/CNN.test', './data/DSCNN.test',
                             max_epochs, pool_size, p, n)
                             
            ct.train(flag, classifier_type)
            data_list_p = []
            data_list_r = []
            data_list_f1 = []
            data_list_auc = []

# Precision            
            p1 = [i[0] for i in ct.prf1]
            p2 = [i[0] for i in ct.prf2]
            p_a = [i[0] for i in ct.prf_all]
            p_c = [i[0] for i in ct.prf_conb]
            P1_value.append(p1[-1])
            P2_value.append(p2[-1])
            Pa_value.append(p_a[-1])
            Pc_value.append(p_c[-1])
            data_list_p = [p1, p2, p_a, p_c]

# Recall            
            r1 = [i[1] for i in ct.prf1]
            r2 = [i[1] for i in ct.prf2]
            r_a = [i[1] for i in ct.prf_all]
            r_c = [i[1] for i in ct.prf_conb]
            R1_value.append(r1[-1])
            R2_value.append(r2[-1])
            Ra_value.append(r_a[-1])
            Rc_value.append(r_c[-1])
            data_list_r = [r1, r2, r_a, r_c]

# F1 scores
            f1 = [i[2] for i in ct.prf1]
            f2 = [i[2] for i in ct.prf2]
            f_a = [i[2] for i in ct.prf_all]
            f_c = [i[2] for i in ct.prf_conb]
            F11_value.append(f1[-1])
            F12_value.append(f2[-1])
            F1a_value.append(f_a[-1])
            F1c_value.append(f_c[-1])
            data_list_f1 = [f1, f2, f_a, f_c]

# AUC            
            auc1 = [i[3] for i in ct.prf1]
            auc2 = [i[3] for i in ct.prf2]
            auc_a = [i[3] for i in ct.prf_all]
            auc_c = [i[3] for i in ct.prf_conb]
            AUC1_value.append(auc1[-1])
            AUC2_value.append(auc2[-1])
            AUCa_value.append(auc_a[-1])
            AUCc_value.append(auc_c[-1])
            data_list_auc = [auc1, auc2, auc_a, auc_c]            
            
            data_list_p = np.array(data_list_p).T
            data_list_r = np.array(data_list_r).T
            data_list_f1 = np.array(data_list_f1).T
            data_list_auc = np.array(data_list_auc).T

            arr_p += data_list_p
            arr_r += data_list_r
            arr_f1 += data_list_f1
            arr_auc += data_list_auc

      arr_p = arr_p / 5.0
      arr_r = arr_r / 5.0
      arr_f1 = arr_f1 / 5.0
      arr_auc = arr_auc / 5.0
 

      stats_P1 = Stats.Stats(P1_value)
      stats_R1 = Stats.Stats(R1_value)
      stats_F11 = Stats.Stats(F11_value)
      stats_AUC1 = Stats.Stats(AUC1_value)
      print "P1_value:", P1_value
      print "R1_value:", R1_value
      print "F11_value:", F11_value
      print "AUC1_value:", AUC1_value
      print 'stats_P1(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_P1.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_P1.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_R1(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_R1.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_R1.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_F11(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_F11.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_F11.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_AUC1(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_AUC1.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_AUC1.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))

      stats_P2 = Stats.Stats(P2_value)
      stats_R2 = Stats.Stats(R2_value)
      stats_F12 = Stats.Stats(F12_value)
      stats_AUC2 = Stats.Stats(AUC2_value)
      print "P2_value:", P2_value
      print "R2_value:", R2_value
      print "F12_value:", F12_value
      print "AUC2_value:", AUC2_value
      print 'stats_P2(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_P2.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_P2.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_R2(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_R2.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_R2.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_F12(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_F12.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_F12.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_AUC2(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_AUC2.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_AUC2.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))

      stats_Pa = Stats.Stats(Pa_value)
      stats_Ra = Stats.Stats(Ra_value)
      stats_F1a = Stats.Stats(F1a_value)
      stats_AUCa = Stats.Stats(AUCa_value)
      print "Pa_value:", Pa_value
      print "Ra_value:", Ra_value
      print "F1a_value:", F1a_value
      print "AUCa_value:", AUCa_value
      print 'stats_Pa(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_Pa.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_Pa.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_Ra(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_Ra.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_Ra.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_F1a(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_F1a.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_F1a.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_AUCa(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_AUCa.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_AUCa.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))

      stats_Pc = Stats.Stats(Pc_value)
      stats_Rc = Stats.Stats(Rc_value)
      stats_F1c = Stats.Stats(F1c_value)
      stats_AUCc = Stats.Stats(AUCc_value)
      print "Pc_value:", Pc_value
      print "Rc_value:", Rc_value
      print "F1c_value:", F1c_value
      print "AUCc_value:", AUCc_value
      print 'stats_Pc(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_Pc.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_Pc.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_Rc(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_Rc.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_Rc.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_F1c(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_F1c.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_F1c.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_AUCc(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_AUCc.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_AUCc.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
