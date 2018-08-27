#coding:utf8                      

def select_pn(view1_file, view2_file, p, n):
      """
      Select p positive examples and n negative examples from each classifier
      View1_file file format：
      labels   The probability of -1   The probability of 1 or
      labels   The probability of 1   The probability of -1      
      """
      in_file1 = open(view1_file)
      data = [line.strip().split() for line in in_file1]
      first_line1 = data[0]
      view1_list = data[1:]
      in_file1.close()

      in_file2 = open(view2_file)
      data = [line.strip().split() for line in in_file2]
      first_line2 = data[0]
      view2_list = data[1:]
      in_file2.close()

      if len(view1_list) != len(view2_list):
            raise ValueError("view1 and view2 must have the same length") 

      temp1 = []
      temp2 = []
      for i in range(len(view1_list)):
            if first_line1[1] == '-1':
                  if float(view1_list[i][1]) >= float(view1_list[i][2]):
                        temp1.append((-1, float(view1_list[i][1])))
                  else:
                        temp1.append((1, float(view1_list[i][2])))
            else:
                  if float(view1_list[i][1]) >= float(view1_list[i][2]):
                        temp1.append((1, float(view1_list[i][1])))
                  else:
                        temp1.append((-1, float(view1_list[i][2])))
                  
            if first_line2[1] == '-1':
                  if float(view2_list[i][1]) >= float(view2_list[i][2]):
                        temp2.append((-1, float(view2_list[i][1])))
                  else:
                        temp2.append((1, float(view2_list[i][2])))
            else:
                  if float(view2_list[i][1]) >= float(view2_list[i][2]):
                        temp2.append((1, float(view2_list[i][1])))
                  else:
                        temp2.append((-1, float(view2_list[i][2])))                
      
      # Separate positive and negative samples
      pos_1 = [(i, temp1[i][1]) for i in range(len(temp1)) if temp1[i][0] == 1 ]
      neg_1 = [(i, temp1[i][1]) for i in range(len(temp1)) if temp1[i][0] == -1 ]
      pos_2 = [(i, temp2[i][1]) for i in range(len(temp2)) if temp2[i][0] == 1 ]
      neg_2 = [(i, temp2[i][1]) for i in range(len(temp2)) if temp2[i][0] == -1 ]

      # Descending sort
      pos_1.sort(key = lambda d: d[1], reverse = True)
      neg_1.sort(key = lambda d: d[1], reverse = True)
      pos_2.sort(key = lambda d: d[1], reverse = True)
      neg_2.sort(key = lambda d: d[1], reverse = True)

      if len(pos_1) < 2*p:
            pos_set1 = set([po[0] for po in pos_1])
      else:
            pos_set1 = set([po[0] for po in pos_1[: 2*p]])

      if len(pos_2) < 2*p:
            pos_set2 = set([po[0] for po in pos_2])
      else:
            pos_set2 = set([po[0] for po in pos_2[: 2*p]])

      pos_index = pos_set1 | pos_set2

      if len(neg_1) < 2*n:
            neg_set1 = set([ne[0] for ne in neg_1])
      else:
            neg_set1 = set([ne[0] for ne in neg_1[: 2*n]])
      if len(neg_2) < 2*n:
            neg_set2 = set([ne[0] for ne in neg_2])
      else:
            neg_set2 = set([ne[0] for ne in neg_2[: 2*n]])

      neg_index = neg_set1 | neg_set2

      # Each classifier picks 2p and 2n positive and negative samples, and finally returns only p+n
      # Remove the example of coincidence of positive and negative samples, and find the difference set for the two sets

      pos_index = pos_index - neg_index
      neg_index = neg_index - pos_index

      if len(pos_index) < 2*p:
            pos_index = list(pos_index)
      else:
            pos_index = list(pos_index)[: 2*p]

      if len(neg_index) < 2*n:
            neg_index = list(neg_index)
      else:
            neg_index = list(neg_index)[: 2*n]

      return pos_index, neg_index
                      
if __name__ == "__main__":
      view1_file = 'CNN.test.result'
      view2_file = 'DSCNN.test.result'
      p, n = select_pn(view1_file, view2_file, 2, 2)
      print p
      print n


   
            
