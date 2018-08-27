#coding:utf-8
import sys
 
class Stats:
 
    def __init__(self, sequence):
        # convert all items to floats for numerical processing
        self.sequence = [float(item) for item in sequence]
 
    def sum(self):
        if len(self.sequence) < 1:
            return None
        else:
            return sum(self.sequence)
 
    def count(self):
        return len(self.sequence)
 
    def min(self):
        if len(self.sequence) < 1:
            return None
        else:
            return min(self.sequence)
 
    def max(self):
        if len(self.sequence) < 1:
            return None
        else:
            return max(self.sequence)
 
    def avg(self):
        if len(self.sequence) < 1:
            return None
        else:
            return sum(self.sequence) / len(self.sequence)    
 
    def stdev(self):
        if len(self.sequence) < 1:
            return None
        else:
            avg = self.avg()
            sdsq = sum([(i - avg) ** 2 for i in self.sequence])
            stdev = (sdsq / len(self.sequence)) ** .5
            return stdev
 