# -*- coding: utf-8 -*-

import datetime
import numpy as np

class sentence:
    def __init__(self):
        self.word = []
        self.tag = []
        self.wordchars = []

class dataset:
    def __init__(self):
        self.sentences = []
        self.name = ""
    
    def open_file(self, inputfile):
        self.inputfile = open(inputfile, mode='r', encoding='utf-8')
        self.name = inputfile.split('.')[0]
        
    def close_file(self):
        self.inputfile.close()

    def read_data(self, sentenceLen):
        sentenceCount = 0
        wordCount = 0
        sen = sentence()
        for s in self.inputfile:
            if(s == '\n'):
                self.sentences.append(sen)
                sentenceCount += 1
                sen = sentence()
                if(sentenceLen !=-1 and sentenceCount >= sentenceLen):
                    break
                continue
            list_s = s.split('\t')
            str_word = list_s[1]#.decode('utf-8')
            str_tag = list_s[3]
            list_wordchars = list(str_word)
            sen.word.append(str_word)
            sen.tag.append(str_tag)
            sen.wordchars.append(list_wordchars)
            wordCount += 1
        print(self.name + ".conll contains " + str(sentenceCount) + " sentences")
        print(self.name + ".conll contains " + str(wordCount) + " words")
        
class HMM():
    def __init__(self):
        self.alpha = 0.5 #平滑指数
        self.train = dataset()
        self.dev = dataset()

        self.train.open_file("train.conll")
        self.train.read_data(-1)
        self.train.close_file()

        #self.dev.open_file("dev.conll")
        self.dev.open_file("train.conll")
        self.dev.read_data(-1)
        self.dev.close_file()
        
        self.tag_dict = dict()
        self.tag_dict["START"] = len(self.tag_dict)
        self.tag_dict["STOP"] = len(self.tag_dict)
        for s in self.train.sentences:
            for t in s.tag:
                if(t in self.tag_dict.keys()):
                    pass
                else:
                    self.tag_dict[t] = len(self.tag_dict)
                    
        self.word_dict = {}
        for s in self.train.sentences:
            for w in s.word:
                if(w in self.word_dict):
                    pass
                else:
                    self.word_dict[w] = len(self.word_dict)
                    
        self.tags = []
        for s in self.train.sentences:
            for t in s.tag:
                self.tags.append(t)
            
        self.words = []
        for s in self.train.sentences:
            for w in s.word:
                self.words.append(w)
        # print(self.tag_dict)
        # print(self.word_dict)
        # print(self.tags[:10])
        # print(self.words[:10])
        
    """
    Count(s; t) 表示st 这个bigram（两个连续出现的词性）在数据集D 中出现的次数
    Count(s) 表示s 这个词性在数据集D 中出现的次数
    Count(t;w) 表示t;w 这个bigram（一个词性及其相对应的词）在数据集中出现的次数
    q(s,t) = p(t|s) = count(s,t) / count(s) 转移概率
    e(t,w) = p(w|t) = count(t,w)  / count(t) 发射概率
    """
    
    def lau_mat(self):#发射概率矩阵p(w|t)
        alpha = self.alpha
        N = len(self.tag_dict)
        M = len(self.words)
        print("shape of lau_mat:" + str(N) + "*" + str(M))
        self.lau_mat = np.zeros([N, M])
        # count(t, w)
        for s in self.train.sentences:
            for i in range(len(s.word)):
                w = s.word[i]
                t = s.tag[i]
                self.lau_mat[self.tag_dict[t]][self.word_dict[w]] += 1
        """
        for w in self.words:
            for t in self.tags:
                self.lau_mat[self.tag_dict[t]][self.word_dict[w]] += 1
        """
        # / count(t)
        for i in range(N):
            count_s = sum(self.lau_mat[i][:])
            for j in range(M):
                self.lau_mat[i][j] = (self.lau_mat[i][j] + alpha) / (count_s + alpha*M)
            
        #for i in range(5):
        #    for j in range(5):
        #            print(self.lau_mat[i, j])
    
    def tra_mat(self):#转移概率矩阵
        alpha = self.alpha
        N = len(self.tag_dict)
        self.tra_mat = np.zeros([N, N])
        print("shape of tra_mat:" + str(N) + "*" + str(N))
        #count(s, t)
        for s in self.train.sentences:
            for i in range(len(s.tag)):
                if i == 0: # "START",s.tag[i]
                    self.tra_mat[self.tag_dict["START"], self.tag_dict[s.tag[i]]] += 1
                elif i == (len(s.tag) - 1): # s.tag[i],"STOP"
                    self.tra_mat[self.tag_dict[s.tag[i]], self.tag_dict["STOP"]] += 1
                else: # s.tag[i - 1], s.tag[i]
                    self.tra_mat[self.tag_dict[s.tag[i -1]], self.tag_dict[s.tag[i]]] += 1
        #/ count(s)
        for i in range(N):
            count_t = sum(self.tra_mat[i][:])
            for j in range(N):
                self.tra_mat[i][j] = (self.tra_mat[i][j] + alpha) / (count_t + alpha*N)
        
        
        #for i in range(5):
        #    for j in range(5):
        #            print(self.tra_mat[i, j])
        
    def viterbi(self):
        # 转换对数
        lau_mat = np.log(self.lau_mat)
        tra_mat = np.log(self.tra_mat)
        self.path_list = [] # list of list 每个list元素对应一个句子的最佳path
        self.path = [] # 测试集的词性序列
        self.tags_used = []
        for t in self.tag_dict:
            if t in ["START", "STOP"]:
                pass
            else:
                self.tags_used.append(t)
                
        for s in self.dev.sentences:
            N = len(self.tag_dict) - 2 # 词性数量
            path = np.zeros([len(s.word), N]) # 用于存放不同的词性序列的index, “START”, "STOP"除外
            max_prop = np.zeros([len(s.word), N])
            prop = np.zeros(N)
            # 初始化
            for i in range(N):
                w = s.word[0]
                path[0][i] = -1
                max_prop[0, i] = tra_mat[0, 2 + i] + lau_mat[2 + i, self.word_dict[w]]# 转移概率[start, t] + 发射概率[t, w]
            for i in range(1, len(s.word)):
                w = s.word[i]
                if(i == (len(s.word) - 1)): # 最后一个词
                    for j in range(N): # 前一个词性
                        max_prop[i, j] = max_prop[i - 1, j] + tra_mat[2 + j, self.tag_dict["STOP"]]
                        path[i][j] = j
                else:
                    for j in range(N): # 当前词性
                        for k in range(N): # 前一个词性
                            prop[k] = max_prop[i - 1][k] + tra_mat[2 + k, 2 + j] + lau_mat[2 + j, self.word_dict[w]]
                        max_prop[i, j] = max(prop)
                        path[i, j] = np.argmax(prop)

            best_path_index = []
            word_index = len(s.word) - 1
            max_index = np.argmax(max_prop[word_index])
            best_path_index.insert(0, max_index)
            while True:
                max_index = int(path[word_index, max_index])
                if max_index == -1:
                    break
                best_path_index.insert(0, max_index)
                word_index -= 1
            #print("*********************\n", word_index)
            #print("*********************\n")
                
            best_path = []   
            for loc in best_path_index:
                best_path.append(self.tags_used[loc])
            self.path_list.append(best_path)
            self.path.extend(best_path)
        
        
    def evaluate(self):
        count = 0
        l1 = self.tags
        l2 = self.path
        for i in range(len(l1)):
            if(l1[i] == l2[i]):
                count = count + 1
        self.accuracy = count / len(l1)
        print(self.accuracy)
        
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    hmm = HMM()
    hmm.lau_mat()
    hmm.tra_mat()
    hmm.viterbi()
    hmm.evaluate()
    endtime = datetime.datetime.now()
    print("executing time is " + str((endtime - starttime).seconds) + " s")