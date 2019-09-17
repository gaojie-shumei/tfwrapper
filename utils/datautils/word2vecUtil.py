from gensim.models import Word2Vec
import itertools
import numpy as np
import random
import os.path as ospath
import os
class Word2vecUtil:
    def __init__(self, pad_word="<pad>",word2vec_path="word2vecmodel",word2vec_model:[dict]=None):
        '''
        :param pad_word: use for padding data,if the data is need padding,use this to pad
        :param word2vec_path: the word2vec_model store path
        :param word2vec_model: the word2vec_model
        '''
        self.pad_word = pad_word
        self.word2vec_path = word2vec_path
        self.word2vec_model = word2vec_model  
    
    def generator_batch(self, batch_size, data_x, data_y=None, shuffle=False,num_parallel_calls=0):
        position = 0
        if num_parallel_calls > 0:
            batch_size = batch_size * num_parallel_calls
        length = self.__type2len(data_x)
        while position < length:
            batch_x = None
            batch_y = None
            batch_len = batch_size
            temp_x = data_x[position:]
            if data_y is not None:
                temp_y = data_y[position:]
            if shuffle:
                temp_x_len = self.__type2len(temp_x)
                shuffle_index = random.sample(range(temp_x_len),temp_x_len)
                temp_x = self.__type2shuffle(temp_x, shuffle_index)
                if data_y is not None:
                    temp_y = self.__type2shuffle(temp_y, shuffle_index)
            data_x = self.__type2concat(data_x[0:position], temp_x)
            if data_y is not None:
                data_y = self.__type2concat(data_y[0:position], temp_y)
            if position + batch_size > length:
                batch_x = data_x[position:]
                batch_len = self.__type2len(data_x[position:])
                if data_y is not None:
                    batch_y = data_y[position:]
                res = batch_size - self.__type2len(data_x[position:])
                while res >= length:
                    batch_x = self.__type2concat(batch_x, data_x)
                    if data_y is not None:
                        batch_y = self.__type2concat(batch_y, data_y)
                    res -= length
                if res > 0:
                    batch_x = self.__type2concat(batch_x, data_x[0:res])
                    if data_y is not None:
                        batch_y = self.__type2concat(batch_y, data_y[0:res])
            else:
                batch_x = data_x[position:position+batch_size]
                if data_y is not None:
                    batch_y = data_y[position:position+batch_size]
            is_one_epoch = False
            position += batch_size
            if position >length:
                is_one_epoch = True
            yield (batch_x, batch_y, batch_len, is_one_epoch)
            
    @staticmethod        
    def __type2shuffle(data, shuffle_index):
        if isinstance(data, list):
            return (np.array(data)[shuffle_index]).tolist()
        elif isinstance(data, np.ndarray):
            return data[shuffle_index]
        else:
            raise TypeError("the data type can't parsed, only support list and numpy.ndarray")
        
    
    @staticmethod
    def __type2len(data):
        if isinstance(data, list):
            length = len(data)
        elif isinstance(data, np.ndarray):
            length = data.shape[0]
        else:
            raise TypeError("the data type can't parsed, only support list and numpy.ndarray")
        return length
    
    @staticmethod
    def __type2concat(data1,data2):
        '''
        :param data1: the data of front
        :param data2: the data of rear
        '''
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if isinstance(data1, list):
            if isinstance(data2, list):
                data = data1 + data2
            else:
                raise TypeError("the data1 and data2 type not match")
        elif isinstance(data1, np.ndarray):
            if isinstance(data2, np.ndarray):
                data = np.r_[data1, data2]
            else:
                raise TypeError("the data1 and data2 type not match")
        else:
            raise TypeError("the data type can't parsed, only support list and numpy.ndarray")
        return data
    
    '''
    batch_data :二维数组  str
    '''
    def padding(self, batch_data: list, batch_y_data: list=None, max_length=None, limit_len=512):
        pad_word = self.pad_word
        max_len = 0
        actual_lengths = []
        for i in range(len(batch_data)):
            if len(batch_data[i])>max_len:
                max_len = len(batch_data[i])
            actual_lengths.append(len(batch_data[i]))
        pad_data = batch_data
        pad_y_data = batch_y_data
        if max_length is not None:
            max_len = max_length
        if max_len > limit_len:
            max_len = limit_len
        for i in range(len(pad_data)):
            if len(batch_data[i]) <= max_len:
                pad_data[i] = pad_data[i] + [pad_word]*(max_len-len(batch_data[i]))
                if batch_y_data is not None:
                    pad_y_data[i] = pad_y_data[i] + [pad_word]*(max_len-len(batch_y_data[i]))
            else:
                pad_data[i] = pad_data[i][0:max_len]
                if batch_y_data is not None:
                    pad_y_data[i] = pad_y_data[i][0:max_len]
                actual_lengths[i] = max_len
    #             print(len(pad_data[i]),len(pad_y_data[i]))
        actual_lengths = np.array(actual_lengths)
        return pad_data, pad_y_data, actual_lengths
    
    '''
    pad_data:二维数组 str
    '''
    def format(self,pad_data:list,pad_y_data:list=None):
        batch_x = []
        if pad_y_data is not None:
            batch_y = []
        else:
            batch_y = None
        word2vec_model = self.word2vec_model
        if word2vec_model is None:
            try:
                word2vec_model = self.word2vec()
            except:
                raise RuntimeError("the word2vec model not find, can't convert the str to vector")
        if word2vec_model is not None:
            for i in range(len(pad_data)):
                x, y = [], []
                for j in range(len(pad_data[i])):
                    x.append(word2vec_model[pad_data[i][j]])
                    if pad_y_data is not None:
                        y.append(self.label_setlist.index(pad_y_data[i][j]))
                batch_x.append(np.array(x))
                if batch_y is not None:
                    batch_y.append(np.array(y))
            batch_x = np.array(batch_x)
            if batch_y is not None:
                batch_y = np.array(batch_y).astype(np.int32)
        return batch_x,batch_y

    def set_mask(self,batch_x:np.ndarray,actual_lengths:np.ndarray=None,mask=0):
        if actual_lengths is None:
            return batch_x
        else:
            for i in range(actual_lengths.shape[0]):
                batch_x[i,actual_lengths[i]:] = mask
        return batch_x
    '''
    sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
    ·  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    ·  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
    ·  window：表示当前词与预测词在一个句子中的最大距离是多少
    ·  alpha: 是学习速率
    ·  seed：用于随机数发生器。与初始化词向量有关。
    ·  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    ·  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
    ·  sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
    ·  workers参数控制训练的并行数。
    ·  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
    ·  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
    ·  cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
    ·  hashfxn： hash函数来初始化权重。默认使用python的hash函数
    ·  iter： 迭代次数，默认为5
    ·  trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
    ·  sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
    ·  batch_words：每一批的传递给线程的单词的数量，默认为10000
    '''
    def word2vec(self, sentences=None, size=128, alpha=0.025, window=5, min_count=5,max_vocab_size=None, sample=1e-3,
                 seed=1, workers=3, min_alpha=0.0001,sg=0, hs=0, negative=5, cbow_mean=1, iter=5,name="data.model"):
        if sentences is not None:
            if isinstance(sentences,list)==False:
                sentences = list(sentences)
            sentences = [[self.pad_word]] + sentences
        try:
            if self.word2vec_model is not None:
                model = self.word2vec_model
            else:
                model = Word2Vec.load(self.word2vec_path+"/"+name)
            if sentences is not None:
                flag = 0
                sentences_set = set(list(itertools.chain.from_iterable(sentences)))
                for word in sentences_set:
                    if word not in model.wv.vocab:
                        flag = 1
                        break
                if flag==1:
                    tte = model.corpus_count + len(sentences)
                    model.build_vocab(sentences, update=True)
                    model.train(sentences,total_examples=tte,epochs=model.iter)
        except:
            if sentences is not None:
                model = Word2Vec(size=size, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
                                 sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,sg=sg, hs=hs, 
                                 negative=negative, cbow_mean=cbow_mean, iter=iter)
                model.build_vocab(sentences)
                model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
            else:
                raise RuntimeError("sentences is None and model not exists!")
        if self.word2vec_path is not None and self.word2vec_path!="":
            if ospath.exists(self.word2vec_path)==False:
                os.mkdir(self.word2vec_path)
            model.save(self.word2vec_path+"/"+name)
        self.word2vec_model = model
        return model
