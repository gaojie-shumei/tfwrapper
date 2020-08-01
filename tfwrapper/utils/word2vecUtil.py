from gensim.models import Word2Vec
import itertools
import numpy as np
import random
import os.path as ospath
import os


class Word2vecUtil:
    def __init__(self, word2vec_path, word2vec_model_name="data.model",pad_word="<pad>",label_setlist=None,word2vec_model:dict=None):
        '''
        :param word2vec_path: the word2vec_model store path
        :param word2vec_model_name: word2vec model name
        :param pad_word: use for padding data,if the data is need padding,use this to pad
        :param label_setlist: the classes labels
        :param word2vec_model: the word2vec_model
        '''
        self.pad_word = pad_word
        self.word2vec_path = word2vec_path
        self.word2vec_model_name = word2vec_model_name
        self.word2vec_model = word2vec_model
        if label_setlist is not None and pad_word not in label_setlist:
            label_setlist = [pad_word] + label_setlist
        self.label_setlist = label_setlist
    
    def generator_batch(self, batch_size, data_x, data_y=None, shuffle=False,num_parallel_calls=0,random_state=2):
        '''
        the batch data generator
        :param batch_size: batch size
        :param data_x: the data
        :param data_y: the result
        :param shuffle: if True, shuffle, False for not shuffle
        :param num_parallel_calls: if not None and bigger than 0, the batch_size should be multiply this,the type is int 
        '''
        position = 0
        if num_parallel_calls > 0 and num_parallel_calls is not None:
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
                random.seed(random_state)
                shuffle_index = random.sample(range(temp_x_len),temp_x_len)
                temp_x = self.__type2shuffle(temp_x, shuffle_index)
                if data_y is not None:
                    temp_y = self.__type2shuffle(temp_y, shuffle_index)
            data_x = self.__type2concat(data_x[0:position], temp_x)
            if data_y is not None:
                data_y = self.__type2concat(data_y[0:position], temp_y)
            if position + batch_size > length:
                batch_x = data_x[position:]
                if data_y is not None:
                    batch_y = data_y[position:]
            else:
                batch_x = data_x[position:position+batch_size]
                if data_y is not None:
                    batch_y = data_y[position:position+batch_size]
            is_one_epoch = False
            position += batch_size
            if position >length:
                is_one_epoch = True
            yield (batch_x, batch_y, is_one_epoch)
            
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
        '''
        padding the data to the same timesteps,for net compute
        :param batch_data: the char data with list,size is [batch,??]
        :param batch_y_data: the label or result for batch_data, if not None, size can be [batch],[batch,??]
        :param max_length: the timestep max length, if not None and less than limit_len,will padding to max_length
        :param limit_len: the timestep limit_len, the timestep can't bigger than limit_len
        '''
        pad_word = self.pad_word
        max_len = 0
        actual_lengths = []
        for i in range(len(batch_data)):
            if len(batch_data[i])>max_len:
                max_len = len(batch_data[i])
            actual_lengths.append(len(batch_data[i]))
        pad_data = list(batch_data)
        if batch_y_data is not None:
            pad_y_data = list(batch_y_data)
        else:
            pad_y_data = None
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
    
    def format(self,pad_data:list,pad_y_data:list=None,zero_padding=True,word2vec_model=None):
        '''
        :param pad_data: the char data with padding   the size is [batch,max_len], the element type is str
        :param pad_y_data: the label for pad_data,can be None,if exists the size can be [batch],[batch,max_len],[batch,max_len,num_classes]
        :param zero_padding: if True use 0 for pad_word, False,use the word2vec model padding vector
        '''
        batch_x = []
        pad_word = self.pad_word
        if pad_y_data is not None:
            batch_y = []
        else:
            batch_y = None
        if word2vec_model is None:
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
                    if pad_data[i][j]==pad_word and zero_padding:
                        w2v = np.array(word2vec_model[pad_data[i][j]])
                        w2v = np.zeros(shape=w2v.shape)
                        x.append(w2v)
                    else:
                        x.append(word2vec_model[pad_data[i][j]])
                    if pad_y_data is not None:
                        y.append(self.label_setlist.index(pad_y_data[i][j]))
                batch_x.append(np.array(x))
                if batch_y is not None:
                    batch_y.append(np.array(y))
            batch_x = np.array(batch_x)
            if batch_y is not None:
                batch_y = np.array(batch_y)
        else:
            raise RuntimeError("the word2vec model not find, can't convert the str to vector")
        return batch_x,batch_y
    
    def __window_merge(self,data1:np.ndarray,data2:np.ndarray,merge_mode="concat"):
        '''
        window merge
        :param data1: a np.ndarray size is [hidden_size],the batch is 1,the timestep is 1
        :param data2: a np.ndarray size is [hidden_size],the batch is 1,the timestep is 1
        :param merge_mode: the mode for merge, only support 'concat','avg','sum'
        '''
        data = None
        mode = ["concat","avg","sum"]
        if merge_mode not in mode:
            raise RuntimeError("the merge_mode is invalid")
        if merge_mode=="concat":
            data = np.array(data1.tolist()+data2.tolist())
        elif merge_mode=="avg" or merge_mode=="sum":
            data = data1 + data2
        return data
    
    def window_format(self,pad_data:list,window:int=1,pad_y_data:list=None,zero_padding=True,merge_mode="concat",word2vec_model=None):
        '''
        :param pad_data: the char data with padding   the size is [batch,max_len], the element type is str
        :param window: the window num data for merge, it is best to set it with an odd number,
                       if current timestep is i,the window timestep is [i - (window-1)/2~i + (window-1)/2],
                       if the timestep is negtive or more than the max_len, it will be set with zero vector
        :param pad_y_data: the label for pad_data
        :param zero_padding: if True use 0 for pad_word, False,use the word2vec model padding vector
        :param merge_mode: the mode for merge, only support 'concat','avg','sum'
        '''
        mode = ["concat","avg","sum"]
        batch_x = []
        pad_word = self.pad_word
        if pad_y_data is not None:
            batch_y = []
        else:
            batch_y = None
        if word2vec_model is None:
            word2vec_model = self.word2vec_model
        if word2vec_model is None:
            try:
                word2vec_model = self.word2vec()
            except:
                raise RuntimeError("the word2vec model not find, can't convert the str to vector")
        if word2vec_model is not None:
            if not isinstance(window, int) or window < 1:
                raise RuntimeError("window parameter is invalid")
            if window > 1:
                if merge_mode not in mode:
                    raise RuntimeError("the merge_mode is invalid")
            res = (window - 1)//2
            n = window - 1 - res
            for i in range(len(pad_data)):
                x, y = [], []
                for j in range(len(pad_data[i])):
                    w2v = np.array(word2vec_model[pad_data[i][j]])
                    shape = w2v.shape
                    if pad_data[i][j]==pad_word and zero_padding:
                        w2v = np.zeros(shape=shape)
                    for k in range(n,0,-1):
                        if j-k < 0:
                            if not zero_padding:
                                w2v = self.__window_merge(word2vec_model[pad_word],w2v, merge_mode)
                            else:
                                w2v = self.__window_merge(np.zeros(shape=shape),w2v, merge_mode)
                        else:
                            if pad_data[i][j-k] == pad_word and zero_padding:
                                w2v = self.__window_merge(np.zeros(shape=shape), w2v, merge_mode)
                            else:
                                w2v = self.__window_merge(word2vec_model[pad_data[i][j-k]],w2v, merge_mode)
                    for k in range(1,res+1):
                        if j+k >= len(pad_data[i]):
                            if not zero_padding:
                                w2v = self.__window_merge(word2vec_model[pad_word],w2v, merge_mode)
                            else:
                                w2v = self.__window_merge(np.zeros(shape=shape),w2v, merge_mode)
                        else:
                            if pad_data[i][j+k] == pad_word and zero_padding:
                                w2v = self.__window_merge(w2v,np.zeros(shape=shape), merge_mode)
                            else:
                                w2v = self.__window_merge(w2v,word2vec_model[pad_data[i][j+k]], merge_mode)
                    if merge_mode=="avg":
                        w2v = w2v / window
                    x.append(w2v)
                    if pad_y_data is not None:
                        y.append(self.label_setlist.index(pad_y_data[i][j]))
                batch_x.append(np.array(x))
                if batch_y is not None:
                    batch_y.append(np.array(y))
            batch_x = np.array(batch_x)
            if batch_y is not None:
                batch_y = np.array(batch_y)
        else:
            raise RuntimeError("the word2vec model not find, can't convert the str to vector")
            
        return batch_x,batch_y
    
    def set_mask(self,batch_x:np.ndarray,actual_lengths:np.ndarray=None,mask=0):
        '''
        when the padding data in batch_x is not mask value padding and know the actual timestep lengths for each x,
        can use this method to change it
        :param batch_x: the net input data
        :param actual_lengths: the timestep length for batch_x
        :param mask: the value for mask, use in net for no compute
        '''
        if actual_lengths is None:
            return batch_x
        else:
            for i in range(actual_lengths.shape[0]):
                batch_x[i,actual_lengths[i]:] = mask
        return batch_x
    
    def load_word2vec_model(self,name=None):
        '''
        :param name: word2vec model name
        '''
        if name is None:
            name = self.word2vec_model_name
        try:
            model = Word2Vec.load(os.path.join(self.word2vec_path,name))
            self.word2vec_model = model
        except:
            raise RuntimeError("the word2vec model not exists")
        return model
    
    def word2vec(self, sentences=None, size=128, alpha=0.025, window=5, min_count=5,max_vocab_size=None, sample=1e-3,
                 seed=1, workers=3, min_alpha=0.0001,sg=0, hs=0, negative=5, cbow_mean=1, iter=5,name=None):
        '''
        :param sentences：可以是一个·ist，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
        :param sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
        :param size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
        :param window：表示当前词与预测词在一个句子中的最大距离是多少
        :param alpha: 是学习速率
        :param seed：用于随机数发生器。与初始化词向量有关。
        :param min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
        :param max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
        :param sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
        :param workers参数控制训练的并行数。
        :param hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
        :param negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
        :param cbow_mean: 如果为0，则采用上下文词向量的和，如果为1（defau·t）则采用均值。只有使用CBOW的时候才起作用。
        :param hashfxn： hash函数来初始化权重。默认使用python的hash函数
        :param iter： 迭代次数，默认为5
        :param trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RU·E_DISCARD,uti·s.RU·E_KEEP或者uti·s.RU·E_DEFAU·T的函数。
        :param sorted_vocab： 如果为1（defau·t），则在分配word index 的时候会先对单词基于频率降序排序。
        :param batch_words：每一批的传递给线程的单词的数量，默认为10000
        :param name: the word2vec model name, if None,use self.word2vec_model_name
        '''
        if name is None:
            name = self.word2vec_model_name
        if sentences is not None:
            if isinstance(sentences,list)==False:
                sentences = list(sentences)
            sentences = [[self.pad_word]] + sentences
        try:
            if self.word2vec_model is not None:
                model = self.word2vec_model
            else:
                model = Word2Vec.load(os.path.join(self.word2vec_path,name))
                if model.vector_size != size:
                    raise RuntimeError("word2vec size not match")
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
                os.makedirs(self.word2vec_path)
            model.save(os.path.join(self.word2vec_path,name))
        self.word2vec_model = model
        return model
