import numpy as np
import random
import copy


class Dataset(object):
    def __init__(self, batch_size, shuffle=True, repeat=True, **kwargs):
        '''
        a base dataset for all dataset to extend, the dataset can iter but can't getitem(such as dataset[0] not support)
        for subclass, usually should rewrite __init__, and __getattribute__ method
        properties:
            batch_size(required): the iter size in once
            shuffle(required): when iter, shuffle or not
            repeat(required):when iter, repeat iter or not
            samples(required): the samples in this Dataset, it should be numpy.ndarray type, if not numpy.ndarray,
                               the method __iter__ , iter_result_process and sub_dataset should be overwrited
            read_len: when use this dataset, the sample num has readed
        :param batch_size: batch size
        :param shuffle: shuffle or not
        :param repeat: repeat iter or not
        :param kwargs(optional): not specified, if want use this class to construct dataset,
                                 the 'samples' key and value should be in this
        '''
        if "samples" in kwargs:
            self.samples = np.array(kwargs["samples"])
        else:
            self.samples = None
        if self.samples is not None and not isinstance(self.samples, np.ndarray):
            raise TypeError("the dataset property samples should be the type numpy.ndarray")
        self.read_len = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        super(Dataset, self).__init__()

    def __len__(self):
        if self.samples is None:
            return 0
        else:
            return self.samples.shape[0]

    def __iter__(self):
        samples = self.samples
        if samples is None:
            raise ValueError("dataset is null!")
        if self.read_len >= samples.shape[0]:
            if not self.repeat:
                raise IndexError("the dataset length is " + self.samples.shape[0] +", but the index is "
                                 + self.read_len)
            else:
                self.read_len = 0
            if self.shuffle:
                sample_index = random.sample(range(samples.shape[0]), samples.shape[0])
                samples = samples[sample_index]

        while self.read_len < samples.shape[0]:
            if self.read_len + self.batch_size < samples.shape[0]:
                batch_sample = samples[self.read_len:self.read_len + self.batch_size]
                self.read_len += self.batch_size
                yield batch_sample

            else:
                batch_sample = samples[self.read_len:]
                self.read_len += self.batch_size
                yield batch_sample


    def iter_result_process(self, iter_result):
        '''
        :param iter_result: the iter result, same to self.samples
        :return: the iter result processed data
        '''
        return iter_result

    def sub_dataset(self, sub_rate=0.2, random_state=2):
        '''
        :param sub_rate: sub rate, 0<sub_rate<1
        :param random_state: random state, for get the same sub dataset
        :return:  train_dataset, test_dataset
        '''
        samples = self.samples
        if samples is None:
            raise ValueError("dataset is null, can't be splitted!")
        sub_dataset = copy.deepcopy(self)
        random.seed(random_state)
        sample_index = random.sample(range(samples.shape[0]), samples.shape[0])
        samples = samples[sample_index]
        length = int(samples.shape[0]*sub_rate)
        sub_dataset.samples = samples[0:length]
        self.samples = samples[length:]
        return self, sub_dataset
