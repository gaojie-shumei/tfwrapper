from tfwrapper.basev2.wrapperv2.entity import FeatureTypingFunctions, InputFeatures
import tensorflow as tf
import numpy as np
import collections
from typing import List, Union
import os


class TFDataWrapper:
    def __init__(self):
        self.dataset = None
        super(TFDataWrapper, self).__init__()
    
    def iter(self):
        try:
            if self.dataset is None:
                raise RuntimeError("NoneType can't iter")
            return iter(self.dataset)
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
    
    def wrapper(self,all_features: List[InputFeatures], batch_size, is_train=True,
                drop_remainder=False, num_parallel_calls=None,padded_shapes=None):
        '''
        the len(dataset) % (batch_size*num_parallel_calls) == 0 can be True or False, 
        if want use multi-GPU(multi-Worker), use tf.estimator recommend 
        :param all_features: the all data for network
        :param batch_size: batch size
        :param is_train:  is train set or not
        :param drop_remainder:  if len(all_features)%batch_size!=0, drop the next data or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :param padded_shapes: if each sample is not same length, this param should provide for padding
        :return:
            tf.data.Dataset, num_batch
        '''
#         if num_parallel_calls is not None and num_parallel_calls > 0:
#             batch_size = batch_size * num_parallel_calls
        net_data = {}
        for i in range(len(all_features)):
            f = all_features[i]
            for x_key in f.net_x:
                if x_key in net_data:
                    net_data[x_key].append(f.net_x[x_key])
                else:
                    net_data[x_key] = []
                    net_data[x_key].append(f.net_x[x_key])
            for y_key in f.net_y:
                if y_key in net_data:
                    net_data[y_key].append(f.net_y[y_key])
                else:
                    net_data[y_key] = []
                    net_data[y_key].append(f.net_y[y_key])
            if "is_real_sample" not in net_data:
                net_data["is_real_sample"] = []
                net_data["is_real_sample"].append(f.is_real_sample)
            else:
                net_data["is_real_sample"].append(f.is_real_sample)
        for key in net_data:
            shape = [(len(all_features))]
            temp = net_data[key][0]
            if isinstance(temp, np.ndarray):
                net_data[key] = np.array(net_data[key])
            while isinstance(temp, list) or isinstance(temp, np.ndarray):
                if isinstance(temp, list):
                    while isinstance(temp, list):
                        shape.append(len(temp))
                        temp = temp[0]
                if isinstance(temp, np.ndarray):
                    while isinstance(temp, np.ndarray):
                        shape.append(temp.shape[0])
                        temp = temp[0]
            net_data[key] = tf.constant(net_data[key], shape=shape)
        tf_data = tf.data.Dataset.from_tensor_slices(net_data)
        if is_train:
            try:
                tf_data = tf_data.shuffle(buffer_size=(len(all_features)))
            except:
                tf_data = tf_data.prefetch(10000).shuffle(10000)
        try:
            tf_data = tf_data.map(lambda x: x, num_parallel_calls)
        except:
            tf_data = tf_data.map(lambda x: x)
        try:
            num_batch = len(all_features) // batch_size
            if not drop_remainder and len(all_features) % batch_size != 0:
                num_batch += 1
            if padded_shapes is None:
                tf_data = tf_data.batch(batch_size, drop_remainder)
            else:
                tf_data = tf_data.padded_batch(batch_size, padded_shapes, drop_remainder=drop_remainder)
        except:
            num_batch = len(all_features) // batch_size
            if len(all_features) % batch_size != 0:
                num_batch += 1
            if padded_shapes is None:
                tf_data = tf_data.batch(batch_size)
            else:
                tf_data = tf_data.padded_batch(batch_size,padded_shapes)
        if is_train:
            tf_data = tf_data.repeat()
        self.dataset = tf_data
        return tf_data, num_batch

    def __call__(self, all_features: List[InputFeatures], batch_size, is_train=True,
                 drop_remainder=False, num_parallel_calls=None,padded_shapes=None):
        '''
        :param all_features: the all data for network
        :param batch_size: batch size
        :param is_train:  is train set or not
        :param drop_remainder:  if len(all_features)%batch_size!=0, drop the next data or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :param padded_shapes: if each sample is not same length, this param should provide for padding
        :return:
            tf.data.Dataset, num_batch
        '''
        return self.wrapper(all_features, batch_size, is_train, drop_remainder,num_parallel_calls,padded_shapes)


class TFRecordWrapper:
    def __init__(self, file_path: str, feature_typing_fn: FeatureTypingFunctions, need_write=True):
        '''
        :param file_path: TFRecord file path
        :param feature_typing_fn:a FeatureTypingFunctions for net_x,net_y,is_real_sample
        '''
        if file_path is None or file_path == "":
            raise ValueError("the file_path should provide")
        else:
            if not os.path.exists(os.path.dirname(file_path)) and os.path.dirname(file_path)!="":
                os.makedirs(os.path.dirname(file_path))
            self.file_path = file_path
        if feature_typing_fn is None:
            raise ValueError("feature_typing_fn should provide")
        self.feature_typing_fn = feature_typing_fn
        if need_write:
            try:
                self.writer = tf.io.TFRecordWriter(self.file_path)
            except:
                try:
                    self.writer = tf.python_io.TFRecordWriter(self.file_path)
                except:
                    self.writer = None
        self.need_write = need_write
        self.dataset = None
        self.num_of_data = 0
        self.num_batch = 0
    
    def iter(self):
        try:
            if self.dataset is None:
                raise RuntimeError("NoneType can't iter")
            return iter(self.dataset)
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
    
    def __feature2dict(self, f):
        features = collections.OrderedDict()
        for x_key in f.net_x:
            if isinstance(f.net_x[x_key], list):
                try:
                    features[x_key] = self.feature_typing_fn.x_fns[x_key](f.net_x[x_key])
                except TypeError as e:
                    raise TypeError(str(e))
            elif isinstance(f.net_x[x_key], np.ndarray):
                try:
                    features[x_key] = self.feature_typing_fn.x_fns[x_key](f.net_x[x_key].tolist())
                except TypeError as e:
                    raise TypeError(str(e))
            else:
                try:
                    features[x_key] = self.feature_typing_fn.x_fns[x_key]([f.net_x[x_key]])
                except TypeError as e:
                    raise TypeError(str(e))
        for y_key in f.net_y:
            if isinstance(f.net_y[y_key], list):
                try:
                    features[y_key] = self.feature_typing_fn.y_fns[y_key](f.net_y[y_key])
                except TypeError as e:
                    raise TypeError(str(e))
            elif isinstance(f.net_y[y_key], np.ndarray):
                try:
                    features[y_key] = self.feature_typing_fn.y_fns[y_key](f.net_y[y_key].tolist())
                except TypeError as e:
                    raise TypeError(str(e))
            else:
                try:
                    features[y_key] = self.feature_typing_fn.y_fns[y_key]([f.net_y[y_key]])
                except TypeError as e:
                    raise TypeError(str(e))
        features["is_real_sample"] = self.feature_typing_fn.is_real_sample_fn([f.is_real_sample])
        return features

    def write(self, input_features: Union[InputFeatures, List[InputFeatures]], is_complete=True):
        '''
        :param input_features: the sample(InputFeatures) list or one for to write to TFRecord
        :param batch_size: batch size
        :param num_of_data: the need write data num
        :param is_complete:  TFRecord is complete
        :return:
        '''
        if self.need_write:
            pass
        else:
            return
#             if num_parallel_calls is not None and num_parallel_calls > 0:
#                 batch_size = batch_size * num_parallel_calls
        if self.writer is None:
            try:
                writer = tf.io.TFRecordWriter(self.file_path)
            except:
                try:
                    writer = tf.python_io.TFRecordWriter(self.file_path)
                except:
                    raise RuntimeError("can't create the TFRecord writer")
            self.writer = writer
        else:
            writer = self.writer
        if isinstance(input_features, list):
            count = len(input_features)
            for f in input_features:
                features = self.__feature2dict(f)
                tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_sample.SerializeToString())
        else:
            count = 1
            features = self.__feature2dict(input_features)
            tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_sample.SerializeToString())
        if is_complete:
            writer.close()
            self.writer = None
        self.num_of_data += count
        return

    def __decode_record(self, record):
        '''
        :param record: one data from TFRcord file
        :return: the train example
        '''
        try:
            sample = tf.io.parse_single_example(record, self.feature_typing_fn.name_to_features)
        except:
            sample = tf.parse_single_example(record, self.feature_typing_fn.name_to_features)
        for name in list(sample.keys()):
            t = sample[name]
            if isinstance(t, tf.SparseTensor):
                tf.sparse.to_dense(t)
            if t.dtype == tf.int64:
                t = tf.cast(t,tf.int32)
            sample[name] = t
        return sample

    def read(self, is_train: bool, batch_size, buffer_size, drop_remainder=False, num_parallel_calls=None, 
             padded_shapes=None):
        '''
        :param is_train: is train set or not,if is train set, it will be repeat and shuffle
        :param batch_size: batch size for cpu or one GPU
        :param drop_remainder:  if the set is less than batch_size or batch_size*gpu_num,drop it or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :param buffer_size: the shuffle buffer size, it bigger than the data num for the best effect
        :param padded_shapes: if each sample is not same length, this param should provide for padding
        :return:
            tf.data.Dataset, num_batch
        '''
#         if num_parallel_calls is not None and num_parallel_calls > 0:
#             batch_size = batch_size * num_parallel_calls
        tf_record = tf.data.TFRecordDataset(self.file_path)
        if is_train:
            # tf_record = tf_record.repeat()
            tf_record = tf_record.prefetch(buffer_size).shuffle(buffer_size=buffer_size)
        try:
            tf_record = tf_record.map(lambda record: self.__decode_record(record), num_parallel_calls)
        except:
            tf_record = tf_record.map(lambda record: self.__decode_record(record))
        try:
            num_batch = self.num_of_data // batch_size
            if not drop_remainder and self.num_of_data % batch_size != 0:
                num_batch += 1
            if padded_shapes is None:
                tf_record = tf_record.batch(batch_size, drop_remainder)
            else:
                tf_record = tf_record.padded_batch(batch_size, padded_shapes, drop_remainder=drop_remainder)
        except:
            num_batch = self.num_of_data // batch_size
            if self.num_of_data % batch_size != 0:
                num_batch += 1
            if padded_shapes is None:
                tf_record = tf_record.batch(batch_size)
            else:
                tf_record = tf_record.padded_batch(batch_size, padded_shapes)
        if is_train:
            tf_record = tf_record.repeat()
        self.dataset = tf_record
        self.num_batch = num_batch
        return tf_record, num_batch
