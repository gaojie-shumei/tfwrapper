import tensorflow as tf
from typing import List, Union, Dict
import collections
import numpy as np
import os


class InputSample:
    def __init__(self, guid, input_x: Dict, input_y: Dict=None):
        '''
        :param guid: for unique the sample
        :param input_x: the sample data, e.g:{key:value,key:value}
        :param input_y: the sample answer e.g:{key:value,key:value}
        '''
        try:
            if guid is None:
                raise ValueError("the guid should provide")
            if input_x is None:
                raise ValueError("the input_x should provide")
            self.guid = guid
            self.input_x = input_x
            self.input_y = input_y
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")


class FeatureTypingFunctions:
    def __init__(self, x_fns: Dict, name_to_features: Dict, y_fns: Dict=None, is_real_sample_fn=None):
        '''
        :param x_fns:  the x_fns should be a Dict with same key to net_x in InputFeatures, x_fns value should be in
                        [int64_feature, float_feature, bytes_feature]
        :param name_to_features: type is Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature,
                                                         tf.FixedLenSequenceFeature, tf.SparseFeature,
                                                         tf.io.FixedLenFeature, tf.io.VarLenFeature,
                                                         tf.io.FixedLenSequenceFeature, tf.io.SparseFeature]]
        :param y_fns: same to x_fns
        :param is_real_sample_fn: a function in [int64_feature, float_feature, bytes_feature]
        '''
        try:
            if x_fns is None:
                raise ValueError("x_fns should provide")
            if name_to_features is None:
                raise ValueError("name_to_features should be provide")
            if is_real_sample_fn is None:
                is_real_sample_fn = FeatureTypingFunctions.int64_feature
            self.x_fns = x_fns
            self.y_fns = y_fns
            self.is_real_sample_fn = is_real_sample_fn
            if "is_real_sample" not in name_to_features:
                try:
                    name_to_features["is_real_sample"] = tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
                except:
                    name_to_features["is_real_sample"] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
            self.name_to_features = name_to_features
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")

    @classmethod
    def int64_feature(cls, values):
        data = np.array(values).reshape(-1).tolist()
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(data)))
        return f

    @classmethod
    def float_feature(cls, values):
        data = np.array(values).reshape(-1).tolist()
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(data)))
        return f

    @classmethod
    def bytes_feature(cls, values):
        data = np.array(values).reshape(-1).tolist()
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(data)))
        return f


class InputFeatures:
    def __init__(self, net_x: Dict, net_y: Dict=None, is_real_sample: bool=True):
        '''
        :param net_x: this should be a Dict converted by input_x in class InputSample, if is PadSample,
                      this should be a Dict same to input_x but value for padsample
        :param net_y: this should be a Dict converted by input_y in class InputSample, if is PadSample,
                      this should be a Dict same to input_x but value for padsample
        :param is_real_sample:  is InputSample True, is PadSample False
        '''
        try:
            if net_x is None:
                raise ValueError("the net_x should provide")
            self.net_x = net_x
            self.net_y = net_y
            self.is_real_sample = is_real_sample
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")

class TFDataWrapper:
    def __init__(self):
        self._tf_data = None
        super(TFDataWrapper, self).__init__()
    
    def iter(self):
        try:
            if self._tf_data is None:
                raise RuntimeError("NoneType can't iter")
            return iter(self._tf_data)
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
    
    def wrapper(self,all_features: List[InputFeatures], batch_size, is_train=True,
                drop_remainder=False, num_parallel_calls=None):
        '''
        :param all_features: the all data for network
        :param batch_size: batch size
        :param is_train:  is train set or not
        :param drop_remainder:  if len(all_features)%batch_size!=0, drop the next data or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :return:
            tf.data.Dataset,data with tensor,iterator_init
        '''
        try:
            if num_parallel_calls is not None and num_parallel_calls > 0:
                batch_size = batch_size * num_parallel_calls
            net_data = {}
            if len(all_features) % batch_size != 0:
                res = batch_size - len(all_features) % batch_size
            else:
                res = 0
            for i in range(len(all_features) + res):
                if i < len(all_features):
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
                else:
                    f = all_features[i-len(all_features)]
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
                        net_data["is_real_sample"].append(False)
                    else:
                        net_data["is_real_sample"].append(False)
            for key in net_data:
                shape = [(len(all_features) + res)]
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
                # tf_data = tf_data.repeat()
                tf_data = tf_data.shuffle(buffer_size=100)
            try:
                tf_data = tf_data.map(lambda x: x, num_parallel_calls)
            except:
                tf_data = tf_data.map(lambda x: x)
            try:
                tf_data = tf_data.batch(batch_size, drop_remainder)
            except:
                tf_data = tf_data.batch(batch_size)
            self._tf_data = tf_data
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
        return

    def __call__(self, all_features: List[InputFeatures], batch_size, is_train=True,
                 drop_remainder=False, num_parallel_calls=None):
        '''
        :param all_features: the all data for network
        :param batch_size: batch size
        :param is_train:  is train set or not
        :param drop_remainder:  if len(all_features)%batch_size!=0, drop the next data or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :return:
            tf.data.Dataset,data with tensor,iterator_init
        '''
        self.wrapper(all_features, batch_size, is_train, drop_remainder,num_parallel_calls)
    
    


class TFRecordWrapper:
    def __init__(self, file_path: str, feature_typing_fn: FeatureTypingFunctions, need_write=True):
        '''
        :param file_path: TFRecord file path
        :param feature_typing_fn:a FeatureTypingFunctions for net_x,net_y,is_real_sample
        '''
        try:
            if file_path is None or file_path == "":
                raise ValueError("the file_path should provide")
            else:
                self.file_path = file_path
            if feature_typing_fn is None:
                raise ValueError("feature_typing_fn should provide")
            self.feature_typing_fn = feature_typing_fn
            if need_write:
                try:
                    self.writer = tf.io.TFRecordWriter(self.file_path)
                except:
                    self.writer = tf.python_io.TFRecordWriter(self.file_path)
            self.need_write = need_write
            self._tf_record = None
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
    
    def iter(self):
        try:
            if self._tf_data is None:
                raise RuntimeError("NoneType can't iter")
            return iter(self._tf_record)
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
    
    def __feature2dict(self, f):
        try:
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
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
        return features

    def write(self, input_features: Union[InputFeatures, List[InputFeatures]], batch_size, num_of_data,
              is_complete=True, num_parallel_calls=None):
        '''
        :param input_features: the sample(InputFeatures) list or one for to write to TFRecord
        :param batch_size: batch size
        :param num_of_data: the need write data num
        :param is_complete:  TFRecord is complete
        :param num_parallel_calls: the parallel nums, if None, one thread
        :return:
        '''
        try:
            if self.need_write:
                pass
            else:
                return num_of_data
            if num_parallel_calls is not None and num_parallel_calls > 0:
                batch_size = batch_size * num_parallel_calls
            if self.writer is None:
                try:
                    writer = tf.io.TFRecordWriter(self.file_path)
                except:
                    writer = tf.python_io.TFRecordWriter(self.file_path)
                self.writer = writer
            else:
                writer = self.writer
            if isinstance(input_features, list):
                for f in input_features:
                    features = self.__feature2dict(f)
                    tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tf_sample.SerializeToString())
            else:
                features = self.__feature2dict(input_features)
                tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_sample.SerializeToString())
            if is_complete:
                if num_of_data % batch_size != 0:
                    res = batch_size - num_of_data % batch_size
                else:
                    res = 0
                for i in range(res):
                    if isinstance(input_features, list):
                        f = input_features[i % len(input_features)]
                    else:
                        f = input_features
                    features = self.__feature2dict(f)
                    features["is_real_sample"] = self.feature_typing_fn.is_real_sample_fn([False])
                    tf_sample = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tf_sample.SerializeToString())
                writer.close()
                self.writer = None
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
        return num_of_data

    def __decode_record(self, record):
        '''
        :param record: one data from TFRcord file
        :return: the train example
        '''
        try:
            try:
                sample = tf.io.parse_single_example(record, self.feature_typing_fn.name_to_features)
            except:
                sample = tf.parse_single_example(record, self.feature_typing_fn.name_to_features)
            for name in list(sample.keys()):
                t = sample[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t,tf.int32)
                sample[name] = t
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
        return sample

    def read(self, is_train: bool, batch_size, drop_remainder=False, num_parallel_calls=None):
        '''
        :param is_train: is train set or not,if is train set, it will be repeat and shuffle
        :param batch_size: batch size for cpu or one GPU
        :param drop_remainder:  if the set is less than batch_size or batch_size*gpu_num,drop it or not
        :param num_parallel_calls: the data process with thread,if None,one thread
        :return:
            tf.data.Dataset,data with tensor,iterator_init
        '''
        try:
            if num_parallel_calls is not None and num_parallel_calls > 0:
                batch_size = batch_size * num_parallel_calls
            tf_record = tf.data.TFRecordDataset(self.file_path)
            if is_train:
                # tf_record = tf_record.repeat()
                tf_record = tf_record.shuffle(buffer_size=100)
            try:
                tf_record = tf_record.map(lambda record: self.__decode_record(record), num_parallel_calls)
            except:
                tf_record = tf_record.map(lambda record: self.__decode_record(record))
            try:
                tf_record = tf_record.batch(batch_size, drop_remainder)
            except:
                tf_record = tf_record.batch(batch_size)
            self._tf_record = tf_record
        except:
            raise RuntimeError("tensorflow version must be more than 2,such as 2.0.0rc1")
        return
    
    
    