import tensorflow as tf
import numpy as np
from typing import Dict


class InputSample:
    def __init__(self, guid, input_x: Dict, input_y: Dict=None):
        '''
        :param guid: for unique the sample
        :param input_x: the sample data, e.g:{key:value,key:value}
        :param input_y: the sample answer e.g:{key:value,key:value}
        '''
        if guid is None:
            raise ValueError("the guid should provide")
        if input_x is None:
            raise ValueError("the input_x should provide")
        self.guid = guid
        self.input_x = input_x
        self.input_y = input_y


class FeatureTypingFunctions:
    def __init__(self, x_fns: Dict, name_to_features: Dict, y_fns: Dict=None, is_real_sample_fn=None,
                 feature_columns=None):
        '''
        :param x_fns:  the x_fns should be a Dict with same key to net_x in InputFeatures, x_fns value should be in
                        [int64_feature, float_feature, bytes_feature]
        :param name_to_features: type is Dict[str, Union[tf.FixedLenFeature, tf.VarLenFeature,
                                                         tf.FixedLenSequenceFeature, tf.SparseFeature,
                                                         tf.io.FixedLenFeature, tf.io.VarLenFeature,
                                                         tf.io.FixedLenSequenceFeature, tf.io.SparseFeature]]
        :param y_fns: same to x_fns
        :param is_real_sample_fn: a function in [int64_feature, float_feature, bytes_feature]
        :param feature_columns: a list of tf.feature_column.**_column, this used to convert dataset features 
                                to a input_layer for network (convert by tf.feature_column.input_layer function)
        '''
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
        self.feature_columns = feature_columns

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
        if net_x is None:
            raise ValueError("the net_x should provide")
        self.net_x = net_x
        self.net_y = net_y
        self.is_real_sample = is_real_sample

