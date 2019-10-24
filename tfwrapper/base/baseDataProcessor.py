from .dataWrapper import *

class BaseDataProcessor:
    def __init__(self, features_typing_fn: FeatureTypingFunctions=None):
        self.features_typing_fn = features_typing_fn

    def create_samples(self,*args,**kwargs)->List[InputSample]:
        '''
        the input_x,input_y of InputSample's key should be same to the self.features_typing_fn
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError

    def samples2features(self,samples: List[InputSample],*args,**kwargs)->List[InputFeatures]:
        '''
        the net_x,net_y of InputFeatures' key should be same to the self.features_typing_fn
        :param samples:
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError

class GeneralDataProcessor(BaseDataProcessor):
    def __init__(self,input_size:int,output_size:int=None):
        if input_size is None:
            raise RuntimeError("input size should be provided")
        input_size = [input_size,]
        if output_size is None:
            output_size = []
        else:
            output_size = [output_size,]
        x_fns = {"x":FeatureTypingFunctions.float_feature}
        name_to_features = {
            "x":tf.io.FixedLenFeature(shape=input_size,dtype="float"),
            "y":tf.io.FixedLenFeature(shape=output_size,dtype=tf.int64),
            "is_real_sample":tf.io.FixedLenFeature(shape=[],dtype=tf.int64)
        }
        y_fns = {"y":FeatureTypingFunctions.int64_feature}
        feature_typing_fn = FeatureTypingFunctions(x_fns=x_fns,name_to_features=name_to_features,y_fns=y_fns)
        super(GeneralDataProcessor, self).__init__(feature_typing_fn)
    
    def create_samples(self, xs,ys):
        samples = []
        for guid,(x,y) in enumerate(zip(xs,ys)):
            sample = InputSample(guid=guid,input_x={"x":x},input_y={"y":y})
            samples.append(sample)
        return samples
    
    def samples2features(self, samples):
        features = []
        for sample in samples:
            feature = InputFeatures(net_x=sample.input_x,net_y=sample.input_y)
            features.append(feature)
        return features


class MnistDataProcessor(BaseDataProcessor):
    def __init__(self, features_typing_fn: FeatureTypingFunctions=None):
        if features_typing_fn is None:
            try:
                name_to_features={
                    "x": tf.io.FixedLenFeature(shape=[784,], dtype="float"),
                    "y": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                    "is_real_sample": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
                }
            except:
                name_to_features = {
                    "x": tf.FixedLenFeature(shape=[784, ], dtype="float"),
                    "y": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                    "is_real_sample": tf.FixedLenFeature(shape=[], dtype=tf.int64)
                }
            features_typing_fn = FeatureTypingFunctions({"x":FeatureTypingFunctions.float_feature},
                                                        name_to_features=name_to_features
                                                        , y_fns={"y": FeatureTypingFunctions.int64_feature})
        if features_typing_fn is None:
            raise ValueError("class FeatureTypingFunctions:features_typing_fn should provide")
        super(MnistDataProcessor, self).__init__(features_typing_fn)

    def create_samples(self, xs,ys):
        samples = []
        for index,(x,y) in enumerate(zip(xs,ys)):
            input_x = {"x": x}
            input_y = {"y": y}
            sample = InputSample(index, input_x, input_y)
            samples.append(sample)
        return samples

    def samples2features(self,samples: List[InputSample]):
        input_features = []
        for sample in samples:
            feature = InputFeatures(sample.input_x, sample.input_y, True)
            input_features.append(feature)
        return input_features