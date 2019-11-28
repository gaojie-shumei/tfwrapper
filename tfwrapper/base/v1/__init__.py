#version for tensorflow version less than 2
from tfwrapper.base.v1.baseNet import BaseNet
from tfwrapper.base.v1.dataWrapper import InputSample,InputFeatures,FeatureTypingFunctions,TFDataWrapper,TFRecordWrapper
from tfwrapper.base.v1.baseDataProcessor import BaseDataProcessor,MnistDataProcessor,GeneralDataProcessor
from tfwrapper.base.v1.tfmodel import TFModel
from tfwrapper.base.v1 import baseDataProcessor,baseNet,tfmodel,dataWrapper