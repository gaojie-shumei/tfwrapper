import tensorflow as tf


class BaseNet(object):
    '''
    net  if you extends this class, please implement this function
    __init__: if you have feature to the net, you can set it with this function
    you can extends this class and implement net function to create your net,
    and use the modelModule to wrapper your net to a model for train, predict, evaluation
    '''
    def __init__(self, layers=None):
        '''
        if the subclass has private features, maybe you can set it in this function
        :param layers: the netlayers to compute the outputs
        '''
        super(BaseNet, self).__init__()
        self._layers = layers
        self._parameters = None

    @property
    def layers(self):
        return self._layers

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self,vars):
        self._parameters = vars

    def net(self, inputs):
        try:
            outputs = inputs
            if self.layers is not None:
                if isinstance(self.layers, list):
                    for layer in self.layers:
                        outputs = layer(outputs)
                else:
                    outputs = self.layers(outputs)
        except:
            raise RuntimeError("tensorflow version must be less than 2,such as 1.13.1")
        return outputs

    def __call__(self, inputs):
        try:
            outputs = self.net(inputs)
            self.parameters = tf.global_variables()
        except:
            raise RuntimeError("tensorflow version must be less than 2,such as 1.13.1")
        return outputs

