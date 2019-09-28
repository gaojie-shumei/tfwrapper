'''
Created on 2019年9月6日

@author: gaojie-202
'''
from tfwrapper.base.v1.dataWrapper import *
from tfwrapper.base.v1 import baseDataProcessor
from tfwrapper.base.v1 import tfmodel,baseNet


class SampleMnistNet(baseNet.BaseNet):
    def __init__(self, layers=None):
        super(SampleMnistNet, self).__init__(layers)

    def net(self, inputs):
        try:
            outputs = inputs
            outputs = tf.layers.dense(inputs=outputs, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.nn.l2_loss)
            outputs= tf.layers.dense(inputs=outputs, 
                                  units=512, 
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.nn.l2_loss)
            outputs= tf.layers.dense(inputs=outputs, 
                                    units=10, 
                                    activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.nn.l2_loss)
            log_outputs = tf.nn.softmax(outputs)
        except:
            raise RuntimeError("tensorflow version should be less than 2")
        return outputs,log_outputs



def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

with tf.device("/cpu:0"):
    x = tf.placeholder("float",shape=[None, 784], name="x")
    is_real_sample = tf.placeholder(tf.int32, shape=[None], name="is_real_sample")
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    lr = tf.placeholder("float")
    gpu_num = 0

def mnist_model():
    try:
        with tf.variable_scope("",reuse=tf.AUTO_REUSE):
            net = SampleMnistNet()
            if gpu_num != 0:
                _x = tf.split(x, num_or_size_splits=gpu_num, axis=0)
                _is_real_sample = tf.split(is_real_sample, num_or_size_splits=gpu_num, axis=0)
                _y = tf.split(y, num_or_size_splits=gpu_num, axis=0)
                out = []
                loss = 0
                acc = 0
                for i in range(gpu_num):
                    with tf.device("/gpu:%d"%i):
                        _out,_log_out = net(_x[i])
                        tf_index = tf.cast(_is_real_sample[i], "float")
    #                     _loss = tf.losses.sparse_softmax_cross_entropy(_y[i], _out, reduction=tf.losses.Reduction.NONE)
                        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y[i], logits=_out)
                        _loss = tf.reduce_sum(tf.multiply(tf_index, _loss)) / tf.reduce_sum(tf_index)
                        _acc = tf.cast(tf.equal(_y[i], tf.argmax(_log_out, axis=-1, output_type=tf.int32)), "float")
                        _acc = tf.reduce_sum(tf.multiply(tf_index, _acc)) / tf.reduce_sum(tf_index)
                    out.append(_out)
                    loss += _loss
                    acc += _acc
                out = tf.concat(out, axis=0)
                loss /= gpu_num
                acc /= gpu_num
            else:
                out,log_out = net(x)
                tf_index = tf.cast(is_real_sample, "float")
    #             loss = tf.losses.sparse_softmax_cross_entropy(y, out, reduction=tf.losses.Reduction.NONE)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=out)
                loss = tf.reduce_sum(tf.multiply(tf_index, loss)) / tf.reduce_sum(tf_index)
                acc = tf.cast(tf.equal(y, tf.argmax(log_out, axis=-1, output_type=tf.int32)), "float")
                acc = tf.reduce_sum(tf.multiply(tf_index, acc)) / tf.reduce_sum(tf_index)
    except:
        raise RuntimeError("tensorflow version should be less than 2")
    return out, loss, acc
try:
    with tf.device("/cpu:0"):
        out, loss, acc = mnist_model()
        optimizer = tf.train.AdamOptimizer(lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_ops = optimizer.minimize(loss)
        model = tfmodel.TFModel([x,is_real_sample], out, y, loss, train_ops, net_configs=lr, metrics=acc, num_parallel_calls=1)
except:
    raise RuntimeError("tensorflow version should be less than 2")

def train(x_train,y_train,x_test,y_test,train_num,learning_rate,batch_size):
    try:
        with tf.device("/cpu:0"):
            mnist_data_processor = baseDataProcessor.MnistDataProcessor()
            train_samples = mnist_data_processor.creat_samples(x_train,y_train)
            train_features = mnist_data_processor.samples2features(train_samples)
            test_samples = mnist_data_processor.creat_samples(x_test, y_test)
            test_features = mnist_data_processor.samples2features(test_samples)
            wrapper = TFDataWrapper()
            _, train_data, train_init = wrapper(train_features, batch_size, is_train=True, drop_remainder=False,
                                                num_parallel_calls=1)
            _, test_data, test_init = wrapper(test_features, batch_size, is_train=False, drop_remainder=False,
                                              num_parallel_calls=1)
        #     wrapper = TFRecordWrapper("../data/train.tf_record",mnist_data_processor.features_typing_fn, False)
        #     wrapper1 = TFRecordWrapper("../data/test.tf_record", mnist_data_processor.features_typing_fn, False)
        #     wrapper.write(train_features, batch_size, len(train_features))
        #     wrapper1.write(test_features, batch_size, len(test_features))
        #     _, train_data, train_init = wrapper.read(True, batch_size)
        #     _, test_data, test_init = wrapper1.read(False, batch_size)
            tr_inputs_feed = [train_data["x"], train_data["is_real_sample"]]
            tr_outputs_feed = train_data["y"]
            tr_net_configs_feed = learning_rate
            
            v_inputs_feed = [test_data["x"], test_data["is_real_sample"]]
            v_outputs_feed = test_data["y"]
            v_net_configs_feed = learning_rate
            
            model.fit(None, train_num, tr_inputs_feed, tr_outputs_feed, tr_net_configs_feed, v_inputs_feed, v_outputs_feed, v_net_configs_feed, 
                      batch_size, return_outputs=False, show_result=True, start_save_model_epoch=None, tr_tf_dataset_init=train_init, 
                      v_tf_dataset_init=test_init)
    except:
        raise RuntimeError("tensorflow version should be less than 2")
    
    
def main():
    with tf.device("/cpu:0"):
        mnist_path = "../data/mnist.npz"
        # get data
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        print(y_train.dtype, y_test.dtype)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        train(x_train, y_train, x_test, y_test, train_num=100, learning_rate=0.001, batch_size=128)


if __name__ == '__main__':
    main()
    
    
