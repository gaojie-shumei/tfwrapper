'''
Created on 2019年9月28日

@author: gaojie-202
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tfwrapper.base.dataWrapper import *
from tfwrapper.base import baseDataProcessor


def create_model():
    inputs = keras.Input(shape=(784,))
    out = inputs
    out = keras.layers.Dense(units=128,activation="relu")(out)
#     out = keras.layers.Dense(units=512,activation="relu")(out)
    out = keras.layers.Dense(units=10,activation="softmax")(out)
    model = keras.Model(inputs=inputs,outputs=out)
    return model

def compute(x,y,model):
    out = model(x)
    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, out))
    acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.argmax(out,axis=-1,output_type=tf.int32)),"float"))
    return out,loss,acc

def train(x_train, y_train, x_test, y_test, train_num=100, learning_rate=0.0005, batch_size=128):
    mnist_data_processor = baseDataProcessor.MnistDataProcessor()
    train_samples = mnist_data_processor.create_samples(x_train,y_train)
    train_features = mnist_data_processor.samples2features(train_samples)
    test_samples = mnist_data_processor.create_samples(x_test, y_test)
    test_features = mnist_data_processor.samples2features(test_samples)
    tr_wrapper,t_wrapper = TFDataWrapper(),TFDataWrapper()
    tr_dataset = tr_wrapper(train_features, batch_size, is_train=True, drop_remainder=False,num_parallel_calls=1)
    t_dataset = t_wrapper(test_features, batch_size, is_train=False, drop_remainder=False,num_parallel_calls=1)
    print(tr_dataset)
    optimizer = keras.optimizers.Adam(learning_rate)
    model = create_model()
    for i in range(train_num):
        for data in tr_wrapper.iter():
            with tf.GradientTape() as tape:
                batch_x,batch_y = data["x"],data["y"]
                out,loss,acc = compute(batch_x,batch_y,model)
                grad = tape.gradient(loss,model.trainable_variables)
                optimizer.apply_gradients(zip(grad,model.trainable_variables))
        t_loss,t_acc,count = 0,0,0
        for data in t_wrapper.iter():
            batch_x,batch_y = data["x"],data["y"]
            _,loss,acc = compute(batch_x,batch_y,model)
            t_loss += float(loss)
            t_acc += float(acc)
            count += 1
        t_loss /= count
        t_acc /= count
        print("epoch={:d}, t_loss={:f}, t_acc={:f}".format(i,t_loss,t_acc))
        

        
            
    
    

def read_mnist_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def main():
    mnist_path = "../data/mnist.npz"
    (x_train, y_train), (x_test, y_test) = read_mnist_data(mnist_path)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(y_train.dtype, y_test.dtype)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    train(x_train, y_train, x_test, y_test, train_num=100, learning_rate=0.0001, batch_size=128)


if __name__ == '__main__':
    main()
    