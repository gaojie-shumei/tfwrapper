import numpy as np
import tensorflow as tf


class Auc(object):
    def __init__(self, tf_mode=False):
        self.tf_mode = tf_mode
        self.parameters = None
        return

    def tf_auc(self, label, score, pos_label=1):
        label_new = tf.reshape(label, (-1, ))
        score_new = tf.reshape(score, (-1, ))
        score_asc_index = tf.argsort(score_new)
        pos_num = tf.count_nonzero(label_new, dtype="float")
        neg_num = tf.count_nonzero(1-label_new, dtype="float")
        # rank = tf.range(1, pos_num+neg_num+1)
        label_new_re_index = tf.gather(label_new, score_asc_index)
        pos_rank = tf.where(tf.equal(label_new_re_index, pos_label)) + 1
        overup = (tf.cast(tf.reduce_sum(pos_rank), "float") - (pos_num*(pos_num+1)/2))
        total = tf.multiply(pos_num, neg_num)
        total = tf.cond(total > 0, lambda: total, lambda: 1.0)
        auc_score = overup / total
        self.parameters = tf.global_variables()
        return auc_score

    def auc(self, label, score, pos_label):
        '''
        :param self:
        :param label: an array, the element is the label for every sample
        :param score: an array, score for judging the sample to positive sample
        :param pos_label: a class,which is the positive label
        :return: the auc score
        '''
        label = np.array(label)
        score = np.array(score)

        pos_num, nvi_num = 0, 0
        rank = 0
        score_label = list(zip(score, label))
        score_label.sort(reverse=True)
        sample_num = label.shape[0]
        for i in range(len(score_label)):
            _,lb = score_label[i]
            if lb == pos_label:
                pos_num += 1
                rank += sample_num-i
            else:
                nvi_num += 1
        auc_score = (rank - pos_num*(pos_num+1)/2)/((pos_num*nvi_num) if pos_num*nvi_num!=0 else 1)
        return auc_score

    def __call__(self, label, score, pos_label=1):
        if self.tf_mode:
            return self.tf_auc(label, score, pos_label)
        else:
            return self.auc(label, score, pos_label)

