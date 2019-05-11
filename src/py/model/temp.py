from utils import config
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
conf = config()

length = tf.constant([4,5,6])
mask = tf.sequence_mask([4,5,6]).eval()

#rez = tf.range(length).eval()

#chars = tf.constant(
#    [
#        ["п","y","о"],
#        ["а", "б", "в", "0"]
#    ]
#)
table = tf.contrib.lookup.index_table_from_tensor(
            mapping=conf['chars'],
            num_oov_buckets=1,
            default_value=0
)
tf.tables_initializer().run()


strs = tf.placeholder(tf.string, shape=(None, ))
rez_strs = tf.reshape(strs, (-1,))
rez = table.lookup(rez_strs)
rez = rez.eval({
    strs: np.asarray([
        ["п","y","о"],
        ["а", "б", "в", "0"]
    ])
})

print()

print()
#t1 = tf.constant([
#    [
#        [8,8], [9,9]
#    ],
#    [
#        [7,7], [6,6]
#    ]
#])
#
#t2 = tf.constant([
#    [
#        [0,0]
#    ],
#    [
#        [0,0]
#    ]
#])
#
#encoder_output = tf.concat(values=[t1, t2], axis=1)


#t1 = tf.constant(
#    [
#        [8,8],
#        [9,9]
#    ]
#)
#
#t2 = tf.constant([[0],[0]])
#
#encoder_output = tf.concat(values=[t1, t2], axis=1)
#
#rez = encoder_output.eval()
sess.close()
print()
