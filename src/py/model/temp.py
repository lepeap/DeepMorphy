import tensorflow as tf

sess = tf.InteractiveSession()

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


t1 = tf.constant(
    [
        [8,8],
        [9,9]
    ]
)

t2 = tf.constant([[0],[0]])

encoder_output = tf.concat(values=[t1, t2], axis=1)

rez = encoder_output.eval()
sess.close()
print()
