import tensorflow as tf

def binarize(x):
	# we also have to reassign the sign gradient otherwise it will be almost everywhere equal to zero
	# using the straight through estimator
	with tf.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
		#return tf.sign(x)				#	<-- wrong sign doesn't return +1 for zero
		return tf.sign(tf.sign(x)+1e-8) #	<-- this should be ok, ugly but okay


A = tf.constant([[1, 20, -13.89], [3.65, -21, 13]])
B = tf.clip_by_value(A, clip_value_min=0, clip_value_max=3)

w = tf.get_variable('weight', [3,10], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
w1 = tf.clip_by_value(w, -1, 1)

w1 = binarize(w1)


#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(w))
    print(sess.run(w1))