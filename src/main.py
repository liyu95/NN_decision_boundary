import tensorflow as tf
import numpy as np
from utils import *
from model_graph import model_graph
from sklearn.cross_validation import train_test_split
import os

test_size = 0.1
batch_size = 20
train_steps = 10000
output_step = 1000

data = load_iris()
feature = data.data
label = data.target
label[label==2]=1

label_hot = to_categorical(label, 2)
feature_train, feature_test, label_train, label_test = train_test_split(
	feature, label_hot, test_size=test_size, random_state=0)

os.environ["CUDA_VISIBLE_DEVICES"]='0'
config = tf.ConfigProto()
config.log_device_placement=False
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

with tf.name_scope('placeholder'):
    x=tf.placeholder(tf.float32,shape=[None, feature.shape[1]])
    y_=tf.placeholder(tf.float32,shape=[None, label_hot.shape[1]])


y_conv_logit, y_conv = model_graph(x, y_)

with tf.name_scope('cross_entropy'):
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit))
	tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    weight_collection=[v for v in 
    	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
    theta_1 = 0
    cross_entropy_with_weight_decay=tf.add(cross_entropy,theta_1*l2_loss)
    train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_with_weight_decay)
    # works for one layer
    # train_op=tf.train.GradientDescentOptimizer(5e-5).minimize(cross_entropy_with_weight_decay)
    # works for two layer
    # train_op=tf.train.GradientDescentOptimizer(3e-5).minimize(cross_entropy_with_weight_decay)

#DEFINE EVALUATION
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        actual_label=tf.argmax(y_,1)
        predicted_label=tf.argmax(y_conv,1)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

merged=tf.summary.merge_all()

if not os.path.isdir('../log'):
    os.system('mkdir ../log')
    os.system('mkdir ../log/train')
    os.system('mkdir ../log/test')

train_writer=tf.summary.FileWriter('../log/train/',sess.graph)
test_writer=tf.summary.FileWriter('../log/test/')

sess.run(tf.global_variables_initializer())

feature_train_obj = batch_object(feature_train, batch_size)
label_train_obj = batch_object(label_train, batch_size)

def train_model(train_op):
	for i in range(train_steps):
		feature_train_batch = feature_train_obj.next_batch()
		label_train_batch = label_train_obj.next_batch()
		if i%output_step == 0:
			summary,cross_entropy_output,acc = sess.run([
				merged ,cross_entropy,accuracy],feed_dict={
				x: feature_train_batch, 
				y_: label_train_batch})
			# print("step %d, training accuracy %g"%(i, acc))
			print("step %d, training loss %g"%(i, cross_entropy_output))

			summary,acc, pre_label= sess.run([merged,accuracy,predicted_label],
				feed_dict={
				x: feature_test, 
				y_: label_test})
			test_writer.add_summary(summary,i)
			print("step %d, test accuracy %g"%(i, acc))
			# print(pre_label)

		summary,_ = sess.run([merged,train_op],feed_dict={
			x: feature_train_batch, 
			y_: label_train_batch})
		train_writer.add_summary(summary,i)

train_model(train_op)

support_ind = get_support_ind(feature_train, np.argmax(label_train,1))
cross_entropy_output = sess.run(
	tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit),
	feed_dict={x: feature_train, 
			   y_: label_train})
loss_rank = np.argsort(cross_entropy_output)
print(support_ind)
print(loss_rank[-10:])