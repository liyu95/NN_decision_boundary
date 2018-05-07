import tensorflow as tf
import numpy as np
from utils import *
from model_graph import model_graph, model_graph_asym
from sklearn.cross_validation import train_test_split
import os

test_size = 0.1
batch_size = 200
train_steps = 0
output_step = 1000

samples = 5000
margin = 0.1

# Iris data
# data = load_iris()
# feature = data.data
# label = data.target
# label[label==2]=1

# Random sample across the space
# feature, label = lin_sep_with_ground_truth(samples, margin, random_state=10)

# Blob data
# feature, label = generate_lin_sep_blobs(samples, random_state=10)

# Non linear data
# feature, label = non_lin_sep(samples, margin, random_state=10)

# sector data
# feature, label = generate_difficult_sectors(samples, random_state=10)
feature, label = generate_sectors_not_sep(samples, random_state=10)


# moon data

# feature, label = draw_from_moons(samples, random_state=10)

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


y_conv_logit, y_conv, x_transform = model_graph(x, y_)
# y_conv_logit, y_conv = model_graph_asym(x, y_)

with tf.name_scope('cross_entropy'):
	cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit))
	hinge_loss=tf.reduce_mean(tf.losses.hinge_loss(
		labels=y_, logits=y_conv_logit))
	mse=tf.reduce_mean(tf.losses.mean_squared_error(
		labels=y_, predictions=y_conv_logit))
	abd = tf.reduce_mean(tf.losses.absolute_difference(
		labels=y_, predictions=y_conv_logit))
	log_loss = tf.reduce_mean(tf.losses.log_loss(
		labels=y_, predictions=y_conv_logit))
	mpse = tf.reduce_mean(tf.losses.mean_pairwise_squared_error(
		labels=y_, predictions=y_conv_logit))
	tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    weight_collection=[v for v in 
    	tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weight_collection])
    theta_1 = 0
    cross_entropy_with_weight_decay=tf.add(cross_entropy,theta_1*l2_loss)

    train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_with_weight_decay)
    # train_op=tf.train.MomentumOptimizer(1e-4, 0.9).minimize(cross_entropy_with_weight_decay)
    # train_op = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy_with_weight_decay)
    # train_op = tf.train.GradientDescentOptimizer(
    # 	1e-4).minimize(cross_entropy_with_weight_decay)
    # train_op = tf.train.GradientDescentOptimizer(5e-4).minimize(hinge_loss)
    # train_op = tf.train.GradientDescentOptimizer(2e-4).minimize(mse)
    # train_op = tf.train.GradientDescentOptimizer(2e-4).minimize(abd)
    # train_op = tf.train.GradientDescentOptimizer(5e-4).minimize(log_loss)
    # train_op = tf.train.GradientDescentOptimizer(5e-4).minimize(mpse)


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

def train_model(train_op, train_steps):
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


clf = get_svm(feature_train, np.argmax(label_train,1))
support_ind = clf.support_
cross_entropy_output = sess.run(
	tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y_conv_logit),
	feed_dict={x: feature_train, 
			   y_: label_train})
loss_rank = np.argsort(cross_entropy_output)
# print(sorted(support_ind))
# print(sorted(loss_rank[-len(support_ind):]))


## visualize the prediction result
# pre_label_nn= sess.run(predicted_label,
# 	feed_dict={
# 	x: feature_test, 
# 	y_: label_test})

# pre_label_svm = clf.predict(feature_test)

# visual_separable_binary(feature_test, pre_label_svm,margin)
# visual_separable_binary(feature_test, pre_label_nn, margin)

# plot_blobs(feature, label,
# 	'../result/non_linear_demo/training_data.png')

## predict the random data, check the exact boundary
def plot_result_transform(start, end):
	feature_random, label_random = random_points(start,
		end, samples*20,
		random_state=100)
	label_random = to_categorical(label_random, 2)
	pre_label_nn, feature_random_transform= sess.run(
		[predicted_label, x_transform],
		feed_dict={
		x: feature_random, 
		y_: label_random})

	feature_transform= sess.run(
		x_transform,
		feed_dict={
		x: feature, 
		y_: label_hot})

	clf = get_svm(feature_transform, label)
	pre_label_svm = clf.predict(feature_random_transform)

	# visual_separable_binary(feature_random, pre_label_svm,0.0001,
	# 	'../result/non_linear_demo/svm_cross_entropy.png')
	# visual_separable_binary(feature_random, pre_label_nn, 0.0001,
	# 	'../result/non_linear_demo/nn_cross_entropy.png')

	# plot_difficult_sectors(feature, label,
	# 	'../result/bolb_softsign/training_data.png')
	plot_blobs(feature, label,
		'../result/exploration/Training_data.png')
	plot_blobs_all_together(feature_random_transform, pre_label_svm,
		feature_transform, label,
		'../result/exploration/SVM_decision_boundary_transform.png')
	plot_blobs_all_together(feature_random_transform, pre_label_nn,
		feature_transform, label,
		'../result/exploration/NN_decision_boundary_transform.png')
	boundary_overlay(feature_random_transform, pre_label_nn,
		feature_random_transform, pre_label_svm,
		'../result/exploration/Decision_boundary_overlay_transform.png')

def plot_result_original(start, end):
	feature_random, label_random = random_points(start,
		end, samples*20,
		random_state=100)
	label_random = to_categorical(label_random, 2)
	pre_label_nn= sess.run(
		predicted_label,
		feed_dict={
		x: feature_random, 
		y_: label_random})

	feature_transform= sess.run(
		x_transform,
		feed_dict={
		x: feature, 
		y_: label_hot})

	pre_label_svm = clf.predict(feature_random)

	# plot_difficult_sectors(feature, label,
	# 	'../result/bolb_softsign/training_data.png')
	plot_blobs(feature, label,
		'../result/exploration/Training_data.png')
	plot_blobs_all_together(feature_random, pre_label_svm,
		feature, label,
		'../result/exploration/SVM_decision_boundary.png')
	plot_blobs_all_together(feature_random, pre_label_nn,
		feature, label,
		'../result/exploration/NN_decision_boundary.png')
	boundary_overlay(feature_random, pre_label_nn,
		feature_random, pre_label_svm,
		'../result/exploration/Decision_boundary_overlay.png')


if __name__ == '__main__':
	train_model(train_op, 100001)
	plot_result_original(-100,500)
	plot_result_transform(-100,500)