#!/usr/bin/env python3

from make_mnist_cnn_tf import build_cnn_mnist_model, reset_graph
import tensorflow as tf
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epsmin', type=float, default=0.01)
parser.add_argument('--epsmax', type=float, default=0.2)
parser.add_argument('--idx', type=int, default=100)
parser.add_argument('--numgens', type=int, default=1000)
parser.add_argument('--impdims', default='true')
parser.add_argument('--perdims', type=float, default=0.8)

args = parser.parse_args()

reset_graph()
x = tf.placeholder(tf.float32, shape=(None, 28, 28))
y = tf.placeholder(tf.int32, shape=(None,))
model = build_cnn_mnist_model(x, y, False)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / np.float32(255)
y_train = y_train.astype(np.int32)
x_test = x_test / np.float32(255)
y_test = y_test.astype(np.int32)

grad, = tf.gradients(model['loss'], x)
mask_pos = tf.placeholder(tf.bool)
rand_fill = tf.placeholder(tf.float32)
grad_sign = tf.sign(grad)
mask_neg = tf.math.logical_not(mask_pos)
grad_zero = tf.multiply(grad_sign, tf.cast(mask_neg,tf.float32))
rand_zero = tf.multiply(rand_fill, tf.cast(mask_pos,tf.float32))
grad_final = tf.math.add(grad_zero, rand_zero)

epsilon = tf.placeholder(tf.float32)
optimal_perturbation = tf.multiply(grad_final, epsilon)
adv_example_unclipped = tf.add(optimal_perturbation, x)
adv_example = tf.clip_by_value(adv_example_unclipped, 0.0, 1.0)

classes = tf.argmax(model['probability'], axis=1)

adv_examples = []
idx = args.idx
epsilon_range = (args.epsmin, args.epsmax)


counts = np.zeros((np.max(y_train)+1,))
averages = np.zeros((np.max(y_train) + 1,) + x_train[0].shape)

for i in range(x_train.shape[0]):
    c = y_train[i]
    averages[c] += x_train[i]
    counts[c] += 1

for i in range(averages.shape[0]):
    averages[i] /= counts[i]

with tf.Session() as sess:
    saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
    acc_test = model['accuracy'].eval(feed_dict={
        x: x_test,
        y: y_test,
    })
    print('Accuracy of model on test data: {}'.format(acc_test))
    print('Correct Class: {}'.format(y_train[idx]))
    class_x = classes.eval(feed_dict={x: x_train[idx:idx + 1]})
    print('Predicted class of input {}: {}'.format(idx, class_x))
    distances = np.zeros((averages.shape[0],))
    for i in range(averages.shape[0]):
        if i == class_x:
            continue
        d = np.linalg.norm((x_train[idx] - averages[i]).flatten())
        distances[i] = d
    distances[class_x] = np.max(distances)+1
    class_min_dist = np.argmin(distances)
    diff_imp = x_train[idx] - averages[class_min_dist]
    sorted_indices = np.array(np.unravel_index(np.argsort(diff_imp, axis=None)[::-1], diff_imp.shape))
    if(args.impdims == 'true'):
        filled_mask = np.zeros_like(diff_imp)
        fill = 1.0
    else:
        filled_mask = np.ones_like(diff_imp)
        fill = 0.0
    for i in range(int(diff_imp.size * args.perdims)):
        filled_mask[tuple(sorted_indices[:,i])] = fill

    filled_mask = filled_mask.astype(bool)
    
    start = time.time()
    for i in range(args.numgens):
        adv = adv_example.eval(
            feed_dict={
                x: x_train[idx:idx + 1],
                y: y_train[idx:idx + 1],
                mask_pos: filled_mask,
                rand_fill: np.random.choice([-1.0,0.0,1.0],size=x_train[idx].shape,p=[0.4,0.2,0.4]),
                epsilon: np.random.uniform(
                    epsilon_range[0], epsilon_range[1],
                    # size=(28, 28)
                    )
            })
        class_adv = classes.eval(feed_dict={x: adv})
        if class_adv != y_train[0]:
            adv_examples += [adv]
    print('Duration (s): {}'.format(time.time() - start))
adv_examples = np.concatenate(adv_examples, axis=0)
print('Found {} adversarial examples.'.format(adv_examples.shape[0]))
print('Percentage true adversarial examples: {}'.format(adv_examples.shape[0]/args.numgens))
avg = np.zeros_like(x_train[idx])
for i in range(adv_examples.shape[0]):
    avg += adv_examples[i]
avg /= adv_examples.shape[0]
stddev = 0
for i in range(adv_examples.shape[0]):
    tmp = adv_examples[i] - avg
    tmp = np.square(tmp)
    stddev += np.sum(tmp) / tmp.size

stddev /= adv_examples.shape[0]
print('Found std dev: {}'.format(stddev))

