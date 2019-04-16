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

args = parser.parse_args()

reset_graph()
x = tf.placeholder(tf.float32, shape=(None, 28, 28))
y = tf.placeholder(tf.int32, shape=(None,))
model = build_cnn_mnist_model(x, y, False)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / np.float32(255)
y_train = y_train.astype(np.int32)
x_test = x_test / np.float32(255)
y_test = y_test.astype(np.int32)

classes = tf.argmax(model['probability'], axis=1)

adv_examples = []
idx = args.idx
epsilon_range = (args.epsmin, args.epsmax)

adv_x = jsma_symbolic(
    x,
    model,
    theta=1.0,
    gamma=0.1,
    clip_min=0.0,
    clip_max=1.0,
)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
with tf.Session(config=config) as sess:
    saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
    acc_test = model['accuracy'].eval(feed_dict={
        x: x_test,
        y: y_test,
    })
    print('Accuracy of model on test data: {}'.format(acc_test))
    print('Correct Class: {}'.format(y_train[idx]))
    class_x = classes.eval(feed_dict={x: x_train[idx:idx + 1]})
    print('Predicted class of input {}: {}'.format(idx, class_x))
    start = time.time()
    adv_example = adv_x.eval(feed_dict={
        x: x_train[idx:idx+1],
        y: y_train[idx:idx+1],
    })
    print('adv_example', adv_example.shape)
    for i in range(args.numgens):

        class_adv = classes.eval(feed_dict={x: adv})
        if class_adv != y_train[0]:
            adv_examples += [adv]
    print('Duration (s): {}'.format(time.time() - start))
adv_examples = np.concatenate(adv_examples, axis=0)
print('Found {} adversarial examples.'.format(adv_examples.shape[0]))
