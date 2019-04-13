from make_mnist_cnn_tf import build_cnn_mnist_model, reset_graph
import tensorflow as tf
import numpy as np
import time

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
epsilon = tf.placeholder(tf.float32)
optimal_perturbation = tf.multiply(tf.sign(grad), epsilon)
adv_example_unclipped = tf.add(optimal_perturbation, x)
adv_example = tf.clip_by_value(adv_example_unclipped,0.0,1.0)

classes = tf.argmax(model['probability'], axis=1)

adv_examples = []
idx = 100
epsilon_range = (0.01,0.2)

config = tf.ConfigProto(
        device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
    saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
    print('Correct Class: {}'.format(y_train[idx]))
    prob_x = model['probability'].eval(
        feed_dict={x: x_train[idx:idx+1], y: y_train[idx:idx+1], epsilon: np.random.uniform(0.001, 0.2)})
    print(prob_x, prob_x.shape)
    class_x = classes.eval(
        feed_dict={x: x_train[idx:idx+1], y: y_train[idx:idx+1], epsilon: np.random.uniform(0.001, 0.2)})
    print('Class by network: {}'.format(class_x))
    start = time.time()
    for i in range(1000):
        adv = adv_example.eval(
            feed_dict={x: x_train[idx:idx+1], y: y_train[idx:idx+1], epsilon: np.random.uniform(epsilon_range[0], epsilon_range[1], size=(28,28))})
        class_adv = classes.eval(
            feed_dict={x: adv, y: y_train[idx:idx+1], epsilon: np.random.uniform(0.001, 0.2)})
        print('Class of adv: {}'.format(class_adv))
        if class_adv != y_train[0]:
            adv_examples += [adv]
    print('duration: {}'.format(time.time() - start))
adv_examples = np.concatenate(adv_examples, axis=0)
