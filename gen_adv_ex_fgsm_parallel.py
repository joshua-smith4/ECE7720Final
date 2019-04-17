#!/usr/bin/env python3

from make_mnist_cnn_tf import build_cnn_mnist_model, reset_graph
import tensorflow as tf
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicSourceModule
import pycuda.curandom as curand
import pycuda.gpuarray as gpuarray
import time
import argparse
import cProfile, pstats, io

pr = cProfile.Profile()
parser = argparse.ArgumentParser()
parser.add_argument('--epsmin', type=float, default=0.01)
parser.add_argument('--epsmax', type=float, default=0.2)
parser.add_argument('--idx', type=int, default=100)
parser.add_argument('--numgens', type=int, default=1000)
parser.add_argument('--prof', type=bool, default=True)
parser.add_argument('--outfile', default="fgsm_parallel.prof")

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


print('Type of x_train',x_train.dtype)

grad, = tf.gradients(model['loss'], x)
gradsign = tf.cast(tf.sign(grad), tf.float32)
epsilon = tf.placeholder(tf.float32)
optimal_perturbation = tf.multiply(gradsign, epsilon)
adv_example_unclipped = tf.add(optimal_perturbation, x)
adv_example = tf.clip_by_value(adv_example_unclipped, 0.0, 1.0)

classes = tf.argmax(model['probability'], axis=1)

idx = args.idx
epsilon_range = (args.epsmin, args.epsmax)

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
    if args.prof:
        pr.enable()
    grad_val = gradsign.eval(feed_dict={
        x: x_train[idx:idx+1],
        y: y_train[idx:idx+1]
    })
    grad_flat = np.squeeze(grad_val).flatten()
    x_flat = x_train[idx].flatten()
    with open('parallel_fgsm.cu') as f:
        src = f.read()
    src_comp = DynamicSourceModule(src)
    grid = (1,1)
    block = (1,1,1)
    gen_examples_fgsm = src_comp.get_function("gen_examples_fgsm")
    # gen_examples_fgsm.prepare("PPPPII")

    start = time.time()
    gen = curand.MRG32k3aRandomNumberGenerator()
    epsilon_gpu = gpuarray.GPUArray((args.numgens,), dtype=np.float32)
    gen.fill_uniform(epsilon_gpu)
    # epsilon_gpu = curand.rand((args.numgens,))
    epsilon_gpu = epsilon_gpu * (args.epsmax - args.epsmin) + args.epsmin
    x_gpu = gpuarray.to_gpu(x_flat)
    grad_gpu = gpuarray.to_gpu(grad_flat)
    res_gpu = gpuarray.GPUArray((args.numgens*28*28,), dtype=np.float32)

    gen_examples_fgsm(
        res_gpu,
        x_gpu,
        grad_gpu,
        epsilon_gpu,
        np.int32(args.numgens),
        np.int32(28*28),
        block=block
    )
    adv_examples = res_gpu.get().reshape((args.numgens,28,28))
    if args.prof:
        pr.disable()
    class_adv = classes.eval(feed_dict={x: adv_examples})
    print('Duration (s): {}'.format(time.time() - start))
    num_adv_examples = np.sum((class_adv != y_train[idx]).astype(np.int32))
print('Found {} adversarial examples.'.format(num_adv_examples))

if args.prof:
    pr.dump_stats(args.outfile)
