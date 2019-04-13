from make_mnist_cnn_tf import build_cnn_mnist_model

if __name__ == '__main__':
    reset_graph()
    x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    y = tf.placeholder(tf.int32, shape=(None,))
    model = build_cnn_mnist_model(x, y, False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/np.float32(255)
    y_train = y_train.astype(np.int32)
    x_test = x_test/np.float32(255)
    y_test = y_test.astype(np.int32)

    with tf.Session() as sess:
        saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
        acc_test = model['accuracy'].eval(feed_dict={x: x_test, y: y_test})
        print(acc_test)
