import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def build_cnn_mnist_model(input_placeholder, labels, training=True):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(input_placeholder, [-1, 28, 28, 1], name='x_reshaped')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_1')

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name='conv_2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=training)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    classes = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return {
        'loss': loss,
        'train': train_op,
        'optimize': optimizer,
        'probability': probabilities,
        'accuracy': accuracy,
    }


if __name__ == '__main__':
    reset_graph()
    x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    y = tf.placeholder(tf.int32, shape=(None,))
    model = build_cnn_mnist_model(x, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/np.float32(255)
    y_train = y_train.astype(np.int32)
    x_test = x_test/np.float32(255)
    y_test = y_test.astype(np.int32)

    print(x_train.shape, x_train.dtype)

    num_epochs = 30
    batch_size = 100

    with tf.Session() as sess:
        init.run()
        acc = 0.0
        for epoch in range(num_epochs):
            for i in range(x_train.shape[0] // batch_size):
                batch_indices = np.random.randint(x_train.shape[0], size=batch_size)
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]
                sess.run(model['train'], feed_dict={x: x_batch, y: y_batch})
        acc_train = model['accuracy'].eval(feed_dict={x: x_batch, y: y_batch})
        acc_test = model['accuracy'].eval(feed_dict={x: x_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        if acc_test > acc:
            print('saving model: {}'.format(epoch))
            acc = acc_test
            saver.save(sess, "./models/mnist_cnn_tf")
