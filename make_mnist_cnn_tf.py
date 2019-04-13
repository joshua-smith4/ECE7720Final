import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(input_placeholder, labels, training=True):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(input_placeholder, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step(),
        name='train_operation')

    return logits, classes, probabilities, loss, train_op


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=(None, 28, 28))
    y = tf.placeholder(tf.int32, shape=(None, 1))
    logits, classes, probabilities, loss, train_op = cnn_model_fn(x, y)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/np.float32(255)
    y_train = y_train.astype(np.int32)
    x_test = x_test/np.float32(255)
    y_test = y_test.astype(np.int32)

    print(x_train.shape, x_train.dtype)

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": x_train},
    #     y=y_train,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    #
    # mnist_classifier.train(
    #     input_fn=train_input_fn,
    #     steps=10)
    #
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": x_test},
    #     y=y_test,
    #     num_epochs=1,
    #     shuffle=False)
    #
    # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)
