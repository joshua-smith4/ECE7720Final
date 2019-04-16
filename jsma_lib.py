import numpy as np
import tensorflow as tf


def jsma_symbolic(x, model, theta, gamma, clip_min, clip_max):

    from random import randint

    def random_targets(gt):
        result = gt.copy()
        nb_s = gt.shape[0]
        nb_classes = gt.shape[1]

        for i in range(nb_s):
            result[i, :] = np.roll(result[i, :],
                                   randint(1, nb_classes - 1))

        return result

    preds_max = tf.reduce_max(model['probability'], 1, keepdims=True)
    original_predictions = tf.to_float(
        tf.equal(model['probability'], preds_max))
    labels = tf.stop_gradient(original_predictions)
    if isinstance(labels, np.ndarray):
        nb_classes = labels.shape[1]
    else:
        nb_classes = labels.get_shape().as_list()[1]
    y_target = tf.py_func(random_targets, [labels],
                          tf.float32)
    y_target.set_shape([None, nb_classes])

    nb_classes = int(y_target.shape[-1].value)
    nb_features = int(np.product(x.shape[1:]).value)

    if x.dtype == tf.float32 and y_target.dtype == tf.int64:
        y_target = tf.cast(y_target, tf.int32)

    if x.dtype == tf.float32 and y_target.dtype == tf.float64:
        warnings.warn("Downcasting labels---this should be harmless unless"
                      " they are smoothed")
        y_target = tf.cast(y_target, tf.float32)

    max_iters = np.floor(nb_features * gamma / 2)
    increase = bool(theta > 0)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, tf.float32)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = tf.reshape(
            tf.cast(x < clip_max, tf.float32), [-1, nb_features])
    else:
        search_domain = tf.reshape(
            tf.cast(x > clip_min, tf.float32), [-1, nb_features])

    # Loop variables
    # x_in: the tensor that holds the latest adversarial outputs that are in
    #       progress.
    # y_in: the tensor for target labels
    # domain_in: the tensor that holds the latest search domain
    # cond_in: the boolean tensor to show if more iteration is needed for
    #          generating adversarial samples
    def condition(x_in, y_in, domain_in, i_in, cond_in):
        # Repeat the loop until we have achieved misclassification or
        # reaches the maximum iterations
        return tf.logical_and(tf.less(i_in, max_iters), cond_in)

    # Same loop variables as above
    def body(x_in, y_in, domain_in, i_in, cond_in):
        # Create graph for model logits and predictions
        logits = model['logits']
        preds = tf.nn.softmax(logits)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        # create the Jacobian graph
        list_derivatives = []
        for class_ind in range(nb_classes):
            derivatives = tf.gradients(logits[:, class_ind], x_in)
            list_derivatives.append(derivatives[0])
        grads = tf.reshape(
            tf.stack(list_derivatives), shape=[nb_classes, -1, nb_features])

        # Compute the Jacobian components
        # To help with the computation later, reshape the target_class
        # and other_class to [nb_classes, -1, 1].
        # The last dimention is added to allow broadcasting later.
        target_class = tf.reshape(
            tf.transpose(y_in, perm=[1, 0]), shape=[nb_classes, -1, 1])
        other_classes = tf.cast(tf.not_equal(target_class, 1), tf.float32)

        grads_target = reduce_sum(grads * target_class, axis=0)
        grads_other = reduce_sum(grads * other_classes, axis=0)

        # Remove the already-used input features from the search space
        # Subtract 2 times the maximum value from those value so that
        # they won't be picked later
        increase_coef = (4 * int(increase) - 2) \
            * tf.cast(tf.equal(domain_in, 0), tf.float32)

        target_tmp = grads_target
        target_tmp -= increase_coef \
            * reduce_max(tf.abs(grads_target), axis=1, keepdims=True)
        target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
            + tf.reshape(target_tmp, shape=[-1, 1, nb_features])

        other_tmp = grads_other
        other_tmp += increase_coef \
            * reduce_max(tf.abs(grads_other), axis=1, keepdims=True)
        other_sum = tf.reshape(other_tmp, shape=[-1, nb_features, 1]) \
            + tf.reshape(other_tmp, shape=[-1, 1, nb_features])

        # Create a mask to only keep features that match conditions
        if increase:
            scores_mask = ((target_sum > 0) & (other_sum < 0))
        else:
            scores_mask = ((target_sum < 0) & (other_sum > 0))

        # Create a 2D numpy array of scores for each pair of candidate features
        scores = tf.cast(scores_mask, tf.float32) \
            * (-target_sum * other_sum) * zero_diagonal

        # Extract the best two pixels
        best = tf.argmax(
            tf.reshape(scores, shape=[-1, nb_features * nb_features]), axis=1)

        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        # Check if more modification is needed for each sample
        mod_not_done = tf.equal(reduce_sum(y_in * preds_onehot, axis=1), 0)
        cond = mod_not_done & (reduce_sum(domain_in, axis=1) >= 2)

        # Update the search domain
        cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
        to_mod = (p1_one_hot + p2_one_hot) * cond_float

        domain_out = domain_in - to_mod

        # Apply the modification to the images
        to_mod_reshape = tf.reshape(
            to_mod, shape=([-1] + x_in.shape[1:].as_list()))
        if increase:
            x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
        else:
            x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)

        # Increase the iterator, and check if all misclassifications are done
        i_out = tf.add(i_in, 1)
        cond_out = reduce_any(cond)

        return x_out, y_in, domain_out, i_out, cond_out

    # Run loop to do JSMA
    x_adv, _, _, _, _ = tf.while_loop(
        condition,
        body, [x, y_target, search_domain, 0, True],
        parallel_iterations=1)

    return x_adv
