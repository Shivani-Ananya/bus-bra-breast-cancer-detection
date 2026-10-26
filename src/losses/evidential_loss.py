import tensorflow as tf


def dirichlet_kl(alpha, num_classes=2):
    beta = tf.ones_like(alpha)
    sum_a, sum_b = tf.reduce_sum(alpha, -1, keepdims=True), tf.reduce_sum(beta, -1, keepdims=True)
    ln_b_a = tf.reduce_sum(tf.math.lgamma(alpha), -1, keepdims=True) - tf.math.lgamma(sum_a)
    ln_b_b = tf.reduce_sum(tf.math.lgamma(beta), -1, keepdims=True) - tf.math.lgamma(sum_b)
    return tf.squeeze(ln_b_b - ln_b_a + tf.reduce_sum(
        (alpha - beta) * (tf.math.digamma(alpha) - tf.math.digamma(sum_a)), -1, keepdims=True), -1)


def evidential_loss(labels, alpha, annealing, num_classes=2):
    y = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
    strength = tf.reduce_sum(alpha, axis=-1, keepdims=True)
    expected_ce = tf.reduce_sum(y * (tf.math.digamma(strength) - tf.math.digamma(alpha)), -1)
    # Remove correct-class evidence before KL, as in Sensoy et al.
    adjusted = y + (1.0 - y) * alpha
    return tf.reduce_mean(expected_ce + tf.cast(annealing, alpha.dtype) * dirichlet_kl(adjusted, num_classes))

