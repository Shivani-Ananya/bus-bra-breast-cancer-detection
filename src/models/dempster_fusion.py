import tensorflow as tf


class DempsterFusion(tf.keras.layers.Layer):
    """Two-view subjective-logic fusion used by TMC."""
    def __init__(self, num_classes=2, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs); self.num_classes = num_classes; self.epsilon = epsilon

    def call(self, inputs):
        left, right = inputs
        b1, u1 = left["belief"], left["uncertainty"]
        b2, u2 = right["belief"], right["uncertainty"]
        pair = tf.expand_dims(b1, 2) * tf.expand_dims(b2, 1)
        # Compute one conflict value per sample. Keeping three dimensions on
        # only the first term broadcasts a batch of B into a BxB matrix.
        conflict = tf.reduce_sum(pair, axis=[1, 2]) - tf.reduce_sum(b1 * b2, axis=1)
        conflict = tf.expand_dims(conflict, -1)
        denom = tf.maximum(1.0 - conflict, self.epsilon)
        belief = (b1 * b2 + b1 * u2 + b2 * u1) / denom
        uncertainty = (u1 * u2) / denom
        # Renormalize defensively after clipping near total conflict.
        total = tf.reduce_sum(belief, axis=-1, keepdims=True) + uncertainty
        belief, uncertainty = belief / total, uncertainty / total
        strength = tf.cast(self.num_classes, belief.dtype) / tf.maximum(uncertainty, self.epsilon)
        evidence = belief * strength
        alpha = evidence + 1.0
        return {"evidence": evidence, "alpha": alpha, "belief": belief,
                "uncertainty": uncertainty, "conflict": conflict}

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes, "epsilon": self.epsilon}

