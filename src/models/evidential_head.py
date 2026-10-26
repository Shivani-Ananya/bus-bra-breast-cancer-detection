import tensorflow as tf


class EvidentialHead(tf.keras.layers.Layer):
    def __init__(self, num_classes=2, activation="softplus", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dense = tf.keras.layers.Dense(num_classes)
        self.activation = activation

    def call(self, inputs):
        logits = self.dense(inputs)
        evidence = tf.nn.softplus(logits) if self.activation == "softplus" else tf.nn.relu(logits)
        alpha = evidence + 1.0
        strength = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        belief = evidence / strength
        uncertainty = tf.cast(self.num_classes, alpha.dtype) / strength
        return {"evidence": evidence, "alpha": alpha,
                "belief": belief, "uncertainty": uncertainty}

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes,
                "activation": self.activation}

