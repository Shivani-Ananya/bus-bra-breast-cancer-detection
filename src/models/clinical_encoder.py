import tensorflow as tf


def build_clinical_encoder(preprocessor, hidden=(64, 32), dropout=.2):
    # Preserve the named dictionary structure expected by the clinical
    # preprocessor. Using ``preprocessor.inputs`` directly turns it into a
    # positional list and mismatches categorical/continuous features in
    # Keras 3.
    inputs = {tensor.name.split(":")[0]: tensor for tensor in preprocessor.inputs}
    x = preprocessor(inputs)
    for width in hidden:
        x = tf.keras.layers.Dense(width, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, x, name="clinical_encoder")

