"""Keras preprocessing layers and image augmentation."""
import tensorflow as tf


def build_image_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.1),
    ], name="image_augmentation")


def build_clinical_preprocessor(frame, continuous, categorical):
    """Return a dict-input Keras model and its encoded feature width."""
    inputs, encoded = {}, []
    for name in continuous:
        layer = tf.keras.layers.Normalization(axis=None, name=f"norm_{name}")
        values = frame[name].fillna(frame[name].median()).astype("float32").to_numpy()
        layer.adapt(values)
        inp = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
        inputs[name] = inp
        encoded.append(layer(inp))
    for name in categorical:
        values = frame[name].fillna("unknown").astype(str).to_numpy()
        lookup = tf.keras.layers.StringLookup(
            output_mode="one_hot", num_oov_indices=1, name=f"lookup_{name}"
        )
        lookup.adapt(values)
        inp = tf.keras.Input(shape=(1,), name=name, dtype=tf.string)
        inputs[name] = inp
        encoded.append(lookup(inp))
    if not encoded:
        raise ValueError("At least one clinical feature must be configured")
    output = encoded[0] if len(encoded) == 1 else tf.keras.layers.Concatenate()(encoded)
    return tf.keras.Model(inputs, output, name="clinical_preprocessor")


def decode_resize(path, image_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, antialias=True)
    return tf.cast(image, tf.float32)
