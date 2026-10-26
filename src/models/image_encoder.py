import tensorflow as tf


def build_image_encoder(image_size=(224, 224), backbone="EfficientNetB0", pretrained=True,
                        trainable=False):
    cls = getattr(tf.keras.applications, backbone)
    base = cls(include_top=False, weights="imagenet" if pretrained else None,
               input_shape=(*image_size, 3))
    base.trainable = trainable
    inputs = tf.keras.Input((*image_size, 3), name="image")
    # EfficientNet includes input rescaling.
    x = base(inputs, training=trainable)
    x = tf.keras.layers.GlobalAveragePooling2D(name="image_gap")(x)
    x = tf.keras.layers.Dropout(.2)(x)
    return tf.keras.Model(inputs, x, name="image_encoder")

