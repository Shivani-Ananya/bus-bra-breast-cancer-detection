"""GradientTape Grad-CAM utilities for image-only/fused comparisons."""
import tensorflow as tf


def gradcam(model, inputs, conv_layer_name, class_index=1):
    image, clinical = inputs
    backbone = model.image_encoder
    conv_layer = backbone.get_layer(index=1).get_layer(conv_layer_name)
    activation_model = tf.keras.Model(backbone.get_layer(index=1).input,
                                      [conv_layer.output, backbone.get_layer(index=1).output])
    with tf.GradientTape() as tape:
        conv, _ = activation_model(image, training=False); tape.watch(conv)
        pooled = tf.reduce_mean(conv, axis=[1, 2])
        # Route the selected activation through the existing image evidence head.
        score = model.image_head(pooled)["alpha"][:, class_index]
    gradients = tape.gradient(score, conv); weights = tf.reduce_mean(gradients, axis=[1, 2])
    heatmap = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))
    return heatmap / tf.maximum(tf.reduce_max(heatmap, axis=[1, 2], keepdims=True), 1e-7)

