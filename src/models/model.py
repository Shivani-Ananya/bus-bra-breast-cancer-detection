import tensorflow as tf
from .evidential_head import EvidentialHead
from .dempster_fusion import DempsterFusion


class BreastMultimodalModel(tf.keras.Model):
    def __init__(self, image_encoder, clinical_encoder, mode="fusion_evidential",
                 num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.image_encoder, self.clinical_encoder = image_encoder, clinical_encoder
        self.mode, self.num_classes = mode, num_classes
        self.image_head = EvidentialHead(num_classes, name="image_evidence")
        self.clinical_head = EvidentialHead(num_classes, name="clinical_evidence")
        self.fusion = DempsterFusion(num_classes, name="dempster_fusion")
        self.concat_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(.2)],
            name="concat_projection")
        self.concat_head = EvidentialHead(num_classes, name="concat_evidence")

    def call(self, inputs, training=False):
        image, clinical = inputs
        # Keras 3 flattens dictionaries nested inside tuple inputs into a list,
        # using sorted dictionary keys. Reconstruct the named clinical mapping
        # before passing it to the Functional encoder.
        if isinstance(clinical, (list, tuple)):
            feature_names = sorted(tensor.name.split(":")[0]
                                   for tensor in self.clinical_encoder.inputs)
            clinical = dict(zip(feature_names, clinical))
        zi = self.image_encoder(image, training=training)
        zc = self.clinical_encoder(clinical, training=training)
        if self.mode == "baseline_image":
            return {"fused": self.image_head(zi), "image": self.image_head(zi)}
        if self.mode == "baseline_clinical":
            return {"fused": self.clinical_head(zc), "clinical": self.clinical_head(zc)}
        if self.mode == "fusion_naive_concat":
            fused = self.concat_head(self.concat_projection(tf.concat([zi, zc], -1), training=training))
            return {"fused": fused}
        image_out, clinical_out = self.image_head(zi), self.clinical_head(zc)
        return {"fused": self.fusion([image_out, clinical_out]),
                "image": image_out, "clinical": clinical_out}

