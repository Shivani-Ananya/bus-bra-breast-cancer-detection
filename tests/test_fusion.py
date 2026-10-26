import tensorflow as tf
from src.models.dempster_fusion import DempsterFusion


def view(b, u):
    return {"belief": tf.constant([b], tf.float32), "uncertainty": tf.constant([[u]], tf.float32)}


def test_vacuous_view_is_identity():
    certain = view([.6, .2], .2); vacuous = view([0., 0.], 1.)
    out = DempsterFusion()([certain, vacuous])
    tf.debugging.assert_near(out["belief"], certain["belief"])
    tf.debugging.assert_near(out["uncertainty"], certain["uncertainty"])


def test_agreement_reduces_uncertainty_and_normalizes():
    x = view([.6, .2], .2); out = DempsterFusion()([x, x])
    assert float(out["uncertainty"][0, 0]) < .2
    tf.debugging.assert_near(tf.reduce_sum(out["belief"], -1, keepdims=True) + out["uncertainty"], [[1.]])


def test_high_conflict_stays_finite():
    out = DempsterFusion()([view([.999, 0.], .001), view([0., .999], .001)])
    tf.debugging.assert_all_finite(out["alpha"], "fusion produced non-finite alpha")


def test_batch_fusion_keeps_one_output_per_sample():
    left = {"belief": tf.constant([[.6, .2], [.2, .6]], tf.float32),
            "uncertainty": tf.constant([[.2], [.2]], tf.float32)}
    right = {"belief": tf.constant([[.5, .3], [.3, .5]], tf.float32),
             "uncertainty": tf.constant([[.2], [.2]], tf.float32)}
    out = DempsterFusion()([left, right])
    assert out["alpha"].shape == (2, 2)
    assert out["conflict"].shape == (2, 1)

