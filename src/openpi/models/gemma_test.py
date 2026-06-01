import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma


def test_detach_prefix():
    values = jnp.ones((1, 4, 1))
    grads = jax.grad(lambda x: jnp.sum(gemma._detach_prefix(x, 2)))(values)

    np.testing.assert_array_equal(grads, jnp.array([[[0.0], [0.0], [1.0], [1.0]]]))
