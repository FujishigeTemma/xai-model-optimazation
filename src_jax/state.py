import jax.numpy as jnp
import optax  # Common loss functions and optimizers
from clu import metrics
from flax import linen as nn
from flax import struct  # Flax dataclasses
from flax.training import train_state  # Useful dataclass to keep train state


@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module: nn.Module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, 
      params=params, 
      tx=tx,
      metrics=Metrics.empty()
    )