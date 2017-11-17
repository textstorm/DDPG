
import tensorflow as tf

class Actor(object):
  def __init__(self, num_actions, layer_norm=True, name="actor"):
    self.num_actions = num_actions
    self.layer_norm = layer_norm
    self.name = name

  def __call__(self, state, reuse=None):
    with tf.variable_scope() as scope:
      if reuse:
        scope.reuse_variables()

      x = state
      x = tf.layers.dense(x, 64)
      if self.layer_norm:
        x = tf.layers.layer_norm(x, center=True, scale=True)
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, 64)
      if self.layer_norm:
        x = tf.layers.layer_norm(x, center=True, scale=True)
      x = tf.nn.relu(x)

      x = tf.layers.dense(x, self.num_actions)
      x = tf.nn.tanh(x)
    return x

class Critic(object):
  def __init__(self, layer_norm=True, name="critic"):
    self.layer_norm = layer_norm
    self.name = name

  def __call__(self, state, action, reuse=None):
    with tf.variable_scope() as scope:
      if reuse:
        scope.reuse_variables()

      x = 
