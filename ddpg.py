
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

      x = tf.layers.dense(
          x, 
          self.num_actions, 
          kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
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

      x = state
      x = tf.layers.dense(x, 64)
      if self.layer_norm:
        x = tf.layers.layer_norm(x, center=True, scale=True)
      x = tf.nn.relu(x)

      x = tf.concat([x, action], axis=-1)
      x = tf.layers.dense(x, 64)
      if self.layer_norm:
        x = tf.layers.layer_norm(x, center=True, scale=True)
      x = tf.nn.relu(x)

      x = tf.layers.dense(
        inputs=x, 
        units=1, 
        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
      return x

class ReplayMemory(object):
  def __init__(self, memory_size):
    self.memory_size = memory_size
    self.replay_memory = deque()

  def get_length(self):
    return len(self.replay_memory)

  def add(self, state, one_hot_action, reward, next_state, done):
    data = (state, one_hot_action, reward, next_state, done)
    self.replay_memory.append(data)
    if self.get_length() > self.memory_size:
      self.replay_memory.popleft()

  def sample(self, batch_size):
    batch_data = random.sample(self.replay_memory, batch_size)
    state_batch = [data[0] for data in batch_data]
    action_batch = [data[1] for data in batch_data]
    reward_batch = [data[2] for data in batch_data]
    next_state_batch = [data[3] for data in batch_data]
    done_batch = [data[4] for data in batch_data]

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch

class DDPG(object):
  def __init__(self, args, env, name="ddpg"):
    self.observation_space = env.observation_space.shape[0]
    self.num_actions = env.action_space.n

    self.memory_size = args.memory_size
    self.batch_size = args.batch_size
    self.layer_norm = args.layer_norm
    self.actor_lr = args.actor_lr
    self.critic_lr = args.critic_lr
    self.target_update = args.target_update
    self.gamma = args.gamma
    self.max_grad_norm = args.max_grad_norm

    self.replay_memory = ReplayMemory(self.memory_size)
    self.actor = Actor(self.num_actions, layer_norm=self.layer_norm)
    self.critic = Critic(layer_norm=self.layer_norm)

    self.add_placeholder()

  def add_placeholder(self):
    self.state_t = tf.placeholder(tf.float32,
                                  [None, observation_space],
                                  name='state_t')
    self.action_t = tf.placeholder(tf.float32,
                                   [None, self.num_actions],
                                   name='action_t')
    self.critic_y = tf.placeholder(tf.float32,
                                  [None],
                                  name='critic_y')

  def 
