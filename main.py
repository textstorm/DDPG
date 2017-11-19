
import numpy as np
import gym
import ddpg

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def main(args):
  env = gym.make(args.env_name)

  config_proto = get_config_proto()
  sess = tf.Session(config=config_proto)

  agent = ddpg.DDPG(args, env, sess, name='ddpg')
  for episode in range(args.nb_episodes):
    state = env.reset()
    for step in range(env.spec.timestep_limit):
      action = agent.noise_action(state)
      next_state, reward, done, _ = env.step(action)
      model.train(state, action, reward, next_state, done)
      state = next_state
      if done:
        return

    if (episode + 1) % 100 == 0
      total_reward = 0
      for i in xrange(args.test_episodes):
        state = env.reset()
        for j in xrange(env.spec.timestep_limit):
          #env.render()
          action = agent.action(state)
          state, reward, done, _ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward / args.test_episodes
      print "Episode: %d, Evaluation Average Reward: %.1f" % (episode+1, ave_reward)

if __name__ == '__main__':
  args = config.get_args()
  main(args)