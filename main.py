
import numpy as np
import gym
import ddpg

def main(args):
  env = gym.make(args.env_name)
  agent = ddpg.DDPG(args, env, name='ddpg')
  for episode in range(args.nb_episodes):
    state = env.reset()
    for step in range(args.nbsteps):
      action = agent.actor()

if __name__ == '__main__':
  args = config.get_args()
  main(args)