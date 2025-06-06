import numpy as np
import gym
from tqdm import tqdm 
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

EPS=1e-14
def process_obs_func(obs):
    if len(obs.shape) > 3:
        raise ValueError
    obs[..., -3:] = 0.0
    return obs

class MetaworldGoalMaskWrapper(gym.Wrapper):

    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return process_obs_func(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return process_obs_func(obs), reward, done, info


def eval_metaworld_sim(policy,
                       goal_type: str = "mw",
                       num_eval: int = 50,
                       max_steps: int = 500,
                       task = "pick-place-wall"):

    # evaluate the policy by rolling out the last checkpoint
    if goal_type in ["mw", "none"]:
        env = ALL_V2_ENVIRONMENTS[f"{task}-v2"]()
        env._partially_observable = goal_type == 'none'
        env._freeze_rand_vec = False
        env._set_task_called = True
    else:
        raise NotImplementedError
    
    n_successes = 0

    for ep in tqdm(range(int(num_eval))):

        ob = env.reset()

        for step_i in range(max_steps):
            act = policy.get_action(ob)
            
            ob, rew, done, info = env.step(act)

            if info["success"]:
                n_successes += 1
                break
        
        # print(n_successes)

    prob = np.float64(n_successes / num_eval)
    prob = np.clip(prob, EPS, 1-EPS)

    simulation_metrics = {
        'prob': prob,
    }
    # print(simulation_metrics)
    return simulation_metrics