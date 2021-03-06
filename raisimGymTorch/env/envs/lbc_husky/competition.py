from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lbc_husky
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."
weight_path = home_path + "/raisimGymTorch/data/husky_navigation/2021-11-24-15-11-40/full_1200.pt"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 200
env = VecEnv(lbc_husky.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    completion_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    max_steps = 80 ## 8 secs
    completed_sum = 0

    for step in range(max_steps):
        # time.sleep(cfg['environment']['control_dt'])
        obs = env.observe(False)
        action = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward, dones, not_completed = env.step(action.cpu().detach().numpy())
        completed_sum = completed_sum + sum(not_completed)

    print('----------------------------------------------------')
    print('{:<40} {:>6}'.format("avg completion time: ", '{:0.6f}'.format(completed_sum / env.num_envs * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')

    env.turn_off_visualization()