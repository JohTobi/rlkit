from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
from rlkit.envs.wrappers import NormalizedBoxEnv
from sawyer_control.envs.sawyer_insertion import SawyerHumanControlEnv
# from sawyer_control.envs.sawyer_insertion_fixed_blocks import SawyerHumanControlEnv
filename = str(uuid.uuid4())

def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    # env = data['env']
    # env = NormalizedBoxEnv(SawyerHumanControlEnv(action_mode='position', position_action_scale = 1))
    # env = SawyerHumanControlEnv(action_mode='joint_space_impd', position_action_scale=1.0, max_speed=0.07)
    env = SawyerHumanControlEnv(action_mode='joint_space_impd', position_action_scale=0.05, max_speed=0.1)
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
    # path = rollout(
    #     env,
    #     policy,
    #     max_path_length=args.H,
    #     animated=False,
    # )
    # if hasattr(env, "log_diagnostics"):
    #     env.log_diagnostics([path])
    # logger.dump_tabular()
    # env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=25,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)

