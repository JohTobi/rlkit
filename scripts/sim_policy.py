from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger
from rlkit.envs.wrappers import NormalizedBoxEnv
# from ss.envs.sawyer_env_contr_learning import SawyerHumanControlEnv
# from ss.envs.sawyer_env_compare_sim_hw import SawyerHumanControlEnv
from ss.envs.sawyer_env_fixed_blocks2 import SawyerHumanControlEnv
# from ss.envs.sawyer_env_learning import SawyerHumanControlEnv

filename = str(uuid.uuid4())

def simulate_policy(args):
    data = joblib.load(args.file)
    policy = data['policy']
    # env = data['env']
    env = NormalizedBoxEnv(SawyerHumanControlEnv())
    # env = NormalizedBoxEnv(SawyerHumanControlEnv(action_mode='position', position_action_scale = 1))
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    # set_gpu_mode(True)
    # policy.cuda()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=True,
            # animated=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
