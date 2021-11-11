import argparse
from dex_envs.configs import *


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--train_names', nargs='*', default=[],
                        type=str, help='the environment name')
    parser.add_argument('--test_names', nargs='*', default=[],
                        type=str, help='the environment name')
    parser.add_argument('--n_cycles', type=int, default=40000,
                        help='the times to collect samples per epoch')
    parser.add_argument('--n_batches', type=int, default=40,
                        help='the times to update the network')
    parser.add_argument('--seed', type=int, default=125, help='random seed')
    parser.add_argument('--replay_strategy', type=str,
                        default='future', help='the HER strategy')
    parser.add_argument('--save_dir', type=str,
                        default='dex_logs/', help='the path to save the models')
    parser.add_argument('--noise_eps', type=float,
                        default=0.2, help='noise eps')
    parser.add_argument('--random_eps', type=float,
                        default=0.3, help='random eps')
    parser.add_argument('--buffer_size', type=int,
                        default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay_k', type=int, default=4,
                        help='ratio to be replace')
    parser.add_argument('--clip_obs', type=float,
                        default=200, help='the clip ratio')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98,
                        help='the discount factor')
    parser.add_argument('--action_l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr_actor', type=float, default=0.001281,
                        help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=0.000884,
                        help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95,
                        help='the average coefficient')
    parser.add_argument('--n_test_rollouts', type=int,
                        default=570, help='the number of tests')
    parser.add_argument('--clip_range', type=float,
                        default=5, help='the clip range')
    parser.add_argument('--no_cuda', action='store_true',
                        help='if use gpu do the acceleration')
    parser.add_argument('--num_rollouts', type=int,
                        default=425, help='the rollouts per cycle')
    parser.add_argument('--expID', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=50,
                        help='number of cycles between eval and logging')
    parser.add_argument('--video_count', type=int, default=1,
                        help='number of videos to record during logging')
    parser.add_argument('--load_path', default=None,
                        type=int, help='exp id to load model/data')
    parser.add_argument('--pointnet_load_path', default=None,
                        type=int, help='exp id to load pointnet model')
    parser.add_argument('--serial', action="store_true",
                        help="do not use multiprocessing")
    parser.add_argument('--no_save', action="store_true",
                        help="do not save anything")
    parser.add_argument('--epoch_per_cycle', type=float, default=22.966,
                        help="alternative way to specify n_batches; it means the number of epochs to train over the size of newly collected data; if newly collected 1000 transitions, 10 epochs means training for 100 batches with batch size being 100")
    parser.add_argument('--num_parallel_envs', type=int, default=425,
                        help='number of parallel VecEnv; if not specified, calculate from the available vCPUs')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='if not equal 0, use SubprocChunkVecEnv as proposed in baselines PR 620 with given chunk size')
    parser.add_argument('--eval', action="store_true",
                        help="only eval the network once")
    parser.add_argument('--finetune_pointnet', action="store_true",
                        help="allow finetuning in dagger training")
    parser.add_argument('--fresh', action="store_true",
                        help="ignore saved checkpoint; start from scratch")
    parser.add_argument('--no_save_buffer', action="store_true",
                        help="do not save buffer when saving general checkpoint (to avoid memory error if buffer is too big)")
    parser.add_argument('--load_cycle', type=int,
                        default=None, help='the cycle index of the loaded eval checkpoint')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--point_cloud', action="store_true",
                        help="enable using pointnet + MLP")
    parser.add_argument('--num_points', type=int, default=128,
                        help="number of points to sample in mujoco")
    parser.add_argument('--pointnet_output_dim', type=int, default=512,
                        help="output dim of the two stream pointnet feature net")

    args = parser.parse_args()

    # default to use train/test split specified in dex_envs/configs
    args.train_names = args.train_names if args.train_names else ALL_TRAIN
    args.test_names = args.test_names if args.test_names else ALL_TEST
    assert len(list(set(args.train_names) & set(args.test_names))
               ) == 0, 'cannot have overlapping train/test envs'

    # if using eval mode, do not start new training and do not save
    if args.eval:
        args.fresh = False
        args.no_save = True

    # alternative way to specify how many batches to update per cycle
    if args.epoch_per_cycle:
        batches_per_epoch = (args.num_rollouts * 100) / args.batch_size
        batches_total = int(batches_per_epoch * args.epoch_per_cycle)
        print('*** overwriting n_batches from {} to {} ***'.format(args.n_batches, batches_total))
        args.n_batches = batches_total

    # cap buffer size at 10M
    total_buffer_size = len(args.train_names) * args.buffer_size
    if total_buffer_size > int(1e7):
        args.buffer_size = int(1e7) // len(args.train_names)
        print('adjusted total buffer size from {:.4e} to 1e7'.format(
            total_buffer_size))

    # require parallization args align with each other
    if args.n_cycles > 0 or args.eval:
        if len(args.train_names) > 0:
            assert args.num_parallel_envs % (len(
                args.train_names)) == 0, 'num_parallel_envs must be multiple of num of train envs ({})'.format(len(args.train_names))
            assert args.num_rollouts % args.num_parallel_envs == 0, 'num_rollouts must be multiple of parallel env number ({})'.format(
                args.num_parallel_envs)

        assert args.n_test_rollouts % (len(args.train_names + args.test_names)) == 0, 'n_test_rollouts must be multiple of num of train envs + test_envs ({} + {} = {})'.format(
            len(args.train_names), len(args.test_names), len(args.train_names + args.test_names))
        # make sure n_batches must be multiple of num of train envs
        if len(args.train_names) > 0:
            if args.epoch_per_cycle:
                args.n_batches = (
                    args.n_batches // len(args.train_names)) * len(args.train_names)
            else:
                assert args.n_batches % len(
                    args.train_names) == 0, 'n_batches must be multiple of num of train envs ({})'.format(len(args.train_names))

    return args
