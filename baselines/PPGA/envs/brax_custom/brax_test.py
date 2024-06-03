import functools
import time

from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax


def brax_test(gym_env):
    # jit compile env.reset
    obs = gym_env.reset()

    # jit compile env.step
    action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
    obs, reward, done, info = gym_env.step(action)

    before = time.time()

    steps = 1000
    for _ in range(steps):
        action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
        obs, rewards, done, info = gym_env.step(action)

    duration = time.time() - before
    env_steps = gym_env.num_envs * steps
    print(f'time for {env_steps} steps: {duration:.2f}s ({int(env_steps / duration)} steps/sec)')


def two_brax_gyms():
    env1 = make_vec_env_brax_ppga(task_name="humanoid", feat_name="angle", batch_size=128, backend="spring", episode_length=1000, clip_obs_rew=False, seed=42)
    env2 = make_vec_env_brax_ppga(task_name="humanoid", feat_name="angle", batch_size=128, backend="spring", episode_length=1000, clip_obs_rew=True, seed=12345)
    print('successfully spawned 2 brax env instances!')
    brax_test(env1)
    brax_test(env2)


if __name__ == '__main__':
    two_brax_gyms()
