import argparse
import numpy as np
from simple_env import PricingEnv
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise


def make_model(algo, env):
    if algo == "TD3":
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return TD3("MlpPolicy", env, action_noise=noise, verbose=1)
    if algo == "PPO":
        return PPO("MlpPolicy", env, verbose=1)
    if algo == "SAC":
        return SAC("MlpPolicy", env, verbose=1)
    raise ValueError(f"Unknown algo {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["TD3", "PPO", "SAC"], default="TD3")
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()

    env = PricingEnv()
    model = make_model(args.algo, env)
    model.learn(total_timesteps=args.timesteps)
    model.save(f"{args.algo}_pricing")

    obs, _ = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        print(f"action {action[0]:.2f}, reward {reward:.2f}")
        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
