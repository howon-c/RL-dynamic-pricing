# RL_pricing
This is a repository for the paper "[Scalable reinforcement learning approaches for dynamic pricing in ride-hailing systems](https://www.sciencedirect.com/science/article/abs/pii/S019126152300173X)".

To run the experiment, call
"bash train.sh [small/large] [full/ablation]"

After models are trained, call 
"bash test.sh [small/large] [full/ablation]"

Simplified pricing example
To quickly experiment with RL methods, run `python simple_train.py --algo TD3 --timesteps 10000`.\nAlgorithms TD3, PPO and SAC from Stable-Baselines3 are supported.
