import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PricingEnv(gym.Env):
    """A lightweight ride-hailing pricing environment."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=200, base_demand=10, vehicles=10,
                 base_price=1.0, cost_per_ride=0.5):
        super().__init__()
        self.max_steps = max_steps
        self.base_demand = base_demand
        self.vehicles = vehicles
        self.base_price = base_price
        self.cost_per_ride = cost_per_ride

        self.action_space = spaces.Box(low=0.5, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.demand = self.np_random.poisson(self.base_demand)
        obs = np.array([self.vehicles, self.demand], dtype=np.float32)
        return obs, {}

    def step(self, action):
        price_mult = float(np.clip(action[0], self.action_space.low, self.action_space.high))
        served = min(self.demand, self.vehicles)
        revenue = served * price_mult * self.base_price
        cost = served * self.cost_per_ride
        reward = revenue - cost
        self.step_count += 1

        self.demand = self.np_random.poisson(self.base_demand / price_mult)
        obs = np.array([self.vehicles, self.demand], dtype=np.float32)
        done = self.step_count >= self.max_steps
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        print(f"Step: {self.step_count}, demand: {self.demand}")
