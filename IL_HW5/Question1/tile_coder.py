import numpy as np
import gym
import tiles3 as tc

env = gym.make('MountainCar-v0').env
class MountainCarTileCoder:
    def __init__(self, iht_size, num_tilings, num_tiles):
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, position, velocity):
        min_position = env.min_position
        max_position = env.max_position
        min_velocity = -env.max_speed
        max_velocity = env.max_speed
        position_scale = self.num_tiles / (max_position - min_position)
        velocity_scale = self.num_tiles / (max_velocity - min_velocity)
        tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, velocity * velocity_scale])
        return np.array(tiles)