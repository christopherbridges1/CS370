# This class stores the episodes, all the states that come in between the initial state and the terminal state. 
# This is later used by the agent for learning by experience, called "exploration". 

import numpy as np
import tensorflow as tf

class GameExperience(object):
    
    # model = neural network model
    # max_memory = number of episodes to keep in memory. The oldest episode is deleted to make room for a new episode.
    # discount = discount factor; determines the importance of future rewards vs. immediate rewards
    
    def __init__(self, model, target_model, max_memory=100, discount=0.95):
        self.model = model
        self.target_model = target_model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]
    
    # Stores episodes in memory
    
    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including pirate cell (see method: observe in TreasureMaze.py)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # Predicts the next action based on the current environment state        
    def predict(self, envstate):
        envstate = np.asarray(envstate, dtype=np.float32)
        if envstate.ndim == 1:
          envstate = np.expand_dims(envstate, axis=0)
        return self.model(envstate, training=False).numpy()[0]

    def sample(self, batch_size=32):
        if len(self.memory) == 0:
            return []
        idx = np.random.choice(
            len(self.memory),
            size=min(batch_size, len(self.memory)),
            replace=False
        )
        return [self.memory[i] for i in idx]

    def get_data(self, batch_size=32):
        if self.model is None or self.target_model is None:
            raise ValueError("Both `model` and `target_model` must be provided")

        batch = self.sample(batch_size)
        if not batch:
            # no data yet
            env_size = self.model.input_shape[-1]
            return (np.zeros((0, env_size), dtype=np.float32),
                    np.zeros((0, self.num_actions), dtype=np.float32))

        # states are (1, N); flatten to (N,)
        env_size = batch[0][0].shape[-1]

        inputs = np.zeros((len(batch), env_size), dtype=np.float32)
        targets = np.zeros((len(batch), self.num_actions), dtype=np.float32)

        states = np.vstack([b[0] for b in batch])  # (B, 1, N) or (B, N)
        next_states = np.vstack([b[3] for b in batch])

        # squeeze if needed
        if states.ndim == 3:
            states = states.reshape(states.shape[0], states.shape[-1])
        if next_states.ndim == 3:
            next_states = next_states.reshape(next_states.shape[0], next_states.shape[-1])

        q_values = self.model(states, training=False).numpy()
        q_next = self.target_model(next_states, training=False).numpy()

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            inputs[i] = state.reshape(-1)
            targets[i] = q_values[i]
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * np.max(q_next[i])

        return inputs, targets