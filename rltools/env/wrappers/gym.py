from gym import Wrapper


class FrameSkip(Wrapper):
    def __init__(self, env, frames_number):
        self.env = env
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return next_obs, R, done, info

    def reset(self):
        return self.env.reset()

    # @property
    # def action_space(self):
    #     return self.env.action_space
