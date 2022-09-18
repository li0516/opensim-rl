import numpy as np


# Ornstein-uhlenbeck process-g correlated noise uusgehed ashiglana

# http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class ActionNoise(object):
    def reset(self):
        pass


# Based on https://github.com/openai/baselines/tree/master/baselines/ddpg
class OrnsteinUhlenbeckActionNoiseBaseline(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1)
    ou_baseline = OrnsteinUhlenbeckActionNoiseBaseline(mu=np.zeros(1), sigma=float(0.2) * np.ones(1))

    states = []
    states2 = []

    for i in range(1000):
        states.append(ou.sample())
        states2.append(ou_baseline())

    import matplotlib.pyplot as plt

    plt.plot(states, label="OU")
    plt.plot(states2, label="OU Baseline")
    plt.legend()
    plt.show()
