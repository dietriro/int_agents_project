import random
import numpy as np

class OU(object):

    def function(self, x, mu, theta, sigma):
        r = np.random.randn(1)
        # print(r)
        return theta * (mu - x) + sigma * r