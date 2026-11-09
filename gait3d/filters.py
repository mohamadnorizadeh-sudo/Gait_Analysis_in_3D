"""One-Euro filter (Casiez et al., CHI 2012).

v1 used a fixed exponential average (SMOOTH_FACTOR = 0.35), which either lags
during fast swing phase or jitters during stance.  The One-Euro filter adapts
its cut-off to the signal speed: heavy smoothing when the knee is still,
almost no lag when it moves fast.
"""

import math


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x, self._dx = _LowPass(), _LowPass()
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x, self._dx, self._t = _LowPass(), _LowPass(), None

    def __call__(self, x, t):
        if self._t is None:
            self._t = t
            self._dx(0.0, 1.0)
            return self._x(x, 1.0)
        dt = max(1e-4, t - self._t)
        self._t = t
        prev = self._x.y
        dx = self._dx((x - prev) / dt, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(dx)
        return self._x(x, self._alpha(cutoff, dt))
