"""ML-accelerated depletion: a learned transport model with physics checks.

The transport response (one-group cross sections of every tracked nuclide
and k_eff) varies smoothly with the salt composition.  A local multi-output
ridge model in relative-composition features (fitted on a sliding window of
recent physics evaluations) replaces most transport solves:

* physics is used during warm-up and whenever the model is not *trusted*;
* every ``check_every`` ML calls a physics evaluation validates the model —
  trust is granted only while the check errors stay below tolerance;
* queries outside the window's feature range (extrapolation) go to physics;
* the flux follows exactly from power normalization (energy conserved).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..constants import MEV_TO_J

KEY_FEATURES = ("Th232", "Pa233", "U233", "U234", "U235", "U236", "U238", "Np237", "Pu239",
                "Pu240", "Pu241", "Xe135", "Sm149", "Sm151", "LFP_NG", "LFP_NM", "LFP_RE",
                "LFP_SS", "Li6")


class MLAcceleratedTransport:
    def __init__(self, base, design, warmup: int = 10, check_every: int = 5,
                 tol_k_pcm: float = 30.0, tol_xs: float = 0.01, window: int = 14,
                 alpha: float = 1e-3, extrapolation_margin: float = 0.15):
        self.base, self.design = base, design
        self.warmup, self.check_every = warmup, check_every
        self.tol_k, self.tol_xs, self.window, self.alpha = tol_k_pcm, tol_xs, window, alpha
        self.margin = extrapolation_margin
        self.V_total = design.salt_volumes_m3()["total"] * 1e6
        self.F, self.T = [], []
        self.keys = None
        self.model = None
        self.trusted = False
        self.calls = 0
        self.physics_calls = 0
        self.ml_calls = 0
        self.check_log = []        # (call index, k error pcm, max rel XS error, passed)
        self._since_check = 0

    def _features(self, salt):
        hm = sum(v for k, v in salt.N.items() if k.startswith(("Th", "U", "Pu", "Pa", "Np")))
        return np.array([salt.N.get(n, 0.0) / hm for n in KEY_FEATURES])

    def _targets(self, og, k):
        if self.keys is None:
            self.keys = [(n, rx) for n, d in og.items() for rx, v in d.items() if v > 0]
        return np.array([np.log(max(og.get(n, {}).get(rx, 1e-300), 1e-300))
                         for n, rx in self.keys] + [k])

    def _from_targets(self, y):
        og = {}
        for (n, rx), v in zip(self.keys, y[:-1]):
            og.setdefault(n, {})[rx] = float(np.exp(v))
        for d in og.values():
            for rx in ("capture", "fission", "nalpha", "n2n", "tritium", "absorption", "nu_fission"):
                d.setdefault(rx, 0.0)
        return og, float(y[-1])

    def _fit(self):
        F = np.array(self.F[-self.window:])
        T = np.array(self.T[-self.window:])
        self.scaler = StandardScaler().fit(F)
        self.model = Ridge(alpha=self.alpha).fit(self.scaler.transform(F), T)
        self.f_lo, self.f_hi = F.min(axis=0), F.max(axis=0)

    def _predict(self, f):
        return self.model.predict(self.scaler.transform(f[None, :]))[0]

    def _inside(self, f) -> bool:
        span = np.maximum(self.f_hi - self.f_lo, 1e-12 * np.maximum(np.abs(self.f_hi), 1e-30))
        lo = self.f_lo - self.margin * span
        hi = self.f_hi + self.margin * span
        return bool(np.all((f >= lo) & (f <= hi)))

    def _phi(self, salt, og, P):
        from ..xs.library import load_library
        lib = load_library()
        kf = sum(salt.N.get(n, 0.0) * lib[n].kappa * d.get("fission", 0.0) for n, d in og.items()
                 if n in salt.N)
        return P / (kf * MEV_TO_J * self.V_total)

    def _physics(self, salt, P, f, check_pred=None):
        og, phi, k = self.base.evaluate(salt, P)
        self.physics_calls += 1
        y = self._targets(og, k)
        if check_pred is not None:
            ek = abs(check_pred[-1] - k) * 1e5
            exs = float(np.max(np.abs(np.exp(check_pred[:-1] - y[:-1]) - 1.0)))
            passed = ek <= self.tol_k and exs <= self.tol_xs
            self.check_log.append((self.calls, ek, exs, passed))
            self.trusted = passed
        self.F.append(f)
        self.T.append(y)
        if len(self.F) >= 3:
            self._fit()
        return og, phi, k

    def evaluate(self, salt, P):
        self.calls += 1
        f = self._features(salt)
        if self.model is None or len(self.F) < self.warmup:
            return self._physics(salt, P, f)
        pred = self._predict(f)
        if not self.trusted or not self._inside(f):
            # validation-by-physics: use the physics answer and test the model on it
            self._since_check = 0
            return self._physics(salt, P, f, check_pred=pred)
        if self._since_check >= self.check_every:
            self._since_check = 0
            return self._physics(salt, P, f, check_pred=pred)
        self.ml_calls += 1
        self._since_check += 1
        og, k = self._from_targets(pred)
        return og, self._phi(salt, og, P), k

    @property
    def speedup(self) -> float:
        return self.calls / max(self.physics_calls, 1)
