"""Rigid-link 3D leg solver: bone lengths are physical constants.

Why the v1 hand-off formula is not enough
-----------------------------------------
The proposed correction ``H_fixed = K + L_thigh * u/|u|`` with ``u = H_raw - K``
keeps the *direction* of u and only changes its *length*.  The knee angle

    theta = arccos( (H-K).(A-K) / (|H-K| |A-K|) )

does not depend on vector lengths, so that correction leaves theta
**exactly unchanged** (see tests/test_rigid_leg.py::test_naive_rescale_does_not_change_angle).
If the measured depth of H is wrong, its direction from K is wrong, and the
angle stays wrong.

What v2 does instead (still closed-form at its core)
---------------------------------------------------
The 2D pixel of a joint is reliable; its depth is not.  So each joint is
constrained to lie on its camera ray ``P(Z) = Z * r`` (r = pixel ray, r_z = 1)
and the *depth* along that ray is what we solve for:

1. Ray-sphere intersection (closed form).  With K known, the hip must satisfy
   |Z r_H - K| = L_thigh, i.e.

       (r.r) Z^2 - 2 (r.K) Z + (K.K - L^2) = 0

   which gives two roots (front/back of the sphere); we take the root closest
   to the measured depth or, if that joint's depth is missing, to the previous
   frame.  The same is done for the ankle with L_shin.  This *does* rotate the
   bone to the right direction, so the angle becomes rotation-invariant, and it
   fills in missing Z without ever falling back to 2D.

2. Choosing the knee depth.  The knee anchors both spheres, so its depth is
   chosen by a tiny 3-variable robust least-squares problem over (Z_H, Z_K, Z_A):

       bone terms        (|P_H-P_K| - L_thigh)/s_L , (|P_A-P_K| - L_shin)/s_L
       measurement terms (Z_j - Z_j_meas)/sigma_Z(j)      (Huber-weighted, skipped if missing)
       temporal terms    (Z_j - Z_j_prev)/s_prior          (weak, keeps the right root)

   then step 1 is applied exactly so the final lengths equal L_thigh/L_shin to
   floating-point precision.  Cost: ~10 Levenberg-Marquardt iterations on 3
   unknowns -> microseconds on the Pi 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

JOINTS = ("hip", "knee", "ankle")


# ---------------------------------------------------------------------- #
# basic maths
# ---------------------------------------------------------------------- #
def joint_angle_deg(hip, knee, ankle) -> Optional[float]:
    """Interior knee angle (180 = straight leg)."""
    v1 = np.asarray(hip, float) - np.asarray(knee, float)
    v2 = np.asarray(ankle, float) - np.asarray(knee, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def naive_rescale(parent, child_raw, length):
    """The hand-off formula, kept for reference/tests: P = C + L * u/|u|."""
    u = np.asarray(child_raw, float) - np.asarray(parent, float)
    n = np.linalg.norm(u)
    if n < 1e-9:
        return np.asarray(child_raw, float)
    return np.asarray(parent, float) + length * u / n


def ray_sphere_depths(ray, center, radius):
    """Depths Z where the point Z*ray is exactly `radius` from `center`.

    Returns (roots, hit).  If the ray misses the sphere (2D keypoint and bone
    length are inconsistent by a few mm) we return the depth of the closest
    approach, hit=False -- the best physically consistent estimate.
    """
    r = np.asarray(ray, float)
    c = np.asarray(center, float)
    a = r @ r
    b = -2.0 * (r @ c)
    cc = c @ c - radius * radius
    disc = b * b - 4 * a * cc
    if disc < 0:
        return np.array([-b / (2 * a)]), False
    s = np.sqrt(disc)
    return np.array([(-b - s) / (2 * a), (-b + s) / (2 * a)]), True


def closest_root(roots, target):
    roots = np.asarray(roots, float)
    roots = roots[roots > 1.0] if np.any(roots > 1.0) else roots
    return float(roots[np.argmin(np.abs(roots - target))])


# ---------------------------------------------------------------------- #
# solver
# ---------------------------------------------------------------------- #
@dataclass
class LegResult:
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    angle_deg: float               # interior angle, 180 = straight
    source: Dict[str, str]         # per joint: "measured" | "constrained" | "prior"
    residual_mm: float             # worst |bone length - L| after solve (should be ~0)

    @property
    def flexion_deg(self) -> float:
        """Clinical knee flexion (0 = straight)."""
        return 180.0 - self.angle_deg

    @property
    def thigh_mm(self) -> float:
        return float(np.linalg.norm(self.hip - self.knee))

    @property
    def shin_mm(self) -> float:
        return float(np.linalg.norm(self.ankle - self.knee))


@dataclass
class RigidLegSolver:
    L_thigh: float
    L_shin: float
    depth_sigma_fn: callable = None     # z -> sigma_z (mm); default 2% of z
    bone_sigma_mm: float = 5.0          # how "hard" the bone constraint is inside LM
    prior_sigma_mm: float = 60.0        # temporal prior on depth (gait moves < ~60 mm per frame at 30 fps)
    huber_k: float = 2.0                # outlier rejection for measured depths
    max_prior_age: int = 15             # frames a prior stays usable
    _prev: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _prev_age: int = field(default=0, init=False, repr=False)

    def reset(self):
        self._prev, self._prev_age = None, 0

    # .................................................................. #
    def _sigma(self, z):
        if self.depth_sigma_fn is not None:
            return max(1.0, float(self.depth_sigma_fn(z)))
        return max(1.0, 0.02 * float(z))

    def solve(self, rays: Dict[str, np.ndarray], meas_z: Dict[str, Optional[float]]) -> Optional[LegResult]:
        """rays: joint -> pixel ray (r_z = 1).  meas_z: joint -> measured depth or None."""
        prev = self._prev if (self._prev is not None and self._prev_age <= self.max_prior_age) else None
        names = JOINTS
        R = [np.asarray(rays[n], float) for n in names]
        zm = [meas_z.get(n) for n in names]
        zm = [None if (z is None or not np.isfinite(z) or z <= 0) else float(z) for z in zm]

        # ---- initial guess ------------------------------------------------
        init = []
        known = [z for z in zm if z is not None]
        for i, n in enumerate(names):
            if zm[i] is not None:
                init.append(zm[i])
            elif prev is not None:
                init.append(prev[n])
            elif known:
                init.append(float(np.median(known)))
            else:
                init.append(None)
        if any(v is None for v in init):
            # Nothing measured this frame and no recent history: no 3D lock.
            # (We deliberately do NOT fall back to a 2D pixel angle.)
            self._prev_age += 1
            return None
        z = np.array(init, float)

        # ---- robust Levenberg-Marquardt over the three depths --------------
        def residuals(zv, w_meas):
            P = [zv[i] * R[i] for i in range(3)]
            res = [
                (np.linalg.norm(P[0] - P[1]) - self.L_thigh) / self.bone_sigma_mm,
                (np.linalg.norm(P[2] - P[1]) - self.L_shin) / self.bone_sigma_mm,
            ]
            for i in range(3):
                if zm[i] is not None:
                    res.append(np.sqrt(w_meas[i]) * (zv[i] - zm[i]) / self._sigma(zm[i]))
                if prev is not None:
                    res.append((zv[i] - prev[names[i]]) / self.prior_sigma_mm)
            return np.array(res)

        w = np.ones(3)
        lam = 1e-2
        for _ in range(25):
            # Huber IRLS weights for measurement terms
            for i in range(3):
                if zm[i] is not None:
                    e = abs(z[i] - zm[i]) / self._sigma(zm[i])
                    w[i] = 1.0 if e <= self.huber_k else self.huber_k / e
            r0 = residuals(z, w)
            J = np.empty((r0.size, 3))
            for k in range(3):
                dz = np.zeros(3)
                dz[k] = 0.5
                J[:, k] = (residuals(z + dz, w) - r0) / 0.5
            A = J.T @ J
            g = J.T @ r0
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            z_new = np.maximum(z + step, 10.0)
            if np.sum(residuals(z_new, w) ** 2) < np.sum(r0 ** 2):
                z, lam = z_new, lam * 0.3
                if np.max(np.abs(step)) < 0.05:
                    break
            else:
                lam *= 10.0

        # ---- exact closed-form projection onto the bone spheres -----------
        K = z[1] * R[1]
        roots_h, hit_h = ray_sphere_depths(R[0], K, self.L_thigh)
        roots_a, hit_a = ray_sphere_depths(R[2], K, self.L_shin)
        if not (hit_h and hit_a):
            # Pixel noise put a ray just outside its bone sphere.  Slide the knee
            # along its own ray to the nearest depth where BOTH spheres are hit,
            # so the bone lengths stay exact.
            for dk in np.concatenate([[0.0], np.ravel(np.column_stack([np.arange(1, 81), -np.arange(1, 81)]))]):
                Kc = (z[1] + dk) * R[1]
                rh, ok_h = ray_sphere_depths(R[0], Kc, self.L_thigh)
                ra, ok_a = ray_sphere_depths(R[2], Kc, self.L_shin)
                if ok_h and ok_a:
                    z[1], K, roots_h, roots_a = z[1] + dk, Kc, rh, ra
                    break
        z_h = closest_root(roots_h, z[0])
        z_a = closest_root(roots_a, z[2])
        H, A = z_h * R[0], z_a * R[2]

        # per-joint provenance
        src = {}
        for i, n, zf in zip(range(3), names, (z_h, z[1], z_a)):
            if zm[i] is None:
                src[n] = "constrained" if any(v is not None for v in zm) else "prior"
            elif abs(zf - zm[i]) <= 2.5 * self._sigma(zm[i]):
                src[n] = "measured"
            else:
                src[n] = "constrained"     # measured depth rejected as outlier

        resid = max(abs(np.linalg.norm(H - K) - self.L_thigh), abs(np.linalg.norm(A - K) - self.L_shin))
        ang = joint_angle_deg(H, K, A)
        if ang is None:
            self._prev_age += 1
            return None
        self._prev = {"hip": z_h, "knee": float(z[1]), "ankle": z_a}
        self._prev_age = 0
        return LegResult(H, K, A, ang, src, float(resid))
