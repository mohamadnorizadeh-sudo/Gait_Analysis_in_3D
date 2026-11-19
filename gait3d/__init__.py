"""gait3d - rigid, rotation-invariant 3D knee-angle tracking for a Pi 5 stereo rig.

v2 package layout
-----------------
stereo_geometry.py   load stereo_calibration.npz, rectification maps, triangulation, pixel rays
rigid_leg.py         bone-length-constrained 3D leg solver (ray-sphere + robust least squares)
filters.py           One-Euro filter for jitter-free angles
pose.py              YOLOv8-Pose wrapper: confidence filter, ghost-leg suppression, stereo matching
camera.py            synchronised dual Picamera2 capture with locked focus/exposure/white balance
profile.py           standing bone-length calibration phase (L_thigh, L_shin)
"""

__version__ = "2.0.0"
