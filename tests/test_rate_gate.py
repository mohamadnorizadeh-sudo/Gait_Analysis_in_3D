import pytest

from gait3d.tracker import AngleRateGate


def test_short_spike_hidden_real_bend_kept():
    g = AngleRateGate(max_rate=350, relock_frames=6)
    # (allowance is capped at max_dt, so a spike never 'earns' its way in)
    dt = 0.09
    ok = [g(i * dt, 10.0) for i in range(5)]
    assert all(ok)
    # 3-frame flip to 110 deg: hidden
    assert not any(g((5 + i) * dt, 110.0) for i in range(3))
    assert g(8 * dt, 12.0)
    # a real swing 10 -> 55 over 0.3 s is within the limit
    assert all(g((9 + i) * dt, 10 + 15 * i) for i in range(4))
    # a real sustained big bend is accepted after relock_frames
    res = [g((13 + i) * dt, 100.0) for i in range(8)]
    assert not res[0] and res[5] and res[7]


def test_after_gap_anything_accepted():
    g = AngleRateGate()
    assert g(0.0, 5.0)
    assert g(1.5, 90.0)


def test_eight_frame_spike_hidden():
    g = AngleRateGate()
    dt = 0.09
    assert all(g(i * dt, 10.0) for i in range(5))
    assert not any(g((5 + i) * dt, 110.0) for i in range(7))
    assert g(12 * dt, 15.0)
