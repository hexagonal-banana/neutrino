"""Clock fits use fresh synthetic domains, never historical calibration files."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.clocks import ClockError, fit_clock


class ClockTests(unittest.TestCase):
    def samples(self, gpu, host):
        return ([[host - 10, gpu, host + 10], [host + 990, gpu + 1000, host + 1010]],
                [[host + 999990, gpu + 1000000, host + 1000010],
                 [host + 1000990, gpu + 1001000, host + 1001010]])

    def test_large_integer_ticks_preserve_one_nanosecond(self):
        gpu, host = 1793631081251474368, 992364653077632
        clock = fit_clock(*self.samples(gpu, host))
        self.assertEqual(clock.map_ns(gpu + 200001), host + 200001)
        self.assertEqual(clock.map_ns(gpu + 200002) - clock.map_ns(gpu + 200001), 1)
        self.assertEqual(clock.residual_ns, 0)
        self.assertGreaterEqual(clock.uncertainty_ns(gpu + 200001), 10)
        self.assertIn('conditional', clock.manifest()['uncertainty'])

    def test_cross_gpu_offset_maps_to_one_host_domain(self):
        one = fit_clock(*self.samples(2**61, 10**15))
        two = fit_clock(*self.samples(2**61 + 123456789, 10**15))
        self.assertEqual(one.map_ns(2**61 + 50000), two.map_ns(2**61 + 123506789))
        self.assertGreater(one.uncertainty_ns(2**61 + 50000) +
                           two.uncertainty_ns(2**61 + 123506789), 0)

    def test_affine_drift_and_odd_midpoints(self):
        before = [[1000, 2**60, 1001], [1999, 2**60 + 1000, 2004]]
        after = [[1001099, 2**60 + 1000000, 1001100],
                 [1002100, 2**60 + 1001000, 1002102]]
        clock = fit_clock(before, after)
        self.assertEqual(clock.map_ns(2**60 + 500000), 501050)
        self.assertGreater(clock.manifest()['drift_ppm'], 0)

    def test_rejects_reset_bad_domain_and_extrapolation(self):
        before, after = self.samples(100000, 1000000)
        for left, right in ((before[:1], after), (before[::-1], after), (after, before),
                            ([[10, 20, 9], [20, 30, 40]], after),
                            ([[1, 100, 2], [3, 101, 4]], [[10000, 102, 10001], [10002, 103, 10003]])):
            with self.assertRaises(ClockError):
                fit_clock(left, right)
        clock = fit_clock(before, after)
        for tick in (clock.first[1] - 1, clock.last[1] + 1, float(clock.first[1])):
            with self.assertRaises(ClockError):
                clock.map_ns(tick)


if __name__ == '__main__':
    unittest.main()
