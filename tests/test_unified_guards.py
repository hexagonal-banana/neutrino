"""CPU tests for capture failures and allocation/launch guards."""
import contextlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'examples/moe_training'))
from neutrino import capture
import gpu_guard
import launch
import worker


class CaptureTests(unittest.TestCase):
    def test_missing_runtime_is_not_silent_success(self):
        with patch.dict(os.environ, {'NEUTRINO_SESSION_ID': 'test-session'}), \
                patch.object(capture.ctypes, 'CDLL', return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, 'not loaded'):
                with capture.capture_range('step', 3):
                    self.fail('Must not enter capture without a collector')

    def test_begin_end_and_exception(self):
        begin, end = Mock(return_value=0), Mock(return_value=0)
        with patch.object(capture, 'require_runtime', return_value=(begin, end)):
            with capture.capture_range('step', 3):
                pass
            begin.assert_called_once_with(b'step', 3)
            end.assert_called_once_with(1)
            with self.assertRaisesRegex(ValueError, 'workload failed'):
                with capture.capture_range('step', 4):
                    raise ValueError('workload failed')
            end.assert_called_with(0)
            with capture.capture_range('step', 5):
                with self.assertRaisesRegex(RuntimeError, 'Nested'):
                    with capture.capture_range('nested', 0):
                        pass

    def test_native_failures(self):
        for begin_status, end_status, expected in ((1, 0, 'begin'), (0, 1, 'end/drain')):
            with patch.object(capture, 'require_runtime',
                              return_value=(Mock(return_value=begin_status), Mock(return_value=end_status))):
                with self.assertRaisesRegex(RuntimeError, expected):
                    with capture.capture_range('step', 1):
                        pass

    def test_invalid_ranges(self):
        for name, ordinal in (('', 1), ('bad\0name', 1), ('step', True), ('step', -1), ('step', 2**63)):
            with self.subTest(name=name, ordinal=ordinal), self.assertRaises(ValueError):
                with capture.capture_range(name, ordinal):
                    pass


class GPUGuardTests(unittest.TestCase):
    def telemetry(self, *, memory=0, utilization=0, count=8):
        return '\n'.join(f'{i}, GPU-{i}, NVIDIA H800, {memory}, {utilization}' for i in range(count))

    def test_idle_and_stable_identity(self):
        runner = Mock(side_effect=[self.telemetry(), ''] * 6)
        sleep = Mock()
        self.assertEqual(len(gpu_guard.require_idle(runner, sleep)), 8)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(runner.call_count, 12)

    def test_busy_and_incomplete_telemetry(self):
        for telemetry in (self.telemetry(memory=1025), self.telemetry(utilization=1),
                          self.telemetry(count=7), 'Not Supported',
                          self.telemetry().replace('GPU-1', 'GPU-0')):
            with self.subTest(telemetry=telemetry), self.assertRaises(gpu_guard.UnsafeGPUState):
                gpu_guard.check_once(Mock(side_effect=[telemetry, '']))
        with self.assertRaises(gpu_guard.UnsafeGPUState):
            gpu_guard.check_once(Mock(side_effect=[self.telemetry(), 'GPU-0, 9999']))

    def test_identity_changes(self):
        runner = Mock(side_effect=[self.telemetry(), '', self.telemetry().replace('GPU-0', 'GPU-new'), ''])
        with self.assertRaisesRegex(gpu_guard.UnsafeGPUState, 'identities changed'):
            gpu_guard.require_idle(runner, Mock())


class LaunchTests(unittest.TestCase):
    def test_dry_run_never_queries_or_launches_gpu(self):
        with patch.object(sys, 'argv', ['launch.py']), contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.object(launch, 'require_idle', side_effect=AssertionError('telemetry forbidden')), \
                patch.object(launch.subprocess, 'Popen', side_effect=AssertionError('workers forbidden')):
            self.assertEqual(launch.main(), 0)
        self.assertIn('"dense_dp": 8', output.getvalue())
        self.assertIn('--nproc-per-node=8', output.getvalue())
        self.assertIn('--max-restarts=0', output.getvalue())

    def test_worker_refuses_unguarded_execution_before_framework_imports(self):
        with patch.dict(os.environ, {}, clear=True), patch('builtins.__import__') as imports:
            with self.assertRaisesRegex(SystemExit, 'confirmed allocation'):
                worker.main()
            imports.assert_not_called()

    def test_execute_requires_allocation(self):
        with patch.object(sys, 'argv', ['launch.py', '--execute']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                launch.main()
            self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
