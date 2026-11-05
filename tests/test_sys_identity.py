"""Identity contracts independent of GPU availability or historical artifacts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.identity import ArtifactIdentity, IdentityError, ProcessIdentity, Registry


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.process = ProcessIdentity('session', 123, 'process-start-token', 0)
        self.registry = Registry(self.process)
        self.artifact = ArtifactIdentity('0' * 64, 'sm_90', 'cubin')
        self.registry.context_created(10, 'GPU-test')
        self.registry.module_loaded(10, 20, self.artifact)
        self.kernel = self.registry.kernel(10, 20, 'shared_symbol')

    def test_code_and_live_instance_are_distinct(self):
        self.registry.module_loaded(10, 21, self.artifact)
        other = self.registry.kernel(10, 21, 'shared_symbol')
        self.assertNotEqual(self.kernel.id, other.id)
        self.assertEqual(self.kernel.module.artifact.id, other.module.artifact.id)
        self.assertIs(self.kernel, self.registry.kernel(10, 20, 'shared_symbol'))
        self.assertNotEqual(self.kernel.id, self.registry.kernel(10, 20, 'shared_symbo').id)
        self.assertNotEqual(self.artifact.id, replace(self.artifact, target='sm_90a').id)
        self.assertNotEqual(self.artifact.id, replace(self.artifact, original_sha256='1' * 64).id)

    def test_reused_handles_and_retained_history(self):
        before = self.registry.launch(self.kernel, 0)
        self.registry.context_retired(10)
        with self.assertRaisesRegex(IdentityError, 'retired'):
            self.registry.launch(self.kernel, 0)
        context = self.registry.context_created(10, 'GPU-test')
        self.assertEqual(context.generation, 2)
        self.registry.module_loaded(10, 20, self.artifact)
        kernel = self.registry.kernel(10, 20, 'shared_symbol')
        self.assertNotEqual(kernel.id, self.kernel.id)
        self.assertEqual(self.registry.launch_history[before.id], before)
        self.assertNotEqual(before.id, self.registry.launch(kernel, 0).id)

    def test_module_unload_does_not_allow_stale_function(self):
        self.registry.module_retired(10, 20)
        with self.assertRaisesRegex(IdentityError, 'retired'):
            self.registry.launch(self.kernel, 1)
        self.registry.module_loaded(10, 20, self.artifact)
        self.assertNotEqual(self.kernel.id, self.registry.kernel(10, 20, 'shared_symbol').id)

    def test_concurrent_launches_are_process_unique(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            launches = list(pool.map(lambda stream: self.registry.launch(self.kernel, stream), range(128)))
        self.assertEqual(len({launch.id for launch in launches}), 128)
        self.assertEqual(sorted(launch.ordinal for launch in launches), list(range(1, 129)))
        self.assertEqual(len(self.registry.launch_history), 128)

    def test_session_pid_and_process_reuse(self):
        ids = {self.process.id}
        for change in ({'session': 'other'}, {'pid': 124}, {'instance': 'new-process'}, {'rank': 1}):
            ids.add(replace(self.process, **change).id)
        self.assertEqual(len(ids), 5)
        # Explicit structure prevents separator/canonicalization collisions.
        self.assertNotEqual(ProcessIdentity('a:b', 1, 'c').id, ProcessIdentity('a', 1, 'b:c').id)

    def test_boolean_handle_cannot_alias_live_integer_handle(self):
        self.registry.context_created(1, 'GPU-test')
        self.registry.module_loaded(1, 1, self.artifact)
        for action in (lambda: self.registry.context_retired(True),
                       lambda: self.registry.module_retired(1, True),
                       lambda: self.registry.kernel(True, 1, 'symbol')):
            with self.assertRaises(IdentityError):
                action()
        self.assertEqual(self.registry.kernel(1, 1, 'symbol').entry, 'symbol')

    def test_fail_closed_and_failed_allocations_do_not_consume_ordinals(self):
        with self.assertRaises(IdentityError):
            self.registry.context_created(10, 'GPU-test')
        with self.assertRaises(IdentityError):
            self.registry.module_loaded(10, 20, self.artifact)
        with self.assertRaises(IdentityError):
            self.registry.kernel(99, 20, 'shared_symbol')
        with self.assertRaises(IdentityError):
            self.registry.launch(self.kernel, True)
        self.assertEqual(self.registry.launch(self.kernel, 0).ordinal, 1)
        for value in (True, -1, 2**64):
            with self.assertRaises(IdentityError):
                ProcessIdentity('session', value, 'instance')
        with self.assertRaises(IdentityError):
            ArtifactIdentity('bad-hash', 'sm_90', 'cubin')
        with self.assertRaises(IdentityError):
            replace(self.artifact, target='sm_90;other')


if __name__ == '__main__':
    unittest.main()
