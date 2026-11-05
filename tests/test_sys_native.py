"""CPU-only native launch marshalling and recursion tests; no CUDA loader calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = r'''
#include "sys_launch.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <thread>
#include <type_traits>
using namespace neutrino::sys;

template<class Function> void rejects(Function function) {
    bool caught = false;
    try { function(); } catch (const ArgumentError&) { caught = true; }
    assert(caught);
}

int main() {
    static_assert(!std::is_copy_constructible_v<PreparedArguments>);
    static_assert(!std::is_move_constructible_v<PreparedArguments>);
    static_assert(!std::is_copy_constructible_v<OwnedLaunchConfig>);
    ParameterLayout layout({{0, 4, 4}, {16, 16, 16}, {32, 1, 1}}, 33, 32764);
    assert(layout.context_offset() == 40 && layout.extended_size() == 48);
    std::uint32_t first = 0x12345678;
    std::array<std::uint8_t, 16> blob{};
    for (std::size_t i = 0; i < blob.size(); ++i) blob[i] = i + 10;
    std::uint8_t last = 77;
    void* params[] = {&first, blob.data(), &last};
    const std::uint64_t context = 0x1020304050607080;
    PreparedArguments args(layout, params, nullptr, context);
    assert(args.size() == 48 && args.extra() == nullptr);
    assert(reinterpret_cast<std::uintptr_t>(args.data()) % 128 == 0);
    assert(std::memcmp(args.data(), &first, 4) == 0);
    assert(std::memcmp(args.data() + 16, blob.data(), 16) == 0);
    assert(std::memcmp(args.data() + 32, &last, 1) == 0);
    assert(std::memcmp(args.parameters()[3], &context, 8) == 0);
    first = 0; blob.fill(0); last = 0;
    assert(*static_cast<const std::uint32_t*>(args.parameters()[0]) == 0x12345678);
    for (std::size_t i = 4; i < 16; ++i) assert(args.data()[i] == std::byte{0});
    for (std::size_t i = 33; i < 40; ++i) assert(args.data()[i] == std::byte{0});

    std::array<std::byte, 40> packed;
    packed.fill(std::byte{0x5a});
    std::size_t bytes = 33;
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes,
                     CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(), CU_LAUNCH_PARAM_END};
    PreparedArguments copied(layout, nullptr, extra, context);
    assert(copied.parameters() == nullptr);
    assert(copied.extra()[0] == CU_LAUNCH_PARAM_BUFFER_POINTER);
    assert(copied.extra()[2] == CU_LAUNCH_PARAM_BUFFER_SIZE);
    assert(*static_cast<std::size_t*>(copied.extra()[3]) == 48);
    assert(copied.extra()[4] == CU_LAUNCH_PARAM_END);
    assert(std::memcmp(copied.data(), packed.data(), bytes) == 0);
    bytes = 40;
    PreparedArguments padded(layout, nullptr, extra, 0);
    assert(std::memcmp(padded.data(), packed.data(), bytes) == 0);
    for (std::size_t i = 40; i < 48; ++i) assert(padded.data()[i] == std::byte{0});
    bytes = 32; rejects([&] { PreparedArguments bad(layout, nullptr, extra, 0); });
    bytes = 41; rejects([&] { PreparedArguments bad(layout, nullptr, extra, 0); });
    bytes = 33; rejects([&] { PreparedArguments bad(layout, params, extra, 0); });
    extra[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    rejects([&] { PreparedArguments bad(layout, nullptr, extra, 0); });
    extra[0] = reinterpret_cast<void*>(999);
    rejects([&] { PreparedArguments bad(layout, nullptr, extra, 0); });
    extra[0] = CU_LAUNCH_PARAM_BUFFER_SIZE; extra[4] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    rejects([&] { PreparedArguments bad(layout, nullptr, extra, 0); });
    rejects([&] { PreparedArguments bad(layout, nullptr, nullptr, 0); });
    params[1] = nullptr;
    rejects([&] { PreparedArguments bad(layout, params, nullptr, 0); });
    ParameterLayout empty({}, 0, 8);
    PreparedArguments zero(empty, nullptr, nullptr, context);
    assert(std::memcmp(zero.parameters()[0], &context, 8) == 0);
    ParameterLayout large({{0, 4096, 16}}, 4096, 32764);
    std::array<std::byte, 4096> large_blob{};
    void* large_params[] = {large_blob.data()};
    PreparedArguments large_args(large, large_params, nullptr, context);
    assert(large_args.size() == 4104);
    ParameterLayout prefix({{0, 4096, 16}}, 4096, 32764, 16);
    std::array<std::byte, 16> live_prefix{};
    live_prefix.fill(std::byte{0x6a});
    std::size_t prefix_size = live_prefix.size();
    void* prefix_extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, live_prefix.data(),
                            CU_LAUNCH_PARAM_BUFFER_SIZE, &prefix_size, CU_LAUNCH_PARAM_END};
    PreparedArguments prefix_args(prefix, nullptr, prefix_extra, context);
    assert(std::memcmp(prefix_args.data(), live_prefix.data(), 16) == 0);
    for (std::size_t i = 16; i < 4096; ++i) assert(prefix_args.data()[i] == std::byte{0});
    assert(std::memcmp(prefix_args.data() + 4096, &context, 8) == 0);
    rejects([&] { PreparedArguments bad(large, nullptr, prefix_extra, context); });
    prefix_size = 15;
    rejects([&] { PreparedArguments bad(prefix, nullptr, prefix_extra, context); });
    rejects([] { ParameterLayout bad({{0, 8, 8}, {8, 8, 8}}, 16, 32764, 8); });
    rejects([] { ParameterLayout bad({{0, 16, 16}}, 16, 32764, 17); });
    rejects([] { ParameterLayout bad({{0, 4096, 16}}, 4096, 4096); });
    rejects([] { ParameterLayout bad({{1, 4, 4}}, 5, 32764); });
    rejects([] { ParameterLayout bad({{0, 4, 3}}, 4, 32764); });
    rejects([] { ParameterLayout bad({}, 4, 32764); });
    rejects([] { ParameterLayout bad({}, 0, 0); });
    rejects([] { ParameterLayout bad({}, std::numeric_limits<std::size_t>::max(),
                                     std::numeric_limits<std::size_t>::max()); });

    CUlaunchAttribute attrs[2]{};
    attrs[0].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attrs[0].value.cooperative = 1;
    attrs[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attrs[1].value.programmaticStreamSerializationAllowed = 1;
    CUlaunchConfig config{2, 3, 4, 32, 2, 1, 1024, reinterpret_cast<CUstream>(123), attrs, 2};
    OwnedLaunchConfig owned(config);
    assert(owned.get()->gridDimZ == 4 && owned.get()->blockDimY == 2);
    assert(owned.get()->sharedMemBytes == 1024 && owned.get()->hStream == config.hStream);
    assert(owned.get()->attrs != attrs && owned.get()->numAttrs == 2);
    assert(std::memcmp(owned.get()->attrs, attrs, sizeof(attrs)) == 0);
    attrs[0].value.cooperative = 0; config.gridDimX = 0;
    assert(owned.get()->gridDimX == 2 && owned.get()->attrs[0].value.cooperative == 1);
    assert(owned.get()->attrs[1].value.programmaticStreamSerializationAllowed == 1);
    rejects([&] { OwnedLaunchConfig bad(config); });
    config.gridDimX = 2; config.attrs = nullptr;
    rejects([&] { OwnedLaunchConfig bad(config); });

    std::atomic<unsigned> outer{0}, inner{0};
    auto submit = [&] {
        SubmissionScope runtime;
        if (runtime.outermost()) ++outer;
        SubmissionScope driver;
        if (!driver.outermost()) ++inner;
    };
    std::thread a([&] { for (int i = 0; i < 100; ++i) submit(); });
    std::thread b([&] { for (int i = 0; i < 100; ++i) submit(); });
    a.join(); b.join();
    assert(outer == 200 && inner == 200);
    try { SubmissionScope throwing; throw 1; } catch (int) {}
    SubmissionScope final;
    assert(final.outermost());
}
'''


BRIDGE = r'''
#include "sys_launch.hpp"
using namespace neutrino::sys;
extern "C" void* fixturePrepare(int blob, void** parameters, void** extra, std::uint64_t context) {
    try {
        ParameterLayout layout(blob ? std::vector<Parameter>{{0, 4096, 16}} : std::vector<Parameter>{},
                               blob ? 4096 : 0, 32764);
        return new PreparedArguments(layout, parameters, extra, context);
    } catch (...) { return nullptr; }
}
extern "C" void** fixtureParameters(void* handle) {
    return static_cast<PreparedArguments*>(handle)->parameters();
}
extern "C" void** fixtureExtra(void* handle) {
    return static_cast<PreparedArguments*>(handle)->extra();
}
extern "C" void fixtureRelease(void* handle) {
    delete static_cast<PreparedArguments*>(handle);
}
'''


def compile_bridge(root, env):
    source, output = root / 'native-arguments.cpp', root / 'native-arguments.so'
    source.write_text(BRIDGE)
    command = [os.environ['CXX'], '-std=c++17', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
               '-Wl,-z,defs', '-I' + str(Path(__file__).resolve().parents[1] / 'neutrino/src'),
               '-I' + str(Path(os.environ['UNIFIED_TOOLCHAIN']) / 'targets/x86_64-linux/include'),
               str(source), '-o', str(output)]
    with (root / 'native-arguments.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    return output, command


class NativeLaunchTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected native compiler and CUDA headers required')
    def test_owned_arguments_attributes_and_thread_local_submission_depth(self):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
        env.pop('LD_PRELOAD', None)
        toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, executable = root / 'launch.cpp', root / 'launch-test'
            source.write_text(SOURCE)
            command = [os.environ['CXX'], '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
                       '-fsanitize=undefined', '-fno-sanitize-recover=all',
                       '-I' + str(Path(__file__).resolve().parents[1] / 'neutrino/src'),
                       '-I' + str(toolchain / 'targets/x86_64-linux/include'), str(source), '-o', str(executable)]
            compiled = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            tested = subprocess.run([str(executable)], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(tested.returncode, 0, tested.stderr)


if __name__ == '__main__':
    unittest.main()
