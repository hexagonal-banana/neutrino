"""Decode function_events maps, optionally aligned to a startup CPU clock sample.

Custom payloads remain readable with their generated read.py. This decoder reads
only inert JSON/TOML metadata and never imports code from a trace directory.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import struct

import toml

VERSION = 1
HEADER = struct.Struct("<8I")
SECTION = struct.Struct("<IIQ")
EVENT = struct.Struct("<QQQ")


class TraceError(ValueError):
    pass


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceError(f"Invalid {label}: {value!r}")
    return value


def calibrate_clock(samples=9):
    """Bracket a tiny %globaltimer kernel with CLOCK_MONOTONIC at initialization.

    Requires an existing current CUDA context. This synchronizes only a private
    nonblocking stream, never the application's streams. The minimum round-trip
    sample gives an approximate offset, not drift correction or exact alignment.
    """
    import ctypes as C
    import os
    import socket
    import time

    _integer(samples, "clock sample count", 1)
    if samples > 100:
        raise TraceError("At most 100 startup clock samples are supported")
    driver = C.CDLL(os.environ.get("NEUTRINO_REAL_DRIVER", "libcuda.so.1"))

    def call(name, *args):
        result = getattr(driver, name)(*args)
        if result:
            raise TraceError(f"Clock calibration {name} failed: CUDA error {result}")

    context, context_id, device = C.c_void_p(), C.c_uint64(), C.c_int()
    call("cuCtxGetCurrent", C.byref(context))
    if not context.value:
        raise TraceError("Clock calibration requires a current CUDA context")
    call("cuCtxGetId", context, C.byref(context_id))
    call("cuCtxGetDevice", C.byref(device))
    module, stream, host, pointer = C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_uint64()
    ptx = b""".version 7.0
.target sm_70
.address_size 64
.visible .entry neutrino_clock_sample(.param .u64 output) {
    .reg .b64 address, stamp;
    ld.param.u64 address, [output];
    mov.u64 stamp, %globaltimer;
    st.global.u64 [address], stamp;
    ret;
}
"""
    rows = []
    try:
        call("cuModuleLoadData", C.byref(module), C.c_char_p(ptx))
        function = C.c_void_p()
        call("cuModuleGetFunction", C.byref(function), module, b"neutrino_clock_sample")
        call("cuStreamCreate", C.byref(stream), C.c_uint(1))
        call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(8))
        call("cuMemAllocHost_v2", C.byref(host), C.c_size_t(8))
        arguments = (C.c_void_p * 1)(C.addressof(pointer))
        for index in range(samples + 1):
            before = time.monotonic_ns()
            call("cuLaunchKernel", function, 1, 1, 1, 1, 1, 1, 0, stream, arguments, None)
            call("cuMemcpyDtoHAsync_v2", host, pointer, C.c_size_t(8), stream)
            call("cuStreamSynchronize", stream)
            after = time.monotonic_ns()
            gpu = C.cast(host, C.POINTER(C.c_uint64)).contents.value
            if index:  # Exclude module/lazy-kernel warmup from the candidates.
                rows.append({"cpu_before_ns": before, "cpu_after_ns": after, "gpu_ns": gpu})
    finally:
        # On a failed launch/copy, prove completion before freeing its storage.
        safe = not stream.value or driver.cuStreamSynchronize(stream) == 0
        if safe:
            if host.value:
                call("cuMemFreeHost", host)
            if pointer.value:
                call("cuMemFree_v2", pointer)
            if stream.value:
                call("cuStreamDestroy_v2", stream)
            if module.value:
                call("cuModuleUnload", module)
    best = min(rows, key=lambda row: row["cpu_after_ns"] - row["cpu_before_ns"])
    before, after = best["cpu_before_ns"], best["cpu_after_ns"]
    return {"version": 1, "method": "minimum-round-trip-midpoint", "cpu_clock": "CLOCK_MONOTONIC",
            "gpu_clock": "%globaltimer", "hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid": os.getpid(), "device": device.value, "context_id": context_id.value,
            **best, "offset_ns": (before + after) // 2 - best["gpu_ns"],
            "uncertainty_ns": (after - before + 1) // 2, "samples": rows,
            "drift_corrected": False}


def _site_id(site):
    identity = _integer(site["site_id"], "site ID")
    if identity >= 2**64 or not isinstance(site.get("function"), str) or not site["function"]:
        raise TraceError("Invalid static site identity")
    if "demangled" in site and not isinstance(site["demangled"], str):
        raise TraceError("Invalid demangled function name")
    return identity


def _product(values):
    result = 1
    for value in values:
        result *= _integer(value, "launch dimension", 1)
    return result


def _json(path):
    with path.open() as file:
        return json.load(file)


def _artifact(session, launch):
    # Trace directories may be moved. Resolve the artifact within this session,
    # not against an arbitrary absolute path supplied by companion metadata.
    name = Path(launch["artifact"]).name
    directory = session / "kernel" / name
    if not name or directory.resolve().parent != (session / "kernel").resolve():
        raise TraceError("Artifact is outside the trace session")
    manifest = _json(directory / "function.json")
    if manifest.get("version") != VERSION or manifest.get("status") != "instrumented":
        raise TraceError("Unsupported or unfinished function artifact")
    if manifest.get("kernel") != launch.get("kernel"):
        raise TraceError("Launch/artifact kernel mismatch")
    if not manifest.get("sites"):
        raise TraceError("Instrumented artifact has no measurable sites")
    return directory, manifest


def _buffers(path, launch, manifest):
    standard = manifest.get("runtime_abi") == 3
    data = path.with_suffix(".bin").read_bytes()
    meta = b"" if standard else path.with_suffix(".meta").read_bytes()
    if len(data) < HEADER.size:
        raise TraceError("Truncated trace header")
    gx, gy, gz, bx, by, bz, shared, count = HEADER.unpack_from(data)
    grid, block = [gx, gy, gz], [bx, by, bz]
    warps_per_block = (_product(block) + 31) // 32
    warps = _product(grid) * warps_per_block
    if grid != launch.get("grid") or block != launch.get("block") or shared != launch.get("shared_mem_bytes"):
        raise TraceError("Binary/companion launch configuration mismatch")
    if launch.get("warps") != warps or launch.get("representative_lane") != 0:
        raise TraceError("Unsupported warp coordinates or representative lane")
    maps = manifest["maps"]
    if count != len(maps) or not maps:
        raise TraceError("Map count mismatch")
    stride = (2 * count + 1) * 8
    if not standard:
        if manifest.get("metadata_stride") != stride or launch.get("metadata_stride") != stride:
            raise TraceError("Metadata stride mismatch")
        if manifest.get("unsupported_offset") != count * 16 or len(meta) != warps * stride:
            raise TraceError("Invalid/truncated per-warp metadata")
    elif (manifest.get("collection_backend") not in {
            "cuda-device-maps-event-sync-dtoh", "cuda-device-maps-event-async-dtoh-pool"} or
          launch.get("mapped_host") is not False or launch.get("metadata_stride") != 0):
        raise TraceError("Invalid standard collector contract")
    offset = HEADER.size + count * SECTION.size
    if len(data) < offset:
        raise TraceError("Truncated section table")
    sections = []
    names = set()
    for index, map_ in enumerate(maps):
        size = _integer(map_["size"], "map size", 1)
        cap = _integer(map_["cap"], "map capacity", 1)
        if map_["name"] in names:
            raise TraceError("Duplicate map name")
        names.add(map_["name"])
        if not standard and (map_.get("count_offset") != index * 16 or map_.get("overflow_offset") != index * 16 + 8):
            raise TraceError("Invalid map metadata offsets")
        section = SECTION.unpack_from(data, HEADER.size + index * SECTION.size)
        if section != (size * cap, 32, offset):
            raise TraceError("Invalid section size, warp divisor, or offset")
        sections.append(section)
        offset += warps * size * cap
    if len(data) != offset:
        raise TraceError("Truncated or oversized map payload")
    return data, meta, sections, stride, warps, warps_per_block


def _standard_map(directory, maps):
    candidates = [i for i, map_ in enumerate(maps) if map_["name"] == "function_events"]
    if not candidates:
        return None
    index = candidates[0]
    raw = toml.load(directory / "probe.toml").get("map", {}).get("function_events", {})
    registers = raw.get("regs", {})
    if (maps[index]["size"] != 24 or raw.get("type") != "array" or raw.get("level") != "warp" or
            list(registers) != ["timestamp", "site", "event"] or
            any(not isinstance(reg, list) or not reg or reg[0] != "u64" for reg in registers.values())):
        return None
    return index


def _timestamp_maps(directory, manifest):
    """Validate inert metadata identifying the timestamp-only buffer slices."""
    layout = manifest.get("record_layout")
    if layout is None:
        return None
    if (layout.get("version") != 1 or layout.get("kind") != "timestamp-only" or
            layout.get("pairing") != "per-site-index-nonnested"):
        raise TraceError("Unsupported timestamp record layout")
    raw_maps = toml.load(directory / "probe.toml").get("map", {})
    sites = {_site_id(site) for site in manifest["sites"]}
    partitions = {}
    for index, map_ in enumerate(manifest["maps"]):
        site = _integer(map_.get("site_id"), "timestamp map site")
        event = _integer(map_.get("event"), "timestamp map event")
        raw = raw_maps.get(map_["name"], {})
        registers = raw.get("regs", {})
        if (site not in sites or event not in (0, 1) or (site, event) in partitions or
                map_["size"] != 8 or raw.get("size") != 8 or raw.get("cap") != map_["cap"] or
                raw.get("level") != "warp" or raw.get("type") != "array" or
                list(registers) != ["timestamp"] or not isinstance(registers["timestamp"], list) or
                not registers["timestamp"] or registers["timestamp"][0] != "u64"):
            raise TraceError("Invalid timestamp map schema or site/event partition")
        partitions[site, event] = index
    if set(partitions) != {(site, event) for site in sites for event in (0, 1)}:
        raise TraceError("Missing timestamp site/event partition")
    return partitions


def _timestamp_rows(data, sections, valid_counts, partitions, warp):
    """Pair by per-site index, then order disjoint intervals without losing ties.

    This format deliberately excludes nested selected sites: timestamp ties
    cannot reconstruct arbitrary cross-site event order without another field.
    """
    intervals = []
    for site in sorted({site for site, _ in partitions}):
        entry, exit_ = partitions[site, 0], partitions[site, 1]
        count = valid_counts[entry][warp]
        if count != valid_counts[exit_][warp]:
            raise TraceError(f"Timestamp entry/exit count mismatch: site={site} warp={warp}")
        previous = None
        for index in range(count):
            values = []
            for map_index in (entry, exit_):
                size, _, offset = sections[map_index]
                values.append(struct.unpack_from("<Q", data, offset + warp * size + index * 8)[0])
            begin, end = values
            if end < begin or (previous is not None and begin < previous):
                raise TraceError(f"Timestamp regression: site={site} warp={warp} index={index}")
            previous = end
            intervals.append((begin, end, site, index))
    intervals.sort()
    previous = None
    for begin, end, site, _ in intervals:
        if previous is not None and begin < previous:
            raise TraceError(f"Overlapping intervals in nonnested timestamp layout: warp={warp}")
        previous = end
    return [(timestamp, site, event) for begin, end, site, _ in intervals
            for event, timestamp in ((0, begin), (1, end))]


def _coordinates(flat, grid):
    x = flat % grid[0]
    y = (flat // grid[0]) % grid[1]
    return [x, y, flat // (grid[0] * grid[1])]


def _training_clocks(path):
    """Load explicit same-run calibration; never guess alignment for old traces."""
    data = _json(Path(path))
    clocks, reports, hosts = {}, {}, set()
    if not isinstance(data, dict) or not isinstance(data.get("ranks"), list) or not data["ranks"]:
        raise TraceError("Training results have no rank calibrations")
    for report in data["ranks"]:
        if not isinstance(report, dict) or not isinstance(report.get("clock_sync"), dict):
            raise TraceError("Missing startup clock calibration")
        rank = str(_integer(report.get("rank"), "training rank"))
        clock = report["clock_sync"]
        if (type(clock.get("version")) is not int or clock["version"] != 1 or
                clock.get("method") != "minimum-round-trip-midpoint" or
                clock.get("cpu_clock") != "CLOCK_MONOTONIC" or clock.get("gpu_clock") != "%globaltimer" or
                clock.get("drift_corrected") is not False):
            raise TraceError("Unsupported clock calibration")
        for key in ("cpu_before_ns", "cpu_after_ns", "gpu_ns", "pid", "context_id"):
            _integer(clock.get(key), key, 1)
        _integer(clock.get("device"), "clock device")
        before, after = clock["cpu_before_ns"], clock["cpu_after_ns"]
        if (after < before or type(clock.get("offset_ns")) is not int or
                clock["offset_ns"] != (before + after) // 2 - clock["gpu_ns"] or
                type(clock.get("uncertainty_ns")) is not int or
                clock["uncertainty_ns"] != (after - before + 1) // 2):
            raise TraceError("Invalid clock bracket or offset")
        if any(not isinstance(clock.get(key), str) or not clock[key] for key in ("boot_id", "hostname")):
            raise TraceError("Missing host clock identity")
        samples = clock.get("samples")
        if not isinstance(samples, list) or not 1 <= len(samples) <= 100:
            raise TraceError("Invalid clock samples")
        for sample in samples:
            if not isinstance(sample, dict):
                raise TraceError("Invalid clock sample")
            for key in ("cpu_before_ns", "cpu_after_ns", "gpu_ns"):
                _integer(sample.get(key), key, 1)
            if sample["cpu_after_ns"] < sample["cpu_before_ns"]:
                raise TraceError("Invalid clock sample bracket")
        best = min(samples, key=lambda row: row["cpu_after_ns"] - row["cpu_before_ns"])
        if any(clock[key] != best[key] for key in ("cpu_before_ns", "cpu_after_ns", "gpu_ns")):
            raise TraceError("Calibration is not the minimum round-trip sample")
        spans = report.get("step_spans")
        if not isinstance(spans, list) or not spans:
            raise TraceError("Missing CPU training step spans")
        previous_end, previous_step = clock["cpu_after_ns"], 0
        for span in spans:
            if not isinstance(span, dict) or span.get("phase") not in ("warmup", "measured"):
                raise TraceError("Invalid CPU step span")
            begin = _integer(span.get("start_ns"), "CPU step start", previous_end)
            previous_end = _integer(span.get("end_ns"), "CPU step end", begin)
            previous_step = _integer(span.get("step"), "CPU step", previous_step + 1)
        hosts.add((clock["hostname"], clock["boot_id"]))
        if rank in clocks:
            raise TraceError("Duplicate clock calibration rank")
        clocks[rank], reports[rank] = clock, report
    if len(hosts) != 1:
        raise TraceError("CPU monotonic clocks from different hosts/boots cannot be aligned")
    return clocks, reports


def _chrome_view(timeline, domains, clocks, reports, overview):
    """Render reusable stream/warp tracks, or explicitly labeled warp envelopes."""
    origin = min((clock["cpu_before_ns"] for clock in clocks.values()), default=0)
    metadata, tracks, grouped = [], {}, {}
    for domain, pid in domains.items():
        rank, device = domain[1:3]
        metadata.append({"name": "process_name", "ph": "M", "pid": pid,
                         "args": {"name": f"rank={rank} GPU={device}"}})
        if clocks:
            metadata.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 0,
                             "args": {"name": "CPU training steps"}})
            for span in reports[str(rank)]["step_spans"]:
                begin, end, step = span["start_ns"], span["end_ns"], span["step"]
                timeline.append({"name": f"step {step} ({span['phase']})", "cat": "cpu-training-step",
                                 "ph": "X", "pid": pid, "tid": 0, "ts": (begin - origin) / 1000,
                                 "dur": (end - begin) / 1000, "args": dict(span)})
    for event in timeline:
        if event["cat"] == "cpu-training-step":
            continue
        args, pid = event["args"], event["pid"]
        if overview:
            key = (pid, args["launch_id"], args["site_id"], args["sequence"])
            if key in grouped:
                envelope = grouped[key]
                envelope["args"]["start_ns"] = min(envelope["args"]["start_ns"], args["start_ns"])
                envelope["args"]["end_ns"] = max(envelope["args"]["end_ns"], args["end_ns"])
                envelope["args"]["warp_intervals"] += 1
                continue
            event = {**event, "args": dict(args), "cat": "warp-envelope"}
            for field in ("cta", "warp", "depth", "entry_index", "exit_index"):
                event["args"].pop(field, None)
            event["args"].update(warp_intervals=1, interpretation=
                                 "min entry to max exit across warps by site/invocation ordinal; not kernel duration")
            grouped[key] = event
        else:
            grouped[len(grouped)] = event
    rendered = [event for event in timeline if event["cat"] == "cpu-training-step"]
    for event in grouped.values():
        args, pid = event["args"], event["pid"]
        clock = clocks.get(str(args["rank"]))
        offset = clock["offset_ns"] if clock else 0
        begin, end = args["start_ns"], args["end_ns"]
        event["ts"], event["dur"] = (begin + offset - origin) / 1000, (end - begin) / 1000
        args["duration_ns"] = end - begin
        if clock:
            args.update(cpu_start_ns=begin + offset, cpu_end_ns=end + offset,
                        alignment_uncertainty_ns=clock["uncertainty_ns"])
        stream = args.get("stream", "unknown")
        lane = (stream, args["kernel"], args["site_id"]) if overview else (stream, tuple(args["cta"]), args["warp"])
        track = (pid, lane)
        if track not in tracks:
            tid = len(tracks) + 1
            tracks[track] = tid
            label = (f"{args['kernel']} site={args['site_id']} [warp envelope]" if overview else
                     f"stream={stream} CTA={args['cta']} warp={args['warp']}")
            metadata.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": label}})
        event["tid"] = tracks[track]
        rendered.append(event)
    return metadata + sorted(rendered, key=lambda event: (event["ts"], event["pid"], event["tid"])), origin


def export(trace_dir, output_dir=None, *, training_results=None, overview=False):
    """Write summary.json and Chrome timeline.json; return the summary dictionary.

    ``status == 'complete'`` means available standard records passed structural
    checks, not that every application launch or every static site executed.
    Coverage gaps, unsupported sites/payloads, overflow, and invalid pairs make
    status incomplete. Static sites with no records are listed separately.
    """
    root = Path(trace_dir).resolve()
    if not root.is_dir():
        raise TraceError(f"Trace directory does not exist: {root}")
    output = Path(output_dir).resolve() if output_dir is not None else root / "decoded"
    counts = Counter({key: 0 for key in (
        "launches", "submitted_launches", "missing_launches", "records", "intervals", "coverage_gaps", "overflow", "unsupported_warps",
        "missing_entries", "missing_exits", "nesting_errors", "timestamp_errors", "unknown_sites",
        "invalid_records", "invalid_traces", "unsupported_payloads", "rejected_sites", "saturated_maps")})
    issues, timeline, launches, domains = [], [], [], {}
    clocks, reports = _training_clocks(training_results) if training_results is not None else ({}, {})
    aligned_domains, alignment_mismatches = set(), []
    sites, observed = {}, set()
    submissions, failed, journals = set(), set(), set()

    def issue(kind, location, **details):
        issues.append({"kind": kind, "location": str(location.relative_to(root)), **details})

    for path in sorted(root.glob("**/result/coverage.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                if record.get("version") != VERSION or record.get("status") != "coverage_gap":
                    raise TraceError("Unsupported coverage record")
                count = record.get("count", 1)
                if type(count) is not int or count < 1:
                    raise TraceError("Invalid coverage gap count")
                counts["coverage_gaps"] += count
                if isinstance(record.get("launch_id"), int):
                    failed.add((str(path.parent.parent.relative_to(root)), record["launch_id"]))
                issue("coverage_gap", path, line=number, record=record)
            except (ValueError, TypeError, AttributeError) as exc:
                counts["invalid_traces"] += 1
                issue("invalid_coverage_record", path, line=number, reason=str(exc))

    for path in sorted(root.glob("**/result/launches.jsonl")):
        session_name = str(path.parent.parent.relative_to(root))
        journals.add(session_name)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                if record.get("version") != VERSION or record.get("status") != "submitted":
                    raise TraceError("Unsupported submission record")
                identity = (session_name, _integer(record["launch_id"], "launch ID", 1))
                if identity in submissions:
                    raise TraceError("Duplicate submitted launch ID")
                submissions.add(identity)
                counts["submitted_launches"] += 1
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                counts["invalid_traces"] += 1
                issue("invalid_submission_record", path, line=number, reason=str(exc))

    for path in sorted(root.glob("**/kernel/*/function.json")):
        try:
            manifest = _json(path)
            if manifest.get("version") != VERSION or manifest.get("status") not in ("instrumented", "unsupported", "unselected"):
                raise TraceError("Unsupported site manifest version/status")
            artifact_id = str(path.parent.relative_to(root))
            for site in manifest.get("sites", []):
                site_id = _site_id(site)
                if (artifact_id, site_id) in sites:
                    raise TraceError("Invalid/duplicate static site ID")
                sites[artifact_id, site_id] = site
            rejected = manifest.get("rejected_regions", [])
            if manifest.get("status") == "unsupported" and not rejected:
                rejected = [{"reason": manifest.get("reason", "Unsupported artifact")}]
            for region in rejected:
                counts["rejected_sites"] += 1
                issue("unsupported_site", path, region=region)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            counts["invalid_traces"] += 1
            issue("invalid_site_manifest", path, reason=str(exc))

    for path in sorted(root.glob("**/result/function-*.bin")):
        if not path.with_suffix(".json").is_file():
            counts["invalid_traces"] += 1
            issue("missing_launch_companion", path)

    identifiers = set()
    for path in sorted(root.glob("**/result/function-*.json")):
        session = path.parent.parent
        try:
            launch = _json(path)
            if launch.get("version") != VERSION or launch.get("status") != "complete":
                raise TraceError("Unsupported or incomplete launch companion")
            launch_id = _integer(launch["launch_id"], "launch ID", 1)
            session_name = str(session.relative_to(root))
            identity = (session_name, launch_id)
            if identity in identifiers:
                raise TraceError("Duplicate launch ID in session")
            identifiers.add(identity)
            if session_name not in journals or identity not in submissions:
                counts["invalid_traces"] += 1
                issue("missing_submission_journal", path)
            directory, manifest = _artifact(session, launch)
            data, meta, sections, stride, warps, wpb = _buffers(path, launch, manifest)
            maps = manifest["maps"]
            event_map = _standard_map(directory, maps)
            timestamp_partitions = _timestamp_maps(directory, manifest)
            by_site = {}
            artifact_id = str(directory.relative_to(root))
            for site in manifest["sites"]:
                site_id = _site_id(site)
                if site_id in by_site:
                    raise TraceError("Invalid/duplicate static site ID")
                by_site[site_id] = site
                sites[(artifact_id, site_id)] = site
            counts["launches"] += 1
            launch_summary = {"session": session_name, "launch_id": launch_id, "kernel": launch["kernel"],
                              "artifact": artifact_id, "rank": launch.get("rank"),
                              "device": launch.get("device"), "maps": []}
            if timestamp_partitions is not None:
                launch_summary["record_layout"] = manifest["record_layout"]
            launches.append(launch_summary)
            valid_counts = []
            for index, map_ in enumerate(maps):
                valid, overflow = [], []
                for warp in range(warps):
                    if manifest.get("runtime_abi") == 3:
                        # Standard Neutrino zeroes maps and does not publish counts.
                        # The standard timestamp event schema reserves zero for
                        # unused records. Saturation is uncertain, never a proven
                        # overflow count or a claim that capacity was sufficient.
                        reserved, _, offset = sections[index]
                        start = offset + warp * reserved
                        n, lost = 0, 0
                        for record in range(map_["cap"]):
                            stamp = struct.unpack_from("<Q", data, start + record * map_["size"])[0]
                            if not stamp:
                                if any(data[start + record * map_["size"]:start + reserved]):
                                    raise TraceError("Nonzero records after zero timestamp sentinel")
                                break
                            n += 1
                        if n == map_["cap"]:
                            counts["saturated_maps"] += 1
                            issue("capacity_not_proven", path, map=map_["name"], warp=warp)
                    else:
                        n, lost = struct.unpack_from("<QQ", meta, warp * stride + index * 16)
                    if n > map_["cap"]:
                        raise TraceError("Valid count exceeds map capacity")
                    valid.append(n)
                    overflow.append(lost)
                valid_counts.append(valid)
                counts["overflow"] += sum(overflow)
                launch_summary["maps"].append({"name": map_["name"], "valid_counts": valid, "overflow": overflow})
            flags = ([0] * warps if manifest.get("runtime_abi") == 3 else
                     [struct.unpack_from("<Q", meta, warp * stride + len(maps) * 16)[0] for warp in range(warps)])
            if manifest.get("runtime_abi") == 3:
                launch_summary["collection_backend"] = manifest["collection_backend"]
                launch_summary["diagnostics"] = {"overflow_detection": False, "missing_lane_detection": False,
                                                  "counts": "zero-terminated-timestamps"}
            launch_summary["unsupported_flags"] = flags
            counts["unsupported_warps"] += sum(bool(flag) for flag in flags)
            if event_map is None and timestamp_partitions is None:
                counts["unsupported_payloads"] += 1
                issue("unsupported_payload", path, reason="Expected function_events(timestamp:u64, site:u64, event:u64)")
                continue
            domain = (session_name, launch.get("rank"), launch.get("device"), launch.get("context_id"))
            if clocks:
                clock = clocks.get(str(launch.get("rank")))
                if not clock or any(type(launch.get(key)) is not int or clock[key] != launch[key]
                                    for key in ("pid", "device", "context_id")):
                    alignment_mismatches.append(str(path))
                    raise TraceError("Launch does not match training calibration pid/device/context/rank")
                aligned_domains.add(domain)
            if domain not in domains:
                domains[domain] = len(domains) + 1
            pid = domains[domain]
            for warp in range(warps):
                if timestamp_partitions is not None:
                    rows = _timestamp_rows(data, sections, valid_counts, timestamp_partitions, warp)
                else:
                    size, _, offset = sections[event_map]
                    rows = [EVENT.unpack_from(data, offset + warp * size + index * EVENT.size)
                            for index in range(valid_counts[event_map][warp])]
                stack, sequence, previous = [], Counter(), None
                cta, local_warp = divmod(warp, wpb)
                coordinate = {"launch_id": launch_id, "cta": _coordinates(cta, launch["grid"]), "warp": local_warp}
                for index, (timestamp, site_id, event) in enumerate(rows):
                    counts["records"] += 1
                    if previous is not None and timestamp < previous:
                        counts["timestamp_errors"] += 1
                        issue("timestamp_regression", path, **coordinate, index=index)
                    previous = timestamp
                    if site_id not in by_site:
                        counts["unknown_sites"] += 1
                        issue("unknown_site", path, **coordinate, index=index, site_id=str(site_id))
                        continue
                    observed.add((artifact_id, site_id))
                    if event == 0:
                        stack.append((site_id, timestamp, index, sequence[site_id], len(stack)))
                        sequence[site_id] += 1
                        continue
                    if event != 1:
                        counts["invalid_records"] += 1
                        issue("invalid_event_kind", path, **coordinate, index=index, event=event)
                        continue
                    if not stack or stack[-1][0] != site_id:
                        if any(item[0] == site_id for item in stack):
                            counts["nesting_errors"] += 1
                            issue("nesting_mismatch", path, **coordinate, index=index)
                            while stack and stack[-1][0] != site_id:
                                missing = stack.pop()
                                counts["missing_exits"] += 1
                                issue("missing_exit", path, **coordinate, index=missing[2], site_id=str(missing[0]))
                        else:
                            counts["missing_entries"] += 1
                            issue("missing_entry", path, **coordinate, index=index, site_id=str(site_id))
                            continue
                    _, begin, begin_index, invocation, depth = stack.pop()
                    if timestamp < begin:
                        # The regression above already counts this inconsistency.
                        issue("negative_duration", path, **coordinate, index=index)
                        continue
                    site = by_site[site_id]
                    counts["intervals"] += 1
                    args = {**coordinate, "session": session_name, "rank": launch.get("rank"),
                            "device": launch.get("device"), "context_id": launch.get("context_id"),
                            "stream": launch.get("stream"),
                            "kernel": launch["kernel"], "artifact": artifact_id,
                            "site_id": str(site_id), "sequence": invocation,
                            "depth": depth, "entry_index": begin_index, "exit_index": index,
                            "start_ns": begin, "end_ns": timestamp, "duration_ns": timestamp - begin,
                            "source_file": site.get("file"), "source_line": site.get("line"),
                            "representative_lane": 0}
                    timeline.append({"name": site.get("demangled", site["function"]), "cat": "inline-function",
                                     "ph": "X", "pid": pid, "tid": f"{launch_id}:{cta}:{local_warp}",
                                     "ts": begin / 1000, "dur": (timestamp - begin) / 1000, "args": args})
                for site_id, _, index, _, _ in stack:
                    counts["missing_exits"] += 1
                    issue("missing_exit", path, **coordinate, index=index, site_id=str(site_id))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error) as exc:
            counts["invalid_traces"] += 1
            issue("invalid_trace", path, reason=str(exc))

    for session_name, launch_id in sorted(submissions - identifiers - failed):
        counts["missing_launches"] += 1
        issue("submitted_launch_without_completion", root / session_name, launch_id=launch_id)
    errors = sum(value for key, value in counts.items() if key not in ("launches", "submitted_launches", "records", "intervals"))
    if not counts["launches"]:
        issue("no_completed_launches", root)
    summary = {"version": VERSION, "status": "incomplete" if errors or not counts["launches"] else "complete",
               "clock_domain": "gpu-local-unaligned", "representative_lane": 0, "counts": dict(counts),
               "launches": launches, "issues": issues,
               "sites": [{"artifact": artifact, "site_id": str(site),
                          "function": sites[artifact, site]["function"]}
                         for artifact, site in sorted(sites)],
               "unobserved_sites": [{"artifact": artifact, "site_id": str(site), "function": sites[artifact, site]["function"]}
                                    for artifact, site in sorted(sites.keys() - observed)]}
    metadata = [{"name": "process_name", "ph": "M", "pid": pid,
                 "args": {"name": f"{domain[0]} rank={domain[1]} device={domain[2]} context={domain[3]} (unaligned)"}}
                for domain, pid in domains.items()]
    chrome = {"traceEvents": metadata + timeline, "displayTimeUnit": "ns", "clock_domain": "gpu-local-unaligned"}
    if clocks:
        if alignment_mismatches:
            raise TraceError(f"Launch does not match training calibration: {alignment_mismatches[0]}")
        if {str(domain[1]) for domain in aligned_domains} != set(clocks):
            raise TraceError("Not every training calibration has matching trace launches")
        if len(aligned_domains) != len(clocks):
            raise TraceError("Multiple trace sessions/contexts per calibration are unsupported")
        summary["clock_domain"] = "cpu-monotonic-approximate"
        summary["clock_sync"] = list(clocks.values())
        summary["clock_alignment_limit"] = "Startup offset only; half-round-trip uncertainty excludes later drift"
    if clocks or overview:
        events, origin = _chrome_view(timeline, domains, clocks, reports, overview)
        chrome.update(traceEvents=events, clock_domain=summary["clock_domain"], time_origin_ns=origin,
                      view="warp-envelope" if overview else "per-warp", clock_sync=list(clocks.values()))
    chrome["collection_status"] = summary["status"]
    if clocks:
        chrome["clock_alignment_limit"] = summary["clock_alignment_limit"]
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("summary.json", summary), ("timeline.json", chrome)):
        temporary = output / (name + ".part")
        temporary.write_text(json.dumps(content, indent=2 if name == "summary.json" else None) + "\n")
        temporary.replace(output / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir")
    parser.add_argument("--output", help="Output directory (default: <trace_dir>/decoded)")
    parser.add_argument("--training-results", help="Same-run training JSON with startup clock calibration and CPU steps")
    parser.add_argument("--overview", action="store_true", help="Show per-launch/site warp envelopes instead of individual warps")
    args = parser.parse_args()
    try:
        summary = export(args.trace_dir, args.output, training_results=args.training_results, overview=args.overview)
    except (OSError, TraceError) as exc:
        parser.exit(2, f"Trace decoding failed: {exc}\n")
    print(json.dumps({"status": summary["status"], **summary["counts"]}))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
