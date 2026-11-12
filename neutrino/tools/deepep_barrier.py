"""Warp-level DeepEP GPU barrier probe."""
from neutrino import probe, Map
import neutrino.language as nl

SOURCE_HINTS = {
    "version": 1,
    "headers": [{
        "path": "deep_ep/comm/barrier.cuh",
        "functions": ["deep_ep::comm::gpu_barrier"],
        "source_sha256": "e4361353f31a2f3a36e019bb2ef013991017a205c16ab92f76f9da2a2652260f",
        "transform_version": 1,
    }],
}

@Map(level="warp", type="array", size=24, cap=64)
class function_events:
    timestamp: nl.u64
    site: nl.u64
    event: nl.u64

@probe(pos="func", funcname="deep_ep::comm::gpu_barrier", level="warp", before=True)
def enter():
    function_events.save(nl.time(), nl.site_id, nl.event)

@probe(pos="func", funcname="deep_ep::comm::gpu_barrier", level="warp", before=False)
def leave():
    function_events.save(nl.time(), nl.site_id, nl.event)


MARKED_KERNEL_INVENTORY = {
    "deep_ep::comm::barrier_impl": (1, 1),
    "deep_ep::ep::combine_impl": (1, 2),
    "deep_ep::ep::dispatch_impl": (2, 2),
}


def validate(trace_dir, output_dir=None, *, expected_ranks=(0, 1),
             require_source_hints=False, expected_inventory=None):
    """Decode and strictly validate this probe's multi-rank workload coverage.

    This is trace acceptance only; the workload must separately verify numerical
    results, gradients, optimizer steps, and teardown. Unexecuted static sites
    remain incomplete, rather than being guessed to be optimized away.
    """
    from collections import Counter, defaultdict
    import json
    from pathlib import Path
    from neutrino.probe.cubin_debug import DebugError, base_name, matches
    from neutrino.probe.function_hints import HINT_ABI, validate_identity
    from neutrino.utils.function_trace import export

    ranks = tuple(expected_ranks)
    if (len(ranks) < 2 or len(set(ranks)) != len(ranks) or
            any(type(rank) is not int or rank < 0 for rank in ranks)):
        raise ValueError("Expected at least two distinct nonnegative integer ranks")
    root = Path(trace_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else root / "decoded"
    summary = export(root, output)
    errors = []
    if summary["status"] != "complete":
        kinds = sorted({issue["kind"] for issue in summary["issues"]})
        errors.append(f"Incomplete function trace: {summary['counts']}; issues={kinds}")
    if summary["unobserved_sites"]:
        errors.append(f"Unobserved measurable sites: {summary['unobserved_sites']}")
    target = "deep_ep::comm::gpu_barrier"
    inventory = MARKED_KERNEL_INVENTORY if expected_inventory is None else expected_inventory
    manifests = {}
    if require_source_hints:
        if (not isinstance(inventory, dict) or not inventory or any(
                not isinstance(name, str) or not isinstance(value, (tuple, list)) or len(value) != 2 or
                any(type(number) is not int or number < 1 for number in value)
                for name, value in inventory.items())):
            raise ValueError("Expected marker inventory requires positive specialization/site counts")
        for path in sorted(root.glob("**/kernel/*/function.json")):
            artifact = str(path.parent.relative_to(root))
            try:
                manifest = json.loads(path.read_text())
                if (not isinstance(manifest, dict) or not isinstance(manifest.get("kernel", ""), str) or
                        not isinstance(manifest.get("sites", []), list)):
                    raise ValueError("Malformed marker manifest")
                family = base_name(manifest.get("kernel", ""))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"Invalid marker manifest {artifact}: {exc}")
                continue
            manifests[artifact] = manifest
            if family not in inventory:
                if manifest.get("sites"):
                    errors.append(f"Unexpected selected kernel family: {family}")
                continue
            expected_count = inventory[family][1]
            sites = manifest.get("sites", [])
            if (manifest.get("status") != "instrumented" or
                    manifest.get("boundary_source") != "source-hints" or
                    manifest.get("hint_abi") != HINT_ABI or manifest.get("rejected_markers") or
                    len(sites) != expected_count):
                errors.append(f"Missing marked coverage for {artifact}: expected {expected_count} sites")
            begins = set()
            for site in sites:
                try:
                    decision = site["boundary_decisions"]
                    begin = decision["entry_hint"]
                    ends = decision["exit_candidates"]
                    identity = validate_identity(begin["identity"])
                    if (decision["boundary_source"] != "source-hints" or decision["hint_abi"] != HINT_ABI or
                            begin["kind"] != 0 or begin["abi"] != HINT_ABI or not ends or
                            identity["function"] != target or identity["source_sha256"] != SOURCE_HINTS["headers"][0]["source_sha256"] or
                            any(end["identity"] != identity or end["kind"] != 1 or end["abi"] != HINT_ABI for end in ends) or
                            decision.get("rule") != "direct-source-marker" or
                            decision.get("pairing") != "runtime-entry-tag" or
                            begin["first_line"] in begins):
                        raise ValueError("Invalid or conflicting marker boundary evidence")
                    begins.add(begin["first_line"])
                except (KeyError, TypeError, ValueError, IndexError, AttributeError, DebugError) as exc:
                    errors.append(f"Invalid source-hint provenance for {artifact}: {exc}")
    for site in summary["sites"]:
        if not matches(target, site["function"]):
            errors.append(f"Unexpected selected function: {site['function']}")

    def rank_of(value):
        # Runtime companions encode rank as a decimal environment string.
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return value if type(value) is int else None

    launch_ranks, paired = defaultdict(set), Counter()
    specializations = defaultdict(set)
    for launch in summary["launches"]:
        rank = rank_of(launch["rank"])
        if rank not in ranks:
            errors.append(f"Unexpected or missing launch rank: {launch['rank']!r}")
        kernel = base_name(launch["kernel"])
        if not kernel.startswith("deep_ep::"):
            errors.append(f"Not a DeepEP launch: {launch['kernel']}")
        else:
            launch_ranks[rank].add(kernel)
        if require_source_hints and kernel in inventory:
            specializations[rank, kernel].add(launch["kernel"])
            if launch["artifact"] not in manifests:
                errors.append(f"Missing launch marker manifest: {launch['artifact']}")
    timeline = json.loads((output / "timeline.json").read_text())
    paired_sites = set()
    for event in timeline["traceEvents"]:
        if event["ph"] != "X":
            continue
        args = event["args"]
        rank = rank_of(args["rank"])
        if not matches(target, event["name"]):
            errors.append(f"Unexpected interval: {event['name']}")
        elif rank in ranks and base_name(args["kernel"]).startswith("deep_ep::"):
            paired[rank] += 1
            paired_sites.add((args["artifact"], args["site_id"]))
    for rank in ranks:
        if not launch_ranks[rank]:
            errors.append(f"Rank {rank} has no actual DeepEP launches")
        if not paired[rank]:
            errors.append(f"Rank {rank} has no paired gpu_barrier intervals")
        if require_source_hints:
            for family, (minimum, _) in inventory.items():
                observed = len(specializations[rank, family])
                if observed < minimum:
                    errors.append(f"Rank {rank} has {observed}/{minimum} expected {family} specializations")
    for site in summary["sites"]:
        if (site["artifact"], site["site_id"]) not in paired_sites:
            errors.append(f"Site has no paired invocation: {site}")
    result = {"version": 1, "status": "incomplete" if errors else "passed",
              "expected_ranks": list(ranks), "require_source_hints": require_source_hints,
              "expected_inventory": {key: list(value) for key, value in inventory.items()} if require_source_hints else {},
              "errors": errors, "counts": summary["counts"],
              "rank_intervals": {str(rank): paired[rank] for rank in ranks},
              "clock_domain": summary["clock_domain"],
              "unverified": ["GIN/RDMA", "multi-node hybrid communication"]}
    temporary = output / "acceptance.json.part"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "acceptance.json")
    return result
