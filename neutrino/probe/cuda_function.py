"""Source-marker selection adapter for Neutrino's CUDA probe backend.

No CFG, debug attribution, edge splitting, or separate payload emitter is used.
The source transformation supplies boundaries; a register carries the selected
entry identity to whichever exit executes, independent of PTX block ordering.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from neutrino.common import Probe, load
from neutrino.probe import Ref
from neutrino.probe import cuda
from neutrino.probe.cubin_debug import Cubin, DebugError, demangle, matches
from neutrino.probe.function_hints import HINT_ABI, canonical, decode_hints

FUNCTION_PROBE_ABI = 1
FUNCTION_RUNTIME_ABI = 3


def entry_declaration(ptx, kernel):
    match = re.search(r"\.entry\s+" + re.escape(kernel) + r"\s*\(", ptx)
    if not match:
        raise DebugError(f"Exact PTX entry not found: {kernel}")
    return match


def entry_bounds(ptx, kernel):
    declaration = entry_declaration(ptx, kernel)
    end = ptx.find(")", declaration.end())
    body = ptx.find("{", end)
    if end < 0 or body < 0:
        raise DebugError("Malformed entry signature")
    depth, close = 1, body + 1
    while close < len(ptx) and depth:
        depth += (ptx[close] == "{") - (ptx[close] == "}")
        close += 1
    if depth:
        raise DebugError("Unterminated entry body")
    return declaration, end, body, close


def check_retained_targets(ptx, selectors):
    for name in re.findall(r"\.func\s+(?:\([^)]*\)\s*)?([\w$]+)\s*\(", ptx):
        if any(matches(selector, name) for selector in selectors):
            raise DebugError(f"Selected out-of-line function is unsupported: {demangle(name)}")


def select_hints(ptx, kernel, selectors, image_sha256=""):
    """Select literal marker coordinates without inferring control flow.

    Exit candidates share source identity, not textual order. Runtime records
    carry the entry tag; the decoder checks executed entry/exit pairing. Recursive
    re-entry of the same source identity is outside this marker ABI's contract.
    """
    hints = decode_hints(ptx)
    declaration, _, body, close = entry_bounds(ptx, kernel)
    first, last = ptx[:body].count("\n") + 1, ptx[:close].count("\n") + 1
    local = [hint for hint in hints if first < hint.first_line <= last]
    groups = {}
    for hint in local:
        for selector in selectors:
            if selector.startswith("_Z") and matches(hint.identity["function"], selector) and "symbol" not in hint.identity:
                raise DebugError("Exact mangled selector requires exact symbol identity in source hints")
        names = [s for s in selectors if s in (hint.identity["function"], hint.identity.get("symbol"))]
        if names:
            groups.setdefault(canonical(hint.identity), []).append(hint)
    sites = []
    for group in groups.values():
        entries = [h for h in group if h.kind == 0]
        exits = [h for h in group if h.kind == 1]
        if not entries or not exits:
            raise DebugError("Selected source identity requires entry and exit markers")
        for hint in entries:
            identity = [image_sha256, kernel, hint.identity, hint.first_line, HINT_ABI]
            site = int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:8], "little")
            sites.append({"site_id": site, "function": hint.identity.get("symbol", hint.identity["function"]),
                          "demangled": hint.identity["function"], "file": hint.identity["file"],
                          "line": hint.identity["line"],
                          "selectors": [s for s in selectors if s in (hint.identity["function"], hint.identity.get("symbol"))],
                          "boundary_decisions": {"rule": "direct-source-marker", "boundary_source": "source-hints",
                              "hint_abi": HINT_ABI, "entry_hint": hint.manifest(),
                              "exit_candidates": [h.manifest() for h in exits],
                              "pairing": "runtime-entry-tag", "recursive_identity": "unsupported"}})
    sites.sort(key=lambda site: site["boundary_decisions"]["entry_hint"]["first_line"])
    if len({site["site_id"] for site in sites}) != len(sites):
        raise DebugError("Source-hint site ID collision")
    return local, sites


def marker_references(ptx, hints, sites, probes, regs):
    """Translate selected markers to ordinary probes and explicit line Refs."""
    if re.search(r"%NR\d|%map_\w+|%leader\b|%joint_pred\b", ptx):
        raise DebugError("Original PTX collides with reserved probe identifiers")
    lines, references = ptx.splitlines(), {}
    groups = {}
    for site in sites:
        hint = site["boundary_decisions"]["entry_hint"]
        groups.setdefault(canonical(hint["identity"]), []).append(site)
    tags = {identity: regs + index + 1 for index, identity in enumerate(groups)}
    for hint in hints:
        original = lines[hint.first_line - 1]
        lines[hint.declaration_line - 1] = ""
        for index in range(hint.first_line - 1, hint.last_line):
            lines[index] = ""
        identity = canonical(hint.identity)
        if identity not in groups:
            continue
        candidates = groups[identity]
        tag = f"%NR{tags[identity]}"
        code = []
        if hint.kind == 0:
            site = next(s for s in candidates if s["boundary_decisions"]["entry_hint"]["first_line"] == hint.first_line)
            code.append(f"mov.u64 {tag}, {site['site_id']};")
        for probe in probes:
            if probe.funcname not in candidates[0]["selectors"]:
                continue
            snippet = probe.before if hint.kind == 0 else probe.after
            if snippet:
                code.append(re.sub(r"\bEVENT_KIND\b", str(hint.kind), re.sub(r"\bSITE_ID\b", tag, snippet)))
        if code:
            probe = Probe(f"marker_{hint.first_line}", "warp", [], before="\n".join(code))
            references[hint.first_line - 1] = Ref(original, probe, True)
    return "\n".join(lines), references, regs + len(tags)


def get_arch(ptx):
    match = re.search(r"(?m)^\s*\.target\s+(sm_\d+[a-z]?)\b", ptx)
    if not match:
        raise DebugError("Embedded PTX has no supported declared target")
    return match[1]


def dump(image, workdir):
    ptx = Cubin(image).ptx()
    (Path(workdir) / "original.ptx").write_text(ptx)
    return ptx


def tool_version():
    digest = hashlib.sha256()
    for name in ("cuda.py", "cuda_function.py", "cubin_debug.py", "ptx_payload.py", "function_hints.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update((Path(__file__).parent.parent / "nvcc_wrapper.py").read_bytes())
    return digest.hexdigest()


def assembler_path():
    found = shutil.which("ptxas")
    if found:
        return found
    home = os.environ.get("CUDA_HOME")
    if home and (Path(home) / "bin/ptxas").is_file():
        return str(Path(home) / "bin/ptxas")
    raise DebugError("ptxas was not found in PATH or CUDA_HOME")


def assemble(workdir, target, assembler=None):
    workdir = Path(workdir)
    command = [assembler or assembler_path(), f"-arch={target}", "-m64", "--verbose",
               str(workdir / "probed.ptx"), "-o", str(workdir / "probed.bin")]
    result = subprocess.run(command, capture_output=True, text=True)
    (workdir / "process.log").write_text(json.dumps(command) + "\n" + result.stdout + result.stderr)
    if result.returncode:
        raise DebugError(f"ptxas failed ({result.returncode}): {result.stderr.strip()}")


def compile_kernel(image, kernel, raw, workdir, *, assembler=None, require_hints=None):
    """Cubin-only adapter; unmarked kernels are never inferred from debug/CFG data."""
    import toml
    from neutrino.utils.trace_reading import gen_reading_code

    if require_hints is None:
        require_hints = os.environ.get("NEUTRINO_REQUIRE_SOURCE_HINTS") == "1"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": FUNCTION_PROBE_ABI, "runtime_abi": FUNCTION_RUNTIME_ABI,
                "kernel": kernel, "status": "unsupported", "boundary_source": "source-hints",
                "image_sha256": hashlib.sha256(image).hexdigest(), "tool_version": tool_version(),
                "sites": [], "rejected_regions": [], "representative_lane": 0,
                "require_source_hints": require_hints, "rejected_markers": [], "hint_abi": HINT_ABI}
    try:
        probes, maps, regs = load(raw)
        if not probes or not all(p.funcname is not None for p in probes):
            raise DebugError("Function compiler requires function selectors")
        if raw.get("function_trace_layout", "events") != "events":
            raise DebugError("The mapped-host timestamp-only layout is retired; use ordinary Neutrino maps")
        selectors = sorted({p.funcname for p in probes})
        manifest["selectors"] = selectors
        ptx = dump(image, workdir)
        check_retained_targets(ptx, selectors)
        hints, sites = select_hints(ptx, kernel, selectors, manifest["image_sha256"])
        manifest["markers"] = [hint.manifest() for hint in hints]
        if not sites:
            if require_hints:
                raise DebugError("Expected selected source hints are missing; marked coverage is incomplete")
            manifest.update(status="unselected", reason="No selected source markers")
            return manifest
        # The existing entry pruner preserves retained device functions. All
        # marker coordinates are consumed before pruning/rebasing the entry.
        declaration, end, _, close = entry_bounds(ptx, kernel)
        entry_start = ptx.rfind("\n", 0, declaration.start()) + 1
        entry_first = ptx[:entry_start].count("\n")
        cleaned, refs, regs = marker_references(ptx, hints, sites, probes, regs)
        cleaned_lines = cleaned.splitlines()
        entry_last = ptx[:close].count("\n")
        entry = "\n".join(cleaned_lines[entry_first:entry_last + 1])
        local_refs = {line - entry_first: ref for line, ref in refs.items()}
        global_section, func_section, _, _ = cuda.prune(ptx, kernel)
        if decode_hints(global_section + "\n" + func_section):
            raise DebugError("Source hints in retained out-of-line functions are unsupported")
        parameters = re.sub(r"//[^\n]*", "", ptx[declaration.end():end]).strip()
        parameters = [p.strip() for p in parameters.split(",")] if parameters else []
        lowered = cuda.probing(entry, [], maps, regs, references=local_refs, registers=raw.get("registers", ()))
        lowered = global_section + "\n" + func_section + "\n" + lowered
        if decode_hints(lowered):
            raise DebugError("Source hints remain after lowering")
        manifest.update(sites=sites, parameters=parameters, target=get_arch(ptx),
                        maps=[{"name": m.name, "size": m.size, "cap": m.cap} for m in maps],
                        appended_parameters=[f"param_{m.name}" for m in maps],
                        recording_backend="neutrino.probe.cuda.probing",
                        collection_backend="cuda-device-maps-event-async-dtoh-pool",
                        valid_records="zero-terminated-timestamps", capacity_policy="caller-sized-unchecked",
                        completion={"signal": "cuda-event-query", "copy": "pinned-async-dtoh",
                                    "reuse": "copy-then-reset-event", "exhaustion": "skip-trace"})
        binary_assembler = assembler or assembler_path()
        version = subprocess.check_output([binary_assembler, "--version"], text=True)
        manifest["assembler"] = version.strip()
        identity = [manifest["image_sha256"], kernel, raw, manifest["tool_version"], version, FUNCTION_RUNTIME_ABI]
        manifest["cache_key"] = hashlib.sha256(canonical(identity)).hexdigest()
        (workdir / "pruned.ptx").write_text(global_section + "\n" + func_section + "\n" + entry)
        (workdir / "probed.ptx").write_text(lowered)
        (workdir / "probe.toml").write_text(toml.dumps(raw))
        (workdir / "read.py").write_text(gen_reading_code(raw))
        assemble(workdir, manifest["target"], binary_assembler)
        info = [kernel, str(len(parameters)), str(len(maps)), *[f"1,{m.size * m.cap}" for m in maps]]
        (workdir / "kernel.info").write_text("\n".join(info) + "\n")
        info = [str(FUNCTION_RUNTIME_ABI), str(len(parameters)), str(len(maps)), "0", *[f"{m.size},{m.cap}" for m in maps]]
        (workdir / "function.info").write_text("\n".join(info) + "\n")
        manifest["status"] = "instrumented"
        return manifest
    except (DebugError, ValueError, OSError, subprocess.SubprocessError) as exc:
        manifest["reason"] = str(exc)
        manifest["rejected_markers"] = [{"reason": str(exc)}]
        raise
    finally:
        (workdir / "function.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    import argparse
    import sys
    import toml

    parser = argparse.ArgumentParser(description="Compile source-marker function probes")
    parser.add_argument("workdir", help="Artifact directory containing original.bin")
    parser.add_argument("kernel", help="Exact mangled launch-kernel symbol")
    parser.add_argument("probe", nargs="?", help="Compiled probe TOML (or NEUTRINO_PROBES)")
    args = parser.parse_args()
    try:
        raw = toml.load(args.probe) if args.probe else toml.loads(os.environ["NEUTRINO_PROBES"])
        result = compile_kernel((Path(args.workdir) / "original.bin").read_bytes(), args.kernel, raw, args.workdir)
        return 0 if result["status"] == "instrumented" else 2
    except (KeyError, DebugError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Function instrumentation unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
