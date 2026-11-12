"""Source-hint preparation and nvcc interposition without editing original files."""
from dataclasses import dataclass
import argparse
import ast
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

from .probe.cubin_debug import DebugError
from .probe.function_hints import HINT_ABI, canonical, marker_cpp

TRANSFORM_VERSION = 1
WRAPPER_VERSION = 1


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def cpp_tokens(source):
    """Restricted C++ lexer: literals/comments/directives cannot supply braces."""
    tokens, position = [], 0
    while position < len(source):
        char = source[position]
        if char.isspace():
            position += 1
            continue
        if source.startswith("//", position):
            end = source.find("\n", position)
            position = len(source) if end < 0 else end + 1
            continue
        if source.startswith("/*", position):
            end = source.find("*/", position + 2)
            if end < 0:
                raise DebugError("Unterminated C++ comment")
            position = end + 2
            continue
        if char == "#" and not source[source.rfind("\n", 0, position) + 1:position].strip():
            end = position
            while True:
                end = source.find("\n", end)
                if end < 0:
                    end = len(source)
                    break
                if end == 0 or source[end - 1] != "\\":
                    break
                end += 1
            tokens.append(Token("#directive", position, end))
            position = end
            continue
        raw = re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(', source[position:])
        if raw:
            close = ")" + raw[1] + '"'
            end = source.find(close, position + raw.end())
            if end < 0:
                raise DebugError("Unterminated C++ raw string")
            end += len(close)
            tokens.append(Token("literal", position, end))
            position = end
            continue
        literal = re.match(r"(?:u8|u|U|L)?([\"'])", source[position:])
        if literal:
            end = position + literal.end()
            while end < len(source) and source[end] != literal[1]:
                end += 2 if source[end] == "\\" else 1
            if end >= len(source):
                raise DebugError("Unterminated C++ literal")
            end += 1
            tokens.append(Token("literal", position, end))
            position = end
            continue
        word = re.match(r"[A-Za-z_]\w*|::|->|&&|\|\||\S", source[position:])
        end = position + word.end()
        tokens.append(Token(word[0], position, end))
        position = end
    return tokens


def delimiter_pairs(tokens):
    stack, pairs = [], {}
    for index, token in enumerate(tokens):
        if token.text in ("(", "[", "{"):
            stack.append(index)
        elif token.text in (")", "]", "}"):
            if not stack or tokens[stack[-1]].text != {")": "(", "]": "[", "}": "{"}[token.text]:
                raise DebugError("Unsupported/unbalanced C++ delimiters")
            opening = stack.pop()
            pairs[opening], pairs[index] = index, opening
    if stack:
        raise DebugError("Unbalanced C++ delimiters")
    return pairs


def function_bodies(tokens, pairs, functions):
    namespace_at, namespaces = {}, []
    for index, token in enumerate(tokens):
        if token.text != "namespace":
            continue
        cursor, names = index + 1, []
        while cursor < len(tokens) and (tokens[cursor].text == "::" or
                                        re.fullmatch(r"[A-Za-z_]\w*", tokens[cursor].text)):
            if tokens[cursor].text != "::":
                names.append(tokens[cursor].text)
            cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "{":
            namespace_at[cursor] = tuple(names)
    result = {name: [] for name in functions}
    scopes = []
    for index, token in enumerate(tokens):
        if token.text == "{":
            scope = namespace_at.get(index)
            scopes.append(scope)
            if scope is not None:
                namespaces.extend(scope)
        elif token.text == "}":
            scope = scopes.pop()
            if scope:
                del namespaces[-len(scope):]
        elif scopes and all(scope for scope in scopes):
            qualified = "::".join([*namespaces, token.text])
            if qualified not in result or index + 1 >= len(tokens) or tokens[index + 1].text != "(":
                continue
            close = pairs[index + 1]
            if close + 1 < len(tokens) and tokens[close + 1].text == "{":
                if index == 0 or tokens[index - 1].text != "void":
                    raise DebugError("Source-hint transformer supports void function definitions only")
                result[qualified].append((index, close + 1, pairs[close + 1]))
    for name, bodies in result.items():
        if len(bodies) != 1:
            raise DebugError(f"Expected one supported definition of {name}, found {len(bodies)}")
    return {name: bodies[0] for name, bodies in result.items()}


def owned_returns(tokens, pairs, opening, closing):
    """Skip recognized nested lambdas; reject unsupported local type scopes."""
    returns, index = [], opening + 1
    while index < closing:
        value = tokens[index].text
        if value == "#directive":
            raise DebugError("Preprocessor directives inside a selected body are unsupported")
        if value in ("class", "struct", "union"):
            raise DebugError("Local type scopes in selected bodies are unsupported")
        if value == "[":
            cursor = pairs[index] + 1
            if cursor < closing and tokens[cursor].text == "(":
                cursor = pairs[cursor] + 1
            while cursor < closing and tokens[cursor].text in (
                    "mutable", "constexpr", "noexcept", "__device__", "__host__"):
                qualifier = tokens[cursor].text
                cursor += 1
                if qualifier == "noexcept" and cursor < closing and tokens[cursor].text == "(":
                    cursor = pairs[cursor] + 1
            if cursor < closing and tokens[cursor].text == "{":
                index = pairs[cursor] + 1
                continue
            # Explicit trailing return types/generic lambdas need a richer parser.
            if cursor < closing and tokens[cursor].text in ("->", "<", "requires"):
                raise DebugError("Unsupported nested lambda signature")
        if value == "return":
            if tokens[index + 1].text != ";":
                raise DebugError("Only target-owned return; statements are supported")
            returns.append((tokens[index].start, tokens[index + 1].end))
        index += 1
    return returns


def transform_source(source, logical_file, functions, expected_sha256):
    """Return a marked private copy and insertion evidence, never write inputs."""
    if not isinstance(source, str):
        raise DebugError("Source must be UTF-8 text")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != expected_sha256:
        raise DebugError("Source fingerprint does not match the approved hint policy")
    if "__nh_word" in source or "0x4e485431" in source.lower():
        raise DebugError("Source already contains or collides with reserved hints")
    if not functions or len(set(functions)) != len(functions):
        raise DebugError("Hint policy requires distinct function names")
    tokens = cpp_tokens(source)
    pairs = delimiter_pairs(tokens)
    bodies = function_bodies(tokens, pairs, functions)
    rule = {"version": TRANSFORM_VERSION, "abi": HINT_ABI, "file": logical_file,
            "functions": sorted(functions), "source_sha256": digest}
    rule_sha = hashlib.sha256(canonical(rule)).hexdigest()
    insertions, evidence = [], []
    for name, (definition, opening, closing) in bodies.items():
        identity = {"function": name, "file": logical_file,
                    "line": source[:tokens[definition].start].count("\n") + 1,
                    "source_sha256": digest, "rule_sha256": rule_sha}
        begin, end = marker_cpp(identity, 0), marker_cpp(identity, 1)
        insertions.append((tokens[opening].end, "\n" + begin + "\n"))
        insertions.append((tokens[closing].start, "\n" + end + "\n"))
        returns = owned_returns(tokens, pairs, opening, closing)
        for first, last in returns:
            insertions.append((first, "{ " + end + " "))
            insertions.append((last, " }"))
        evidence.append({"identity": identity, "body_offsets": [tokens[opening].start, tokens[closing].end],
                         "return_offsets": returns})
    marked = source
    for position, text in sorted(insertions, reverse=True):
        marked = marked[:position] + text + marked[position:]
    return marked, {"version": TRANSFORM_VERSION, "source_sha256": digest,
                    "transformed_sha256": hashlib.sha256(marked.encode()).hexdigest(),
                    "rule_sha256": rule_sha, "functions": evidence,
                    "insertions": [{"offset": p, "text": t} for p, t in sorted(insertions)]}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_identity():
    return {"wrapper": file_sha256(__file__),
            "codec": file_sha256(Path(__file__).parent / "probe/function_hints.py"),
            "wrapper_version": WRAPPER_VERSION, "transform_version": TRANSFORM_VERSION,
            "hint_abi": HINT_ABI}


def load_policy(path):
    """Read a literal policy without importing the tool or executing probe code."""
    path = Path(path)
    if not path.is_file() and re.fullmatch(r"[A-Za-z_]\w*", str(path)):
        path = Path(__file__).parent / "tools" / (str(path) + ".py")
    text = path.read_text()
    if path.suffix == ".json":
        policy = json.loads(text)
    else:
        values = [node.value for node in ast.parse(text).body
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == "SOURCE_HINTS"
                      for target in node.targets)]
        if len(values) != 1:
            raise DebugError("Policy requires exactly one literal SOURCE_HINTS assignment")
        try:
            policy = ast.literal_eval(values[0])
        except (ValueError, TypeError) as exc:
            raise DebugError("SOURCE_HINTS must be declarative literal data") from exc
    validate_policy(policy)
    return policy


def validate_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != {"version", "headers"} or
            type(policy["version"]) is not int or policy["version"] != 1 or
            not isinstance(policy["headers"], list) or not policy["headers"]):
        raise DebugError("Unsupported source-hint policy schema/version")
    seen = set()
    for header in policy["headers"]:
        if not isinstance(header, dict) or set(header) != {
                "path", "functions", "source_sha256", "transform_version"}:
            raise DebugError("Invalid source-hint header rule")
        name = header["path"]
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or
                any(part in ("", ".", "..") for part in name.split("/")) or
                "\\" in name or name in seen):
            raise DebugError("Policy header paths must be distinct safe relative paths")
        seen.add(name)
        if (type(header["transform_version"]) is not int or
                header["transform_version"] != TRANSFORM_VERSION or
                not isinstance(header["source_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", header["source_sha256"]) or
                not isinstance(header["functions"], list) or not header["functions"] or
                any(not isinstance(f, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+", f)
                    for f in header["functions"]) or
                len(set(header["functions"])) != len(header["functions"])):
            raise DebugError("Invalid function names, source fingerprint, or transformation version")


def real_compiler(path):
    path = Path(path)
    if not path.is_absolute():
        raise DebugError("The real nvcc path must be absolute")
    path = path.resolve(strict=True)
    if (not path.is_file() or not os.access(path, os.X_OK) or path.name == "neutrino-nvcc" or
            path == Path(sys.argv[0]).resolve()):
        raise DebugError("Invalid real compiler or recursive wrapper configuration")
    return path


def prepare(policy, include_root, real_nvcc, artifact_root, *, jit_cache_root=None):
    """Atomically publish a content-addressed overlay and return its config path."""
    if not isinstance(policy, dict):
        policy = load_policy(policy)
    validate_policy(policy)
    compiler = real_compiler(real_nvcc)
    version = subprocess.run([str(compiler), "--version"], capture_output=True, text=True, check=True,
                             env=dict(os.environ, NEUTRINO_NVCC_ACTIVE="1"))
    root = Path(include_root).resolve(strict=True)
    artifacts = Path(artifact_root).resolve()
    copies, headers = {}, []
    for rule in policy["headers"]:
        original = (root / rule["path"]).resolve(strict=True)
        if not original.is_relative_to(root):
            raise DebugError("Policy header resolves outside the actual include root")
        source = original.read_bytes().decode("utf-8")
        marked, evidence = transform_source(source, rule["path"], rule["functions"], rule["source_sha256"])
        copies[rule["path"]] = marked.encode()
        headers.append(dict(evidence, path=rule["path"], original=str(original),
                            original_size=original.stat().st_size))
    cache_root = Path(jit_cache_root).resolve() if jit_cache_root else artifacts / "neutrino-marker-profile"
    identity = {"implementation": implementation_identity(), "policy": policy, "headers": headers,
                "artifact_root": str(artifacts), "jit_cache_root": str(cache_root),
                "include_root": str(root), "compiler": str(compiler),
                "compiler_sha256": file_sha256(compiler),
                "compiler_version": version.stdout + version.stderr}
    digest = hashlib.sha256(canonical(identity)).hexdigest()
    destination = artifacts / "neutrino-source-hints" / digest
    config = {"version": WRAPPER_VERSION, "digest": digest, "identity": identity,
              "real_nvcc": str(compiler), "overlay_include": str(destination / "include"),
              "jit_cache_dir": str(cache_root / digest),
              "invocation_dir": str(artifacts / "neutrino-nvcc-invocations" / digest),
              "headers": headers}
    files = {"include/" + name: content for name, content in copies.items()}
    files["config.json"] = canonical(config) + b"\n"
    files["preparation.json"] = json.dumps(config, indent=2).encode() + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for name, content in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            os.rename(staging, destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
        # A concurrent publisher must have produced identical complete contents.
        for name, content in files.items():
            if (destination / name).read_bytes() != content:
                raise DebugError(f"Published overlay is inconsistent: {destination / name}")
    return destination / "config.json"


def validate_config(path):
    config = json.loads(Path(path).read_text())
    identity = config["identity"]
    if (config["version"] != WRAPPER_VERSION or identity["implementation"] != implementation_identity() or
            config["digest"] != hashlib.sha256(canonical(identity)).hexdigest()):
        raise DebugError("Stale/corrupt source-hint configuration; run prepare again")
    compiler = real_compiler(config["real_nvcc"])
    if str(compiler) != identity["compiler"] or file_sha256(compiler) != identity["compiler_sha256"]:
        raise DebugError("Real compiler changed; run prepare again")
    validate_policy(identity["policy"])
    artifacts, digest = Path(identity["artifact_root"]), config["digest"]
    expected_paths = {
        "overlay_include": str(artifacts / "neutrino-source-hints" / digest / "include"),
        "jit_cache_dir": str(Path(identity["jit_cache_root"]) / digest),
        "invocation_dir": str(artifacts / "neutrino-nvcc-invocations" / digest),
    }
    if config["headers"] != identity["headers"] or any(config[k] != v for k, v in expected_paths.items()):
        raise DebugError("Corrupt source-hint configuration paths/headers")
    for header in config["headers"]:
        for path, expected in ((Path(header["original"]), header["source_sha256"]),
                               (Path(config["overlay_include"]) / header["path"], header["transformed_sha256"])):
            if file_sha256(path) != expected:
                raise DebugError(f"Source/overlay fingerprint changed: {path}; run prepare again")
    return config


def dependency_arguments(argv):
    """Keep preprocessing semantics, removing only compile mode and output flags."""
    result, modes, sources, index = [], [], [], 0
    while index < len(argv):
        value = argv[index]
        key = value.split("=", 1)[0]
        host_options = value if "=" in value else (argv[index + 1] if index + 1 < len(argv) else "")
        if (value.startswith("@") or key in ("-optf", "--options-file") or
                key in ("-Xcompiler", "--compiler-options") and any(
                    x in host_options for x in ("-include", "-imacros", "-MF", "-MD"))):
            raise DebugError("Response files/hidden preprocessing options are unsupported; expand them in nvcc argv")
        if key in ("-M", "-MM", "-MD", "-MMD", "-MF", "-MT", "-MQ", "--generate-dependencies",
                   "--generate-nonsystem-dependencies", "--generate-dependencies-with-compile",
                   "--generate-nonsystem-dependencies-with-compile", "--dependency-output", "--dependency-target-name"):
            raise DebugError("Explicit dependency-output options are unsupported in hint compilation")
        if key in ("--cubin", "-cubin", "--ptx", "-ptx", "-c", "--compile", "-dc", "--device-c", "--fatbin", "-fatbin"):
            modes.append(key)
        elif key in ("-o", "--output-file"):
            if "=" not in value:
                index += 1
                if index == len(argv):
                    raise DebugError("Missing nvcc output path")
        elif value.startswith("-o") and not value.startswith("--") and len(value) > 2:
            raise DebugError("Use -o PATH or --output-file PATH, not an attached output option")
        else:
            result.append(value)
            if not value.startswith("-") and Path(value).suffix in (".cu", ".cpp", ".cc", ".cxx", ".c"):
                sources.append(value)
        index += 1
    if len(modes) != 1 or len(sources) != 1:
        raise DebugError("Dependency checking requires one source and exactly one compilation mode")
    return [*result, "-M"]


def dependencies(compiler, argv, env):
    result = subprocess.run([compiler, *dependency_arguments(argv)], capture_output=True, text=True, env=env)
    if result.returncode:
        raise DebugError(f"nvcc dependency check failed ({result.returncode}): {result.stderr.strip()}")
    text = result.stdout.replace("\\\n", " ")
    if ":" not in text:
        raise DebugError("nvcc dependency check returned no make target")
    names = shlex.split(text.split(":", 1)[1])
    resolved = sorted({str(Path(name).resolve(strict=True)) for name in names})
    return resolved


def target_inputs(config, inputs):
    """Find canonical targets and alternate/absolute header copies in dependencies."""
    found = {}
    for header in config["headers"]:
        target = Path(config["overlay_include"]) / header["path"]
        matches = []
        for name in inputs:
            path = Path(name)
            same_name = path.as_posix().endswith("/" + header["path"])
            same_bytes = (path.stat().st_size == header["original_size"] and
                          file_sha256(path) == header["source_sha256"])
            if path == target or str(path) == header["original"] or same_name or same_bytes:
                matches.append(name)
        if matches:
            found[header["path"]] = matches
    return found


def invoke(config_path, argv):
    if os.environ.get("NEUTRINO_NVCC_ACTIVE"):
        raise DebugError("Recursive nvcc wrapper invocation rejected")
    config = validate_config(config_path)
    compiler = config["real_nvcc"]
    env = dict(os.environ, NEUTRINO_NVCC_ACTIVE="1")
    if not argv or all(arg in ("--version", "-V", "--help", "-h", "--list-gpu-arch", "--list-gpu-code") for arg in argv):
        return subprocess.run([compiler, *argv], env=env).returncode
    # Unsupported compilation forms must not quietly produce an unmarked image.
    original_inputs = dependencies(compiler, argv, env)
    original_targets = target_inputs(config, original_inputs)
    effective = list(argv)
    effective_inputs = original_inputs
    if original_targets:
        if not any(mode in argv for mode in ("--cubin", "-cubin", "--ptx", "-ptx")):
            raise DebugError("Targeted compilation supports only --cubin or --ptx")
        for header in config["headers"]:
            if header["path"] in original_targets and original_targets[header["path"]] != [header["original"]]:
                raise DebugError(f"Target header bypasses configured include root: {original_targets[header['path']]}")
        effective = ["--include-path", config["overlay_include"], *argv]
        effective_inputs = dependencies(compiler, effective, env)
        resolved = target_inputs(config, effective_inputs)
        for name in set(original_targets) | set(resolved):
            expected = str(Path(config["overlay_include"]) / name)
            if resolved.get(name) != [expected]:
                raise DebugError(f"Target header bypasses overlay: {name}: {resolved.get(name)}; use the logical include path")
        if any(mode in argv for mode in ("--cubin", "-cubin")) and not any(
                flag in argv for flag in ("--generate-line-info", "-lineinfo")):
            raise DebugError("Targeted cubin compilation requires --generate-line-info to retain embedded PTX")
    audit = {"version": WRAPPER_VERSION, "digest": config["digest"], "cwd": os.getcwd(),
             "original_argv": [compiler, *argv], "effective_argv": [compiler, *effective],
             "original_inputs": original_inputs, "resolved_inputs": effective_inputs,
             "targeted_headers": sorted(original_targets)}
    validate_config(config_path)
    result = subprocess.run([compiler, *effective], env=env)
    audit["returncode"] = result.returncode
    root = Path(config["invocation_dir"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / (uuid.uuid4().hex + ".json")
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".audit-", delete=False) as stream:
        json.dump(audit, stream, indent=2)
        temporary = Path(stream.name)
    temporary.replace(target)
    validate_config(config_path)
    return result.returncode


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv and argv[0] == "prepare":
            parser = argparse.ArgumentParser(description="Prepare immutable source hints for nvcc")
            parser.add_argument("--policy", required=True, help="Tool name or literal Python/JSON policy path")
            parser.add_argument("--include-root", required=True)
            parser.add_argument("--real-nvcc", required=True)
            parser.add_argument("--artifact-root", required=True, help="Workspace output/cache directory")
            parser.add_argument("--jit-cache-root")
            args = parser.parse_args(argv[1:])
            print(prepare(args.policy, args.include_root, args.real_nvcc, args.artifact_root,
                          jit_cache_root=args.jit_cache_root))
            return 0
        config = os.environ.get("NEUTRINO_NVCC_CONFIG")
        if not config:
            raise DebugError("Set NEUTRINO_NVCC_CONFIG to a prepared config.json")
        status = invoke(config, argv)
        return status if status >= 0 else 128 - status
    except (DebugError, ValueError, KeyError, TypeError, SyntaxError, OSError, subprocess.SubprocessError) as exc:
        print(f"neutrino-nvcc: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
