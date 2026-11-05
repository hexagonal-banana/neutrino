#!/usr/bin/env python3
"""Record reproducible NVCC inputs and intermediates without changing sources.

Set NEUTRINO_REAL_NVCC and NEUTRINO_COMPILER_RECORDS, then use this executable
as the build's NVCC. Each output has its own keep directory, including device
link steps, so parallel translation units cannot overwrite one another.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    arguments = sys.argv[1:]
    compiler = os.environ['NEUTRINO_REAL_NVCC']
    if '-o' not in arguments or any(a in arguments for a in ('-M', '-MM', '--version')):
        return subprocess.call([compiler, *arguments])
    output = Path(arguments[arguments.index('-o') + 1]).resolve()
    key = hashlib.sha256(str(output).encode()).hexdigest()[:24]
    root = Path(os.environ['NEUTRINO_COMPILER_RECORDS']).resolve()
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    optimization = [] if any(a.startswith('-O') for a in arguments) else ['-O3']
    effective = [compiler, '--verbose', '--keep', '--keep-dir', str(directory),
                 *optimization, *arguments]
    metadata = {'schema': 1, 'cwd': str(Path.cwd()), 'output': str(output),
                'arguments': arguments, 'command': effective, 'keep_dir': str(directory),
                'compiler': compiler, 'status': 'running'}
    record = directory / 'command.json'
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    with (directory / 'compiler.log').open('w') as log:
        process = subprocess.Popen(effective, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)
        for line in process.stdout:
            log.write(line)
        status = process.wait()
    metadata.update(status='passed' if status == 0 else 'failed', exit_code=status)
    record.write_text(json.dumps(metadata, indent=2) + '\n')
    if status:
        print(f'NVCC failed ({status}); see {directory / "compiler.log"}', file=sys.stderr)
        print((directory / 'compiler.log').read_text()[-12000:], file=sys.stderr)
    return status


if __name__ == '__main__':
    sys.exit(main())
