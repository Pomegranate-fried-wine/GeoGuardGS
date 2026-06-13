#!/usr/bin/env python3
"""GeoGuardGS training wrapper.

The current release keeps the runnable Street Gaussian training code under
third_party/street_gaussian for compatibility. This wrapper launches that
entrypoint from the correct working directory and passes through all arguments.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    streetgs_root = repo_root / "third_party" / "street_gaussian"
    train_entry = streetgs_root / "train.py"
    if not train_entry.exists():
        raise SystemExit(f"Missing Street Gaussian train entry: {train_entry}")
    env = os.environ.copy()
    pythonpath = [str(streetgs_root), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    args = []
    for arg in sys.argv[1:]:
        if arg.endswith(".yaml") or arg.endswith(".yml"):
            p = Path(arg)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            args.append(str(p))
        else:
            args.append(arg)
    cmd = [sys.executable, str(train_entry)] + args
    raise SystemExit(subprocess.call(cmd, cwd=str(streetgs_root), env=env))


if __name__ == "__main__":
    main()
