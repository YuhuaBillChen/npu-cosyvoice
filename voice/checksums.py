"""Check downloaded files against a sha256sum list before anything loads them.

    python voice/checksums.py SUMS DIR [EXACT_SUBDIR...]

Every file SUMS names must exist under DIR with that hash, and an EXACT_SUBDIR may hold no other
files: a model loader reads whatever is in its directory (an extra tokenizer.json or config file
would be used too), so an unlisted file there counts as tampering. Exits 1 with the first problem.
"""
import hashlib
import os
import sys


def problem(sums, root, exact=()):
    listed = {}
    with open(sums) as f:
        for line in f:
            if line.strip():
                digest, name = line.split(maxsplit=1)
                listed[os.path.normpath(name.strip().lstrip("*"))] = digest.lower()
    for name, digest in listed.items():
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            return f"missing {name}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != digest:
            return f"checksum mismatch: {name}"
    for sub in exact:
        for dirpath, _, files in os.walk(os.path.join(root, sub)):
            for file in files:
                name = os.path.relpath(os.path.join(dirpath, file), root)
                if name not in listed:
                    return f"unexpected file {name}"
    return None


if __name__ == "__main__":
    error = problem(sys.argv[1], sys.argv[2], sys.argv[3:])
    if error:
        print(error, file=sys.stderr)
    sys.exit(1 if error else 0)
