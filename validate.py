#!/usr/bin/env python3
"""Check a Tropalm model catalog before it is published.

    python validate.py v1/models.json
    python validate.py v1/models.json --against base.json   # a change must raise the revision

A file of the voice pack the catalog serves itself (source {"catalog": "voices/x.wav"}) must be
in this repository beside the document, at the entry's length and sha256: it is published from
here, so a mismatch is a pack that fails every player's download.

The same rules the game reads the file by (ModelCatalog.Parse in the game's client), stated as
errors rather than skips: the game drops one bad entry and lists the rest, which is right for a
player and wrong for a publisher, who should never ship the bad entry at all. Standard library
only, so the Pages workflow needs nothing installed.

Exit 0 when the file is publishable, 1 with one line per problem otherwise.
"""

import hashlib
import json
import os
import re
import sys

SCHEMA = 1
TIERS = {"recommended", "supported", "experimental"}
SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_BUILDS = {"vulkan", "cuda", "metal", "cpu"}
# What a voice-pack file may end in: weights llama.cpp loads, the transcriber's ggml model, and a
# recording a cloned voice is taken from. The game reads the same list (ModelCatalog.FileEndings).
VOICE_ENDINGS = (".gguf", ".bin", ".wav")


def segment(value):
    return isinstance(value, str) and bool(SEGMENT.match(value)) and value not in (".", "..")


def bare_name(name, endings):
    if not isinstance(name, str) or not name or len(name) > 200:
        return False
    if not name.lower().endswith(endings) or name.startswith(".") or ".." in name:
        return False
    return not any(c in name for c in '/\\:<>"|?*') and all(ord(c) >= 32 for c in name)


def bare_gguf(name):
    return bare_name(name, (".gguf",))


def check_models(models, problems, ids=None, files=None, list_name="models", base_dir=None):
    ids = set() if ids is None else ids
    files = set() if files is None else files
    voice = list_name == "voice"
    for i, m in enumerate(models, 1):
        where = "%s[%d] (%s)" % (list_name, i, m.get("id", "?") if isinstance(m, dict) else "?")
        if not isinstance(m, dict):
            problems.append(where + ": not an object")
            continue
        if not isinstance(m.get("id"), str) or not m["id"]:
            problems.append(where + ": no id")
        elif m["id"] in ids:
            problems.append(where + ": id used twice")
        else:
            ids.add(m["id"])
        if not isinstance(m.get("name"), str) or not m["name"]:
            problems.append(where + ": no name")
        if not (bare_name(m.get("file"), VOICE_ENDINGS) if voice else bare_gguf(m.get("file"))):
            problems.append(where + ": file is not a bare %s name" % (" / ".join(VOICE_ENDINGS) if voice else ".gguf"))
        elif m["file"].lower() in files:
            problems.append(where + ": file used twice")
        else:
            files.add(m["file"].lower())
        src = m.get("source")
        if not isinstance(src, dict):
            problems.append(where + ": no source")
        elif voice and "catalog" in src:
            check_served(where, m, src.get("catalog"), base_dir, problems)
        else:
            repo = src.get("repo", "")
            if not (isinstance(repo, str) and len(repo.split("/")) == 2 and all(segment(p) for p in repo.split("/"))):
                problems.append(where + ": source.repo is not owner/name")
            rev = src.get("revision")
            if not segment(rev):
                problems.append(where + ": source.revision is not one path segment")
            elif not re.match(r"^[0-9a-f]{40}$", rev):
                problems.append(where + ": source.revision should be a commit hash, so the bytes never move")
            path = src.get("path", "")
            if not (isinstance(path, str) and path and all(segment(p) for p in path.split("/"))):
                problems.append(where + ": source.path is not a path inside the repo")
        if not (isinstance(m.get("sha256"), str) and SHA256.match(m["sha256"])):
            problems.append(where + ": sha256 is not 64 lower-case hex digits")
        if not (isinstance(m.get("bytes"), int) and m["bytes"] > 0):
            problems.append(where + ": bytes is not a positive integer")
        for key in ("vramMiB", "contextTokens"):
            if key in m and not (isinstance(m[key], int) and m[key] > 0):
                problems.append(where + ": %s is not a positive integer" % key)
        # A voice file is not a model anybody chooses between, so it has no tier.
        if not voice and m.get("tier") not in TIERS:
            problems.append(where + ": tier is not one of %s" % ", ".join(sorted(TIERS)))
        if not (isinstance(m.get("minGameVersion"), int) and m["minGameVersion"] >= 0):
            problems.append(where + ": minGameVersion is not a non-negative integer")


def check_served(where, m, path, base_dir, problems):
    # Served from beside the catalog: a relative path of plain segments, so it cannot leave the
    # published folder, and -- when the repository is at hand -- the very bytes the entry names.
    if not (isinstance(path, str) and path and all(segment(p) for p in path.split("/"))):
        problems.append(where + ": source.catalog is not a path beside the catalog")
        return
    if base_dir is None:
        return
    local = os.path.join(base_dir, *path.split("/"))
    if not os.path.isfile(local):
        problems.append(where + ": source.catalog %s is not in the repository beside the catalog" % path)
        return
    with open(local, "rb") as f:
        data = f.read()
    if len(data) != m.get("bytes"):
        problems.append(where + ": %s is %d bytes, the entry says %r" % (path, len(data), m.get("bytes")))
    if hashlib.sha256(data).hexdigest() != m.get("sha256"):
        problems.append(where + ": %s does not hash to the entry's sha256" % path)


def check_runtimes(runtimes, problems):
    # Designed for the optional CUDA download; no build reads it yet, so the shape is held here.
    ids = set()
    for i, r in enumerate(runtimes, 1):
        where = "runtimes[%d] (%s)" % (i, r.get("id", "?") if isinstance(r, dict) else "?")
        if not isinstance(r, dict):
            problems.append(where + ": not an object")
            continue
        if not segment(r.get("id")) or r["id"] in ids:
            problems.append(where + ": id missing or used twice")
        else:
            ids.add(r["id"])
        if r.get("build") not in RUNTIME_BUILDS:
            problems.append(where + ": build is not one of %s" % ", ".join(sorted(RUNTIME_BUILDS)))
        for key in ("os", "arch"):
            if not segment(r.get(key)):
                problems.append(where + ": no %s" % key)
        parts = r.get("parts")
        if not isinstance(parts, list) or not parts:
            problems.append(where + ": no parts")
            continue
        for j, p in enumerate(parts, 1):
            pw = "%s.parts[%d]" % (where, j)
            url = p.get("url", "") if isinstance(p, dict) else ""
            if not (isinstance(url, str) and url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/")):
                problems.append(pw + ": url is not a llama.cpp release asset")
            if not (isinstance(p, dict) and isinstance(p.get("sha256"), str) and SHA256.match(p["sha256"])):
                problems.append(pw + ": sha256 is not 64 lower-case hex digits")
            if not (isinstance(p, dict) and isinstance(p.get("bytes"), int) and p["bytes"] > 0):
                problems.append(pw + ": bytes is not a positive integer")
        if not (isinstance(r.get("minGameVersion"), int) and r["minGameVersion"] >= 0):
            problems.append(where + ": minGameVersion is not a non-negative integer")


def check(doc, base=None, base_dir=None):
    problems = []
    ids, files = set(), set()
    if not isinstance(doc, dict):
        return ["the catalog is not a JSON object"]
    if doc.get("schema") != SCHEMA:
        problems.append("schema is %r; this validator checks schema %d" % (doc.get("schema"), SCHEMA))
    if not (isinstance(doc.get("revision"), int) and doc["revision"] >= 0):
        problems.append("revision is not a non-negative integer")
    if not isinstance(doc.get("models"), list):
        problems.append("models is not a list")
    else:
        check_models(doc["models"], problems, ids, files)
    # The voice pack. Optional: a catalog written before it has none. Ids and file names are
    # unique across both lists, because both land in one folder.
    if not isinstance(doc.get("voice", []), list):
        problems.append("voice is not a list")
    else:
        check_models(doc.get("voice", []), problems, ids, files, list_name="voice", base_dir=base_dir)
    if not isinstance(doc.get("runtimes", []), list):
        problems.append("runtimes is not a list")
    else:
        check_runtimes(doc.get("runtimes", []), problems)
    # A base with no revision is a first publish: there is nothing to have risen from.
    if isinstance(base, dict) and "revision" in base:
        changed = {k: v for k, v in doc.items() if k != "revision"} != {k: v for k, v in base.items() if k != "revision"}
        if changed and not (isinstance(doc.get("revision"), int) and isinstance(base.get("revision"), int)
                            and doc["revision"] > base["revision"]):
            problems.append("the catalog changed but revision did not rise (was %r, is %r) -- an offline "
                            "build decides between its cached and shipped copies by it"
                            % (base.get("revision"), doc.get("revision")))
    return problems


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        doc = json.load(f)
    base = None
    if "--against" in argv:
        with open(argv[argv.index("--against") + 1], encoding="utf-8") as f:
            base = json.load(f)
    problems = check(doc, base, base_dir=os.path.dirname(os.path.abspath(argv[0])))
    for p in problems:
        print("catalog: " + p)
    if not problems:
        print("catalog: revision %d, %d model(s), %d voice file(s), %d runtime(s) -- publishable"
              % (doc["revision"], len(doc["models"]), len(doc.get("voice", [])), len(doc.get("runtimes", []))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
