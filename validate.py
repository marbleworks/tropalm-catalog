#!/usr/bin/env python3
"""Check a Tropalm model catalog before it is published.

    python validate.py v1/models.json
    python validate.py v1/models.json --against base.json   # a change must raise the revision
    python validate.py v1/models.json --check-urls          # and ask GitHub for every runtime part

A file of the voice pack the catalog serves itself (source {"catalog": "voices/x.wav"}) must be
in this repository beside the document, at the entry's length and sha256: it is published from
here, so a mismatch is a pack that fails every player's download.

The same rules the game reads the file by (ModelCatalog.Parse in the game's client), stated as
errors rather than skips: the game drops one bad entry and lists the rest, which is right for a
player and wrong for a publisher, who should never ship the bad entry at all. Standard library
only, so the Pages workflow needs nothing installed.

A runtime (a program the voice engines run, downloaded as GitHub release assets) is checked for
its shape here; with --check-urls each part is also asked for on GitHub. A part that is not
published yet (404: the release is the owner's to push) is a warning and does not fail the run; a
part published at another length than the entry says is a problem.

Exit 0 when the file is publishable, 1 with one line per problem otherwise.
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

SCHEMA = 1
TIERS = {"recommended", "supported", "experimental"}
SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_BUILDS = {"vulkan", "cuda", "metal", "cpu"}
RUNTIME_SYSTEMS = {"windows", "macos", "linux"}
RUNTIME_ARCHES = {"x64", "arm64"}
RELEASE_ASSET = re.compile(r"^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/releases/download/"
                           r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
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
        # A voice file is not a model anybody chooses between, so it has no tier. It is what a
        # companion speaks with, though, and the game's credits name it by its licence: the
        # speech model a cloned voice is spoken by and the recording it is cloned from.
        if voice:
            check_licence(where, m.get("licence"), problems)
        if not voice and m.get("tier") not in TIERS:
            problems.append(where + ": tier is not one of %s" % ", ".join(sorted(TIERS)))
        if not (isinstance(m.get("minGameVersion"), int) and m["minGameVersion"] >= 0):
            problems.append(where + ": minGameVersion is not a non-negative integer")


def check_licence(where, licence, problems):
    # {"name": "MIT", "url": "https://..."} -- the shape a runtime and a voice file share, and what
    # the game's credits list shows beside the thing it licenses.
    if not (isinstance(licence, dict) and isinstance(licence.get("name"), str) and licence["name"]):
        problems.append(where + ": licence needs a name")
    elif "url" in licence and not (isinstance(licence["url"], str) and licence["url"].startswith("https://")):
        problems.append(where + ": licence.url is not https")


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


def release_asset(url):
    """Whether url is a GitHub release asset and nothing else -- the game reads a runtime part by
    the same rule (ModelCatalog.IsReleaseAsset) and drops any other address."""
    return isinstance(url, str) and bool(RELEASE_ASSET.match(url))


def check_runtimes(runtimes, problems, ids=None, warnings=None, check_urls=False):
    # The programs the voice engines run (owner ruling Q-voice6 = A): one entry per build of a
    # program for one system and processor, fetched as GitHub release assets and unpacked into a
    # folder named by the id. The game drops a malformed entry; a publisher never ships one.
    ids = set() if ids is None else ids
    warnings = [] if warnings is None else warnings
    for i, r in enumerate(runtimes, 1):
        where = "runtimes[%d] (%s)" % (i, r.get("id", "?") if isinstance(r, dict) else "?")
        if not isinstance(r, dict):
            problems.append(where + ": not an object")
            continue
        # The id is the folder a runtime unpacks into, and a lookup key across every list.
        if not segment(r.get("id")) or r["id"] in ids:
            problems.append(where + ": id missing, not one plain segment, or used twice")
        else:
            ids.add(r["id"])
        if not isinstance(r.get("name"), str) or not r["name"]:
            problems.append(where + ": no name")
        if not segment(r.get("program")):
            problems.append(where + ": program is not a plain word")
        if r.get("build") not in RUNTIME_BUILDS:
            problems.append(where + ": build is not one of %s" % ", ".join(sorted(RUNTIME_BUILDS)))
        if r.get("os") not in RUNTIME_SYSTEMS:
            problems.append(where + ": os is not one of %s" % ", ".join(sorted(RUNTIME_SYSTEMS)))
        if r.get("arch") not in RUNTIME_ARCHES:
            problems.append(where + ": arch is not one of %s" % ", ".join(sorted(RUNTIME_ARCHES)))
        unpack = r.get("unpack")
        if unpack not in ("zip", "none"):
            problems.append(where + ": unpack is not zip or none")
        exe = r.get("exe", "")
        if not (isinstance(exe, str) and exe and all(segment(p) for p in exe.split("/"))):
            problems.append(where + ": exe is not a path inside the runtime's folder")
        if unpack == "zip" and not (isinstance(r.get("unpackedBytes"), int) and r["unpackedBytes"] > 0):
            problems.append(where + ": an archive needs unpackedBytes, a positive integer (the room it takes)")
        check_licence(where, r.get("licence"), problems)
        if not (isinstance(r.get("minGameVersion"), int) and r["minGameVersion"] >= 0):
            problems.append(where + ": minGameVersion is not a non-negative integer")
        parts = r.get("parts")
        if not isinstance(parts, list) or not parts:
            problems.append(where + ": no parts")
            continue
        if unpack == "none" and len(parts) != 1:
            problems.append(where + ": a program placed as it is comes as one part")
        for j, p in enumerate(parts, 1):
            pw = "%s.parts[%d]" % (where, j)
            url = p.get("url", "") if isinstance(p, dict) else ""
            if not release_asset(url):
                problems.append(pw + ": url is not https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>")
            if not (isinstance(p, dict) and isinstance(p.get("sha256"), str) and SHA256.match(p["sha256"])):
                problems.append(pw + ": sha256 is not 64 lower-case hex digits")
            if not (isinstance(p, dict) and isinstance(p.get("bytes"), int) and p["bytes"] > 0):
                problems.append(pw + ": bytes is not a positive integer")
            elif check_urls and release_asset(url):
                check_published(pw, url, p["bytes"], problems, warnings)


def check_published(where, url, size, problems, warnings):
    """Ask GitHub for the asset's first byte. Not there yet (a release the owner has still to push)
    is a warning: the entry is right and the download will work the day the release exists. There
    at another length is a problem: the entry names some other file."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "tropalm-catalog-validate"})
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            content_range = answer.headers.get("Content-Range", "")
            total = content_range.rsplit("/", 1)[-1] if "/" in content_range else answer.headers.get("Content-Length")
            if total and total.isdigit() and int(total) != size:
                problems.append("%s: the published asset is %s bytes, the entry says %d" % (where, total, size))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            warnings.append("%s: not published yet (404) -- %s" % (where, url))
        else:
            warnings.append("%s: GitHub answered %d -- %s" % (where, e.code, url))
    except (urllib.error.URLError, OSError) as e:
        warnings.append("%s: could not be asked (%s)" % (where, e))


def check(doc, base=None, base_dir=None, warnings=None, check_urls=False):
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
        check_runtimes(doc.get("runtimes", []), problems, ids, warnings, check_urls)
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
    warnings = []
    problems = check(doc, base, base_dir=os.path.dirname(os.path.abspath(argv[0])), warnings=warnings,
                     check_urls="--check-urls" in argv)
    for w in warnings:
        print("catalog: warning: " + w)
    for p in problems:
        print("catalog: " + p)
    if not problems:
        print("catalog: revision %d, %d model(s), %d voice file(s), %d runtime(s) -- publishable"
              % (doc["revision"], len(doc["models"]), len(doc.get("voice", [])), len(doc.get("runtimes", []))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
