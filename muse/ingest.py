"""python -m muse.ingest notes/*.md  — chunk markdown by heading, embed, link, cluster."""
import re
import sys
from pathlib import Path

from muse.graph import Memory

MAX_CHARS = 1200  # ~300 tokens: one idea per node, well under every embedder's context


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split on headings, then pack paragraphs up to MAX_CHARS. Returns (heading, chunk) pairs.
    The heading is prepended to each chunk so a bare paragraph keeps its topic when embedded."""
    out, heading = [], ""
    for section in re.split(r"(?m)^(?=#{1,6} )", text):
        lines = section.strip().splitlines()
        if not lines:
            continue
        if lines[0].startswith("#"):
            heading, lines = lines[0].lstrip("#").strip(), lines[1:]
        buf = ""
        for para in re.split(r"\n\s*\n", "\n".join(lines).strip()):
            if buf and len(buf) + len(para) > MAX_CHARS:
                out.append((heading, buf))
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
        if buf:
            out.append((heading, buf))
    return [(h, f"{h}\n\n{c}" if h else c) for h, c in out]


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    m = Memory()
    texts, sources = [], []
    for p in paths:
        for heading, chunk in chunk_markdown(Path(p).read_text(encoding="utf-8")):
            texts.append(chunk)
            sources.append(f"{Path(p).as_posix()}#{heading}")
    new = m.add(texts, sources)
    s = m.stats()
    print(f"{len(texts)} chunks from {len(paths)} files, {len(new)} new "
          f"-> {s['nodes']} nodes, {s['edges']} edges, {s['clusters']} clusters")


if __name__ == "__main__":
    main(sys.argv[1:])
