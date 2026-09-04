"""Local snapshots of primary authority, read a section at a time.

Two sources, one shape. ``build_cfr`` fetches a CFR title from the eCFR versioner
endpoint as it stood on a date. ``build_usc`` fetches a U.S. Code title from the House's
release point in force on that date -- the Code changes only when a law passes, so a
release point covers the span until the next one. Both write two files::

    <name>.xml    the document, unmodified
    index.json    metadata plus {section name: [offset, length]}

Offsets are byte positions, so ``Corpus.section`` seeks and reads one section rather than
parsing the whole title. It returns raw markup; the caller renders it, which keeps a
corpus section and a fetched section on the same conversion.

    python -m tax_agent.corpus build --kind cfr --dest data/cfr
    python -m tax_agent.corpus build --kind usc --dest data/usc
"""

import argparse
import gzip
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

INDEX_NAME = "index.json"
FETCH_TIMEOUT_SECONDS = 600
KINDS = ("cfr", "usc")

# Matches the User-Agent the agent's other fetches send. Defined here so this module
# has no import-time dependencies.
USER_AGENT = "ValsAI/tax-agent"

# A corpus with no later bound covers every date from its start onward.
OPEN_ENDED = "9999-12-31"

# eCFR wraps each section as <DIV8 N="1.163(j)-7" TYPE="SECTION">. The U.S. Code uses
# USLM, <section identifier="/us/usc/t26/s163">, and those nest -- a section's notes can
# quote another act's section -- so spans are found by tag depth, not by first close.
_CFR_OPEN = re.compile(rb'<DIV8[^>]*\bN="([^"]+)"[^>]*\bTYPE="SECTION"[^>]*>')
_CFR_TAG = re.compile(rb"</?DIV8\b")
_CFR_ROOT_CLOSE = b"</ECFR>"

_USC_TAG = re.compile(rb"</?section\b")
_USC_ROOT_CLOSE = b"</uscDoc>"


def _fetch(url: str) -> bytes:
    # eCFR refuses (406) any request that does not accept a compressed response.
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310
        body = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            return gzip.decompress(body)
        return body


def _spans(xml: bytes, opens: re.Pattern[bytes], tag: re.Pattern[bytes]) -> dict[bytes, list[int]]:
    """Map each opening tag's captured name to the [offset, length] of its whole element.

    Walks tag depth from each opening tag, so an element of the same name nested inside
    does not end the span early.
    """
    found: dict[bytes, list[int]] = {}
    for match in opens.finditer(xml):
        depth = 0
        end = len(xml)
        for boundary in tag.finditer(xml, match.start()):
            depth += -1 if boundary.group(0).startswith(b"</") else 1
            if depth == 0:
                closed = xml.find(b">", boundary.end())
                end = len(xml) if closed == -1 else closed + 1
                break
        found[match.group(1)] = [match.start(), end - match.start()]
    return found


def index_cfr(xml: bytes) -> dict[str, list[int]]:
    """Index an eCFR title response by section name."""
    spans = _spans(xml, _CFR_OPEN, _CFR_TAG)
    return {name.decode("utf-8", "replace"): span for name, span in spans.items()}


def index_usc(xml: bytes, title: str) -> dict[str, list[int]]:
    """Index a USLM U.S. Code title by section number."""
    prefix = f"/us/usc/t{title}/s".encode()
    opens = re.compile(rb'<section\b[^>]*\bidentifier="(' + re.escape(prefix) + rb'[^"/]+)"')
    spans = _spans(xml, opens, _USC_TAG)
    return {name[len(prefix) :].decode("utf-8", "replace"): span for name, span in spans.items()}


@dataclass(frozen=True)
class Corpus:
    """A built corpus, read one section at a time."""

    directory: Path
    kind: str
    title: str
    currency: str
    valid_from: str
    valid_until: str
    xml: str
    sections: dict[str, list[int]]

    @classmethod
    def load(cls, directory: Path) -> "Corpus | None":
        """Load the corpus in ``directory``, or None if it is absent or unreadable."""
        try:
            meta = json.loads((directory / INDEX_NAME).read_text())
            return cls(
                directory=directory,
                kind=meta["kind"],
                title=meta["title"],
                currency=meta["currency"],
                valid_from=meta["valid_from"],
                valid_until=meta["valid_until"],
                xml=meta["xml"],
                sections=meta["sections"],
            )
        except (OSError, ValueError, KeyError):
            return None

    def covers(self, title: str, as_of_date: str) -> bool:
        """Whether this corpus holds that title as the law stood on that date."""
        return self.title == title and self.valid_from <= as_of_date < self.valid_until

    def section(self, name: str) -> bytes | None:
        """The raw markup of one section, or None if the corpus does not carry it."""
        located = self.sections.get(name)
        if located is None:
            return None
        offset, length = located
        with open(self.directory / self.xml, "rb") as handle:
            _ = handle.seek(offset)
            return handle.read(length)


def _require_closed(url: str, body: bytes, root_close: bytes) -> None:
    if not body.rstrip().endswith(root_close):
        raise ValueError(
            f"{url} returned {len(body):,} bytes not ending in "
            f"{root_close.decode()}; the response was truncated"
        )


def _write(dest: Path, xml_name: str, xml: bytes, meta: dict[str, object]) -> Path:
    """Write a corpus and read it back. Raises rather than leaving a partial one."""
    sections = meta["sections"]
    assert isinstance(sections, dict)
    if not sections:
        raise ValueError("no sections found; the response format may have changed")

    dest.mkdir(parents=True, exist_ok=True)
    _ = (dest / xml_name).write_bytes(xml)
    _ = (dest / INDEX_NAME).write_text(json.dumps({**meta, "xml": xml_name}))

    written = Corpus.load(dest)
    if written is None or len(written.sections) != len(sections):
        raise ValueError(f"corpus at {dest} did not survive being written")
    probe = next(iter(sections))
    if not written.section(probe):
        raise ValueError(f"corpus at {dest} cannot read back section {probe}")

    print(
        f"wrote {len(sections):,} {meta['kind']} sections of title {meta['title']} "
        f"({len(xml):,} bytes, {written.currency}) to {dest}"
    )
    return dest / INDEX_NAME


def cfr_title_url(title: str, as_of_date: str) -> str:
    """The eCFR endpoint for a whole title as it stood on a date."""
    return f"https://www.ecfr.gov/api/versioner/v1/full/{as_of_date}/title-{title}.xml"


def build_cfr(dest: Path, as_of_date: str, title: str = "26") -> Path:
    """Download a CFR title as of a date and index its sections."""
    url = cfr_title_url(title, as_of_date)
    xml = _fetch(url)
    _require_closed(url, xml, _CFR_ROOT_CLOSE)
    return _write(
        dest,
        f"title-{title}.xml",
        xml,
        {
            "kind": "cfr",
            "title": title,
            "currency": f"eCFR as of {as_of_date}",
            "valid_from": as_of_date,
            # The endpoint is exact to the day, so this corpus speaks for that day only.
            "valid_until": (date.fromisoformat(as_of_date) + timedelta(days=1)).isoformat(),
            "sections": index_cfr(xml),
        },
    )


_RELEASE_POINT_PAGES = (
    "https://uscode.house.gov/download/download.shtml",
    "https://uscode.house.gov/download/priorreleasepoints.htm",
)
# Rows pair a public law with the date its release point was published.
_RELEASE_POINT = re.compile(r"(\d{3})-(\d{1,3})[^0-9]{0,60}?(\d{2})/(\d{2})/(\d{4})")


def release_points() -> list[tuple[str, str, str]]:
    """Every published release point as (congress, law number, ISO date), newest first."""
    seen: dict[tuple[str, str], str] = {}
    for page in _RELEASE_POINT_PAGES:
        body = _fetch(page).decode("utf-8", "ignore")
        for congress, number, month, day, year in _RELEASE_POINT.findall(body):
            _ = seen.setdefault((congress, number), f"{year}-{month}-{day}")
    return sorted(
        ((congress, number, published) for (congress, number), published in seen.items()),
        key=lambda row: (row[2], int(row[0]), int(row[1])),
        reverse=True,
    )


def resolve_release_point(as_of_date: str, points: list[tuple[str, str, str]]) -> tuple[str, str, str, str]:
    """The release point in force on ``as_of_date``, as (congress, number, from, until).

    ``until`` is the next release point's date, or OPEN_ENDED when none is newer.
    """
    if not points:
        raise ValueError("no release points were found")
    newer = [published for _, _, published in points if published > as_of_date]
    for congress, number, published in points:
        if published <= as_of_date:
            return congress, number, published, min(newer) if newer else OPEN_ENDED
    raise ValueError(f"no U.S. Code release point was published on or before {as_of_date}")


def usc_release_url(title: str, congress: str, number: str) -> str:
    """The House's XML archive for one title at one release point."""
    return (
        f"https://uscode.house.gov/download/releasepoints/us/pl/{congress}/{number}"
        f"/xml_usc{title.zfill(2)}@{congress}-{number}.zip"
    )


def build_usc(dest: Path, as_of_date: str, title: str = "26") -> Path:
    """Download the U.S. Code title in force on a date and index its sections."""
    congress, number, valid_from, valid_until = resolve_release_point(as_of_date, release_points())
    url = usc_release_url(title, congress, number)
    member = f"usc{title.zfill(2)}.xml"
    xml = zipfile.ZipFile(io.BytesIO(_fetch(url))).read(member)
    _require_closed(url, xml, _USC_ROOT_CLOSE)
    return _write(
        dest,
        member,
        xml,
        {
            "kind": "usc",
            "title": title,
            "currency": f"U.S. Code release point P.L. {congress}-{number} ({valid_from})",
            "valid_from": valid_from,
            "valid_until": valid_until,
            "sections": index_usc(xml, title),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build", help="download a title and index its sections")
    _ = builder.add_argument("--kind", choices=KINDS, required=True)
    _ = builder.add_argument("--dest", type=Path, required=True)
    _ = builder.add_argument(
        "--as-of-date",
        default=None,
        help="defaults to the agent's DEFAULT_AS_OF_DATE, which owns that value",
    )
    _ = builder.add_argument("--title", default="26")
    args = parser.parse_args()

    # Imported here, not at module scope, so `python -m tax_agent.corpus` does not find
    # this module already loaded via the package's import chain.
    from .tools import DEFAULT_AS_OF_DATE

    as_of_date = args.as_of_date or DEFAULT_AS_OF_DATE
    build = build_cfr if args.kind == "cfr" else build_usc
    _ = build(args.dest, as_of_date, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
