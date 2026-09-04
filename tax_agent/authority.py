"""Resolve a tax citation to the document that carries it, and to the part it names.

A citation is a path into a document: "P.L. 119-21 § 70303(a)" names an act, a section
within it, and a subsection within that. These functions perform no I/O:

    parse("IRC § 163(j)(10)")            -> Citation(kind="usc", section="163", path=("j", "10"))
    canonical_url(citation, as_of_date)  -> the document's stable URL
    narrow(document_text, citation)      -> just the cited provision, if it can be located

Fetching lives in ``tools.LookupAuthority``.

Each kind resolves to its publisher: the House's U.S. Code, the eCFR service for
regulations, and govinfo for public laws. The eCFR endpoint takes a date and returns the
regulation as it stood then, so ``canonical_url`` takes one.
"""

import re
from dataclasses import dataclass
from typing import Literal

CitationKind = Literal["usc", "cfr", "public_law"]

# Title 26 unless a citation names another.
DEFAULT_TITLE = "26"


@dataclass(frozen=True)
class Citation:
    """A parsed citation: which document, and the path to the provision inside it."""

    kind: CitationKind
    section: str
    path: tuple[str, ...] = ()
    title: str = DEFAULT_TITLE
    act: str | None = None  # public laws only, e.g. "119-21"
    raw: str = ""

    def __str__(self) -> str:
        path = "".join(f"({p})" for p in self.path)
        if self.kind == "public_law":
            return f"P.L. {self.act} § {self.section}{path}"
        if self.kind == "cfr":
            prefix = f"{self.title} CFR"
        else:
            prefix = "IRC" if self.title == DEFAULT_TITLE else f"{self.title} U.S.C."
        return f"{prefix} § {self.section}{path}"


# A trailing run of parenthesised parts is the path into the section: (j)(10), (c)(3).
_PATH = r"(?P<path>(?:\([0-9A-Za-z]{1,4}\))*)"
# Section numbers: "163" for the Code, "1.163(j)-7" for regulations. A letter can sit
# inside the number itself (1.951A-2, 1400Z-2). The embedded parentheses in a regulation
# number are part of the number, not the path, so the regulation pattern is greedy up to
# the final hyphenated element. A hyphen in a Code section is taken only after a letter,
# leaving a span such as "1561-1563" to resolve as its first section.
_PATTERNS: tuple[tuple[CitationKind, re.Pattern[str]], ...] = (
    (
        "public_law",
        re.compile(
            r"(?:P\.?\s?L\.?|Pub(?:lic)?\.?\s?L(?:aw)?\.?)\s*(?P<act>\d{2,3}-\d{1,3})"
            rf"[\s,]*(?:§+|[Ss]ec(?:tion)?\.?)?\s*(?P<section>\d{{3,6}}){_PATH}",
        ),
    ),
    (
        "cfr",
        re.compile(
            r"(?:Treas(?:ury)?\.?\s*Reg(?:ulation)?s?\.?|(?P<title>\d{1,2})\s*C\.?F\.?R\.?)"
            r"\s*§*\s*(?P<section>\d+\.\d+[A-Za-z]?(?:\([0-9A-Za-z]{1,3}\)(?=-))?(?:-\d+[A-Za-z]?)?)"
            rf"{_PATH}",
        ),
    ),
    (
        "usc",
        re.compile(
            r"(?:I\.?R\.?C\.?|Internal\s+Revenue\s+Code|(?P<title>\d{1,2})\s*U\.?S\.?C\.?)"
            rf"\s*§*\s*(?P<section>\d+[A-Za-z]+-\d+|\d+[A-Za-z]?){_PATH}",
        ),
    ),
)


def parse(text: str) -> Citation | None:
    """Parse the first citation in ``text``, or None if it names no supported authority.

    Patterns are tried in order: a public law reference also contains a section sign, and
    a regulation number contains parentheses that would otherwise read as a path.
    """
    for kind, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groupdict()
        return Citation(
            kind=kind,
            section=groups["section"],
            path=tuple(re.findall(r"\(([0-9A-Za-z]{1,4})\)", groups.get("path") or "")),
            title=groups.get("title") or DEFAULT_TITLE,
            act=groups.get("act"),
            raw=match.group(0).strip(),
        )
    return None


def _cfr_part(section: str) -> str:
    """The CFR part a section belongs to: 1.163(j)-7 is in part 1, 301.6751(b)-1 in 301."""
    return section.split(".", 1)[0]


def canonical_url(citation: Citation, as_of_date: str) -> str:
    """The URL of the document carrying the citation, as of ``as_of_date``.

    The date applies to regulations only; the U.S. Code endpoint serves the current
    edition and a public law is fixed as enacted.
    """
    if citation.kind == "public_law":
        act = (citation.act or "").replace("-", "publ")
        return f"https://www.govinfo.gov/content/pkg/PLAW-{act}/pdf/PLAW-{act}.pdf"
    if citation.kind == "cfr":
        return (
            f"https://www.ecfr.gov/api/versioner/v1/full/{as_of_date}"
            f"/title-{citation.title}.xml"
            f"?part={_cfr_part(citation.section)}&section={citation.section}"
        )
    return (
        "https://uscode.house.gov/view.xhtml?req=granuleid:"
        f"USC-prelim-title{citation.title}-section{citation.section}"
        "&num=0&edition=prelim"
    )


def _slice_to_next(text: str, start: int, pattern: re.Pattern[str]) -> str:
    following = pattern.search(text, start + 1)
    return text[start : following.start() if following else len(text)]


_ROMAN_SEQUENCE = (
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
)  # fmt: skip
_ROMAN_RANK = {numeral: rank for rank, numeral in enumerate(_ROMAN_SEQUENCE, start=1)}


def _successors(part: str, window: int = 4) -> set[str]:
    """Markers that could follow ``part`` at its own level.

    A short run rather than the next marker alone, so a repealed or reserved number does
    not carry the provision on to the end of its parent. "(i)" is both a letter and a
    roman numeral, so both readings contribute; taking a bounded run rather than every
    later marker is what keeps a nested clause "(i)" from closing subsection "(b)".
    """
    if part.isdigit():
        return {str(int(part) + step) for step in range(1, window + 1)}
    cased = str.upper if part.isupper() else str.lower
    lowered = part.lower()
    out: set[str] = set()
    if len(part) == 1 and part.isalpha():
        letters = (chr(ord(lowered) + step) for step in range(1, window + 1))
        out |= {cased(letter) for letter in letters if letter.isalpha()}
    if lowered in _ROMAN_RANK:
        index = _ROMAN_RANK[lowered]
        out |= {cased(numeral) for numeral in _ROMAN_SEQUENCE[index : index + window]}
    return out


_MARKER = re.compile(r"^[ \t]*\(\s*([0-9A-Za-z]{1,4})\s*\)", re.MULTILINE)


def _find_marker(text: str, part: str, inside_parent: bool) -> re.Match[str] | None:
    """Locate the paragraph that ``part`` opens.

    A paragraph normally starts its own line, so that is where it is looked for: a
    mid-sentence marker is a cross-reference ("Paragraph (c) of this section provides"),
    not a paragraph. The exception is the first child of a paragraph, which is set run-in
    on its parent's line: "(b) Definitions-(1) Adjusted taxable income". That position is
    considered only once ``text`` has been narrowed to the parent, and it wins when it
    comes first, so a deeper paragraph of the same number cannot stand in for it.
    """
    marker = rf"\(\s*{re.escape(part)}\s*\)"
    anchored = re.compile(rf"^[ \t]*{marker}", re.MULTILINE).search(text)
    if not inside_parent:
        return anchored
    run_in = re.compile(marker).search(text.split("\n", 1)[0], 1)
    if anchored and run_in:
        return run_in if run_in.start() < anchored.start() else anchored
    return anchored or run_in


def _provision_end(text: str, start: int, part: str) -> int:
    """Where the provision opened by ``part`` ends: at the next marker of its own level."""
    successors = _successors(part)
    if not successors:
        return len(text)
    for match in _MARKER.finditer(text, start + 1):
        if match.group(1) in successors:
            return match.start()
    return len(text)


_ACT_SECTION = re.compile(r"^\s*SEC\.\s*\d{3,6}\.", re.MULTILINE)
# Running head on every page of a slip law: "139 STAT. 207" is volume 139, page 207.
_STAT_PAGE = re.compile(r"(\d{1,3})\s+STAT\.?\s*(\d+)", re.IGNORECASE)


def statutes_at_large_cite(document: str, citation: Citation) -> str | None:
    """The Statutes at Large reference a public-law section begins at, e.g. "139 Stat. 207".

    Taken from the last page running head before the section heading. None for citations
    that are not public laws, and for documents carrying no running heads.
    """
    if citation.kind != "public_law":
        return None
    start = _find_act_section(document, citation.section)
    if start is None:
        return None
    heads = _STAT_PAGE.findall(document[:start])
    if not heads:
        return None
    volume, page = heads[-1]
    return f"{volume} Stat. {page}"


def _find_act_section(document: str, section: str) -> int | None:
    match = re.search(rf"^\s*SEC\.\s*{re.escape(section)}\.", document, re.MULTILINE)
    return match.start() if match else None


def _narrow_to_path(text: str, path: tuple[str, ...]) -> tuple[str, int]:
    """Walk successive ``(x)`` markers, returning the deepest span and how deep it got."""
    depth = 0
    for part in path:
        match = _find_marker(text, part, inside_parent=depth > 0)
        if match is None:
            break
        start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
        text = text[start : _provision_end(text, start, part)]
        depth += 1
    return text, depth


def narrow(document: str, citation: Citation) -> tuple[str, int] | None:
    """Reduce ``document`` to the cited provision.

    Returns the text and how many of the citation's path parts were located, or None
    when the document does not carry the cited section at all. A depth short of the full
    path leaves the nearest enclosing provision, so callers must report the depth rather
    than present that text as the citation.
    """
    if citation.kind == "public_law":
        start = _find_act_section(document, citation.section)
        if start is None:
            return None
        section_text = _slice_to_next(document, start, _ACT_SECTION)
        return _narrow_to_path(section_text, citation.path)

    # One CFR section per URL, so a regulation arrives already scoped to its section.
    return _narrow_to_path(document, citation.path)


def uslm_identifiers(citation: Citation) -> list[tuple[str, int]]:
    """USLM ``identifier`` values for the citation and each of its ancestors, deepest first.

    A U.S. Code release-point document tags every provision with its own path, as
    ``identifier="/us/usc/t26/s382/h/3/B"``. Each entry pairs an identifier with how many
    path parts it covers, so a caller that finds only an ancestor reports the depth it
    reached, as ``narrow`` does.

    Empty for kinds whose documents carry no identifiers: CFR and public laws.
    """
    if citation.kind != "usc":
        return []
    base = f"/us/usc/t{citation.title}/s{citation.section}"
    return [
        (base + "".join(f"/{part}" for part in citation.path[:depth]), depth)
        for depth in range(len(citation.path), -1, -1)
    ]


# A U.S. Code page opens with its own label, "26 USC 163: Interest". The House site
# answers an unknown section with a 200 and a "Document not Found" page, so the label is
# what separates a provision from that.
def carries_citation(document: str, citation: Citation) -> bool:
    """Whether ``document`` is the document the citation names.

    Only the U.S. Code is checked; the other publishers answer an unknown citation with
    an HTTP error, which the fetch raises.
    """
    if citation.kind != "usc":
        return True
    return f"{citation.title} USC {citation.section}" in document
