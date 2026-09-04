import asyncio
import io
import json
import logging
import math
import os
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from pypdf import PdfReader
from tavily import AsyncTavilyClient

from simpleeval import SimpleEval

from . import authority
from .exceptions import (
    RetryExhaustedError,
    get_retry_policy,
    retry_http_errors,
    retry_with_policy,
)

if TYPE_CHECKING:
    from .corpus import Corpus


# The date the agent answers as of: the cutoff for published guidance it may rely on,
# and the ceiling web_search end dates are clamped to. Only a question's own
# context.as_of_date moves it, reaching the tools through Parameters.as_of_date; a tax
# year or effective date named in the question does not.
DEFAULT_AS_OF_DATE = "2026-04-30"
VALID_TOOLS = [
    "web_search",
    "retrieve_information",
    "fetch_document",
    "lookup_authority",
    "calculator",
]

# The regulation endpoint serves XML, which html.parser renders to the same text an XML
# parser would. Suppresses the parser-mismatch warning on every regulation lookup.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date_format(field_name: str, value: str) -> None:
    if not _DATE_REGEX.match(value):
        raise ValueError(f"Invalid {field_name} format: '{value}'. Expected YYYY-MM-DD.")


# Page ceiling for PDF extraction, set high because primary sources run long
# (P.L. 119-21 is ~330 pages). Truncation is reported in the stored text.
MAX_PDF_PAGES = 2_000

# Ceiling on each document substituted into a retrieve_information prompt. Documents
# are stored whole; this bounds what one call can put in front of the LLM. Large enough
# for the longest Code sections (IRC §168 is ~400k characters).
MAX_RETRIEVED_CHARS = 500_000

# Ceiling on text lookup_authority returns, which goes into the conversation rather
# than into data storage. Beyond this the tool truncates and names fetch_document.
MAX_PROVISION_CHARS = 50_000


def _bounded(text: str, start: int = 0) -> str:
    """Cap text at MAX_RETRIEVED_CHARS, saying so rather than truncating silently.

    ``start`` is where this span begins in the stored document, so the notice names
    absolute positions the caller can ask for again.
    """
    if len(text) <= MAX_RETRIEVED_CHARS:
        return text
    return text[:MAX_RETRIEVED_CHARS] + (
        f"\n[TRUNCATED: characters {start:,} to {start + MAX_RETRIEVED_CHARS:,} of this "
        f"document are shown; it continues to character {start + len(text):,}.]"
    )


def _looks_like_pdf(url: str, content_type: str, content: bytes) -> bool:
    """A response is a PDF if its content-type says so, its URL path ends in .pdf, or
    its bytes start with the %PDF- magic number (covers mislabeled octet-stream/text)."""
    return (
        "application/pdf" in content_type
        or url.lower().split("?", 1)[0].endswith(".pdf")
        or content[:5] == b"%PDF-"
    )


def _pdf_to_text(data: bytes) -> str:
    """Extract text from a text-based PDF. Best-effort per page; scanned image-only
    PDFs yield little or nothing (there is no OCR). Synchronous and CPU-bound — call
    it via a thread so it does not block the event loop."""
    reader = PdfReader(io.BytesIO(data))
    total_pages = len(reader.pages)
    parts: list[str] = []
    for page in reader.pages[:MAX_PDF_PAGES]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # a single malformed page shouldn't sink the whole document
            continue
    if total_pages > MAX_PDF_PAGES:
        parts.append(
            f"\n[TRUNCATED: this PDF has {total_pages} pages; only the first {MAX_PDF_PAGES} were extracted.]"
        )
    text = "\n".join(parts)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


class Calculator(Tool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression and return the result. "
        "Use this tool for all arithmetic calculations instead of computing by hand. "
        "Supports: +, -, *, /, ** (exponentiation), % (modulo), "
        "and parentheses for grouping. "
        "Available functions: abs(), min(), max(), sqrt(), log(), log10(). "
        "Examples: '(5000000 - 3200000) * 0.21', '(2865507 / 1905871) ** 0.5 - 1', '14060 / 2148'."
    )
    parameters: dict[str, Any] = {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate",
        }
    }
    required: list[str] = ["expression"]

    def __init__(self) -> None:
        self._evaluator = SimpleEval(
            functions={
                "abs": abs,
                "min": min,
                "max": max,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
            }
        )

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        expression = args.get("expression", "")
        if not expression:
            return ToolOutput(output="Error: expression must not be empty", error="empty expression")
        try:
            result = self._evaluator.eval(expression)
            return ToolOutput(output=str(result))
        except ZeroDivisionError:
            error_msg = f"Error: division by zero in '{expression}'"
            logger.warning(error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
        except OverflowError:
            error_msg = f"Error: numerical overflow in '{expression}'"
            logger.warning(error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
        except Exception as e:
            logger.warning(f"Calculator error for '{expression}': {e}")
            error_msg = f"Error: invalid expression '{expression}'"
            return ToolOutput(output=error_msg, error=error_msg)


class SubmitFinalResult(Tool):
    name = "submit_final_result"
    description = (
        "Submits the final answer to the user. You should include your final answer, as well as any necessary "
        "reasoning, justification, calculations, and explanation. Finally, you should provide any sources used to answer the question. "
        "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {
        "final_result": {
            "type": "string",
            "description": "The final result to submit to the user",
        }
    }
    required: list[str] = ["final_result"]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            final_result = args["final_result"]
            if not final_result:
                raise ValueError("Final result must not be empty")
            return ToolOutput(output=final_result, done=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Submission failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg, done=False)


class TavilyWebSearch(Tool):
    name = "web_search"
    description = "Search the public internet for information. Each result will contain a url, a title, and one excerpt taken directly from the page."
    parameters: dict[str, Any] = {
        "search_query": {
            "type": "string",
            "description": "The query to search for",
        },
        "start_date": {
            "type": "string",
            "description": "(optional) The start date for the search range in the format YYYY-MM-DD. Must not be equal to end_date.",
        },
        "end_date": {
            "type": "string",
            "description": "(optional) The end date for the search range in the format YYYY-MM-DD. If it is later than the question's as-of date, it is set to that date.",
        },
        "number_of_results": {
            "type": "integer",
            "description": "(optional) The number of search results to return.",
            "maximum": 20,
            "minimum": 1,
            "default": 10,
        },
    }
    required = ["search_query"]

    def __init__(self, tavily_api_key: str | None = None, as_of_date: str | None = None):
        if not tavily_api_key:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        self.client = AsyncTavilyClient(api_key=tavily_api_key)

        self._as_of_date = as_of_date or DEFAULT_AS_OF_DATE
        validate_date_format("as_of_date", self._as_of_date)

    @retry_http_errors(429, 503, max_tries=8)
    async def _execute_search(
        self,
        search_query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        number_of_results: int = 10,
    ) -> list[dict[str, Any]]:
        kwargs = {}
        if not end_date:
            end_date = self._as_of_date

        if end_date:
            validate_date_format("end_date", end_date)
            end_date = min(end_date, self._as_of_date)

        if start_date:
            validate_date_format("start_date", start_date)
            requested_start = start_date
            start_date = min(start_date, self._as_of_date)
            # Equal dates are rejected by the search API, and capping to the as-of date
            # can collapse a range the model gave as a valid one.
            if start_date >= end_date:
                raise ValueError(
                    f"Parameter start_date '{requested_start}' leaves no range to search "
                    f"before end_date '{end_date}'. Dates later than {self._as_of_date} "
                    f"are capped to it; retry with an earlier start_date."
                )

            kwargs["start_date"] = start_date

        response = await self.client.search(
            search_depth="fast",
            end_date=end_date,
            max_results=number_of_results,
            chunks_per_source=1,
            query=search_query,
            **kwargs,
        )

        return response.get("results", [])

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            kwargs = {k: v for k, v in args.items() if k in self.parameters}
            results = await self._execute_search(**kwargs)
            return ToolOutput(output=json.dumps(results, default=str))
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Web search failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


FETCH_TIMEOUT_SECONDS = 60


async def _fetch_bytes(url: str) -> tuple[bytes, str, str | None]:
    """GET a URL, returning (body, content-type, declared charset).

    Retries per the host's policy.
    """

    @retry_with_policy(get_retry_policy(url))
    async def _get(fetch_url: str) -> tuple[bytes, str, str | None]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    fetch_url,
                    timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECONDS),
                    headers={"User-Agent": "ValsAI/tax-agent"},
                ) as response:
                    response.raise_for_status()
                    return (
                        await response.read(),
                        (response.headers.get("content-type") or "").lower(),
                        response.charset,
                    )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Timeout error when fetching the source after {FETCH_TIMEOUT_SECONDS} "
                    "seconds. The URL might be blocked or the server is taking too long to "
                    "respond."
                )
            except Exception:
                raise

    try:
        return await _get(url)
    except aiohttp.ClientResponseError as e:
        if e.status in (429, 503) and "irs.gov" in url:
            raise RetryExhaustedError(f"irs.gov retry attempts exhausted for HTTP {e.status}") from e
        raise


def _markup_to_text(content: bytes, declared_charset: str | None) -> str:
    """Strip HTML or XML to readable text.

    Takes raw bytes so BeautifulSoup can fall back from the server-declared charset
    through <meta charset> and chardet, rather than raising on non-UTF-8 pages.
    """
    soup = BeautifulSoup(content, "html.parser", from_encoding=declared_charset)
    for script_or_style in soup(["script", "style"]):
        _ = script_or_style.extract()

    lines = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


async def _fetch_source(url: str) -> str:
    """Fetch a URL and return its text, whether it is served as a PDF or as markup."""
    content, content_type, declared_charset = await _fetch_bytes(url)
    if _looks_like_pdf(url, content_type, content):
        # pypdf is synchronous and CPU-bound, so keep it off the event loop.
        try:
            text = await asyncio.to_thread(_pdf_to_text, content)
        except Exception:
            # pypdf's own errors ("stream has ended", "xref table read error") read like
            # a truncated download and invite a pointless retry of a URL that will keep
            # serving the same thing. Reading the body as markup instead would store a
            # "page not found" page under the agent's key and let it be cited as
            # authority, so name what arrived and send the agent looking elsewhere.
            raise ValueError(
                "This URL was requested as a PDF, but the response body is not a "
                f"readable PDF (declared type: {content_type or 'none'}). The document "
                "has most likely moved or been removed, or a redirect or interstitial "
                "page was served in its place. Locate the document's current URL "
                "rather than retrying this one."
            )
        if not text:
            raise ValueError(
                "No extractable text found in the PDF. It may be a scanned image-only "
                "document, which cannot be read (there is no OCR)."
            )
        return text
    return _markup_to_text(content, declared_charset)


# Single-flight cache: each entry holds the fetch rather than its result, so concurrent
# callers asking for one URL await the same request. Least-recently-used entries are
# dropped past MAX_CACHED_DOCUMENTS.
MAX_CACHED_DOCUMENTS = 64
_document_cache: OrderedDict[str, asyncio.Task[str]] = OrderedDict()


def clear_document_cache() -> None:
    """Drop every cached document."""
    _document_cache.clear()


async def _read_source(url: str) -> str:
    """Fetch a URL's text, reusing a completed or in-flight fetch of the same URL."""
    task = _document_cache.get(url)
    if task is None:
        task = asyncio.ensure_future(_fetch_source(url))
        _document_cache[url] = task
        while len(_document_cache) > MAX_CACHED_DOCUMENTS:
            _ = _document_cache.popitem(last=False)
    else:
        _document_cache.move_to_end(url)

    try:
        # Shielded so a caller that gives up does not cancel a fetch others are awaiting.
        return await asyncio.shield(task)
    finally:
        if task.done() and not task.cancelled() and task.exception() is not None:
            # Failures are not cached, so the next caller retries instead of inheriting it.
            _ = _document_cache.pop(url, None)


class FetchDocument(Tool):
    name = "fetch_document"
    description = (
        "Fetch a document from a URL, convert it to plain text, and save it to the agent's data storage "
        "system under the key you provide. "
        "The text is NOT returned to you: the tool returns a confirmation and the current list of storage "
        "keys, and you then use the retrieve_information tool to query the stored text. This is what lets "
        "you work with documents far larger than your context window. "
        "Both HTML pages and PDF documents are supported; PDFs are detected automatically, so a URL ending "
        "in .pdf can be passed directly. Scanned image-only PDFs cannot be read (there is no OCR). "
        "This is useful for reading long primary sources such as IRS publications and forms, the Internal "
        "Revenue Code, Treasury regulations, revenue rulings, public laws, court opinions, and state tax "
        "authority pages, many of which are published only as PDFs."
    )
    parameters: dict[str, Any] = {
        "url": {"type": "string", "description": "The URL of the web page or PDF to parse"},
        "key": {
            "type": "string",
            "description": "The key to use when saving the result in the conversation's data storage.",
        },
    }
    required = ["url", "key"]

    async def _fetch_document(self, url: str) -> str:
        return await _read_source(url)

    async def _save_tool_output(self, output: str, key: str, state: dict[str, Any]) -> str:
        if not output:
            raise ValueError("The document produced no text")

        tool_result = ""
        if key in state:
            tool_result = "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        tool_result += f"SUCCESS: The result has been saved to the data storage under the key: {key}." + "\n"

        state[key] = output

        keys_list = "\n".join(state.keys())
        tool_result += (
            f"""
        The data_storage currently contains the following keys:
        {keys_list}
        """.strip()
            + "\n"
        )

        return tool_result

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            url = args["url"]
            key = args["key"]
            text_output = await self._fetch_document(url)
            tool_result = await self._save_tool_output(text_output, key, state)
            return ToolOutput(output=tool_result)
        except RetryExhaustedError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Parse source failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


# Each names a directory built by ``python -m tax_agent.corpus build``. When one is
# absent, that authority is fetched over the network instead.
CFR_CORPUS_ENV = "TAX_AGENT_CFR_CORPUS"
USC_CORPUS_ENV = "TAX_AGENT_USC_CORPUS"
CORPUS_ENV = {"cfr": CFR_CORPUS_ENV, "usc": USC_CORPUS_ENV}

# Keyed by kind as well as path: whether a corpus answers depends on the kind asked
# for, so two kinds pointed at one directory must not share an entry.
_corpus_cache: "dict[tuple[str, str], Corpus | None]" = {}


def _corpus(kind: str) -> "Corpus | None":
    """The configured corpus for a citation kind, loaded once per process.

    Imported here rather than at module scope so that ``python -m tax_agent.corpus``
    does not find the module already loaded via this package's import chain.
    """
    from .corpus import Corpus

    variable = CORPUS_ENV.get(kind)
    path = os.environ.get(variable, "").strip() if variable else ""
    if not path:
        return None
    key = (kind, path)
    if key not in _corpus_cache:
        loaded = Corpus.load(Path(path))
        _corpus_cache[key] = loaded if loaded is not None and loaded.kind == kind else None
    return _corpus_cache[key]


def clear_corpus_cache() -> None:
    """Drop the memoised corpus so a changed environment takes effect."""
    _corpus_cache.clear()


# USLM label elements, after which a separator is needed to render readable text.
_USLM_LABEL_END = re.compile(rb"</(?:num|heading)>")

# What a fetched document is current to, by citation kind, when it is not served from a
# dated corpus. Kinds absent here are already fixed to the as-of date.
_LIVE_CURRENCY = {
    "usc": (
        "the U.S. Code as published today, NOT limited to {as_of_date}. It may include "
        "amendments enacted after that date -- check the amendment credits before relying "
        "on this text."
    ),
}


class LookupAuthority(Tool):
    name = "lookup_authority"
    description = (
        "Look up the text of a cited authority by its citation, instead of searching for it. "
        "Give a citation such as 'IRC § 163(j)(10)', 'Treas. Reg. § 1.163(j)-7(c)(3)', "
        "'26 CFR § 1.168(k)-2', or 'P.L. 119-21 § 70303(a)', and the tool returns that "
        "provision's text together with the canonical URL to cite. "
        "Prefer this over web_search whenever you already know which provision you need: it "
        "returns the provision rather than a page about it, and for a public law it also "
        "reports the Statutes at Large page the section begins on. "
        "Supported: the Internal Revenue Code and other U.S. Code titles, Treasury regulations "
        "and other CFR titles, and public laws. For anything else -- revenue rulings and "
        "procedures, notices, IRS forms and publications, court opinions, state authority -- "
        "use web_search and fetch_document."
    )
    parameters: dict[str, Any] = {
        "citation": {
            "type": "string",
            "description": (
                "The citation to resolve, e.g. 'IRC § 163(j)(10)', "
                "'Treas. Reg. § 1.163(j)-7(c)', 'P.L. 119-21 § 70303(a)'. "
                "Cite to the subsection you need; the tool narrows to it when it can."
            ),
        }
    }
    required = ["citation"]

    def __init__(self, as_of_date: str | None = None) -> None:
        self._as_of_date = as_of_date or DEFAULT_AS_OF_DATE
        validate_date_format("as_of_date", self._as_of_date)

    def _corpus_section(self, citation: authority.Citation) -> bytes | None:
        """The cited section's markup from a local corpus, or None to fetch.

        Returns None unless a corpus covers this exact citation: the citation's kind, in
        the corpus's title, as the law stood on the tool's date, in a section it carries.
        """
        corpus = _corpus(citation.kind)
        if corpus is None or not corpus.covers(citation.title, self._as_of_date):
            return None
        return corpus.section(citation.section)

    def _from_corpus(self, citation: authority.Citation) -> tuple[str, str] | None:
        """The cited section as text, and what the corpus is current to."""
        corpus = _corpus(citation.kind)
        section = self._corpus_section(citation)
        if corpus is None or section is None:
            return None
        # USLM leaves no whitespace after a <num> or <heading>, so the label would fuse
        # onto what follows: "(a)General rule", "In generalThe amount allowed".
        section = _USLM_LABEL_END.sub(rb"\g<0> ", section)
        # Otherwise the same conversion the network response goes through.
        return _markup_to_text(section, "utf-8"), corpus.currency

    def _locate_in_corpus(self, citation: authority.Citation) -> tuple[str, int] | None:
        """The cited provision from the corpus, selected by its USLM identifier.

        Returns the provision and how many of the citation's path parts were located, the
        same contract as ``authority.narrow``. None when the corpus does not carry the
        section, or when this kind of document carries no identifiers, leaving the caller
        to match the text.
        """
        identifiers = authority.uslm_identifiers(citation)
        section = self._corpus_section(citation)
        if not identifiers or section is None:
            return None
        soup = BeautifulSoup(_USLM_LABEL_END.sub(rb"\g<0> ", section), "html.parser", from_encoding="utf-8")
        for identifier, depth in identifiers:
            element = soup.find(attrs={"identifier": identifier})
            if element is not None:
                return _markup_to_text(str(element).encode("utf-8"), "utf-8"), depth
        return None

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            raw = str(args.get("citation") or "").strip()
            citation = authority.parse(raw)
            if citation is None:
                message = (
                    f"Could not read '{raw}' as a U.S. Code, CFR, or public law citation. "
                    "This tool resolves those three only -- use web_search and fetch_document "
                    "for rulings, notices, forms, publications, cases, and state authority."
                )
                return ToolOutput(output=message, error="unsupported citation")

            url = authority.canonical_url(citation, self._as_of_date)
            local = self._from_corpus(citation)
            if local is None:
                document, currency = await _read_source(url), _LIVE_CURRENCY.get(citation.kind)
            else:
                document, currency = local
            # A corpus section was found by name in the index; only a fetched document
            # can be the publisher's "not found" page.
            fetched_wrong_document = local is None and not authority.carries_citation(document, citation)
            # A corpus U.S. Code document tags each provision with its path, so the cited
            # element is selected directly. Text matching covers every other document.
            located = None if local is None else self._locate_in_corpus(citation)
            if located is None:
                located = authority.narrow(document, citation)
            if located is None or fetched_wrong_document:
                message = (
                    f"No provision found at {citation}. Check the section number against the "
                    "authority you mean, or use web_search to find it."
                )
                return ToolOutput(output=message, error="citation not found")
            provision, depth = located
            resolved = depth == len(citation.path)
            statutes_at_large = authority.statutes_at_large_cite(document, citation)

            # The URL stands as the citation's source only when the whole path resolved;
            # otherwise it is the document the text was taken from.
            header = [f"{citation}", f"{'Source' if resolved else 'Document'}: {url}"]
            if currency:
                header.append(f"Currency: {currency.format(as_of_date=self._as_of_date)}")
            if statutes_at_large:
                header.append(f"Statutes at Large: {statutes_at_large}")
            if not resolved:
                reached = citation.section + "".join(f"({p})" for p in citation.path[:depth])
                missing = "".join(f"({p})" for p in citation.path[depth:])
                header.append(
                    f"NOTE: could not locate {missing} within {reached}. The text below is "
                    f"{reached}; do not cite it as {citation}. The section may number the "
                    "provision differently -- check the text, or use web_search."
                )
            if len(provision) > MAX_PROVISION_CHARS:
                # A citation with no path asks for a whole section, so the cheap fix is a
                # narrower citation rather than storing the document and querying it.
                remedy = (
                    "Look it up again citing the subsection you need."
                    if not citation.path
                    else "Cite a deeper subsection, or use fetch_document on the URL above to "
                    "store the whole document and query it with retrieve_information."
                )
                header.append(
                    f"NOTE: the provision is {len(provision):,} characters; the first "
                    f"{MAX_PROVISION_CHARS:,} follow. {remedy}"
                )
                provision = provision[:MAX_PROVISION_CHARS]

            return ToolOutput(output="\n".join(header) + "\n\n" + provision.strip())
        except RetryExhaustedError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Authority lookup failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)


class RetrieveInformation(Tool):
    name = "retrieve_information"
    description = (
        "This tool allows you to retrieve data from previously saved documents from the agent's data storage system, by applying an LLM prompt to the stored document.\n"
        "\n"
        "To use the tool, you will need to provide a prompt. This prompt will include both the query to be sent to the LLM, "
        "as well as the keys of files you have previously saved to the data storage system.\n"
        "\n"
        'For example, if you want to analyze data stored under the key "irs_publication", your prompt should look like the following:\n'
        '"Analyze the following IRS publication and extract the applicable depreciation rules: {{irs_publication}}"\n'
        "\n"
        "The {{key_name}} will be replaced with the full text of the document stored under that key before the query is sent.\n"
        "\n"
        "IMPORTANT: Your prompt MUST include at least one key from the data storage using this exact format: {{key_name}}. "
        "If you don't use this exact format with double braces, the tool will fail to retrieve the information.\n"
        "\n"
        "You can also optionally only pass *a portion* of each document to the LLM, rather than the entire document. This can be used to avoid token limit errors or improve efficiency. "
        "To do so, use the input_character_ranges parameter to specify which portions of documents to extract. "
        'For example, if "irs_publication" contains "Annual Report 2023" and you specify:  [{"key": "irs_publication", "start": 1, "end": 6}], '
        'then only "nnual" will be inserted into the prompt (characters 1 through 5, as end is exclusive).'
    )
    parameters: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} - for example: 'Summarize this revenue ruling: {{rev_ruling}}'. The content stored under each key will replace the {{key_name}} placeholder.",
        },
        "input_character_ranges": {
            "type": "array",
            "description": "An optional list of character range specifications for extracting only portions of documents. Each object should have 'key' (the document key), 'start' (start character index, inclusive), and 'end' (end character index, exclusive). By default, the full document is used if this parameter is not provided or if a key is not included in the list.",
            "items": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The document key from data storage",
                    },
                    "start": {
                        "type": "integer",
                        "description": "The starting character index (inclusive)",
                    },
                    "end": {
                        "type": "integer",
                        "description": "The ending character index (exclusive)",
                    },
                },
                "required": ["key", "start", "end"],
            },
        },
    }
    required = ["prompt"]

    def __init__(self, llm: LLM):
        self._llm = llm

    def _validate_inputs(
        self, prompt: str, input_character_ranges: list, state: dict[str, Any]
    ) -> dict[str, tuple[int, int]]:
        """Validate prompt placeholders, character ranges, and data storage keys. Returns the parsed ranges dict."""
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError(
                "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. Please try again with the correct format. You can add documents to the data storage with fetch_document."
            )

        ranges_dict = {}
        for range_spec in input_character_ranges:
            if not isinstance(range_spec, dict):
                raise ValueError(
                    "ERROR: Each item in input_character_ranges must be an object with 'key', 'start', and 'end' fields."
                )
            if "key" not in range_spec or "start" not in range_spec or "end" not in range_spec:
                raise ValueError(
                    "ERROR: Each range specification must have 'key', 'start', and 'end' fields."
                )
            key, start, end = range_spec["key"], range_spec["start"], range_spec["end"]
            if key in ranges_dict:
                raise ValueError(
                    f"ERROR: '{key}' appears twice in input_character_ranges. One range per "
                    "key; call the tool again for another part of the same document."
                )
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"ERROR: The range for '{key}' must give 'start' and 'end' as integers.")
            # An empty span would substitute an empty document and produce an answer
            # grounded in nothing.
            if start < 0 or start >= end:
                raise ValueError(
                    f"ERROR: The range for '{key}' selects no text: start={start:,}, "
                    f"end={end:,}. 'start' counts from the beginning of the document and "
                    f"must be less than 'end', which is exclusive."
                )
            ranges_dict[key] = (start, end)

        keys = re.findall(r"{{([^{}]+)}}", prompt)
        keys_set = set(keys)

        for range_key in ranges_dict.keys():
            if range_key not in keys_set:
                raise ValueError(
                    f"ERROR: The key '{range_key}' is specified in input_character_ranges but is not referenced in the prompt. "
                    f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                )

        for key in keys:
            if key not in state:
                raise KeyError(
                    f"ERROR: The key '{key}' was not found in the data storage. Available keys are: {', '.join(state.keys())}. Use the fetch_document tool to add keys to the data storage."
                )

        for key, (start, _) in ranges_dict.items():
            if start >= len(state[key]):
                raise ValueError(
                    f"ERROR: The range for '{key}' starts at character {start:,}, but the "
                    f"document is {len(state[key]):,} characters long."
                )

        return ranges_dict

    def _format_prompt(
        self,
        prompt: str,
        ranges_dict: dict[str, tuple[int, int]],
        state: dict[str, Any],
        truncated: dict[str, tuple[int, int, int]] | None = None,
    ) -> str:
        """Substitute data storage content into prompt placeholders, applying character ranges.

        Uses a single re.sub pass so that document content is never rescanned — this
        preserves literal curly braces (e.g. JSON the LLM included) and prevents
        {{key}}-looking sequences inside one document from triggering substitution of
        another key's placeholder.
        """

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            doc_content = state[key]
            start = 0
            if key in ranges_dict:
                start, end_idx = ranges_dict[key]
                doc_content = doc_content[start:end_idx]
            if truncated is not None and len(doc_content) > MAX_RETRIEVED_CHARS:
                truncated[key] = (start, start + MAX_RETRIEVED_CHARS, len(state[key]))
            return _bounded(doc_content, start)

        return re.sub(r"{{([^{}]+)}}", replace, prompt)

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            prompt: str = args["prompt"]
            input_character_ranges = args.get("input_character_ranges", [])
            if input_character_ranges is None:
                input_character_ranges = []

            ranges_dict = self._validate_inputs(prompt, input_character_ranges, state)
            truncated: dict[str, tuple[int, int, int]] = {}
            prompt = self._format_prompt(prompt, ranges_dict, state, truncated)
            response = await self._llm.query(prompt)
            # The answer came from part of a document. Say so here rather than only in
            # the prompt: this tool's caller is the one that can ask for another range.
            notes = [
                f"NOTE: '{key}' was read from characters {start:,} to {end:,} of "
                f"{total:,}. Call again with input_character_ranges to read another part."
                for key, (start, end, total) in truncated.items()
            ]
            return ToolOutput(
                output="\n".join([response.output_text_str, *notes]),
                metadata=response.metadata,
            )
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Retrieve information failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg)
