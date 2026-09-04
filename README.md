# Tax Agent Bench

This repository contains the agentic harness used to run the Tax Agent Bench benchmark and, in
`data/public.json`, the public sample of the question set. The scored set is held out.

Tax Agent Bench evaluates LLMs on their ability to use tools to research and answer complex **enterprise / corporate US tax** questions spanning federal corporate tax, state & local tax (SALT), partnership tax, tax controversy and procedure, transfer pricing, reporting and disclosure, and corporate employment tax.

The agent has access to the following tools:

- `web_search`: Search the web for information (via Tavily)
- `fetch_document`: Fetch a web page or PDF and store its text for later retrieval (e.g. the Internal Revenue Code, Treasury regulations, IRS publications and form instructions, public laws, revenue rulings, court opinions, state tax authority pages). PDFs are detected automatically; scanned image-only PDFs cannot be read (no OCR)
- `retrieve_information`: Access stored information from previous steps via LLM-based prompts
- `lookup_authority`: Resolve a citation to the text of the provision it names, for the U.S. Code, the CFR, and public laws (e.g. `IRC § 163(j)(10)`, `Treas. Reg. § 1.163(j)-7(c)`, `P.L. 119-21 § 70303(a)`). Resolves against the publisher of each: the House's U.S. Code, the eCFR service, and govinfo. Returns the provision's text and the URL to cite, plus the Statutes at Large reference for a public law section. Regulations are served as of the question's as-of date. Other authority — rulings, notices, forms, publications, court opinions, state material — goes through `web_search` and `fetch_document`
- `calculator`: Evaluate arithmetic expressions for tax computations

For more details on the benchmark, please refer to our [public website](https://www.vals.ai/benchmarks/tax_agent_bench).

## Set up

### Dependencies

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. Then run:

```
make install
source .venv/bin/activate
```

### Platform (optional)

Running this agent locally requires only the API keys in the next section — no Vals account is needed.

Benchmark results are published on the [Vals platform](https://www.vals.ai/). Access is gated and requires approval, so please reach out to us if you would like it.

### Environment Variables

Copy the example file and fill in the keys you need:

```
cp .env.example .env
```

Two things are required: a key for the provider of the model you run (the model library resolves it from
`--model`, so `anthropic/claude-sonnet-4-6` needs `ANTHROPIC_API_KEY`), and `TAVILY_API_KEY` for the
`web_search` tool, which will not start without it. You can create a Tavily API key
[here](https://tavily.com/).

The `.env` takes precedence over set environment variables.

### Authority corpora

`lookup_authority` fetches regulations from the eCFR service, which rate-limits
per-section requests, and statutes from the House's current U.S. Code, which has no as-of
date. Downloading each title once avoids both:

```
python -m tax_agent.corpus build --kind cfr --dest data/cfr
python -m tax_agent.corpus build --kind usc --dest data/usc
export TAX_AGENT_CFR_CORPUS=data/cfr TAX_AGENT_USC_CORPUS=data/usc
```

Each writes a title plus a section index — the CFR as of the agent's default as-of date
(~87 MB), and the U.S. Code from the House release point in force on that date (~56 MB).
Lookups then read from disk. Two reasons to do it: throughput, since nothing is shared
when each question runs in its own container; and currency, since the U.S. Code corpus is
frozen at a release point whereas the live edition includes every later amendment.

Coverage is exact. A CFR corpus answers only for the day it was built for. A U.S. Code
corpus answers for the span between its release point and the next one, because the Code
changes only when a law passes. Anything outside is fetched, and a fetched statute says so
in its `Currency:` line.

## Running the benchmark

For a list of command line options, run `tax-agent --help`

To run, for example, a single question on openai/gpt-5.2-2025-12-11:

```
tax-agent --questions "For the 2025 tax year, how does the §163(j) business interest limitation interact with a domestic C corporation's GILTI inclusion?" --model openai/gpt-5.2-2025-12-11
```

You can specify multiple questions at once:

```
tax-agent --questions "Question 1?" "Question 2?"
```

You can also point the agent at a question file — the public sample shipped here, or a text file
containing one question per line:

```
tax-agent --question-file data/public.json
tax-agent --question-file my-questions.txt
```

The default time budget matches the benchmark run (`--max-time`, two hours). The
`--max-turns` default of 50 is a local-testing limit; the benchmark used time limits
only, so pass `--max-turns 0` to reproduce it.

### Public questions

`data/public.json` holds five questions (`P-001`..`P-005`) that are representative of the
shape and difficulty of the benchmark but are not part of the scored set. Each entry has an `id` and a
`question`:

```json
{
  "dataset_name": "public",
  "suite_title": "Tax Agent Bench public sample",
  "tests": [
    { "id": "P-001", "question": "ABC Corporation is a domestic C corporation ..." }
  ]
}
```

Questions are graded against expert-written rubrics with severity-weighted partial credit, so answers are expected to identify the governing authority at the level of the specific provision relied on, not merely reach the right conclusion.

### List of Models

A list of available models can be found at our [model library](https://github.com/vals-ai/model-library/blob/main/model_library/config/all_models.json), and also by running `make browse_models` in the model library repository.

Any model in that registry can be selected with `--model`. To run against your own endpoint, build the
`LLM` yourself and pass it to `get_agent`, which uses it instead of resolving the model from the registry:

```python
from model_library.base import LLMConfig
from model_library.registry_utils import get_raw_model

from tax_agent import Parameters, get_agent

config = LLMConfig(custom_endpoint="https://my-endpoint.example/v1", supports_tools=True)
llm = get_raw_model("openai/gpt-5.2-2025-12-11", config=config)

parameters = Parameters(model_name="openai/gpt-5.2-2025-12-11", llm_config=config)
agent = get_agent(parameters, llm=llm)
result = await agent.run(parameters.build_input("Your question here"))
```

The agent requires a model that supports tool calling.

## Logs

The agent writes detailed logs to the `logs/` directory. Each run creates a timestamped directory with per-question log files containing tool usage, token counts, and error tracking.
