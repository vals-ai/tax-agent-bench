import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from model_library.agent import AgentResult
from model_library.base import LLMConfig
from tqdm.asyncio import tqdm

from .get_agent import Parameters, get_agent, MAX_TIME_SECONDS
from .tools import DEFAULT_AS_OF_DATE, VALID_TOOLS


async def run_tests_parallel(
    questions: list[str],
    max_concurrent: int,
    parameters: Parameters,
    log_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run multiple questions in parallel using the agent"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question: str, question_index: int):
        # tqdm.gather has no return_exceptions: it awaits each task, so one raise ends
        # the run and discards every other question's result. Return the exception
        # instead and let the caller report it alongside the successes.
        async with semaphore:
            try:
                agent = get_agent(parameters, log_dir=log_dir)
                return await agent.run(
                    parameters.build_input(question),
                    question_id=f"q{question_index:03d}",
                    atif_export=True,
                )
            except Exception as e:
                return e

    tasks = [process_question(question, i + 1) for i, question in enumerate(questions)]

    results: list[AgentResult | Exception] = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for question, result in zip(questions, results):
        if isinstance(result, Exception):
            formatted_results.append({"question": question, "success": False, "error": str(result)})
            print(f"\nFAIL Question failed: {question}\n   Error: {result}\n")
        else:
            formatted_results.append(
                {"question": question, "success": result.success, "result": result.model_dump(mode="json")}
            )
            if not result.success and result.final_error:
                print(
                    f"\nFAIL Question failed: {question}\n   Turns: {result.total_turns}\n   Error: [{result.final_error.type}] {result.final_error.message}\n"
                )
            else:
                print(
                    f"\nOK Question succeeded: {question}\n   Turns: {result.total_turns}\n   Result: {result.final_answer}\n"
                )

    # Next to the agent logs. When every question raised there is no agent output
    # directory to sit beside, so the configured log directory stands in.
    succeeded = [r for r in results if not isinstance(r, Exception)]
    results_dir = succeeded[0].output_dir.parent if succeeded else log_dir
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "results.json"
        _ = results_file.write_text(json.dumps(formatted_results, indent=2))
        print(f"\nResults saved to: {results_file}")

    return formatted_results


async def main():
    parser = argparse.ArgumentParser(description="Run the harness for the Tax Agent Bench agent benchmark")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum number of tokens for completion generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation",
    )
    parser.add_argument("--questions", type=str, nargs="+", help="List of questions to process")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-6",
        help="Model to use to generate completions",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to a question file: a benchmark JSON (data/*.json) or plain text with one question per line",
    )
    parser.add_argument(
        "--tools",
        type=str,
        nargs="+",
        default=VALID_TOOLS,
        choices=VALID_TOOLS,
        help="List of tools to make available to the agent",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=MAX_TIME_SECONDS,
        help="Maximum time in seconds for the agent to run before stopping (default: 120 minutes, matching the benchmark budget)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum number of agent turns for local testing (default: 50). The benchmark evaluation workflow uses time limits only.",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=DEFAULT_AS_OF_DATE,
        help=(
            f"Date the agent answers as of, in YYYY-MM-DD format (default: {DEFAULT_AS_OF_DATE}). "
            "Clamps web_search end dates and is stated in the question turn. Applies to every question "
            "in this run, so set it only when the questions you pass state a later as-of date themselves. "
            "In the benchmark it comes from each test's context instead, per question."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of parallel requests to make to the model",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where per-question agent logs are written",
    )
    args = parser.parse_args()

    env_file = Path(".env")
    load_dotenv(override=True, dotenv_path=env_file)

    if args.question_file:
        question_file = Path(args.question_file)
        if question_file.suffix == ".json":
            questions = [test["question"] for test in json.loads(question_file.read_text())["tests"]]
        else:
            questions = [line.strip() for line in question_file.read_text().splitlines() if line.strip()]
    elif args.questions:
        questions = args.questions
    else:
        raise Exception("No questions provided. One of --question-file or --questions must be used.")

    parameters = Parameters(
        model_name=args.model,
        max_time_seconds=args.max_time,
        max_turns=args.max_turns,
        tools=args.tools,
        as_of_date=args.as_of_date,
        llm_config=LLMConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )

    await run_tests_parallel(
        questions=questions,
        max_concurrent=args.parallelism,
        parameters=parameters,
        log_dir=args.log_dir,
    )


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
