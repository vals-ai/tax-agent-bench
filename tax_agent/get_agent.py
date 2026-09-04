from pathlib import Path

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    DEFAULT_SUMMARY_PREFIX,
    HistoryCompaction,
    TimeLimit,
    ToolCallRecord,
    TurnLimit,
    TurnResult,
    default_before_query,
    llm_summary_compactor,
    truncate_oldest,
)
from model_library.base import LLM, LLMConfig, RawResponse, TextInput
from model_library.base.input import InputItem, SystemInput
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_registry_model
from pydantic import BaseModel, field_validator

from .prompt import QUESTION_PROMPT, SYSTEM_PROMPT, TAX_COMPACTION_PROMPT
from .exceptions import RetryExhaustedError
from .tools import (
    DEFAULT_AS_OF_DATE,
    VALID_TOOLS,
    Calculator,
    FetchDocument,
    LookupAuthority,
    RetrieveInformation,
    SubmitFinalResult,
    TavilyWebSearch,
    Tool,
    validate_date_format,
)


# Governs local runs; the benchmark runner passes max_time_seconds explicitly.
MAX_TIME_SECONDS = 120 * 60  # 2 hours

# Compact when the next query is projected to reach this share of the context window.
COMPACTION_THRESHOLD_PERCENTAGE = 0.80


class Parameters(BaseModel):
    model_name: str
    max_time_seconds: int = MAX_TIME_SECONDS
    max_turns: int | None = None
    tools: list[str] = VALID_TOOLS
    llm_config: LLMConfig
    # Sets this question's web_search ceiling and is stated in the question turn, so
    # build_input lives here: both derive from one object and cannot disagree.
    as_of_date: str = DEFAULT_AS_OF_DATE

    @field_validator("as_of_date")
    @classmethod
    def _validate_as_of_date(cls, value: str) -> str:
        validate_date_format("as_of_date", value)
        return value

    def build_input(self, question: str) -> list[InputItem]:
        """The system prompt plus the question, stating a non-default as-of date."""
        question_text = QUESTION_PROMPT.format(question=question)
        if self.as_of_date != DEFAULT_AS_OF_DATE:
            question_text = f"Answer this question as of {self.as_of_date}.\n\n{question_text}"
        return [SystemInput(text=SYSTEM_PROMPT), TextInput(text=question_text)]


def create_llm(parameters: Parameters) -> LLM:
    """Create an LLM instance from parameters using the model registry."""
    return get_registry_model(parameters.model_name, parameters.llm_config)


def get_agent(
    parameters: Parameters,
    llm: LLM | None = None,
    log_dir: Path | None = None,
) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    if llm is None:
        llm = create_llm(parameters)

    available_tools: dict[str, type[Tool]] = {
        "web_search": TavilyWebSearch,
        "retrieve_information": RetrieveInformation,
        "fetch_document": FetchDocument,
        "calculator": Calculator,
        "lookup_authority": LookupAuthority,
    }

    selected_tools: list[Tool] = []
    for tool_name in parameters.tools:
        if tool_name not in available_tools:
            raise Exception(f"Tool {tool_name} not found in tools. Available tools: {available_tools.keys()}")
        tool_cls = available_tools[tool_name]
        if tool_name == "retrieve_information":
            selected_tools.append(tool_cls(llm=llm))  # type: ignore[call-arg]
        elif tool_name in ("web_search", "lookup_authority"):
            selected_tools.append(tool_cls(as_of_date=parameters.as_of_date))  # type: ignore[call-arg]
        else:
            selected_tools.append(tool_cls())  # type: ignore[call-arg]

    selected_tools.append(SubmitFinalResult())

    def _before_query(history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        """Truncate on context window overflow, re-raise all other errors (stops the loop).

        Also injects a nudge to call a tool when the previous turn had no tool calls
        (last item in history is a RawResponse, meaning no ToolResult was appended).
        """
        if isinstance(last_error, MaxContextWindowExceededError):
            return truncate_oldest(history)
        if history and isinstance(history[-1], RawResponse):
            history.append(
                TextInput(
                    text=(
                        "Your last response produced no tool call. "
                        "Call `submit_final_result` if you have a final result, "
                        "otherwise continue with the next tool call."
                    )
                )
            )
        return default_before_query(history, last_error)

    def _on_tool_result(record: ToolCallRecord, state: dict) -> None:
        if record.error and record.error.type == "RetryExhaustedError":
            raise RetryExhaustedError(record.error.message)

    def _should_stop(turn_result: TurnResult) -> bool:
        """Never stop on text-only responses.

        The model library default stops on text-only responses (no tool calls), but the tax agent
        should keep looping until the model calls submit_final_result or a configured limit is hit.
        """
        return False

    # None when the model exposes no context window, as with get_raw_model custom
    # endpoints; the run then falls back to _before_query's truncate_oldest.
    history_compaction = HistoryCompaction(
        threshold_percentage=COMPACTION_THRESHOLD_PERCENTAGE,
        compact_on_max_context=True,
    )
    compaction_hook = llm_summary_compactor(
        llm,
        history_compaction,
        prompt=TAX_COMPACTION_PROMPT,
        summary_prefix=DEFAULT_SUMMARY_PREFIX,
    )

    return Agent(
        llm=llm,
        tools=selected_tools,
        name="tax",
        log_dir=log_dir or Path("logs"),
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=parameters.max_turns) if parameters.max_turns else None,
            time_limit=TimeLimit(max_seconds=parameters.max_time_seconds),
            history_compaction=history_compaction,
        ),
        hooks=AgentHooks(
            before_query=_before_query,
            should_stop=_should_stop,
            on_tool_result=_on_tool_result,
            compaction=compaction_hook,
        ),
    )
