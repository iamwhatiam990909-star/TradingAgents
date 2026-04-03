# TradingAgents/graph/setup.py

import logging
import threading
import time
from typing import Callable, Dict, Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic

logger = logging.getLogger(__name__)

# ── Retry + Request Size Logging ──────────────────────────────────────

_TRANSIENT_ERROR_KEYWORDS = (
    "disconnected", "connection", "timeout", "reset by peer",
    "broken pipe", "eof occurred", "remoteerror", "remoteprotocolerror",
    "server disconnected", "connection aborted", "incomplete read",
)

_MAX_RETRIES = 2
_RETRY_BACKOFF = [5, 15]

# Thread-local storage for tracking which agent is currently invoking
_tls = threading.local()


def set_current_agent(name: str) -> None:
    """Set the current agent name for logging (called by agent nodes)."""
    _tls.agent_name = name


def _get_current_agent() -> str:
    return getattr(_tls, "agent_name", "unknown")


def _estimate_tokens(input_data: Any) -> int:
    """Rough token estimate: chars / 3 for mixed CJK+English text."""
    if isinstance(input_data, str):
        text = input_data
    elif isinstance(input_data, list):
        text = " ".join(str(m) for m in input_data)
    else:
        text = str(input_data)
    return len(text) // 3


def _is_transient_error(exc: BaseException) -> bool:
    err_str = f"{type(exc).__name__} {exc}".lower()
    return any(kw in err_str for kw in _TRANSIENT_ERROR_KEYWORDS)


def _inject_retry_and_logging(llm: Any, label: str) -> None:
    """Monkey-patch llm.invoke with retry on transient errors + size logging."""
    original_invoke = llm.invoke

    def retrying_invoke(input_data: Any, *args: Any, **kwargs: Any) -> Any:
        est_tokens = _estimate_tokens(input_data)
        agent = _get_current_agent()

        for attempt in range(_MAX_RETRIES + 1):
            try:
                logger.info(
                    "LLM invoke [%s/%s]: ~%d tokens (attempt %d/%d)",
                    label, agent, est_tokens, attempt + 1, _MAX_RETRIES + 1,
                )
                result = original_invoke(input_data, *args, **kwargs)
                return result
            except Exception as exc:
                is_transient = _is_transient_error(exc)
                logger.error(
                    "LLM invoke FAILED [%s/%s]: ~%d tokens | %s: %s | attempt %d/%d | transient=%s",
                    label, agent, est_tokens,
                    type(exc).__name__, str(exc)[:200],
                    attempt + 1, _MAX_RETRIES + 1, is_transient,
                )
                if not is_transient or attempt == _MAX_RETRIES:
                    raise
                wait = _RETRY_BACKOFF[attempt]
                logger.warning(
                    "Retrying in %ds (attempt %d/%d)...",
                    wait, attempt + 2, _MAX_RETRIES + 1,
                )
                time.sleep(wait)

    object.__setattr__(llm, "invoke", retrying_invoke)


def _skip_analyst_if_done(node_fn: Callable, report_field: str) -> Callable:
    """Wrap an analyst node to skip LLM call if its report already exists."""
    def wrapped(state):
        set_current_agent(report_field)
        existing = state.get(report_field)
        if existing:
            logger.info("Checkpoint skip: %s already populated", report_field)
            return {
                "messages": [AIMessage(content=f"[Checkpoint: {report_field} restored]")],
                report_field: existing,
            }
        return node_fn(state)
    return wrapped


def _skip_debate_if_done(node_fn: Callable, debate_field: str) -> Callable:
    """Wrap a debate node to skip if judge_decision already exists."""
    def wrapped(state):
        set_current_agent(debate_field)
        debate = state.get(debate_field, {})
        if debate.get("judge_decision"):
            logger.info("Checkpoint skip: %s debate already judged", debate_field)
            return {}
        return node_fn(state)
    return wrapped


def _skip_if_done(node_fn: Callable, field: str) -> Callable:
    """Wrap a node to skip if its output field already exists."""
    def wrapped(state):
        set_current_agent(field)
        existing = state.get(field)
        if existing:
            logger.info("Checkpoint skip: %s already populated", field)
            return {}
        return node_fn(state)
    return wrapped

_LANG_NAMES: Dict[str, str] = {
    "zh-TW": "Traditional Chinese (繁體中文)",
    "zh-CN": "Simplified Chinese (简体中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
}


def _inject_language(llm: Any, language: str) -> None:
    """Monkey-patch llm.invoke to prepend a language instruction.

    Works for both direct invoke (non-tool agents) and via
    RunnableBinding (tool-using agents created by create_react_agent),
    because RunnableBinding.invoke delegates to self.bound.invoke.
    """
    lang_name = _LANG_NAMES.get(language, language)
    instruction = (
        f"CRITICAL INSTRUCTION: You MUST write all analysis, reasoning, conclusions, "
        f"and recommendations in {lang_name}. "
        f"HOWEVER, you MUST keep ALL structured field labels, section headers, and "
        f"format markers in English exactly as specified in the prompt. "
        f"Examples of labels to keep in English: Entry, Stop-Loss, Take-Profit, "
        f"Direction, Strategy, Strike, Expiry, Catalyst, Max Risk, Position Sizing, "
        f"Timeframe, Risk/Reward, Scaling Strategy. "
        f"Keep financial terms, ticker symbols, and numerical data as-is."
    )
    original_invoke = llm.invoke

    reminder = (
        f"\n\nREMINDER: Write all analysis content in {lang_name}. "
        f"Keep ALL structured field labels in English exactly as shown in the prompt. "
        f"請用{lang_name}撰寫分析內容，但保留所有結構化欄位標籤為英文。"
    )

    def locale_invoke(input: Any, *args: Any, **kwargs: Any) -> Any:
        from langchain_core.messages import SystemMessage

        if isinstance(input, str):
            input = f"{instruction}\n\n{input}{reminder}"
        elif isinstance(input, list):
            input = [SystemMessage(content=instruction)] + list(input) + [SystemMessage(content=reminder)]
        return original_invoke(input, *args, **kwargs)

    object.__setattr__(llm, "invoke", locale_invoke)


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"],
        language: str = "en",
        skip_options: bool = False,
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Inject retry + request-size logging (before language patch).
        _inject_retry_and_logging(self.quick_thinking_llm, "quick")
        _inject_retry_and_logging(self.deep_thinking_llm, "deep")

        # Inject language instruction into LLMs for non-English reports.
        # Save original deep_thinking invoke BEFORE patching so Options
        # Strategist can use an un-injected LLM (it handles language itself).
        _original_deep_invoke = None
        if language and language != "en":
            _original_deep_invoke = self.deep_thinking_llm.invoke
            _inject_language(self.quick_thinking_llm, language)
            _inject_language(self.deep_thinking_llm, language)

        # Mapping: analyst type -> report state field
        _ANALYST_REPORT_FIELDS = {
            "market": "market_report",
            "social": "sentiment_report",
            "news": "news_report",
            "fundamentals": "fundamentals_report",
        }

        # Create analyst nodes (wrapped with checkpoint skip)
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = _skip_analyst_if_done(
                create_market_analyst(self.quick_thinking_llm),
                _ANALYST_REPORT_FIELDS["market"],
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = _skip_analyst_if_done(
                create_social_media_analyst(self.quick_thinking_llm),
                _ANALYST_REPORT_FIELDS["social"],
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = _skip_analyst_if_done(
                create_news_analyst(self.quick_thinking_llm),
                _ANALYST_REPORT_FIELDS["news"],
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = _skip_analyst_if_done(
                create_fundamentals_analyst(self.quick_thinking_llm),
                _ANALYST_REPORT_FIELDS["fundamentals"],
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes (wrapped with checkpoint skip)
        bull_researcher_node = _skip_debate_if_done(
            create_bull_researcher(self.quick_thinking_llm, self.bull_memory),
            "investment_debate_state",
        )
        bear_researcher_node = _skip_debate_if_done(
            create_bear_researcher(self.quick_thinking_llm, self.bear_memory),
            "investment_debate_state",
        )
        research_manager_node = _skip_if_done(
            create_research_manager(self.deep_thinking_llm, self.invest_judge_memory),
            "investment_plan",
        )
        trader_node = _skip_if_done(
            create_trader(self.quick_thinking_llm, self.trader_memory),
            "trader_investment_plan",
        )

        # Create parallel risk debate node (3 analysts run concurrently)
        parallel_risk_debate_node = _skip_debate_if_done(
            create_parallel_risk_debate(self.quick_thinking_llm),
            "risk_debate_state",
        )
        risk_manager_node = _skip_if_done(
            create_risk_manager(self.deep_thinking_llm, self.risk_manager_memory),
            "final_trade_decision",
        )

        # Create options strategist node (skipped when skip_options=True
        # to speed up pipeline; options generated on-demand via API).
        if not skip_options:
            if _original_deep_invoke is not None:
                class _RawInvokeProxy:
                    """Proxy that delegates to the original un-patched invoke."""
                    def __init__(self, llm, raw_invoke):
                        self._llm = llm
                        self.invoke = raw_invoke
                    def __getattr__(self, name):
                        return getattr(self._llm, name)
                options_llm = _RawInvokeProxy(self.deep_thinking_llm, _original_deep_invoke)
            else:
                options_llm = self.deep_thinking_llm

            options_strategist_node = _skip_if_done(
                create_options_strategist(options_llm, language=language),
                "options_strategist_output",
            )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Parallel Risk Debate", parallel_risk_debate_node)
        workflow.add_node("Risk Judge", risk_manager_node)
        if not skip_options:
            workflow.add_node("Options Strategist", options_strategist_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Parallel Risk Debate")
        workflow.add_edge("Parallel Risk Debate", "Risk Judge")

        if skip_options:
            workflow.add_edge("Risk Judge", END)
        else:
            workflow.add_edge("Risk Judge", "Options Strategist")
            workflow.add_edge("Options Strategist", END)

        # Compile and return
        return workflow.compile()
