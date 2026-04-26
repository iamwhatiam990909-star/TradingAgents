import logging

from .aggressive_debator import create_aggressive_debator
from .conservative_debator import create_conservative_debator
from .neutral_debator import create_neutral_debator

logger = logging.getLogger(__name__)


def _run_debator(fn, state, agent_label: str):
    """Run a debator in a thread, setting the agent name for logging."""
    from tradingagents.graph.setup import set_current_agent
    set_current_agent(agent_label)
    return fn(state)


def create_parallel_risk_debate(llm):
    aggressive_fn = create_aggressive_debator(llm)
    conservative_fn = create_conservative_debator(llm)
    neutral_fn = create_neutral_debator(llm)

    def parallel_risk_debate(state):
        role_fns = {
            "aggressive": aggressive_fn,
            "conservative": conservative_fn,
            "neutral": neutral_fn,
        }

        results: dict[str, dict] = {}

        for role, fn in role_fns.items():
            try:
                results[role] = _run_debator(fn, state, f"risk_{role}")
            except Exception:
                logger.exception("Risk debator '%s' failed", role)
                raise

        merged: dict[str, str | int] = {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 3,
            "latest_speaker": "All Risk Analysts",
        }

        for role in ["aggressive", "conservative", "neutral"]:
            rds = results[role]["risk_debate_state"]
            response = rds.get(f"current_{role}_response", "")
            merged[f"current_{role}_response"] = response
            merged[f"{role}_history"] = rds.get(f"{role}_history", "")
            merged["history"] += response + "\n\n"

        logger.info("Parallel risk debate completed: 3 analysts in parallel")
        return {"risk_debate_state": merged}

    return parallel_risk_debate
