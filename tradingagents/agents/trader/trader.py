import functools
import time
import json
import re


_FINAL_PROPOSAL_RE = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*[:：]?\s*\**\s*(?:STRONG[\s_]+)?(BUY|SELL|HOLD)",
    re.IGNORECASE,
)


def _field_first_number(block, pattern):
    """First numeric value on the line captured by ``pattern`` (group 1)."""
    m = re.search(pattern, block, re.IGNORECASE)
    if not m:
        return None
    n = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", m.group(1))
    if not n:
        return None
    try:
        return float(n.group(1).replace(",", ""))
    except ValueError:
        return None


def parse_trade_plan(content):
    """Best-effort parse of the ---TRADE PLAN--- block: direction + levels.

    Pure function (stdlib only) so it can be unit-tested without LLM deps.
    Missing pieces come back as None — callers must treat that as "cannot
    judge", never as "consistent".
    """
    if not isinstance(content, str):
        content = str(content) if content else ""
    m = re.search(r"-{2,}\s*TRADE\s*PLAN\s*-{2,}", content, re.IGNORECASE)
    block = content[m.end():] if m else content
    pm = _FINAL_PROPOSAL_RE.search(content)
    return {
        "proposal": pm.group(1).upper() if pm else None,
        "entry": _field_first_number(block, r"entry[^:：\n]*[:：]([^\n]*)"),
        "stop": _field_first_number(block, r"stop.?loss[^:：\n]*[:：]([^\n]*)"),
        "take_profit": _field_first_number(block, r"take.?profit[^:：\n]*[:：]([^\n]*)"),
    }


def trade_plan_issue(plan):
    """Name the internal contradiction in a parsed plan, or None if coherent.

    BUY must take profit ABOVE entry with the stop BELOW; SELL is the mirror.
    HOLD is exempt: its levels are monitoring bounds, not a directional trade.
    """
    p, e = plan.get("proposal"), plan.get("entry")
    t, s = plan.get("take_profit"), plan.get("stop")
    if p not in ("BUY", "SELL") or e is None:
        return None
    if p == "SELL":
        if t is not None and t > e:
            return "you recommended SELL but Take-Profit $%g is ABOVE Entry $%g" % (t, e)
        if s is not None and s < e:
            return "you recommended SELL but Stop-Loss $%g is BELOW Entry $%g" % (s, e)
    else:
        if t is not None and t < e:
            return "you recommended BUY but Take-Profit $%g is BELOW Entry $%g" % (t, e)
        if s is not None and s > e:
            return "you recommended BUY but Stop-Loss $%g is ABOVE Entry $%g" % (s, e)
    return None


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"""Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.

Proposed Investment Plan: {investment_plan}

Key Market Data (USE THESE ACTUAL PRICES for your trade plan):
{market_research_report}

Leverage these insights to make an informed and strategic decision.""",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}

CRITICAL: Use the ACTUAL stock prices from the Key Market Data section provided by the user. Do NOT guess or hallucinate prices. Your Entry, Stop-Loss, and Take-Profit levels MUST be based on the real current price provided in the market data.

PRICE LOGIC RULES (you MUST follow):
- If recommending BUY:
  Take-Profit MUST be ABOVE Entry (you expect price to RISE)
  Stop-Loss MUST be BELOW Entry (cut loss if price drops below support)
  Example: Entry $100, Stop-Loss $92 (-8%), Take-Profit $115 (+15%)
- If recommending SELL:
  Take-Profit MUST be BELOW Entry (you expect price to FALL)
  Stop-Loss MUST be ABOVE Entry (cut loss if price rises above resistance)
  Example: Entry $100, Stop-Loss $108 (+8%), Take-Profit $85 (-15%)
- If recommending HOLD:
  Entry = current price as reference point (not a new entry)
  Provide upside target and downside stop as monitoring levels
- BEFORE outputting the TRADE PLAN, SELF-CHECK: for BUY verify Stop-Loss < Entry < Take-Profit; for SELL verify Take-Profit < Entry < Stop-Loss. If a level violates the rule for your direction, FIX the level — do NOT keep mismatched levels and do NOT flip the direction to excuse them.

After your analysis, you MUST include the following structured section at the end of your response. Fill in concrete numbers based on your analysis (use actual price levels, not placeholders):

---TRADE PLAN---
Entry: [price or short price range ONLY, e.g. "$303.69" or "$300-$310"]
Stop-Loss: [price level with percentage from entry, e.g. "$270.00 (-11.10%)"]
Take-Profit Target: [ONE price level + percentage from entry + REQUIRED short basis for this level in a second parenthesis, e.g. "$400.00 (+31.74%) (key resistance)" or "$85.00 (-15.0%) (200-day SMA support)". The basis is mandatory — it must name the technical/fundamental level the target sits on, not just restate the number.]
Position Sizing: [absolute % of portfolio ONLY — single number or range, e.g. "15%" or "10-15%". DO NOT use words like Small/Medium/Large/中等. Choose based on the SIZING RUBRIC below.]
Timeframe: [expected holding period, e.g. "2-4 weeks"]
Risk/Reward: [ratio ONLY, e.g. "2.86" or "2.86 / 4.34"]
Scaling Strategy: [entry method, e.g. single entry or tranches]

FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**

POSITION SIZING RUBRIC (pick one tier and output the numeric range or a number inside it):
- High Conviction (15-25%): multiple aligned signals across technicals/fundamentals/catalyst, clear stop, catalyst confirmed
- Medium Conviction (8-15%): primary thesis clear with 1-2 noise items, or catalyst timing uncertain
- Speculative (3-8%): good risk/reward but lower win-rate, event-driven single-side bet
- Pilot / Scout (1-3%): test position, awaiting confirmation, or extremely high volatility

Position Sizing output format examples (allowed):
  Position Sizing: 15% (medium conviction)
  Position Sizing: 10-15%
  Position Sizing: 5% (speculative, earnings play)

Position Sizing values that are NEVER acceptable: "Medium", "Small", "Large", "中等倉位", "中等", "中型", or any word-only label without a percentage number.

FORMAT RULES (MUST follow):
- Each field value MUST be SHORT and CONCISE — numbers/prices only, no sentences or paragraphs.
- Entry: ONLY a dollar price or price range. Do NOT write explanations. Wrong: "建議在當前市場價格附近入場..." Right: "$303.69" or "$300-$310 (near 10-day EMA)"
- Stop-Loss / Take-Profit: price + percentage in parentheses. One line max.
- Risk/Reward: ratio number(s) only. Wrong: "The risk reward ratio is approximately..." Right: "2.86 / 4.34"
- You may add a SHORT parenthetical note (under 30 chars) after the number, e.g. "$400.00 (+31.74%) (Needham PT)"

CRITICAL: You MUST keep ALL field labels (Entry, Stop-Loss, Take-Profit Target, Position Sizing, Timeframe, Risk/Reward, Scaling Strategy, FINAL TRANSACTION PROPOSAL) and the header "---TRADE PLAN---" in English exactly as shown above. Do NOT translate these labels into any other language.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        # Deterministic self-consistency gate: a directional plan whose levels
        # form the opposite trade (e.g. SELL with take-profit above entry) gets
        # ONE corrective re-invoke. Keep the direction, fix the levels. If the
        # retry is unparseable, keep the original — the app layer flags the
        # residual mismatch instead of us looping.
        issue = trade_plan_issue(parse_trade_plan(result.content))
        if issue:
            correction = {
                "role": "user",
                "content": (
                    f"Your TRADE PLAN is internally inconsistent: {issue}. "
                    "A SELL profits when price FALLS: Take-Profit MUST be BELOW "
                    "Entry and Stop-Loss MUST be ABOVE Entry. A BUY profits when "
                    "price RISES: Take-Profit MUST be ABOVE Entry and Stop-Loss "
                    "MUST be BELOW Entry. Re-output your ENTIRE response (full "
                    "analysis, the complete ---TRADE PLAN--- section and FINAL "
                    "TRANSACTION PROPOSAL) with self-consistent price levels. "
                    "Keep your recommendation direction unless the market data "
                    "truly supports flipping it."
                ),
            }
            retry = llm.invoke(
                messages
                + [{"role": "assistant", "content": result.content}, correction]
            )
            retry_plan = parse_trade_plan(retry.content)
            if retry_plan["proposal"] and retry_plan["entry"] is not None:
                result = retry

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
