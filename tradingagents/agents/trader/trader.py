import functools
import time
import json


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

After your analysis, you MUST include the following structured section at the end of your response. Fill in concrete numbers based on your analysis (use actual price levels, not placeholders):

---TRADE PLAN---
Entry: [price range or condition]
Stop-Loss: [price level with percentage from entry]
Take-Profit Target 1: [price level with percentage from entry]
Take-Profit Target 2: [price level with percentage from entry]
Position Sizing: [recommended allocation]
Timeframe: [expected holding period]
Risk/Reward: [ratio]
Scaling Strategy: [entry method, e.g. single entry or tranches]

FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**

CRITICAL: You MUST keep ALL field labels (Entry, Stop-Loss, Take-Profit Target, Position Sizing, Timeframe, Risk/Reward, Scaling Strategy, FINAL TRANSACTION PROPOSAL) and the header "---TRADE PLAN---" in English exactly as shown above. Do NOT translate these labels into any other language.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
