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
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}

After your analysis, you MUST include the following structured section at the end of your response. Fill in concrete numbers based on your analysis (use actual price levels, not placeholders):

---TRADE PLAN---
Entry: [price range or condition]
Stop-Loss: [price level with percentage from entry]
Take-Profit Target 1: [price level with percentage gain]
Take-Profit Target 2: [price level with percentage gain]
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
