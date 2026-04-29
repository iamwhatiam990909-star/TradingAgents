def create_options_strategist(llm, language: str = "en"):
    def options_strategist_node(state) -> dict:
        recommendation = state.get("final_trade_decision", "")
        investment_plan = state.get("investment_plan", "")
        trader_plan = state.get("trader_investment_plan", "")

        invest_debate = state.get("investment_debate_state", {})
        bull_history = invest_debate.get("bull_history", "")
        bear_history = invest_debate.get("bear_history", "")
        judge_decision = invest_debate.get("judge_decision", "")

        risk_debate = state.get("risk_debate_state", {})
        risk_verdict = risk_debate.get("judge_decision", "")

        market_report = state.get("market_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")

        prompt = f"""You are an Options Strategist. Based on the completed analysis pipeline outputs below, produce a structured options trading recommendation.

=== FINAL TRADE DECISION ===
{recommendation}

=== INVESTMENT PLAN (from Research Manager) ===
{investment_plan[:3000]}

=== TRADER EXECUTION PLAN ===
{trader_plan[:3000]}

=== BULL ARGUMENTS ===
{bull_history[:2000]}

=== BEAR ARGUMENTS ===
{bear_history[:2000]}

=== INVESTMENT JUDGE DECISION ===
{judge_decision[:2000]}

=== RISK JUDGE VERDICT ===
{risk_verdict[:2000]}

=== SHORT-TERM DIRECTIONAL SIGNALS (PRIMARY — drives options direction) ===

--- Market / Technical Report ---
{market_report[:2000]}

--- Social Sentiment Report ---
{sentiment_report[:1500]}

=== BACKGROUND CONTEXT (SECONDARY — for catalyst identification only) ===

--- News Report (for upcoming event catalysts) ---
{news_report[:1000]}

--- Fundamentals Report (for earnings / catalyst context) ---
{fundamentals_report[:1500]}

---

RULES — you MUST follow every rule below:

1. DIRECTION RULES:
   - Determine direction from the SHORT-TERM DIRECTIONAL SIGNALS section (Market/Technical Report + Social Sentiment Report), NOT from the final trade decision.
   - The long-term recommendation (BUY/SELL/HOLD) is background context only — options are a SHORT-TERM instrument.
   - Output "bullish" if short-term technical indicators show upward momentum (e.g. RSI rising above 50, price above key moving averages, bullish chart patterns, positive social sentiment trend)
   - Output "bearish" if short-term technical indicators show downward momentum (e.g. RSI falling below 50, price below key moving averages, bearish chart patterns, negative social sentiment trend)
   - Output "neutral" if short-term technical indicators show sideways/range-bound price action (e.g. price oscillating between support and resistance, RSI near 50, no clear trend, Bollinger Bands flat or contracting, low ADX)
   - Output "no_trade" ONLY if:
     * No identifiable catalyst within 6 weeks AND no clear range to sell premium against
     * Liquidity is too thin to enter/exit options safely (wide bid-ask spread, very low open interest)

     Note: high IV with priced-in catalyst is NOT by itself a "no_trade" signal —
     it is the textbook setup for "neutral + iron_condor" (sell premium to capture
     post-event IV crush). Default to neutral + iron_condor in that scenario, not no_trade.

2. STRATEGY SELECTION MATRIX:
   Use this decision table based on direction and IV environment:

   BULLISH + Low/Normal IV  → buy_call (single-leg, max upside exposure)
   BULLISH + Elevated IV    → bull_call_spread (debit spread, offset high premium)
   BEARISH + Low/Normal IV  → buy_put (single-leg, max downside exposure)
   BEARISH + Elevated IV    → bear_put_spread (debit spread, offset high premium)
   NEUTRAL + Normal/High IV → iron_condor (sell OTM call spread + sell OTM put spread; profit from time decay in range-bound market; defined risk)
   NEUTRAL + Low IV         → calendar_spread (sell near-month ATM, buy far-month ATM; profit from near-term time decay while IV is low)

   How to assess IV from the Technical Report:
   - Low/Normal IV: ATR is contracting, Bollinger Bands are narrow, no recent gap or spike
   - Elevated IV: ATR is expanding, Bollinger Bands are wide, recent earnings/event, large daily ranges

   If direction is no_trade → leave strategy empty.

3. STRIKE RULES:
   - Express as a percentage range from current price (OTM)
   - Directional (bullish/bearish):
     * Default: 5%-10% OTM
     * Strong directional signal: 2%-5% OTM
     * Speculative catalyst: up to 15% OTM
   - Neutral (iron_condor): express as range width, e.g. "sell 5% OTM call + sell 5% OTM put, wings 10% OTM"
   - Neutral (calendar_spread): "ATM"
   - Never exceed 20% OTM
   - If no_trade, output "—"

4. EXPIRY RULES:

   Output format (alignment tag REQUIRED):
   - "X weeks (monthly)"                          — default for all strategies
   - "X-Y weeks (weekly, <catalyst>)"             — catalyst-driven only
   - "X weeks (monthly), Y weeks (monthly)"       — calendar spread two legs
   - "—"                                          — if no_trade

   A) BASE RANGES (before IV adjustment):
      - Directional (buy_call/put, debit spreads): 2-4 weeks
      - Iron condor: 3-5 weeks
      - Calendar spread: sell 2-3 weeks / buy 5-6 weeks

   B) MONTHLY vs WEEKLY ALIGNMENT:
      - DEFAULT: monthly expiry (3rd Friday of the month) — best liquidity, tightest spreads
      - REQUIRED monthly: iron_condor, calendar_spread (both legs)
      - WEEKLY allowed ONLY when ALL of these hold:
        * Strategy is directional debit (buy_call/put or debit spread), AND
        * A specific catalyst (earnings/FDA/Fed/product launch) falls before the next monthly expiry, AND
        * Catalyst date falls within the chosen week range
      - NEVER use weekly for iron_condor or calendar_spread

   C) IV-BASED EXPIRY ADJUSTMENT (apply within the base range):
      - ELEVATED IV (ATR expanding, BB wide, recent earnings/event):
        * Premium SELLERS (iron_condor) → pick the SHORTER end of the range (e.g. 3 weeks, not 5).
          Reason: theta decay accelerates near expiry; capture IV crush faster.
        * Premium BUYERS (buy_call/put, debit spreads) → pick the LONGER end of the range (e.g. 4 weeks, not 2).
          Reason: give the thesis time to play out before vega/theta erodes value.
      - LOW/NORMAL IV (ATR contracting, BB narrow):
        * Premium BUYERS → pick the SHORTER end of the range (e.g. 2 weeks, not 4).
          Reason: premium is cheap; maximize leverage with shorter-dated contracts.
        * Premium SELLERS → pick the LONGER end of the range (e.g. 5 weeks, not 3).
          Reason: IV is low, more time = more premium collected.
      - Calendar spread legs: do NOT apply IV adjustment (structure is already vega/theta hedged).

   D) HARD LIMITS:
      - Never suggest 0DTE or 1DTE
      - Never exceed 6 weeks for the short leg
      - If no_trade, output "—"

5. CATALYST:
   - One sentence describing the primary catalyst driving the trade
   - If no_trade, explain why in one sentence

6. MAX RISK:
   - Debit strategies (buy_call, buy_put, bull_call_spread, bear_put_spread, calendar_spread): output "全部權利金"
   - Credit strategies (iron_condor): output "價差寬度 − 收到權利金"

OUTPUT FORMAT — output EXACTLY this structure, nothing else:

---OPTIONS STRATEGY---
Direction: [bullish / bearish / neutral / no_trade]
Strategy: [buy_call / buy_put / bull_call_spread / bear_put_spread / iron_condor / calendar_spread / (empty if no_trade)]
Strike: [X%-Y% OTM / —]
Expiry: [X weeks (monthly) / X-Y weeks (weekly, catalyst) / X weeks (monthly), Y weeks (monthly) / —]
Catalyst: [one sentence]
Max Risk: 全部權利金

CRITICAL FORMAT RULES:
- You MUST keep ALL field labels (Direction, Strategy, Strike, Expiry, Catalyst, Max Risk) in English exactly as shown above.
- The header "---OPTIONS STRATEGY---" must appear exactly as shown.
- Do NOT translate field labels into any other language."""

        # If non-English, instruct catalyst content in target language
        if language and language != "en":
            _LANG_NAMES = {
                "zh-TW": "Traditional Chinese (繁體中文)",
                "zh-CN": "Simplified Chinese (简体中文)",
                "ja": "Japanese (日本語)",
                "ko": "Korean (한국어)",
            }
            lang_name = _LANG_NAMES.get(language, language)
            prompt += f"\n- Write the Catalyst sentence in {lang_name}."

        result = llm.invoke(prompt)

        return {
            "options_strategist_output": result.content,
        }

    return options_strategist_node
