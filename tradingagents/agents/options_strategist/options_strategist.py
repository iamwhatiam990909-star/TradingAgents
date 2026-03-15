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

=== MARKET / TECHNICAL REPORT (for IV context) ===
{market_report[:2000]}

=== FUNDAMENTALS REPORT (for catalyst context) ===
{fundamentals_report[:1500]}

---

RULES — you MUST follow every rule below:

1. DIRECTION RULES:
   - Output "bullish" if the final decision is BUY with clear conviction
   - Output "bearish" if the final decision is SELL with clear conviction
   - Output "no_trade" if ANY of these apply:
     * Bull and bear arguments are roughly equal in strength (no clear edge)
     * IV appears elevated or the catalyst is already priced in
     * The catalyst is more than 6 weeks away
     * The catalyst occurs today (0DTE not allowed)
     * The final decision is HOLD

2. STRATEGY RULES:
   - If IV is low or normal → prefer single-leg: "buy_call" (bullish) or "buy_put" (bearish)
   - If IV is elevated → prefer spread: "call_spread" (bullish) or "put_spread" (bearish)
   - Only output one of: buy_call, buy_put, call_spread, put_spread
   - If direction is no_trade, leave strategy empty

3. STRIKE RULES:
   - Express as a percentage range from current price (OTM)
   - Default: 5%-10% OTM
   - Strong directional signal: 2%-5% OTM
   - Speculative catalyst: up to 15% OTM
   - Never exceed 20% OTM
   - If no_trade, output "—"

4. EXPIRY RULES:
   - Express as a week range (e.g., "2-3 weeks")
   - If catalyst is clear: 1-2 weeks after the catalyst
   - If catalyst is unclear: 3-4 weeks
   - Never suggest 0DTE or 1DTE
   - Never exceed 6 weeks
   - If no_trade, output "—"

5. CATALYST:
   - One sentence describing the primary catalyst driving the trade
   - If no_trade, explain why in one sentence

6. MAX RISK:
   - Always output: "全部權利金"

OUTPUT FORMAT — output EXACTLY this structure, nothing else:

---OPTIONS STRATEGY---
Direction: [bullish / bearish / no_trade]
Strategy: [buy_call / buy_put / call_spread / put_spread / (empty if no_trade)]
Strike: [X%-Y% OTM / —]
Expiry: [X-Y weeks / —]
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
