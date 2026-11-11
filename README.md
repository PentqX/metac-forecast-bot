# metac-forecast-bot

A forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/) bot tournaments
(the seasonal bot tournament and MiniBench).

The bot is fully automated. It was designed and written, and is maintained, by an AI agent (Claude) on behalf
of the account owner. The owner does not review, edit or influence individual forecasts; there is no human in
the loop.

## How it works

For every open tournament question the bot has not answered yet:

1. **Research.** Independent research agents search the web and read pages (the resolution source first),
   using different search indexes. They report facts with dates and sources — resolution details, current
   state, base rates, scheduled events, market prices and expert expectations — and are told not to forecast.
   If no agent returns live research, the bot does not answer the question in that run.
2. **Forecast.** Several frontier models from different labs forecast independently from the same research,
   with reasoning effort set high. Each writes out a resolution check, an outside view with a base rate, an
   inside view, scenarios and a calibration check before giving a final answer.
3. **Adjudicate.** If the forecasters disagree materially, another model is shown their answers and reasoning,
   identifies the crux and gives its own forecast, which is blended with the ensemble.
4. **Aggregate.**
   - Binary: weighted mean and weighted median of log-odds, averaged, then capped away from 0% and 100%.
   - Multiple choice: weighted linear pool with a floor on every option.
   - Numeric and discrete: each forecaster states 13 percentiles. They are turned into a smooth distribution
     by monotone cubic interpolation of the probit of the CDF on the question's own axis (so tails fall off
     like a normal distribution), mixed with a wider copy of itself to offset overconfidence, mixed across
     forecasters, and standardised to the constraints of the Metaculus API.
5. **Submit.** One forecast per question, plus a private comment with the ensemble's answers, the lead
   reasoning and a research excerpt.

Final answers are read strictly: only the requested format, at the very end of the text, is accepted. Anything
else goes to a small model that copies the stated numbers and is told to report "missing" when in doubt, because
a guessed number (a market price, a draft, the probability of the opposite outcome) is worse than no answer.
Answers in the wrong units or with values out of order are discarded. If too few forecasters succeed, nothing is
submitted and the question is retried on a later run. Before submission every forecast is checked against the
same rules as the Metaculus server.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Entry point used by the workflows |
| `bot/config.py` | Models, spending tiers, caps and tournament ids |
| `bot/research.py`, `bot/prompts.py` | Research agents and all prompts |
| `bot/pipeline.py` | Research → ensemble → adjudication → aggregation for one question |
| `bot/numeric.py`, `bot/aggregate.py`, `bot/parsing.py` | Distribution building, aggregation, answer parsing |
| `bot/metaculus.py`, `bot/llm.py`, `bot/http.py` | Metaculus and OpenRouter clients |
| `bot/runner.py` | Run modes, budget pacing, logging |
| `tests/` | Unit tests (no network) |

The bot uses only the Python standard library. There is nothing to install.

## Running

Secrets are read from the environment (or a local `.env` file, see `.env.template`):
`METACULUS_TOKEN` and `OPENROUTER_API_KEY`.

```bash
python main.py --mode offline         # offline checks of the numeric core
python main.py --mode selftest        # checks both APIs and every configured model (a few cents)
python main.py --mode test_questions  # forecasts the bot testing area
python main.py                        # forecasts the seasonal tournament and MiniBench
python main.py --post-id 123 --verbose  # dry run on one post, for closed or test questions only
python -m unittest discover -s tests -t .
```

In GitHub Actions the scheduled workflow runs `main.py` every 20 minutes; questions are open for about three
hours. Logs never contain forecasts or reasoning, because the logs of a public repository are public.

## Cost control

Each run reads the remaining OpenRouter credits and picks a spending tier per question: `full` for seasonal
tournament questions, `lite` for MiniBench, `economy` when credits run low; below a minimum it stops. A single
run stops starting questions once a fixed amount has been spent. A question that keeps failing is tried at most
five times during its three-hour window instead of on every run. Parallel model calls are throttled so that
OpenRouter's in-flight spending budget is not exceeded on a small balance.

## Tournament rules this bot is built around

- No human in the loop: forecasts and comments go straight from the bot to the API.
- One forecast per question in the bot tournaments; already answered questions are skipped.
- The bot is never test-run on open or upcoming tournament questions; the bot testing area and closed
  questions are used instead.
- Every forecast is accompanied by a comment explaining the reasoning.

The workflow files come from the [Metaculus bot template](https://github.com/Metaculus/metac-bot-template),
of which this repository is a fork. Everything else was rewritten.
