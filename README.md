# Dynamic Poker Decision Engine

**A probabilistic decision-analysis engine built in Python.**  
*Monte Carlo simulation · Combinatorics · Expected value · Uncertainty analysis*

You know your cards, but not your opponent's. **How do you compare your options when critical information is missing — and how might the next card change your assessment?**

This project uses heads-up Texas Hold'em as a concrete setting for **decision-making under uncertainty**. It estimates the value of a hand against a modeled range of possible opponent hands, compares the expected value of *folding* and *calling*, and examines how the distribution of outcomes changes when new cards are revealed.

**Poker is the environment; probability and decision analysis are the point.** This is an analytical tool, not a poker bot or a game-theoretic optimal (GTO) solver.

## What the engine does

| Question | How the engine addresses it |
|---|---|
| What might the opponent hold? | Maintains a weighted distribution of legal two-card combinations. |
| How strong is my position? | Estimates showdown equity using Monte Carlo simulation; switches to exact enumeration on the river. |
| How uncertain is the estimate? | Reports Monte Carlo standard error and an approximate 95% interval. |
| Is calling worth its cost? | Displays break-even equity, `EV(call)`, `EV(fold)` and related metrics. |
| What could the next card change? | Evaluates every possible next community card and summarizes the resulting equity distribution. |

The application is **interactive and terminal-based**. The user supplies cards, the pot, and observed actions; the engine reports quantitative evidence, but **the user chooses whether to fold or call**.

## A concrete example

Suppose you hold **A♥ K♥**, the flop is **Q♥ J♥ 4♣**, and the opponent is modeled with the heuristic `balanced` range. The current pot is **100**, and the opponent bets **25**.

A reproducible calculation with **2,000 Monte Carlo simulations** and seed `42` gives:

| Quantity | Result |
|---|---:|
| Estimated equity | **72.73%** |
| Monte Carlo standard error | 0.98 percentage points |
| Approximate 95% Monte Carlo interval | 70.80%–74.65% |
| Break-even equity to call | 16.67% |
| `EV(fold)` | 0.00 |
| `EV(call)` | **+84.09** |

Here, *equity* means the expected share of the pot at showdown, counting a tie as half a win. The positive call EV is **conditional on the modeled opponent range and the simplified one-call calculation**: it is not a guaranteed outcome or a complete poker strategy.

<details>
<summary>Reproduce these exact numbers in Python</summary>

```python
from poker_engine import (
    GameState,
    cards_from_text,
    monte_carlo_equity,
    make_call_decision,
)

game = GameState(
    cards_from_text("Ah", "Kh"),
    opponent_profile="balanced",
    seed=42,
)
game.add_flop(cards_from_text("Qh", "Jh", "4c"))

result = monte_carlo_equity(game, 2_000)
decision = make_call_decision(result, pot_before_bet=100, opponent_bet=25)

print(f"Equity: {result['equity']:.2%}")
print(f"95% MC interval: {result['ci_low']:.2%}–{result['ci_high']:.2%}")
print(f"Break-even equity: {decision['required_equity']:.2%}")
print(f"EV(call): {decision['call_ev']:+.2f}")
```

This example calls the underlying analytical functions directly. A full interactive session also evaluates earlier streets and forward scenarios, so its sequence of random draws — and therefore its Monte Carlo estimates — can differ.

</details>

## Quick start

**Requirements:** Python 3; no third-party packages are needed.

```bash
git clone https://github.com/irrerafrancesco/dynamic-poker-decision-engine.git
cd dynamic-poker-decision-engine
python3 poker_engine.py
```

The terminal prompts for your hole cards, community cards, pot values, observed opponent actions, and your own fold/call choices. Card notation uses the rank followed by the suit: `Ah` is the ace of hearts, `10s` is the ten of spades, and `Qc` is the queen of clubs.

To choose a different opponent model and control simulation settings:

```bash
python3 poker_engine.py \
  --profile balanced \
  --simulations 20000 \
  --branch-simulations 750 \
  --seed 42
```

Opponent profiles: `random`, `loose`, `balanced`, `tight`. The non-random profiles assign **heuristic weights**, not learned or empirically calibrated opponent probabilities.

For all command-line options, use `python3 poker_engine.py --help`.

## How the mathematics works

### 1. Representing incomplete information

At each stage (*preflop → flop → turn → river*), known cards are removed from the possible opponent hands and future boards. With two private cards known, the opponent could hold **1,225** two-card combinations; on the river, with seven cards visible to the player, **990** remain.

A probability weight is assigned to each compatible opponent hand and normalized:

$$
p_i = \frac{w_i}{\sum_j w_j}.
$$

The effective number of opponent combinations, $N_{\mathrm{eff}}=1/\sum_i p_i^2$, describes the concentration of that distribution.

### 2. Estimating equity

Before the river, the engine samples a legal opponent hand and the remaining community cards without replacement, evaluates the resulting showdown, and repeats the experiment.

$$
\widehat{\mathrm{Equity}} = \widehat{P}(\mathrm{win}) + \tfrac12\widehat{P}(\mathrm{tie}).
$$

It estimates the Monte Carlo standard error from the simulated outcomes and reports a normal-approximation interval, $\widehat E \pm 1.96\,SE$. **That interval measures simulation noise, not uncertainty about whether the opponent model is correct.**

On the **river**, there are no future board cards to simulate. The engine enumerates all remaining weighted opponent hands and computes showdown equity exactly, conditional on the selected range.

### 3. Quantifying a call

Let $P$ be the pot *before* the opponent's bet and $C$ the amount required to call. Under the project's simplified showdown-value model:

$$
\mathrm{BreakEvenEquity}=\frac{C}{P+2C},
$$

$$
EV(\mathrm{call})=\mathrm{Equity}\cdot(P+2C)-C,
\qquad EV(\mathrm{fold})=0.
$$

The engine presents these values side by side; it does **not** choose an action automatically. Further betting rounds, folds induced by future bets, and strategic bet sizing are outside this EV formula.

### 4. Looking one card ahead

From the flop or turn, the engine constructs a **one-step forward scenario tree** over every legal next community card. Branch probabilities account for the weighted opponent range. Each branch has a resulting equity estimate.

The report includes the expected next-street equity, its dispersion, weighted quantiles, the probability of an equity decrease, and the probability of a decrease of at least ten percentage points. At the turn, each hypothetical river branch can be evaluated by exact enumeration.

These scenarios help distinguish **the current estimate** from **the range of outcomes still possible**.

## Reproducibility and tests

A fixed random seed makes equivalent simulation runs reproducible. The repository includes a separate test suite and internal mathematical checks:

```bash
python3 -m unittest -v test_poker_engine.py
python3 poker_engine.py --self-test
```

The tests cover card handling, hand rankings, opponent-range combinatorics, probability normalization, expected-value arithmetic, exact river calculations, and Monte Carlo reproducibility.

### Project structure

```text
dynamic-poker-decision-engine/
├── poker_engine.py       # Probability, equity, scenarios, EV, terminal interface
├── test_poker_engine.py  # Automated unit tests
├── README.md
├── LICENSE
└── .gitignore
```

## Scope and limitations

This is a **transparent quantitative model, not a production poker strategy**:

- It models a **single opponent** in heads-up Texas Hold'em.
- `loose`, `balanced`, and `tight` ranges are heuristic; betting actions do **not** update them through empirical or Bayesian inference.
- The simplified call EV does not account for future betting decisions, stack constraints, rake, fold equity, opponent adaptation, or optimal bet sizing.
- Monte Carlo intervals quantify **sampling error only**; exact river calculations are exact **under the assumed range**, not a claim of perfect knowledge.
- The engine is not a GTO solver and does not use machine learning or reinforcement learning.

## Why this project

The broader lesson is that a decision under incomplete information can be decomposed into a **state**, a **model of what is unknown**, a **distribution of possible outcomes**, and a **transparent comparison of alternatives**. Poker makes those steps visible, testable, and mathematically precise.

## License

See [LICENSE](LICENSE).
