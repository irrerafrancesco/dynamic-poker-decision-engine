# Dynamic Poker Decision Engine

A Python-based probabilistic decision-analysis engine for Texas Hold'em.

Poker is only the environment.

The project is really about a more general problem:

> **How does the distribution of possible outcomes change as new information becomes available, and how can that information be translated into a transparent quantitative analysis of the available decisions?**

The same structure appears in many problems involving uncertainty:

```text
state
  ↓
unknown variables
  ↓
possible future scenarios
  ↓
probability distribution
  ↓
expected outcomes
  ↓
decision analysis
```

This project uses Texas Hold'em as a compact laboratory for exploring:

```text
combinatorics
+ probability distributions
+ Monte Carlo simulation
+ uncertainty quantification
+ scenario analysis
+ expected value
```

The engine starts from the player's hole cards, updates its state as the flop, turn and river are revealed, estimates equity against a weighted opponent range, evaluates fold/call outcomes, and explores how future cards could change the current position.

It is not a GTO solver and it is not an automated poker-playing system.

Its purpose is to study **sequential decision making under incomplete information**.

---

## Why poker?

Texas Hold'em provides a useful probabilistic environment because information arrives progressively.

At the beginning of a hand, the player knows only two private cards.

Then information is revealed through:

```text
PREFLOP
   ↓
FLOP
   ↓
TURN
   ↓
RIVER
```

At every step:

- some uncertainty disappears;
- the set of possible opponent hands changes;
- the set of possible future boards changes;
- estimated equity changes;
- the economic value of a decision may change.

This makes poker a natural example of a dynamic stochastic system.

The project therefore focuses not only on estimating the probability of winning, but on understanding **how the entire probability landscape evolves as the state changes**.

---

# Main Ideas

## 1. Dynamic state representation

The engine maintains a complete representation of the currently known state:

- hero hole cards;
- community cards;
- remaining deck;
- current street;
- pot size;
- opponent range;
- previous states;
- user-selected actions.

Known cards are physically removed from the set of possible future states.

For example, once the hero's two cards are known, 50 cards remain unknown.

The opponent can therefore hold:

```text
C(50, 2) = 1,225
```

different two-card combinations.

After the flop, five cards are known:

```text
2 hero cards + 3 community cards
```

so 47 cards remain unknown and the opponent range contains:

```text
C(47, 2) = 1,081
```

possible combinations.

The number continues to shrink as new cards are revealed.

| Street | Unknown cards | Possible opponent hands |
|---|---:|---:|
| Preflop | 50 | 1,225 |
| Flop | 47 | 1,081 |
| Turn | 46 | 1,035 |
| River | 45 | 990 |

From the hero's initial information alone, there are also:

```text
C(50, 3) = 19,600
```

possible flops.

---

## 2. Weighted opponent ranges

The opponent is not represented by a single hidden hand.

Instead, the engine maintains a probability distribution over every compatible two-card combination.

If a possible opponent hand has weight `w_i`, its normalized probability is:

```text
P(hand_i) = w_i / sum(all weights)
```

The engine provides four configurable profiles:

| Profile | Description |
|---|---|
| `random` | Every compatible opponent hand has equal weight |
| `loose` | Mild heuristic preference for stronger starting hands |
| `balanced` | Moderate heuristic concentration on stronger hands |
| `tight` | Stronger heuristic concentration on stronger hands |

Example:

```bash
python3 poker_engine.py --profile balanced
```

The non-uniform profiles are deliberately simple heuristic models.

They are **not empirical population estimates and are not GTO ranges**.

Their purpose is to study how the results change when probability mass is distributed differently over the opponent's possible holdings.

---

## 3. Effective range size

A weighted range may contain 1,000 possible hands without behaving like a uniform distribution over 1,000 hands.

If most of the probability is concentrated on a smaller subset, the effective range is smaller.

The engine therefore computes:

```text
N_eff = 1 / sum(p_i²)
```

where `p_i` is the normalized probability of opponent hand `i`.

For a uniform range:

```text
N_eff = number of possible hands
```

For a concentrated range:

```text
N_eff < number of possible hands
```

This provides a compact measure of how concentrated the modeled opponent distribution is.

---

# Equity Engine

## 4. Hand evaluation

The project includes a complete Texas Hold'em hand evaluator.

Given seven available cards, the program evaluates every possible five-card subset:

```text
C(7, 5) = 21
```

and selects the strongest hand.

Supported categories include:

```text
Straight Flush
Four of a Kind
Full House
Flush
Straight
Three of a Kind
Two Pair
Pair
High Card
```

An Ace-high Straight Flush is displayed as a Royal Flush.

The evaluator also correctly handles the wheel straight:

```text
A - 2 - 3 - 4 - 5
```

---

## 5. Equity

Poker equity is treated as the expected share of the showdown.

The model uses:

```text
Equity = P(win) + 0.5 × P(tie)
```

A win contributes `1`.

A tie contributes `0.5`.

A loss contributes `0`.

Equity is therefore an expected-value quantity rather than simply a probability of winning.

---

## 6. Monte Carlo simulation

Before the river, many hidden-card configurations remain possible.

Instead of exhaustively evaluating all of them during an interactive session, the engine uses Monte Carlo simulation.

For every simulation:

```text
1. Sample an opponent hand from the weighted range
2. Remove those cards from the available deck
3. Sample the missing community cards
4. Build the complete board
5. Evaluate hero and opponent hands
6. Record win, tie or loss
```

After `N` simulations:

```text
Estimated equity
    =
(wins + 0.5 × ties) / N
```

The default is:

```text
10,000 simulations
```

but this can be changed:

```bash
python3 poker_engine.py --simulations 50000
```

---

# Quantifying Simulation Uncertainty

Monte Carlo produces an estimate, not an exact value.

The engine therefore also measures the uncertainty generated by the simulation itself.

Each simulated showdown is represented by:

```text
X = 1.0    if hero wins
X = 0.5    if the hand ties
X = 0.0    if hero loses
```

The equity estimate is the sample mean of `X`.

The Monte Carlo standard error is estimated as:

```text
SE = sample standard deviation / sqrt(N)
```

The program also reports an approximate 95% Monte Carlo interval:

```text
equity ± 1.96 × SE
```

For example:

```text
Equity: 67.12%
MC standard error: 0.46%
95% MC interval: [66.21%, 68.03%]
```

This interval represents **Monte Carlo sampling uncertainty**.

It does not measure uncertainty caused by incorrect assumptions about the opponent model.

Increasing the number of simulations reduces Monte Carlo noise approximately according to:

```text
SE ∝ 1 / sqrt(N)
```

---

# Exact River Enumeration

Once the river is known, no community cards remain unknown.

The remaining uncertainty consists only of the opponent's two hidden cards.

At this point:

```text
52 total cards
- 2 hero cards
- 5 board cards
= 45 unknown cards
```

so there are:

```text
C(45, 2) = 990
```

possible opponent combinations.

The engine therefore stops using Monte Carlo and evaluates the complete weighted opponent range exactly.

The computational strategy is:

```text
PREFLOP  → Monte Carlo
FLOP     → Monte Carlo
TURN     → Monte Carlo
RIVER    → Exact weighted enumeration
```

The river result consequently has no Monte Carlo sampling error.

---

# Decision Analysis

## 7. Pot odds and break-even equity

Suppose:

```text
P = pot before the opponent bet
C = opponent bet / cost of calling
```

If the player calls, the final pot becomes:

```text
Final pot = P + 2C
```

The break-even equity is:

```text
Break-even equity = C / (P + 2C)
```

This is the equity at which the prospective expected value of calling is zero.

The program displays both current equity and break-even equity.

---

## 8. Expected value

Folding is used as the prospective reference point:

```text
EV(fold) = 0
```

Calling has expected value:

```text
EV(call) = Equity × Final pot - Call cost
```

The engine also calculates:

```text
Equity edge
    =
Current equity - Break-even equity
```

and:

```text
ROI(call)
    =
EV(call) / Call cost
```

A typical output may therefore look like:

```text
---------- FOLD / CALL ANALYSIS ----------

Current equity:     67.12%
Break-even equity:  20.00%
Equity minus B/E:   +47.12%

EV(fold):           +0.00
EV(call):           +94.24
ROI(call):          +188.48%
```

The program deliberately **does not automatically select an action**.

It reports the quantities associated with the available alternatives and leaves the action to the user.

---

# Forward Scenario Analysis

This is one of the main features of the project.

The engine does not evaluate only the currently observed state.

After the flop or turn, it also asks:

> What could happen to the current equity distribution when the next card is revealed?

---

## 9. Flop → Turn tree

After the flop there are 47 unknown cards.

Each physically possible turn card becomes a new branch:

```text
CURRENT FLOP
│
├── Turn card 1
│   └── new equity
│
├── Turn card 2
│   └── new equity
│
├── Turn card 3
│   └── new equity
│
└── ...
```

Mathematically:

```text
Current state
    ↓
{possible next state 1,
 possible next state 2,
 ...
 possible next state n}
```

For every branch, the engine calculates the equity associated with that new state.

---

## 10. Turn → River tree

The same mechanism is applied after the turn.

Every possible river card creates a new state.

Because the board becomes complete at that point, each resulting river equity is computed using exact opponent-range enumeration rather than Monte Carlo.

---

# Scenario Probabilities

The next community card is not always treated as simply uniform over the visible remaining deck.

Why?

Because some cards may themselves be more likely to be hidden inside high-weight opponent hands.

The probability that a candidate card becomes the next community card therefore depends on the weighted opponent range.

The engine accounts for this interaction before normalizing the complete next-card distribution so that:

```text
sum(probability of every next-card branch) = 1
```

---

# Scenario Risk Metrics

For every possible next card, the engine obtains:

```text
branch probability
+
resulting equity
```

This creates a complete distribution of possible future equities.

The expected next-street equity is calculated as:

```text
Expected future equity
    =
sum(branch probability × branch equity)
```

The engine also measures the dispersion of future equity:

```text
Equity volatility
    =
sqrt(
    sum(
        branch probability
        ×
        (branch equity - expected equity)²
    )
)
```

Additional statistics include:

```text
5% downside-tail equity

Median future equity

95% upside-tail equity

Probability that the next card lowers equity

Probability that equity drops by at least 10 percentage points

Highest-equity next-card branches

Lowest-equity next-card branches
```

This turns the next community card from a single unknown event into a full distribution of possible future states.

---

# Actual Path and Counterfactual Information

The user chooses the action actually taken:

```text
fold
```

or:

```text
call
```

The selected action becomes part of the historical path.

For example:

```text
Hero A♥ K♥
│
└── FLOP Q♥ J♥ 4♣
    │
    ├── FOLD
    │   EV = 0
    │
    └── CALL
        EV = ...
        [SELECTED PATH]
        │
        └── TURN ...
```

If the user folds, the real hand stops.

However, the forward scenario analysis calculated at that decision point remains available.

The model can therefore preserve information about what uncertainty existed **before the path terminated**.

This distinction between:

```text
realized path
```

and:

```text
possible but unrealized states
```

is one of the central ideas of the project.

---

# Reproducibility

Monte Carlo simulation is stochastic.

For reproducibility, the engine uses an explicit random seed.

Default:

```text
42
```

It can be changed with:

```bash
python3 poker_engine.py --seed 123
```

Using the same:

```text
cards
opponent profile
number of simulations
seed
```

produces the same Monte Carlo sequence.

A reproducible run can therefore be started with:

```bash
python3 poker_engine.py \
    --profile balanced \
    --simulations 20000 \
    --branch-simulations 750 \
    --seed 42
```

---

# Example

One possible demonstration hand is:

```text
Hero: Ah Kh

Flop:
Qh Jh 4c

Pot:
100

Opponent action:
bet

Opponent bet:
20

Turn:
10h

River:
2c
```

The cards are entered using compact notation:

```text
Ah   = Ace of hearts
Kd   = King of diamonds
10s  = Ten of spades
Qc   = Queen of clubs
```

The terminal displays them using suit symbols:

```text
A♥ K♥
Q♥ J♥ 4♣
```

A typical equity section looks like:

```text
---------- EQUITY ----------

Method: Monte Carlo
Win:    ...
Tie:    ...
Loss:   ...
Equity: ...

Samples: 20,000
MC standard error: ...
95% MC interval: [...]
```

The decision-analysis section then reports:

```text
---------- FOLD / CALL ANALYSIS ----------

Current equity:     ...
Break-even equity:  ...
Equity minus B/E:   ...
Reward/Risk:        ...

EV(fold):           ...
EV(call):           ...
ROI(call):          ...
```

And the scenario engine may produce:

```text
---------- FORWARD SCENARIO ANALYSIS ----------

Possible next cards: 47

Expected next-street equity: ...
Equity volatility:          ...

5% downside equity:         ...
Median equity:              ...
95% upside equity:          ...

P(next card lowers equity): ...
P(drop >= 10 pp):           ...

BEST BRANCHES
...

WORST BRANCHES
...
```

A complete real terminal output from the final version will be added here.

---

# Running the Project

Clone the repository and enter the project directory.

Then run:

```bash
python3 poker_engine.py
```

Example with custom parameters:

```bash
python3 poker_engine.py \
    --profile balanced \
    --simulations 20000 \
    --branch-simulations 750 \
    --seed 42
```

Display the available command-line options:

```bash
python3 poker_engine.py --help
```

---

# Command-Line Parameters

```text
--profile
    random
    loose
    balanced
    tight

--simulations
    Number of Monte Carlo simulations for the current game state

--branch-simulations
    Number of Monte Carlo simulations for each hypothetical
    flop-to-turn branch

--seed
    Random seed for reproducibility

--self-test
    Run internal validation checks and exit
```

---

# Testing

The project includes both internal self-tests and an independent unit-test file.

Internal validation:

```bash
python3 poker_engine.py --self-test
```

Expected output:

```text
Running self-tests...
All self-tests passed.
```

External test suite:

```bash
python3 -m unittest -v test_poker_engine.py
```

The tests verify several core properties, including:

- deck size;
- card handling;
- Royal Flush recognition;
- wheel straight recognition;
- hand-ranking consistency;
- opponent-range combinatorics;
- probability normalization;
- effective range size;
- pot-odds calculations;
- expected-value calculations;
- Monte Carlo reproducibility;
- exact river probabilities;
- forward scenario construction.

---

# Project Structure

```text
dynamic-poker-decision-engine/
│
├── poker_engine.py
├── test_poker_engine.py
├── README.md
├── LICENSE
└── .gitignore
```

### `poker_engine.py`

Contains the complete analytical engine:

```text
card representation
deck management
opponent-range model
hand evaluator
Monte Carlo engine
exact river enumeration
uncertainty estimation
EV analysis
scenario tree
command-line interface
```

### `test_poker_engine.py`

Contains automated validation of the core mathematical and computational properties of the project.

---

# Dependencies

The project uses only the Python standard library.

No external Python packages are required.

```bash
python3 poker_engine.py
```

is sufficient to run the application.

---

# Assumptions and Limitations

The project is intentionally designed as an interpretable probabilistic model rather than a complete poker solver.

The current version does not model:

- GTO equilibrium strategies;
- empirical opponent populations;
- Bayesian range updates from betting behaviour;
- stack sizes;
- all-in constraints;
- player position;
- rake;
- fold equity;
- proactive betting;
- raise optimization;
- optimal bet sizing;
- multiple opponents;
- opponent databases;
- reinforcement learning.

The `loose`, `balanced` and `tight` profiles are heuristic probability-weighting mechanisms.

All numerical results should therefore be interpreted **conditional on the assumptions of the selected opponent model**.

---

# Possible Extensions

Future versions could investigate:

- Bayesian updating of opponent ranges;
- opponent-action likelihood models;
- stack-aware expected value;
- fold equity;
- bet-size optimization;
- position-dependent ranges;
- empirical calibration of opponent profiles;
- multiple opponents;
- historical hand databases;
- exact enumeration on earlier streets where computationally practical;
- comparison between heuristic and data-driven distributions.

These extensions are deliberately outside the current scope in order to keep the present model understandable and auditable.

---

# Final Perspective

The interesting part of the project is not poker itself.

The central structure is:

```text
incomplete information
        ↓
probability model
        ↓
new information arrives
        ↓
probability distribution changes
        ↓
future scenarios are generated
        ↓
risk and expected outcomes are measured
        ↓
available decisions can be compared quantitatively
```

Poker simply provides a clear environment in which all of these elements appear together.

The project is therefore best understood as a small experiment in:

**probability, simulation, uncertainty and sequential decision analysis.**

---

## License

Released under the MIT License.
