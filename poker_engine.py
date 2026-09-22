"""
Dynamic Poker Decision Engine
=============================

A small quantitative decision-analysis project built around heads-up
Texas Hold'em.

The program is intentionally designed as a probability/risk engine rather
than as a poker "advisor".  It tracks the public game state, estimates hero
equity against a weighted opponent range, compares the prospective value of
FOLD and CALL when facing a bet, and builds a one-step forward scenario tree
for every physically possible next community card.

The command-line output is deliberately descriptive:
    * probabilities and uncertainty are reported;
    * FOLD and CALL values are displayed side by side;
    * the user chooses the action;
    * the program does not print a prescriptive action recommendation.

The engine intentionally stops at the quantitative comparison. A commented
example in the decision section shows how one could derive a higher-EV label,
but that label is not computed or exposed during normal execution.

Main quantitative components
----------------------------
1. Combinatorial state-space reduction as cards become known.
2. Weighted opponent-hand distributions.
3. Monte Carlo equity estimation before the river.
4. Exact weighted enumeration on the river.
5. Standard error and an approximate 95% Monte Carlo interval.
6. Pot odds, break-even equity, expected value and ROI.
7. Forward scenario analysis over every possible next board card.
8. Counterfactual reporting after the user's chosen action.

The non-random opponent profiles are transparent heuristics.  They are not
empirical population estimates and they are not GTO ranges.
"""

import argparse
import math
import random
from collections import Counter
from enum import Enum
from itertools import combinations


# ============================================================
# CONFIGURATION
# ============================================================

DECK_SIZE = 52
HOLE_CARDS = 2
FLOP_CARDS = 3
MAX_BOARD_CARDS = 5

# Main equity estimate per observed street.
DEFAULT_SIMULATIONS = 10_000

# Per-branch simulations used only for hypothetical FLOP -> TURN scenarios.
# TURN -> RIVER branches are evaluated exactly because the board becomes complete.
DEFAULT_BRANCH_SIMULATIONS = 750

# A fixed seed makes demonstrations and tests reproducible.
DEFAULT_SEED = 42
DEFAULT_PROFILE = "random"

# 1.96 is the standard-normal multiplier for an approximate 95% interval.
CONFIDENCE_Z = 1.96

# To keep terminal output readable, only the extreme branches are printed.
# All branches are still analyzed internally.
BRANCHES_TO_SHOW = 5


# ============================================================
# FIXED ELEMENTS OF TEXAS HOLD'EM
# ============================================================

class Suit(Enum):
    HEARTS = ("h", "♥")
    DIAMONDS = ("d", "♦")
    CLUBS = ("c", "♣")
    SPADES = ("s", "♠")

    def __init__(self, code, symbol):
        self.code = code
        self.symbol = symbol


class Rank(Enum):
    TWO = ("2", 2)
    THREE = ("3", 3)
    FOUR = ("4", 4)
    FIVE = ("5", 5)
    SIX = ("6", 6)
    SEVEN = ("7", 7)
    EIGHT = ("8", 8)
    NINE = ("9", 9)
    TEN = ("10", 10)
    JACK = ("J", 11)
    QUEEN = ("Q", 12)
    KING = ("K", 13)
    ACE = ("A", 14)

    def __init__(self, symbol, value):
        self.symbol = symbol
        self.value_strength = value


class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


HAND_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "Pair",
    0: "High Card",
}


OPPONENT_PROFILES = {
    "random": "Uniform range: every physically possible hand has equal weight.",
    "loose": "Heuristic range: mild preference for stronger preflop holdings.",
    "balanced": "Heuristic range: moderate preference for stronger preflop holdings.",
    "tight": "Heuristic range: strong concentration on stronger preflop holdings.",
}


# ============================================================
# CARD
# ============================================================

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @classmethod
    def from_text(cls, text):
        text = text.strip()

        if len(text) < 2:
            raise ValueError(f"Invalid card: {text}")

        rank_code = text[:-1].upper()
        suit_code = text[-1].lower()

        rank = next(
            (rank for rank in Rank if rank.symbol == rank_code),
            None,
        )
        suit = next(
            (suit for suit in Suit if suit.code == suit_code),
            None,
        )

        if rank is None or suit is None:
            raise ValueError(f"Invalid card: {text}")

        return cls(rank, suit)

    def __str__(self):
        return f"{self.rank.symbol}{self.suit.symbol}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit


# ============================================================
# DECK
# ============================================================

class Deck:
    def __init__(self):
        self.cards = [
            Card(rank, suit)
            for rank in Rank
            for suit in Suit
        ]

    def __len__(self):
        return len(self.cards)

    def remove(self, card):
        self.cards.remove(card)

    def remove_cards(self, cards):
        for card in cards:
            self.remove(card)


# ============================================================
# OPPONENT RANGE
# ============================================================

class OpponentRange:
    """
    Weighted probability model for the opponent's two hidden cards.

    If k public/hero cards are known, every two-card combination that does not
    contain one of those cards remains physically possible.  The class stores
    those combinations together with a non-negative weight.

    For the ``random`` profile all combinations have equal weight.  The other
    profiles reshape the distribution using a transparent preflop-strength
    heuristic.  After normalization,

        p_i = w_i / sum_j w_j

    is the probability assigned to opponent hand i.

    These profiles are deliberately illustrative: they are useful for
    sensitivity analysis, but they are not empirical population estimates
    and they are not GTO ranges.
    """

    def __init__(self, available_cards, profile=DEFAULT_PROFILE):
        if profile not in OPPONENT_PROFILES:
            raise ValueError(f"Unknown opponent profile: {profile}")

        self.profile = profile
        self.hands = []

        for card1, card2 in combinations(available_cards, HOLE_CARDS):
            cards = (card1, card2)
            self.hands.append(
                {
                    "cards": cards,
                    "label": self.hand_label(cards),
                    "strength": self.preflop_strength(cards),
                    "weight": 1.0,
                }
            )

        self.apply_profile(profile)

    def __len__(self):
        return len(self.hands)

    @staticmethod
    def rank_notation(card):
        return "T" if card.rank == Rank.TEN else card.rank.symbol

    @classmethod
    def hand_label(cls, cards):
        card1, card2 = cards
        ordered = sorted(
            (card1, card2),
            key=lambda card: card.rank.value_strength,
            reverse=True,
        )
        high, low = ordered
        high_symbol = cls.rank_notation(high)
        low_symbol = cls.rank_notation(low)

        if high.rank == low.rank:
            return f"{high_symbol}{low_symbol}"

        suited_suffix = "s" if high.suit == low.suit else "o"
        return f"{high_symbol}{low_symbol}{suited_suffix}"

    @staticmethod
    def preflop_strength(cards):
        """
        Return a transparent heuristic strength score in [0, 1].

        The score rewards features that are commonly associated with stronger
        preflop holdings: high ranks, pairs, suitedness, connectedness and
        broadway structure.  Its only purpose is to shape the optional
        opponent profiles.  It must not be interpreted as exact hand equity.
        """
        card1, card2 = cards
        high, low = sorted(
            (card1.rank.value_strength, card2.rank.value_strength),
            reverse=True,
        )

        pair = high == low
        suited = card1.suit == card2.suit
        gap = max(0, high - low - 1)

        high_component = (high - 2) / 12
        low_component = (low - 2) / 12

        score = 0.42 * high_component + 0.23 * low_component

        if pair:
            score += 0.28 + 0.07 * high_component
        if suited:
            score += 0.08

        if not pair:
            if gap == 0:
                score += 0.10
            elif gap == 1:
                score += 0.07
            elif gap == 2:
                score += 0.04
            elif gap >= 4:
                score -= 0.05

        if high >= 11 and low >= 10:
            score += 0.08
        if high == 14:
            score += 0.05

        return min(1.0, max(0.01, score))

    @staticmethod
    def profile_weight(strength, profile):
        if profile == "random":
            return 1.0
        if profile == "loose":
            return 0.35 + 0.65 * strength
        if profile == "balanced":
            return 0.08 + 0.92 * (strength ** 2)
        if profile == "tight":
            return 0.01 + 0.99 * (strength ** 4)
        raise ValueError(f"Unknown opponent profile: {profile}")

    def apply_profile(self, profile):
        if profile not in OPPONENT_PROFILES:
            raise ValueError(f"Unknown opponent profile: {profile}")

        self.profile = profile
        for hand in self.hands:
            hand["weight"] = self.profile_weight(
                hand["strength"],
                profile,
            )

    def remove_known_cards(self, known_cards):
        self.hands = [
            hand
            for hand in self.hands
            if not any(card in known_cards for card in hand["cards"])
        ]

    def sample_hand(self, rng):
        if not self.hands:
            raise ValueError("Opponent range is empty.")

        weights = [hand["weight"] for hand in self.hands]
        if sum(weights) <= 0:
            raise ValueError("Opponent range weights must have a positive total.")

        selected = rng.choices(
            self.hands,
            weights=weights,
            k=1,
        )[0]

        return list(selected["cards"])

    def effective_size(self):
        """
        Return the effective number of opponent combinations.

        Let p_i denote normalized hand probabilities.  The concentration
        measure

            N_eff = 1 / sum_i p_i^2

        equals the raw number of combinations under a uniform distribution
        and decreases as probability mass becomes concentrated on fewer hands.
        """
        if not self.hands:
            return 0.0

        weights = [hand["weight"] for hand in self.hands]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        return 1.0 / sum(probability ** 2 for probability in probabilities)

    def next_board_card_probability(self, card, unknown_card_count):
        """
        Return P(next board card = ``card``) under the weighted hidden range.

        The opponent's two cards are hidden, so a candidate board card is not
        available in scenarios where the opponent already holds it.  We first
        sum the probability mass of opponent hands that leave the card
        available; conditional on such a hand, the next community card is
        uniform over the remaining unknown cards.

        This gives a properly weighted next-card distribution instead of
        assuming that every visible candidate card is automatically equally
        likely under a non-uniform opponent range.
        """
        if unknown_card_count <= HOLE_CARDS:
            return 0.0

        total_weight = sum(hand["weight"] for hand in self.hands)
        eligible_weight = sum(
            hand["weight"]
            for hand in self.hands
            if card not in hand["cards"]
        )

        return (
            eligible_weight / total_weight
        ) / (unknown_card_count - HOLE_CARDS)


# ============================================================
# GAME STATE
# ============================================================

class GameState:
    """
    Mutable public state of one heads-up Texas Hold'em hand.

    The object owns the hero cards, public board, remaining deck, pot,
    opponent range and random-number generator.  Whenever a board card becomes
    known it is removed from both the physical deck and the opponent's
    admissible hidden-hand combinations.  This keeps all later calculations
    consistent with the information currently available.
    """

    def __init__(self, hero_cards, opponent_profile=DEFAULT_PROFILE, seed=DEFAULT_SEED):
        if len(hero_cards) != HOLE_CARDS:
            raise ValueError("Texas Hold'em requires exactly 2 hole cards.")

        if hero_cards[0] == hero_cards[1]:
            raise ValueError("The two hole cards cannot be identical.")

        self.hero_cards = list(hero_cards)
        self.board = []
        self.street = Street.PREFLOP
        self.pot = 0.0
        self.history = []
        self.seed = seed
        self.rng = random.Random(seed)

        self.deck = Deck()
        self.deck.remove_cards(self.hero_cards)

        self.opponent_range = OpponentRange(
            self.deck.cards,
            profile=opponent_profile,
        )

    def _add_board_cards(self, cards):
        card_keys = [(card.rank, card.suit) for card in cards]

        if len(card_keys) != len(set(card_keys)):
            raise ValueError("Duplicate cards are not allowed.")

        for card in cards:
            if card not in self.deck.cards:
                raise ValueError(f"{card} is already known or unavailable.")

        self.board.extend(cards)
        self.deck.remove_cards(cards)
        self.opponent_range.remove_known_cards(cards)

    def add_flop(self, cards):
        if self.street != Street.PREFLOP:
            raise ValueError("The flop can only be added after preflop.")
        if len(cards) != FLOP_CARDS:
            raise ValueError("The flop must contain exactly 3 cards.")

        self._add_board_cards(cards)
        self.street = Street.FLOP

    def add_turn(self, card):
        if self.street != Street.FLOP:
            raise ValueError("The turn can only be added after the flop.")

        self._add_board_cards([card])
        self.street = Street.TURN

    def add_river(self, card):
        if self.street != Street.TURN:
            raise ValueError("The river can only be added after the turn.")

        self._add_board_cards([card])
        self.street = Street.RIVER


# ============================================================
# HAND EVALUATOR
# ============================================================

def get_straight_high(values):
    """
    Return the high-card value of a five-card straight, or ``None``.

    Ace is normally high (14), except in the wheel A-2-3-4-5 where it acts as
    a low card and the straight is therefore scored as five-high.
    """
    unique_values = sorted(set(values), reverse=True)

    if unique_values == [14, 5, 4, 3, 2]:
        return 5

    if len(unique_values) != 5:
        return None

    if unique_values[0] - unique_values[4] == 4:
        return unique_values[0]

    return None


def evaluate_five_cards(cards):
    """
    Encode the strength of exactly five cards as a comparable tuple.

    The first tuple component is the hand category:

        8 straight flush, 7 quads, 6 full house, 5 flush,
        4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.

    Remaining tuple components are tie-break values ordered by importance.
    Python's lexicographic tuple comparison can therefore compare two hands
    directly and deterministically.
    """
    values = sorted(
        [card.rank.value_strength for card in cards],
        reverse=True,
    )
    suits = [card.suit for card in cards]
    counts = Counter(values)

    is_flush = len(set(suits)) == 1
    straight_high = get_straight_high(values)

    if is_flush and straight_high is not None:
        return (8, straight_high)

    four_values = [value for value, count in counts.items() if count == 4]
    if four_values:
        four = four_values[0]
        kicker = max(value for value in values if value != four)
        return (7, four, kicker)

    three_values = [value for value, count in counts.items() if count == 3]
    pair_values = [value for value, count in counts.items() if count == 2]

    if three_values and pair_values:
        return (6, max(three_values), max(pair_values))

    if is_flush:
        return (5, *values)

    if straight_high is not None:
        return (4, straight_high)

    if three_values:
        three = max(three_values)
        kickers = sorted(
            [value for value in values if value != three],
            reverse=True,
        )
        return (3, three, *kickers)

    if len(pair_values) == 2:
        pairs = sorted(pair_values, reverse=True)
        kicker = max(value for value in values if value not in pairs)
        return (2, pairs[0], pairs[1], kicker)

    if len(pair_values) == 1:
        pair = pair_values[0]
        kickers = sorted(
            [value for value in values if value != pair],
            reverse=True,
        )
        return (1, pair, *kickers)

    return (0, *values)


def evaluate_best_hand(cards):
    """
    Evaluate the strongest five-card subset available.

    A Texas Hold'em player can have up to seven available cards
    (two private + five community).  The evaluator checks every five-card
    subset and keeps the lexicographically largest score.  With seven cards
    this means only C(7, 5) = 21 subsets.
    """
    if len(cards) < 5:
        raise ValueError("At least 5 cards are required to evaluate a poker hand.")

    best_score = None
    best_cards = None

    for five_cards in combinations(cards, 5):
        score = evaluate_five_cards(five_cards)
        if best_score is None or score > best_score:
            best_score = score
            best_cards = five_cards

    return best_score, best_cards


def hand_name(score):
    if score == (8, 14):
        return "Royal Flush"
    return HAND_NAMES[score[0]]


def compare_hands(hero_cards, opponent_cards, board):
    """Compare hero and opponent at showdown on a complete five-card board."""
    if len(board) != MAX_BOARD_CARDS:
        raise ValueError("A complete board must contain exactly 5 cards.")

    hero_score, hero_best = evaluate_best_hand(hero_cards + board)
    opponent_score, opponent_best = evaluate_best_hand(opponent_cards + board)

    if hero_score > opponent_score:
        result = "hero"
    elif hero_score < opponent_score:
        result = "opponent"
    else:
        result = "tie"

    return result, hero_score, hero_best, opponent_score, opponent_best


# ============================================================
# EQUITY ENGINE
# ============================================================

def simulate_once(game):
    """
    Simulate one showdown from the current information set.

    Step 1: sample an opponent hand from the current weighted range.
    Step 2: remove those hidden cards from the future-card pool.
    Step 3: sample the missing community cards without replacement.
    Step 4: compare the completed seven-card holdings at showdown.
    """
    missing_board_cards = MAX_BOARD_CARDS - len(game.board)

    opponent_cards = game.opponent_range.sample_hand(game.rng)

    available_future_cards = [
        card
        for card in game.deck.cards
        if card not in opponent_cards
    ]

    future_board = game.rng.sample(
        available_future_cards,
        missing_board_cards,
    )

    complete_board = game.board + future_board

    result, _, _, _, _ = compare_hands(
        game.hero_cards,
        opponent_cards,
        complete_board,
    )

    return result


def monte_carlo_equity(game, simulations):
    """
    Estimate hero equity by Monte Carlo simulation.

    Each simulated payoff X is

        X = 1     for a win
        X = 1/2   for a tie
        X = 0     for a loss

    so the sample mean estimates

        equity = P(win) + 0.5 * P(tie).

    The sample variance of X is also used to estimate the Monte Carlo standard
    error.  The reported 95% interval is the usual normal approximation

        equity_hat +/- 1.96 * SE.

    This interval describes simulation noise only; it does not account for
    model uncertainty in the chosen opponent range.
    """
    if simulations <= 1:
        raise ValueError("The number of simulations must be greater than 1.")

    wins = 0
    ties = 0
    losses = 0

    for _ in range(simulations):
        result = simulate_once(game)

        if result == "hero":
            wins += 1
        elif result == "tie":
            ties += 1
        else:
            losses += 1

    win_probability = wins / simulations
    tie_probability = ties / simulations
    loss_probability = losses / simulations
    # Expected showdown share: a tie contributes half of the pot share.
    equity = win_probability + 0.5 * tie_probability

    # Since X^2 is 1 for a win, 0.25 for a tie and 0 for a loss,
    # the second empirical moment is available without storing simulations.
    sum_x2 = wins + 0.25 * ties
    sample_variance = (
        sum_x2 - simulations * (equity ** 2)
    ) / (simulations - 1)
    sample_variance = max(0.0, sample_variance)
    standard_error = math.sqrt(sample_variance / simulations)

    ci_low = max(0.0, equity - CONFIDENCE_Z * standard_error)
    ci_high = min(1.0, equity + CONFIDENCE_Z * standard_error)

    return {
        "method": "Monte Carlo",
        "samples": simulations,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_probability": win_probability,
        "tie_probability": tie_probability,
        "loss_probability": loss_probability,
        "equity": equity,
        "standard_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def exact_river_equity(game):
    """
    Compute weighted river equity exactly.

    Once all five community cards are known there are no future board cards to
    simulate.  The only uncertainty is the opponent's hidden two-card hand, so
    every admissible opponent combination can be enumerated directly.

    The result therefore has zero Monte Carlo sampling error.  It can still
    depend on the modeling assumption encoded by the opponent-range weights.
    """
    if game.street != Street.RIVER:
        raise ValueError("Exact river equity is only available on the river.")

    total_weight = 0.0
    win_weight = 0.0
    tie_weight = 0.0
    loss_weight = 0.0

    for hand in game.opponent_range.hands:
        weight = hand["weight"]
        opponent_cards = list(hand["cards"])

        result, _, _, _, _ = compare_hands(
            game.hero_cards,
            opponent_cards,
            game.board,
        )

        total_weight += weight
        if result == "hero":
            win_weight += weight
        elif result == "tie":
            tie_weight += weight
        else:
            loss_weight += weight

    if total_weight <= 0:
        raise ValueError("Opponent range has zero total weight.")

    win_probability = win_weight / total_weight
    tie_probability = tie_weight / total_weight
    loss_probability = loss_weight / total_weight
    equity = win_probability + 0.5 * tie_probability

    return {
        "method": "Exact enumeration",
        "samples": len(game.opponent_range),
        "wins": None,
        "ties": None,
        "losses": None,
        "win_probability": win_probability,
        "tie_probability": tie_probability,
        "loss_probability": loss_probability,
        "equity": equity,
        "standard_error": 0.0,
        "ci_low": equity,
        "ci_high": equity,
    }


def estimate_equity(game, simulations=DEFAULT_SIMULATIONS):
    """
    Select the appropriate equity method for the current information state.

    Earlier streets use Monte Carlo because unknown future community cards
    enlarge the state space.  The river switches to exact enumeration.
    """
    if game.street == Street.RIVER:
        return exact_river_equity(game)
    return monte_carlo_equity(game, simulations)


# ============================================================
# DECISION ENGINE
# ============================================================

def make_check_decision():
    """
    Represent the only no-cost continuation modeled after an opponent check.

    V1 does not model proactive BET/RAISE decisions, so CHECK is stored as the
    observed continuation rather than as a strategic recommendation.
    """
    return {
        "opponent_action": "check",
        "required_equity": None,
        "fold_ev": None,
        "call_ev": None,
        "equity_edge": None,
        "roi_on_call": None,
        "reward_to_risk": None,
        "decision_confidence": "No call decision required.",
    }


def make_call_decision(equity_results, pot_before_bet, opponent_bet):
    """
    Compute the quantitative FOLD/CALL comparison after an opponent bet.

    If C is the call cost and P_final is the pot after hero calls,

        required_equity = C / P_final
        EV(call) = equity * P_final - C
        EV(fold) = 0

    The function intentionally returns the quantitative ingredients only.
    A higher-EV action can be derived from the two EV values, but the runtime
    interface does not compute or print that label. This keeps the program
    descriptive rather than prescriptive.
    """
    if pot_before_bet < 0:
        raise ValueError("Pot cannot be negative.")
    if opponent_bet <= 0:
        raise ValueError("Opponent bet must be positive.")

    equity = equity_results["equity"]
    call_cost = opponent_bet
    pot_before_call = pot_before_bet + opponent_bet
    final_pot = pot_before_call + call_cost

    required_equity = call_cost / final_pot
    call_ev = equity * final_pot - call_cost
    fold_ev = 0.0
    equity_edge = equity - required_equity
    roi_on_call = call_ev / call_cost
    reward_to_risk = pot_before_call / call_cost

    # Optional extension, deliberately NOT executed in this project:
    #
    # higher_ev_action = Action.CALL if call_ev > fold_ev else Action.FOLD
    #
    # The terminal reports both EVs and leaves the action choice to the user.

    ci_low = equity_results["ci_low"]
    ci_high = equity_results["ci_high"]

    if equity_results["method"] == "Exact enumeration":
        decision_confidence = "Exact river calculation: no Monte Carlo sampling error."
    elif ci_low > required_equity:
        decision_confidence = "Call EV remains positive across the 95% Monte Carlo interval."
    elif ci_high < required_equity:
        decision_confidence = "Call EV remains negative across the 95% Monte Carlo interval."
    else:
        decision_confidence = "The 95% Monte Carlo interval crosses break-even equity."

    return {
        "opponent_action": "bet",
        "call_cost": call_cost,
        "pot_before_call": pot_before_call,
        "final_pot": final_pot,
        "required_equity": required_equity,
        "fold_ev": fold_ev,
        "call_ev": call_ev,
        "equity_edge": equity_edge,
        "roi_on_call": roi_on_call,
        "reward_to_risk": reward_to_risk,
        "decision_confidence": decision_confidence,
    }


# ============================================================
# FORWARD SCENARIO TREE
# ============================================================

def clone_game_state(game, seed):
    """Rebuild the current public state without mutating the live hand."""
    clone = GameState(
        game.hero_cards,
        opponent_profile=game.opponent_range.profile,
        seed=seed,
    )

    if len(game.board) >= 3:
        clone.add_flop(list(game.board[:3]))
    if len(game.board) >= 4:
        clone.add_turn(game.board[3])
    if len(game.board) >= 5:
        clone.add_river(game.board[4])

    clone.pot = game.pot
    return clone


def scenario_seed(base_seed, card):
    """
    Derive a deterministic seed for one hypothetical next-card branch.

    Giving each branch a stable seed makes repeated runs reproducible while
    preventing every branch from consuming the exact same random stream.
    """
    suit_index = list(Suit).index(card.suit) + 1
    return base_seed * 1000 + card.rank.value_strength * 10 + suit_index


def weighted_quantile(branches, q):
    """
    Compute a probability-weighted quantile of next-street equity.

    Branches are sorted by equity and accumulated by their next-card
    probabilities until cumulative mass reaches q.
    """
    if not branches:
        return None

    ordered = sorted(branches, key=lambda branch: branch["equity"])
    cumulative = 0.0

    for branch in ordered:
        cumulative += branch["probability"]
        if cumulative >= q:
            return branch["equity"]

    return ordered[-1]["equity"]


def analyze_next_card_scenarios(
    game,
    current_equity,
    branch_simulations=DEFAULT_BRANCH_SIMULATIONS,
):
    """
    Build a one-step forward scenario tree from the current public state.

    FLOP -> every physically possible TURN card becomes one branch.  For each
    branch, hero equity is estimated while the river is still unknown.

    TURN -> every physically possible RIVER card becomes one branch.  Because
    the board is then complete, each branch uses exact opponent-hand
    enumeration.

    Each branch stores:
        * probability of that next card,
        * resulting hero equity,
        * resulting made hand.

    The branch distribution is then summarized by a probability-weighted mean,
    standard deviation, quantiles and downside probabilities.  This is a
    scenario/risk report, not a prescribed poker strategy.
    """
    if game.street not in {Street.FLOP, Street.TURN}:
        return None

    branches = []
    unknown_card_count = len(game.deck)

    for card in list(game.deck.cards):
        probability = game.opponent_range.next_board_card_probability(
            card,
            unknown_card_count,
        )

        branch_game = clone_game_state(
            game,
            seed=scenario_seed(game.seed, card),
        )

        if game.street == Street.FLOP:
            branch_game.add_turn(card)
            result = monte_carlo_equity(
                branch_game,
                branch_simulations,
            )
            next_street = Street.TURN
        else:
            branch_game.add_river(card)
            result = exact_river_equity(branch_game)
            next_street = Street.RIVER

        branches.append(
            {
                "card": card,
                "probability": probability,
                "equity": result["equity"],
                "method": result["method"],
                "current_hand": current_hand_name(branch_game),
            }
        )

    probability_sum = sum(branch["probability"] for branch in branches)
    if probability_sum > 0:
        for branch in branches:
            branch["probability"] /= probability_sum

    # Probability-weighted expectation of next-street equity.
    mean_equity = sum(
        branch["probability"] * branch["equity"]
        for branch in branches
    )

    # Dispersion across possible next cards: a simple state-risk measure.
    variance = sum(
        branch["probability"] * ((branch["equity"] - mean_equity) ** 2)
        for branch in branches
    )

    # Probability that the next public card reduces equity relative to now.
    downside_probability = sum(
        branch["probability"]
        for branch in branches
        if branch["equity"] < current_equity
    )

    # Probability of an equity drawdown of at least ten percentage points.
    severe_drop_probability = sum(
        branch["probability"]
        for branch in branches
        if branch["equity"] <= current_equity - 0.10
    )

    q05 = weighted_quantile(branches, 0.05)
    q50 = weighted_quantile(branches, 0.50)
    q95 = weighted_quantile(branches, 0.95)

    best = sorted(
        branches,
        key=lambda branch: branch["equity"],
        reverse=True,
    )[:BRANCHES_TO_SHOW]

    worst = sorted(
        branches,
        key=lambda branch: branch["equity"],
    )[:BRANCHES_TO_SHOW]

    return {
        "from_street": game.street,
        "next_street": next_street,
        "branch_count": len(branches),
        "probability_sum": sum(branch["probability"] for branch in branches),
        "mean_equity": mean_equity,
        "equity_std": math.sqrt(max(0.0, variance)),
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "downside_probability": downside_probability,
        "severe_drop_probability": severe_drop_probability,
        "best": best,
        "worst": worst,
        "branches": branches,
    }


# ============================================================
# HISTORY AND TREE DISPLAY
# ============================================================

def cards_to_text(cards):
    return " ".join(str(card) for card in cards) if cards else "-"


def current_hand_name(game):
    known_cards = game.hero_cards + game.board
    if len(known_cards) < 5:
        return "-"

    score, _ = evaluate_best_hand(known_cards)
    return hand_name(score)


def save_history(
    game,
    equity_results,
    decision_results=None,
    opponent_bet=None,
    actual_action=None,
    scenario_report=None,
):
    snapshot = {
        "street": game.street,
        "board": list(game.board),
        "current_hand": current_hand_name(game),
        "opponent_profile": game.opponent_range.profile,
        "opponent_combos": len(game.opponent_range),
        "effective_range": game.opponent_range.effective_size(),
        "method": equity_results["method"],
        "samples": equity_results["samples"],
        "win_probability": equity_results["win_probability"],
        "tie_probability": equity_results["tie_probability"],
        "loss_probability": equity_results["loss_probability"],
        "equity": equity_results["equity"],
        "standard_error": equity_results["standard_error"],
        "ci_low": equity_results["ci_low"],
        "ci_high": equity_results["ci_high"],
        "pot_before_action": game.pot,
        "opponent_action": (
            decision_results["opponent_action"]
            if decision_results is not None
            else None
        ),
        "opponent_bet": opponent_bet,
        "actual_action": actual_action,
        "required_equity": (
            decision_results["required_equity"]
            if decision_results is not None
            else None
        ),
        "fold_ev": (
            decision_results["fold_ev"]
            if decision_results is not None
            else None
        ),
        "call_ev": (
            decision_results["call_ev"]
            if decision_results is not None
            else None
        ),
        "equity_edge": (
            decision_results["equity_edge"]
            if decision_results is not None
            else None
        ),
        "roi_on_call": (
            decision_results["roi_on_call"]
            if decision_results is not None
            else None
        ),
        "reward_to_risk": (
            decision_results["reward_to_risk"]
            if decision_results is not None
            else None
        ),
        "decision_confidence": (
            decision_results["decision_confidence"]
            if decision_results is not None
            else None
        ),
        "scenario_report": scenario_report,
    }

    game.history.append(snapshot)


def print_equity_report(equity_results):
    print()
    print("---------- EQUITY ----------")
    print(f"Method: {equity_results['method']}")
    print(f"Win:    {equity_results['win_probability']:.2%}")
    print(f"Tie:    {equity_results['tie_probability']:.2%}")
    print(f"Loss:   {equity_results['loss_probability']:.2%}")
    print(f"Equity: {equity_results['equity']:.2%}")

    if equity_results["method"] == "Monte Carlo":
        print(f"Samples: {equity_results['samples']:,}")
        print(f"MC standard error: {equity_results['standard_error']:.4%}")
        print(
            "95% MC interval: "
            f"[{equity_results['ci_low']:.2%}, "
            f"{equity_results['ci_high']:.2%}]"
        )
    else:
        print(f"Enumerated opponent combos: {equity_results['samples']:,}")
        print("Sampling error: 0 (exact river enumeration)")


def print_scenario_report(report, indent="    "):
    if report is None:
        return

    print(f"{indent}└── IF WE CONTINUE: {report['branch_count']} possible {report['next_street'].value.upper()} cards")
    print(f"{indent}    ├── Expected next-street equity: {report['mean_equity']:.2%}")
    print(f"{indent}    ├── Equity volatility across cards: {report['equity_std']:.2%}")
    print(f"{indent}    ├── 5% downside-tail equity: {report['q05']:.2%}")
    print(f"{indent}    ├── Median next-card equity: {report['q50']:.2%}")
    print(f"{indent}    ├── 95% upside-tail equity: {report['q95']:.2%}")
    print(f"{indent}    ├── P(next card lowers equity): {report['downside_probability']:.2%}")
    print(f"{indent}    └── P(equity drops by >= 10 pp): {report['severe_drop_probability']:.2%}")

    print(f"{indent}        ├── HIGHEST-EQUITY BRANCHES")
    for branch in report["best"]:
        print(
            f"{indent}        │   ├── {branch['card']}  "
            f"p={branch['probability']:.2%}  "
            f"equity={branch['equity']:.2%}  "
            f"hand={branch['current_hand']}"
        )

    print(f"{indent}        └── LOWEST-EQUITY BRANCHES")
    for branch in report["worst"]:
        print(
            f"{indent}            ├── {branch['card']}  "
            f"p={branch['probability']:.2%}  "
            f"equity={branch['equity']:.2%}  "
            f"hand={branch['current_hand']}"
        )


def print_decision_tree(game):
    """
    Print the observed path plus counterfactual FOLD/CALL branches.

    The tree deliberately avoids labels such as "recommended".  For every bet
    it reports the same quantitative information for both actions and marks
    only the path actually selected by the user.
    """
    print()
    print("=" * 84)
    print("                         PYTHON DECISION / SCENARIO TREE")
    print("=" * 84)
    print()

    print(f"ROOT  Hero: {cards_to_text(game.hero_cards)}")
    print(f"├── Initial unknown deck: {DECK_SIZE - HOLE_CARDS} cards")
    print(f"├── Initial opponent combinations: {math.comb(50, 2):,}")
    print(f"├── Opponent profile: {game.opponent_range.profile}")
    print(f"└── Possible flops from hero information: {math.comb(50, 3):,}")

    previous_equity = None

    for index, state in enumerate(game.history):
        prefix = "    " * (index + 1)
        street = state["street"].value.upper()

        print(f"{prefix}└── {street}  board=[{cards_to_text(state['board'])}]")
        print(f"{prefix}    ├── Equity: {state['equity']:.2%}")
        print(
            f"{prefix}    ├── W/T/L: "
            f"{state['win_probability']:.2%} / "
            f"{state['tie_probability']:.2%} / "
            f"{state['loss_probability']:.2%}"
        )
        print(f"{prefix}    ├── Opponent combos: {state['opponent_combos']:,}")
        print(f"{prefix}    ├── Effective weighted range: {state['effective_range']:.1f}")

        if state["current_hand"] != "-":
            print(f"{prefix}    ├── Current made hand: {state['current_hand']}")

        if previous_equity is not None:
            delta = state["equity"] - previous_equity
            print(f"{prefix}    ├── Δ Equity from previous street: {delta:+.2%}")

        if state["opponent_action"] == "check":
            print(f"{prefix}    └── Opponent CHECK")
            print(f"{prefix}        └── Recorded continuation: CHECK")

        elif state["opponent_action"] == "bet":
            actual = state["actual_action"]

            fold_tag = "  [SELECTED PATH]" if actual == Action.FOLD else ""
            call_tag = "  [SELECTED PATH]" if actual == Action.CALL else ""

            print(
                f"{prefix}    ├── Opponent BET {state['opponent_bet']:.2f} "
                f"| break-even={state['required_equity']:.2%} "
                f"| edge={state['equity_edge']:+.2%}"
            )
            print(f"{prefix}    ├── FOLD  EV={state['fold_ev']:+.2f}{fold_tag}")
            print(
                f"{prefix}    └── CALL  EV={state['call_ev']:+.2f} "
                f"ROI={state['roi_on_call']:+.2%}{call_tag}"
            )

            print_scenario_report(
                state["scenario_report"],
                indent=prefix + "        ",
            )

            if actual == Action.FOLD:
                print(f"{prefix}        × OBSERVED PATH STOPS HERE")
                print(
                    f"{prefix}        Counterfactual continuation above "
                    f"summarizes the remaining next-card risk."
                )
                break

        previous_equity = state["equity"]


def print_strategy_summary(game):
    """
    Print a neutral trace of the actions actually taken.

    If the user wants to compare alternatives, the tree exposes both FOLD and
    CALL EVs without turning that comparison into an instruction.
    """
    decisions = [
        state
        for state in game.history
        if state["opponent_action"] is not None
    ]

    if not decisions:
        return

    print()
    print("=" * 84)
    print("                                ACTION TRACE")
    print("=" * 84)

    for state in decisions:
        street = state["street"].value.upper()
        actual = state["actual_action"]
        actual_text = actual.value.upper() if actual is not None else "-"

        if actual == Action.CALL:
            selected_ev = state["call_ev"]
            alternative_ev = state["fold_ev"]
        elif actual == Action.FOLD:
            selected_ev = state["fold_ev"]
            alternative_ev = state["call_ev"]
        else:
            selected_ev = None
            alternative_ev = None

        print(
            f"{street:<8} action={actual_text:<5} "
            f"equity={state['equity']:.2%}"
        )

        if selected_ev is not None:
            print(
                f"         selected EV={selected_ev:+.2f} | "
                f"counterfactual EV={alternative_ev:+.2f}"
            )


# ============================================================
# USER INPUT
# ============================================================

def read_cards(prompt, expected_count):
    while True:
        try:
            texts = input(prompt).split()

            if len(texts) != expected_count:
                raise ValueError(
                    f"Please enter exactly {expected_count} card(s)."
                )

            cards = [Card.from_text(text) for text in texts]
            card_keys = [(card.rank, card.suit) for card in cards]

            if len(card_keys) != len(set(card_keys)):
                raise ValueError("You cannot enter the same card twice.")

            return cards

        except ValueError as error:
            print(f"Error: {error}")
            print("Example format: Ah Kh, Qd, 10s")
            print()


def read_non_negative_number(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a valid non-negative number.")


def read_positive_number(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a valid positive number.")


def read_opponent_action():
    while True:
        action = input("Opponent action [check/bet]: ").strip().lower()
        if action in {"check", "bet"}:
            return action
        print("Please enter 'check' or 'bet'.")


def read_actual_action():
    """
    Read the user's FOLD/CALL choice without supplying a default action.

    This is intentionally neutral: the terminal shows both alternatives and
    requires an explicit choice instead of accepting a model-selected default.
    """
    while True:
        value = input("Your action [fold/call]: ").strip().lower()

        if value == "fold":
            return Action.FOLD
        if value == "call":
            return Action.CALL

        print("Please enter 'fold' or 'call'.")


# ============================================================
# STREET ANALYSIS
# ============================================================

def analyze_current_street(
    game,
    simulations=DEFAULT_SIMULATIONS,
    branch_simulations=DEFAULT_BRANCH_SIMULATIONS,
):
    """
    Analyze one observed street and record the user's action.

    The screen reports estimates, uncertainty, action values and future-card
    scenario risk.  It never prints an instruction to FOLD or CALL.
    """
    print()
    print("Running equity engine...")

    equity_results = estimate_equity(
        game,
        simulations=simulations,
    )

    print_equity_report(equity_results)

    scenario_report = analyze_next_card_scenarios(
        game,
        current_equity=equity_results["equity"],
        branch_simulations=branch_simulations,
    )

    print()
    opponent_action = read_opponent_action()

    if opponent_action == "check":
        decision_results = make_check_decision()
        opponent_bet = None
        actual_action = Action.CHECK

        print()
        print("---------- ACTION STATE ----------")
        print("Opponent action: CHECK")
        print("Recorded continuation: CHECK")
        print("Scope note: proactive BET/RAISE actions are outside V1.")

        if scenario_report is not None:
            print()
            print("---------- FORWARD SCENARIO TREE ----------")
            print_scenario_report(scenario_report, indent="")

    else:
        opponent_bet = read_positive_number("Opponent bet: ")
        decision_results = make_call_decision(
            equity_results,
            game.pot,
            opponent_bet,
        )

        print()
        print("---------- FOLD / CALL ANALYSIS ----------")
        print(f"Current equity:     {equity_results['equity']:.2%}")
        print(f"Break-even equity:  {decision_results['required_equity']:.2%}")
        print(f"Equity minus B/E:   {decision_results['equity_edge']:+.2%}")
        print(f"Reward/Risk:        {decision_results['reward_to_risk']:.2f}:1")
        print(f"EV(fold):           {decision_results['fold_ev']:+.2f}")
        print(f"EV(call):           {decision_results['call_ev']:+.2f}")
        print(f"ROI(call):          {decision_results['roi_on_call']:+.2%}")
        print(f"EV robustness:      {decision_results['decision_confidence']}")

        if scenario_report is not None:
            print()
            print("---------- FORWARD SCENARIO TREE ----------")
            print_scenario_report(scenario_report, indent="")

        actual_action = read_actual_action()

    save_history(
        game,
        equity_results,
        decision_results,
        opponent_bet,
        actual_action=actual_action,
        scenario_report=scenario_report,
    )

    if actual_action == Action.CALL:
        game.pot = decision_results["final_pot"]

    print_decision_tree(game)

    return actual_action


# ============================================================
# SELF TESTS
# ============================================================

def cards_from_text(*texts):
    return [Card.from_text(text) for text in texts]


def run_self_tests():
    """
    Lightweight deterministic checks that can run without unittest.

    These assertions protect the mathematical invariants most likely to break:
    deck size, hand ranking, range combinatorics, probability normalization,
    EV arithmetic, exact river probabilities and reproducible simulation.
    """
    print("Running self-tests...")

    assert len(Deck()) == DECK_SIZE

    royal = cards_from_text("Ah", "Kh", "Qh", "Jh", "10h")
    royal_score = evaluate_five_cards(royal)
    assert royal_score == (8, 14)
    assert hand_name(royal_score) == "Royal Flush"

    wheel = cards_from_text("Ah", "2d", "3c", "4s", "5h")
    assert evaluate_five_cards(wheel) == (4, 5)

    full_house = cards_from_text("Ah", "Ad", "Ac", "7s", "7h")
    assert evaluate_five_cards(full_house) == (6, 14, 7)

    game = GameState(cards_from_text("Ah", "Kh"), opponent_profile="random", seed=1)
    assert len(game.opponent_range) == math.comb(50, 2) == 1225

    game.add_flop(cards_from_text("Qh", "Jh", "4c"))
    assert len(game.opponent_range) == math.comb(47, 2) == 1081

    probability_sum = sum(
        game.opponent_range.next_board_card_probability(card, len(game.deck))
        for card in game.deck.cards
    )
    assert abs(probability_sum - 1.0) < 1e-12

    game.add_turn(Card.from_text("8s"))
    assert len(game.opponent_range) == math.comb(46, 2) == 1035

    game.add_river(Card.from_text("2c"))
    assert len(game.opponent_range) == math.comb(45, 2) == 990

    assert abs(game.opponent_range.effective_size() - 990.0) < 1e-9

    equity_results = {
        "equity": 0.40,
        "ci_low": 0.39,
        "ci_high": 0.41,
        "method": "Monte Carlo",
    }
    decision = make_call_decision(equity_results, 100.0, 50.0)
    assert abs(decision["required_equity"] - 0.25) < 1e-12
    assert abs(decision["call_ev"] - 30.0) < 1e-12

    exact = exact_river_equity(game)
    probability_sum = (
        exact["win_probability"]
        + exact["tie_probability"]
        + exact["loss_probability"]
    )
    assert abs(probability_sum - 1.0) < 1e-12

    random_game = GameState(
        cards_from_text("As", "Kd"),
        opponent_profile="random",
        seed=7,
    )
    tight_game = GameState(
        cards_from_text("As", "Kd"),
        opponent_profile="tight",
        seed=7,
    )
    assert (
        tight_game.opponent_range.effective_size()
        < random_game.opponent_range.effective_size()
    )

    mc_game_1 = GameState(
        cards_from_text("As", "Ks"),
        opponent_profile="balanced",
        seed=123,
    )
    mc_game_2 = GameState(
        cards_from_text("As", "Ks"),
        opponent_profile="balanced",
        seed=123,
    )
    result_1 = monte_carlo_equity(mc_game_1, 250)
    result_2 = monte_carlo_equity(mc_game_2, 250)
    assert result_1["wins"] == result_2["wins"]
    assert result_1["ties"] == result_2["ties"]
    assert result_1["losses"] == result_2["losses"]

    turn_game = GameState(
        cards_from_text("Ah", "Kh"),
        opponent_profile="random",
        seed=9,
    )
    turn_game.add_flop(cards_from_text("Qh", "Jh", "4c"))
    turn_game.add_turn(Card.from_text("8s"))
    current = monte_carlo_equity(turn_game, 200)
    report = analyze_next_card_scenarios(
        turn_game,
        current_equity=current["equity"],
        branch_simulations=50,
    )
    assert report["branch_count"] == 46
    assert abs(report["probability_sum"] - 1.0) < 1e-12
    assert 0.0 <= report["q05"] <= 1.0
    assert 0.0 <= report["q95"] <= 1.0

    print("All self-tests passed.")


# ============================================================
# CLI
# ============================================================

def parse_arguments():
    """Parse reproducibility and simulation controls for the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic Poker Decision Engine: weighted opponent ranges, "
            "Monte Carlo equity, exact river enumeration, EV analysis and "
            "forward scenario trees."
        )
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"Monte Carlo simulations per street (default: {DEFAULT_SIMULATIONS}).",
    )
    parser.add_argument(
        "--branch-simulations",
        type=int,
        default=DEFAULT_BRANCH_SIMULATIONS,
        help=(
            "Monte Carlo simulations for each hypothetical next-card branch "
            f"on the flop (default: {DEFAULT_BRANCH_SIMULATIONS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(OPPONENT_PROFILES),
        default=DEFAULT_PROFILE,
        help=(
            "Opponent preflop range profile. Non-random profiles are heuristic "
            f"(default: {DEFAULT_PROFILE})."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic internal tests and exit.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run one interactive hand from preflop through river or until hero folds.

    The terminal is the complete interface: no HTML, browser, JSON or external
    visualization layer is required.
    """
    args = parse_arguments()

    if args.self_test:
        run_self_tests()
        return

    if args.simulations <= 1:
        raise ValueError("--simulations must be greater than 1.")
    if args.branch_simulations <= 1:
        raise ValueError("--branch-simulations must be greater than 1.")

    print()
    print("=" * 84)
    print("                         DYNAMIC POKER DECISION ENGINE")
    print("=" * 84)
    print()
    print(f"Monte Carlo simulations: {args.simulations:,}")
    print(f"Scenario simulations/branch: {args.branch_simulations:,}")
    print(f"Random seed: {args.seed}")
    print(f"Opponent profile: {args.profile}")
    print(f"Range model: {OPPONENT_PROFILES[args.profile]}")
    print()

    # HERO / PREFLOP
    hero_cards = read_cards(
        "Enter your cards (example: Ah Kh): ",
        HOLE_CARDS,
    )

    game = GameState(
        hero_cards,
        opponent_profile=args.profile,
        seed=args.seed,
    )

    print()
    print("---------- ROOT / PREFLOP ----------")
    print(f"Hero: {cards_to_text(game.hero_cards)}")
    print(f"Unknown cards: {len(game.deck)}")
    print(f"Opponent combinations: {len(game.opponent_range):,}")
    print(f"Effective weighted range: {game.opponent_range.effective_size():.1f}")
    print(f"Possible flops: {math.comb(50, 3):,}")

    preflop_results = estimate_equity(
        game,
        simulations=args.simulations,
    )
    print_equity_report(preflop_results)
    save_history(game, preflop_results)
    print_decision_tree(game)

    # FLOP
    print()
    while True:
        flop = read_cards(
            "Enter the flop (example: Qh Jh 4c): ",
            FLOP_CARDS,
        )
        try:
            game.add_flop(flop)
            break
        except ValueError as error:
            print(f"Error: {error}")
            print()

    print()
    print("---------- OBSERVED FLOP ----------")
    print(f"Hero:  {cards_to_text(game.hero_cards)}")
    print(f"Board: {cards_to_text(game.board)}")
    print(f"Unknown cards: {len(game.deck)}")
    print(f"Opponent combinations: {len(game.opponent_range):,}")

    game.pot = read_non_negative_number(
        "Current pot before opponent action: "
    )

    flop_action = analyze_current_street(
        game,
        simulations=args.simulations,
        branch_simulations=args.branch_simulations,
    )

    if flop_action == Action.FOLD:
        print("\nActual hand ended on the flop.")
        print_strategy_summary(game)
        return

    # TURN
    print()
    while True:
        turn = read_cards(
            "Enter the turn (example: 8s): ",
            1,
        )
        try:
            game.add_turn(turn[0])
            break
        except ValueError as error:
            print(f"Error: {error}")
            print()

    print()
    print("---------- OBSERVED TURN ----------")
    print(f"Hero:  {cards_to_text(game.hero_cards)}")
    print(f"Board: {cards_to_text(game.board)}")
    print(f"Current hand: {current_hand_name(game)}")

    turn_action = analyze_current_street(
        game,
        simulations=args.simulations,
        branch_simulations=args.branch_simulations,
    )

    if turn_action == Action.FOLD:
        print("\nActual hand ended on the turn.")
        print_strategy_summary(game)
        return

    # RIVER
    print()
    while True:
        river = read_cards(
            "Enter the river (example: 2c): ",
            1,
        )
        try:
            game.add_river(river[0])
            break
        except ValueError as error:
            print(f"Error: {error}")
            print()

    print()
    print("---------- OBSERVED RIVER ----------")
    print(f"Hero:  {cards_to_text(game.hero_cards)}")
    print(f"Board: {cards_to_text(game.board)}")
    print(f"Current hand: {current_hand_name(game)}")

    river_action = analyze_current_street(
        game,
        simulations=args.simulations,
        branch_simulations=args.branch_simulations,
    )

    if river_action == Action.FOLD:
        print("\nActual hand ended on the river.")
    else:
        print("\nDecision path completed through the river.")

    print_strategy_summary(game)


if __name__ == "__main__":
    main()
