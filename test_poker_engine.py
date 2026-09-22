"""
Unit tests for poker_engine.py.

The tests focus on mathematical and structural invariants rather than on one
specific interactive transcript.  Their purpose is to detect regressions in
hand ranking, combinatorics, probability normalization, EV arithmetic,
reproducibility and scenario-tree construction.
"""

import math
import unittest

import poker_engine as pe


class TestPokerEngine(unittest.TestCase):
    def test_deck_has_52_cards(self):
        self.assertEqual(len(pe.Deck()), 52)

    def test_royal_flush(self):
        cards = pe.cards_from_text("Ah", "Kh", "Qh", "Jh", "10h")
        score = pe.evaluate_five_cards(cards)
        self.assertEqual(score, (8, 14))
        self.assertEqual(pe.hand_name(score), "Royal Flush")

    def test_wheel_straight(self):
        cards = pe.cards_from_text("Ah", "2d", "3c", "4s", "5h")
        self.assertEqual(pe.evaluate_five_cards(cards), (4, 5))

    def test_full_house(self):
        cards = pe.cards_from_text("Ah", "Ad", "Ac", "7s", "7h")
        self.assertEqual(pe.evaluate_five_cards(cards), (6, 14, 7))

    def test_range_combinatorics(self):
        game = pe.GameState(pe.cards_from_text("Ah", "Kh"), seed=1)
        self.assertEqual(len(game.opponent_range), math.comb(50, 2))

        game.add_flop(pe.cards_from_text("Qh", "Jh", "4c"))
        self.assertEqual(len(game.opponent_range), math.comb(47, 2))

        game.add_turn(pe.Card.from_text("8s"))
        self.assertEqual(len(game.opponent_range), math.comb(46, 2))

        game.add_river(pe.Card.from_text("2c"))
        self.assertEqual(len(game.opponent_range), math.comb(45, 2))

    def test_random_profile_effective_size(self):
        game = pe.GameState(pe.cards_from_text("Ah", "Kh"), seed=1)
        self.assertAlmostEqual(
            game.opponent_range.effective_size(),
            len(game.opponent_range),
            places=9,
        )

    def test_tight_range_is_more_concentrated(self):
        random_game = pe.GameState(
            pe.cards_from_text("As", "Kd"),
            opponent_profile="random",
            seed=7,
        )
        tight_game = pe.GameState(
            pe.cards_from_text("As", "Kd"),
            opponent_profile="tight",
            seed=7,
        )
        self.assertLess(
            tight_game.opponent_range.effective_size(),
            random_game.opponent_range.effective_size(),
        )

    def test_next_card_probabilities_sum_to_one(self):
        game = pe.GameState(pe.cards_from_text("Ah", "Kh"), seed=1)
        game.add_flop(pe.cards_from_text("Qh", "Jh", "4c"))

        total = sum(
            game.opponent_range.next_board_card_probability(card, len(game.deck))
            for card in game.deck.cards
        )
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_call_ev_example(self):
        equity_results = {
            "equity": 0.40,
            "ci_low": 0.39,
            "ci_high": 0.41,
            "method": "Monte Carlo",
        }
        decision = pe.make_call_decision(equity_results, 100.0, 50.0)
        self.assertAlmostEqual(decision["required_equity"], 0.25)
        self.assertAlmostEqual(decision["call_ev"], 30.0)

    def test_exact_river_probabilities_sum_to_one(self):
        game = pe.GameState(pe.cards_from_text("Ah", "Kh"), seed=1)
        game.add_flop(pe.cards_from_text("Qh", "Jh", "4c"))
        game.add_turn(pe.Card.from_text("8s"))
        game.add_river(pe.Card.from_text("2c"))

        result = pe.exact_river_equity(game)
        total = (
            result["win_probability"]
            + result["tie_probability"]
            + result["loss_probability"]
        )
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_monte_carlo_reproducibility(self):
        game_1 = pe.GameState(
            pe.cards_from_text("As", "Ks"),
            opponent_profile="balanced",
            seed=123,
        )
        game_2 = pe.GameState(
            pe.cards_from_text("As", "Ks"),
            opponent_profile="balanced",
            seed=123,
        )

        result_1 = pe.monte_carlo_equity(game_1, 300)
        result_2 = pe.monte_carlo_equity(game_2, 300)

        self.assertEqual(result_1["wins"], result_2["wins"])
        self.assertEqual(result_1["ties"], result_2["ties"])
        self.assertEqual(result_1["losses"], result_2["losses"])

    def test_turn_scenario_tree(self):
        game = pe.GameState(
            pe.cards_from_text("Ah", "Kh"),
            opponent_profile="random",
            seed=9,
        )
        game.add_flop(pe.cards_from_text("Qh", "Jh", "4c"))
        game.add_turn(pe.Card.from_text("8s"))

        current = pe.monte_carlo_equity(game, 200)
        report = pe.analyze_next_card_scenarios(
            game,
            current_equity=current["equity"],
            branch_simulations=50,
        )

        self.assertEqual(report["branch_count"], 46)
        self.assertAlmostEqual(report["probability_sum"], 1.0, places=12)
        self.assertGreaterEqual(report["q05"], 0.0)
        self.assertLessEqual(report["q95"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
