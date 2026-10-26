"""Phase 4 RL-confirmation package: league training, tournament, and non-transitivity measurement."""

from phase4.tournament import ConfirmResult, measure, tournament
from phase4.training import distinct_checkpoints, train_league

__all__ = ["ConfirmResult", "distinct_checkpoints", "measure", "tournament", "train_league"]
