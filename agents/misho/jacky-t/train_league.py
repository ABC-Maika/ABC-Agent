"""Jacky-T, league stage: continue the TQC run against the teammates' trained agents.

Seat 0 is Jacky-T (TQC, expansion-masked). The other five seats are the trained checkpoints on
``main`` (Krischo twice, Yavkata, orcho, and Jack), shuffled over the six metro homes each episode.
Each opponent is loaded with the ladder's own ``_materialize_predicts``, so a recurrent opponent
keeps its LSTM state across the match.

Reward per tick:
* the world's z-normalized weighted sum (profit 0.3, revenue 0.3, market share 0.3, loyalty 0.1);
* minus ``--penalty`` while Jacky-T is not in the top 2 by cumulative profit;
* plus ``--bonus`` while it is first and its lead over second place exceeds half of its own
  cumulative profit (second place below 50% of Jacky-T's score).
The two rank terms start after ``--rank-warmup`` ticks, when cumulative profits are not yet noise.
Scoring is still plain profit; the rank terms only shape training.

It resumes a snapshot of ``train.py``'s run (the obs scale saved beside it), trains up to ``--cap``
TOTAL steps, folds the obs scale into the network and writes a scorable zip.

    python agents/misho/jacky-t/train_league.py jacky_t --resume runs/jacky_t/ckpt/<snapshot>.zip
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
sys.path.insert(0, str(HERE))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import train as jacky  # noqa: E402  (the TQC trainer next to this file)
from gymnasium.utils import seeding  # noqa: E402
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from aggressive_npcs import register as register_personas  # noqa: E402
from anchored_tqc import AnchoredTQC, load_demos  # noqa: E402
from retail_simulator.envs import arena_match  # noqa: E402
from gymnasium import spaces  # noqa: E402
from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config, arena_home_regions  # noqa: E402
from retail_simulator.envs.arena_match import _materialize_predicts  # noqa: E402
from retail_simulator.envs.geo_archetypes import _name_to_index as geo_name_to_index  # noqa: E402
from retail_simulator.envs.geo_archetypes import geo_archetypes  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402
from retail_simulator.envs.spaces import build_action_space, build_observation_space  # noqa: E402

N_SEATS = 6
LEARNER = 0
OPPONENTS: dict[str, str] = {  # the trained agents on main
    "krischo": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "krischo-v8": "agents/Krischo/rppo-lstm/checkpoint_v8.zip",
    "yavkata": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "orcho": "agents/orcho/market-share/checkpoint.zip",
    "jack": "agents/misho/jack/checkpoint.zip",
    "kmlp6": "agents/Krischo/mlp-v6/checkpoint.zip",
    "kmlp8": "agents/Krischo/mlp-v8/checkpoint.zip",
    "orcho-v5": "agents/orcho/v5/checkpoint.zip",
    "yav-sac": "agents/Yavkata/sac/checkpoint.zip",
    # Hall of fame: a frozen copy of Jacky-T v1.1, so a new candidate cannot win by forgetting it.
    "jacky-v11": "runs/hof/jacky_v11.zip",
}
YARDSTICKS = ("cherry-picker",)  # a scripted archetype that never changes
# The teammates' scripted sparring partners (scripts/aggressive_npcs.py): distinct, strong styles.
PERSONAS = ("cherry-picker+", "fast-follower+", "sprawler+", "orcho-recipe2")
EPISODE_OPPONENTS = N_SEATS - 1


def capped_action_space(research_cap: float | None) -> spaces.Box:
    """The bounded-logit training space, with the research lever's upper bound lowered to the cap.

    The bound is stored in the saved zip (a plain ``Box``) and the policy rescales its tanh output
    into it, so a saved policy can never ask for more research than the cap.
    """
    space = jacky.base.bounded_logit_space(build_action_space())
    if research_cap is None:
        return space
    high = space.high.copy()
    for slot in flatten_action_space_layout().levers:
        if slot.spec.name == "research":
            high[slot.offset] = min(float(high[slot.offset]), research_cap)
    return spaces.Box(low=space.low, high=high, dtype=np.float32)


MARGIN = 0.39  # profit / revenue before running costs, read from KMlp8's trace (rough)
STORE_OPEX = 50.0  # per store per round
CAPEX = 5_000.0
BUFFER_CASH = 8_000.0  # after the opening, open a store only with this much cash and a profit
OPENING_ROUNDS = 5
MAX_LOW_REGIONS = 2  # LOYAL / FRONTIER stores held at most
EXPECTED_RIVALS = {0: 0, 1: 3, 2: 1, 3: 0}  # region index % 4: FRONTIER, METRO, AFFLUENT, LOYAL


def region_scores(config: Any, retailers: Any, learner: int, retaliate: float = 0.0) -> np.ndarray:
    """Estimated profit per round of a new store of ``learner``'s in each region.

    With equal levers the agents in a region split it evenly, and one agent alone is capped at 60%:
    share = 0.6 alone, else 1 / (agents there + 1). Profit = demand * share * MARGIN - STORE_OPEX. A rival
    that holds a store in one of my regions is an intruder; a region it is in gets ``retaliate`` times my
    gross gain on top (that gain is exactly what the rivals there lose). Region 0 is never playable.
    """
    demand = np.array([r.base_regional_demand for r in config.demand.regions], dtype=float)
    presence = [np.asarray(r.stores_per_region) > 0 for r in retailers]
    mine = presence[learner]
    rivals = sum(p.astype(int) for k, p in enumerate(presence) if k != learner)
    intruders = [k for k, p in enumerate(presence) if k != learner and (p & mine).any()]
    # Crowds grow: by round ~40 every agent is in every METRO (3 others), about one in each AFFLUENT,
    # almost none in LOYAL / FRONTIER (seen in the battle replays), so score against the expected crowd.
    expected = np.array([EXPECTED_RIVALS[i % 4] for i in range(len(demand))])
    rivals = np.maximum(rivals, expected)
    share = np.where(rivals == 0, 0.6, 1.0 / (rivals + 1))
    gain = demand * share * MARGIN
    scores = gain - STORE_OPEX
    if retaliate and intruders:
        hit = np.zeros(len(demand), dtype=bool)
        for k in intruders:
            hit |= presence[k]
        scores = scores + retaliate * gain * hit
    scores[0] = -np.inf
    return scores


def may_open(retailer: Any, region_scores_row: np.ndarray, tick: int) -> np.ndarray:
    """Regions the cash-buffer rule allows opening (all of them during the opening rounds).

    After ``OPENING_ROUNDS`` a store needs cash >= BUFFER_CASH and a positive last profit (session 10
    expanded to 15 stores on a 5,000 gate alone, ran out of cash and was cut back to 3), and at most
    ``MAX_LOW_REGIONS`` LOYAL / FRONTIER stores may be held (they pay least per store).
    """
    n = len(region_scores_row)
    if tick < OPENING_ROUNDS:
        return np.ones(n, dtype=bool)
    if retailer.cash < BUFFER_CASH or retailer.last_profit <= 0:
        return np.zeros(n, dtype=bool)
    stores = np.asarray(retailer.stores_per_region) > 0
    low = np.array([(i % 4) in (0, 3) for i in range(n)])
    if int((stores & low).sum()) >= MAX_LOW_REGIONS:
        return ~low
    return np.ones(n, dtype=bool)


class LeagueEnv(gym.Env):
    """Jacky-T against the five trained agents, with rank-based reward shaping."""

    def __init__(
        self,
        seed: int,
        obs_scale: np.ndarray,
        weights: tuple[float, float, float, float],
        bonus: float,
        penalty: float,
        rank_warmup: int,
        ticks: int = 1000,
        neg_penalty: float = 0.0,
        pool: Sequence[str] | None = None,
        any_home: float = 0.0,
        prime_weight: float = 0.0,
        first_bonus: float = 0.0,
        open_bonus: float = 0.0,
        open_bonus_ticks: int = 200,
        happy_penalty: float = 0.0,
        research_pulse: bool = False,
        open_value: bool = False,
        debt_penalty: float = 0.0,
        opening_bonus: float = 0.0,
        open_early_mult: float = 1.0,
        fixed_homes: Sequence[int] | None = None,
        crowd_rule: int = 0,
        owner_rule: bool = False,
        score_rule: bool = False,
        retaliate: float = 0.0,
        research_cap: float | None = None,
        raw_profit_scale: float | None = None,
        lead_shaping: float = 0.0,
        shaping_fade_episodes: int = 0,
        shaping_gamma: float = 0.99,
    ) -> None:
        super().__init__()
        register_personas()  # the teammates' scripted personas (cherry-picker+, orcho-recipe2, ...)
        # ``raw_profit_scale``: train on raw post-penalty profit x this factor. The weighted mode
        # z-scores every component against the seat's own running mean in the episode, so steady
        # high profit earns ~0 there; raw profit pays for the level that scoring ranks on.
        self.raw_profit_scale = raw_profit_scale
        # ``lead_shaping``: potential-based shaping on the lead over the best rival,
        # F = k * (gamma * lead' - lead) * scale. Summed over a match it pays the final margin over
        # the best rival (the "finish first" goal) without changing which policy is optimal. It
        # fades linearly to 0 over ``shaping_fade_episodes`` episodes of this env copy.
        self.lead_shaping, self.shaping_fade = lead_shaping, shaping_fade_episodes
        self.shaping_gamma = shaping_gamma
        self.episodes = -1
        # ``first_bonus``: per tick while Jacky-T is first by cumulative profit (after the warm-up).
        self.first_bonus = first_bonus
        self.open_bonus = open_bonus
        self.open_bonus_ticks = open_bonus_ticks
        self.happy_penalty = happy_penalty
        self.research_pulse = research_pulse
        self.open_value = open_value
        self.debt_penalty = debt_penalty
        self.opening_bonus = opening_bonus
        self.crowd_rule = crowd_rule
        self.owner_rule = owner_rule
        self.score_rule = score_rule
        self.retaliate = retaliate
        self.fixed_homes = tuple(fixed_homes) if fixed_homes is not None else None
        self.open_early_mult = open_early_mult
        self._owned = np.zeros(0, dtype=bool)
        self._research_at = next(
            s.offset for s in flatten_action_space_layout().levers if s.spec.name == "research"
        )
        # ``prime_weight``: extra reward per tick for the demand-weighted market share held in the
        # METRO (1, 5, 9, ...) and AFFLUENT (2, 6, 10, ...) regions, the two biggest-demand classes
        # (mean base demand 1,400 and 900, against 600 and 350 for LOYAL and FRONTIER).
        self.prime_weight = prime_weight
        # ``any_home``: probability that an episode draws its six homes from all enterable regions
        # (1..24) instead of the six published metro homes, so the policy is not tied to one
        # starting geography.
        self.any_home = any_home
        # ``pool``: opponent candidates (OPPONENTS keys or geo archetype names). Five are drawn per
        # episode, so line-ups vary; the default is every trained agent plus the yardstick.
        self.pool = list(pool) if pool is not None else [*OPPONENTS, *YARDSTICKS, *PERSONAS]
        self.neg_penalty = neg_penalty
        self.np_random, _ = seeding.np_random(seed)
        profit, revenue, share, loyalty = weights
        base_config = arena_balanced_config()
        if raw_profit_scale is not None:
            self.config = replace(base_config, reward=replace(base_config.reward, mode="profit"))
        else:
            self.config = replace(
                base_config,
                reward=replace(
                    base_config.reward,
                    mode="weighted",
                    weight_profit=profit,
                    weight_revenue=revenue,
                    weight_market_share=share,
                    weight_loyalty=loyalty,
                ),
            )
        self.scale = obs_scale.astype(np.float32)
        self.bonus, self.penalty, self.rank_warmup, self.ticks = bonus, penalty, rank_warmup, ticks
        self.homes = arena_home_regions(N_SEATS)
        regions = self.config.demand.regions
        obs_index = geo_name_to_index(len(regions))
        prime = [r for r in range(1, len(regions)) if (r - 1) % 4 in (0, 1)]  # METRO, AFFLUENT
        demand = np.array([regions[r].base_regional_demand for r in prime], dtype=np.float64)
        self._prime_idx = [obs_index[f"market_share_region_{r}"] for r in prime]
        self._prime_w = demand / demand.sum()
        self.observation_space = build_observation_space(self.config)
        self.action_space = capped_action_space(research_cap)
        self._predicts: dict[str, Any] = {}

    def _predict(self, name: str) -> Any:
        if name not in OPPONENTS:  # a scripted archetype: a fresh closure, it carries a tick counter
            # ``arena_match.geo_archetypes`` includes the personas once ``register_personas`` ran.
            roster = arena_match.geo_archetypes(
                len(self.config.demand.regions), self.config.demand.regions
            )
            return roster[name]
        if name not in self._predicts:
            self._predicts.update(
                _materialize_predicts(
                    {name: OPPONENTS[name]},
                    action_space=build_action_space(),
                    config=self.config,
                )
            )
        return self._predicts[name]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(0, 2**31 - 1)))
        if self.any_home and self.np_random.random() < self.any_home:
            n_regions = len(self.config.demand.regions)
            drawn = self.np_random.choice(np.arange(1, n_regions), size=N_SEATS, replace=False)
            homes = tuple(int(h) for h in drawn)
        elif self.fixed_homes is not None:
            homes = self.fixed_homes
        else:
            homes = tuple(int(h) for h in self.np_random.permutation(self.homes))
        names = rng.sample(self.pool, EPISODE_OPPONENTS)
        self.opponents = [self._predict(name) for name in names]
        self.world = RetailParallelEnv(
            self.config,
            n_learning_agents=N_SEATS,
            home_regions=homes,
            seed=int(self.np_random.integers(0, 2**31 - 1)),
            mask_expansion_seats=tuple(range(N_SEATS)),
        )
        self.agents_ = self.world.possible_agents
        self._obs, _ = self.world.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        self.cum = np.zeros(N_SEATS)
        self.t = 0
        self.episodes += 1
        self._lead_phi = 0.0
        self._owned = np.asarray(self.world._state.retailers[LEARNER].stores_per_region) > 0
        return self._learner_obs(), {"homes": homes, "opponents": names}

    def step(self, action: np.ndarray) -> Any:
        acts = {self.agents_[LEARNER]: np.asarray(action, dtype=np.float32)}
        for seat, predict in enumerate(self.opponents, start=1):
            agent = self.agents_[seat]
            acts[agent] = np.asarray(
                predict(self._obs[agent], episode_start=self.t == 0), dtype=np.float32
            ).ravel()
        if self.research_pulse and self.t % 10 == 0:
            # Training-side rule: every 10th round, while not first by cumulative profit, spend 0.4 on
            # research (the stored action stays the policy's own, so this nudges play, it is not shipped).
            if int((np.delete(self.cum, LEARNER) > self.cum[LEARNER]).sum()) > 0:
                acts[self.agents_[LEARNER]] = self._with_research(acts[self.agents_[LEARNER]])
        self._obs, rewards, _, _, infos = self.world.step(acts)
        self.t += 1
        self.cum += [infos[a]["components"]["profit"] for a in self.agents_]
        reward = float(rewards[self.agents_[LEARNER]])
        if self.raw_profit_scale is not None:
            reward *= self.raw_profit_scale  # profit mode: the world returns raw profit
        reward += self._rank_shaping() + self._lead_shaping()
        if self.open_bonus and self.t <= self.open_bonus_ticks:
            # Aggression: a one-off reward each time the learner opens a new region (the clone stalls
            # at 3 stores when a rival contests its regions; the teachers keep expanding to 12).
            retailers = self.world._state.retailers
            mine = np.asarray(retailers[LEARNER].stores_per_region) > 0
            if self.open_value:
                # Value of a region = its demand split over the agents already there (with equal
                # levers an agent gets 1/n of a region): METRO 1400, AFFLUENT 900, LOYAL 600, FRONTIER
                # 350, divided by (rivals present + 1). The bonus is scaled so a METRO with one rival
                # (700) pays the base bonus; crowded regions pay less, so a lower class wins there.
                demand = np.array([r.base_regional_demand for r in self.config.demand.regions])
                rivals = sum(
                    (np.asarray(retailers[k].stores_per_region) > 0).astype(int)
                    for k in range(N_SEATS)
                    if k != LEARNER
                )
                gained = mine & ~self._owned
                reward += self.open_bonus * float((demand / (1 + rivals) / 700.0)[gained].sum())
            elif self.score_rule:
                # One score for every region (see ``region_scores``): the bonus is the score of the
                # region opened relative to the best one that was still open to me, so the best region
                # pays the full bonus and a poorer one proportionally less (negative if it loses money).
                scores = region_scores(self.config, retailers, LEARNER, self.retaliate)
                gained = mine & ~self._owned
                candidates = ~self._owned
                candidates[0] = False
                best = float(scores[candidates].max()) if candidates.any() else 0.0
                mult = self.open_early_mult if self.t <= 10 else 1.0
                if best > 0 and gained.any():
                    rel = np.clip(scores[gained] / best, -1.0, 1.0)
                    # The cash-buffer rule is judged on the state before the purchase (cash + capex).
                    me = retailers[LEARNER]
                    before = self._buffer_view(me, int(gained.sum()))
                    allowed = may_open(before, scores, self.t - 1)
                    rel = np.where(allowed[gained], rel, -0.5)
                    reward += self.open_bonus * mult * float(rel.sum())
            elif self.owner_rule:
                # Owner rule: open the highest-class region nobody else owns; once no region of that
                # class is free, the next class down, and so on. Reward = base * demand / demand of the
                # best free region (so a free AFFLUENT scores 0.64 while a free METRO exists); a region a
                # rival already owns is penalised while any free region is left, otherwise it pays
                # demand / (rivals + 1).
                demand = np.array([r.base_regional_demand for r in self.config.demand.regions])
                rivals = sum(
                    (np.asarray(retailers[k].stores_per_region) > 0).astype(int)
                    for k in range(N_SEATS)
                    if k != LEARNER
                )
                gained = mine & ~self._owned
                free = (rivals == 0) & ~(self._owned) & (np.arange(len(demand)) >= 1)
                best = float(demand[free].max()) if free.any() else 0.0
                mult = self.open_early_mult if self.t <= 10 else 1.0
                total = 0.0
                for r in np.flatnonzero(gained):
                    if rivals[r] == 0 and best > 0:
                        total += demand[r] / best
                    elif best > 0:
                        total -= 1.0
                    else:
                        total += demand[r] / 1400.0 / (1 + rivals[r])
                reward += self.open_bonus * mult * total
            elif self.crowd_rule:
                # Crowd rule: open a region only while at most ``crowd_rule`` rivals are already in it;
                # a more crowded region (a METRO with 3+ owners) is penalised, so the policy goes to
                # the next class down (AFFLUENT, then LOYAL, then FRONTIER). The reward is scaled by the
                # region's demand (METRO 1.0, AFFLUENT 0.64, LOYAL 0.43, FRONTIER 0.25).
                demand = np.array([r.base_regional_demand for r in self.config.demand.regions])
                rivals = sum(
                    (np.asarray(retailers[k].stores_per_region) > 0).astype(int)
                    for k in range(N_SEATS)
                    if k != LEARNER
                )
                gained = mine & ~self._owned
                sign = np.where(rivals <= self.crowd_rule, 1.0, -1.0)
                mult = self.open_early_mult if self.t <= 10 else 1.0
                reward += self.open_bonus * mult * float((sign * demand / 1400.0)[gained].sum())
            else:
                mult = self.open_early_mult if self.t <= 10 else 1.0
                reward += self.open_bonus * mult * int((mine & ~self._owned).sum())
            if self.retaliate and not self.score_rule:
                # Retaliation: a rival is an intruder when it holds a store in one of my regions. Opening
                # in a region the intruder is in pulls D / (agents there + 1) of that region's demand
                # away from the rivals there (zero-sum split), at the price of the store's running cost
                # (~50 a round). The bonus pays the surplus, so I go where the intruder loses more than
                # the store costs me.
                demand = np.array([r.base_regional_demand for r in self.config.demand.regions])
                presence = [np.asarray(retailers[k].stores_per_region) > 0 for k in range(N_SEATS)]
                intruders = [k for k in range(N_SEATS) if k != LEARNER and (presence[k] & self._owned).any()]
                rivals_n = sum(presence[k].astype(int) for k in range(N_SEATS) if k != LEARNER)
                for r in np.flatnonzero(mine & ~self._owned):
                    if any(presence[k][r] for k in intruders):
                        reward += self.retaliate * max(0.0, demand[r] / (1 + rivals_n[r]) - 50.0) / 700.0
            self._owned = self._owned | mine
        if self.opening_bonus and self.t == 4:
            reward += self.opening_bonus * float(
                int(np.sum(self.world._state.retailers[LEARNER].stores_per_region)) >= 4
            )
        if self.debt_penalty:
            # The world closes all but 3 stores once cash stays below -10,000 for 50 rounds; start
            # pushing back at -5,000 (cash guard).
            cash = float(self.world._state.retailers[LEARNER].cash)
            reward -= self.debt_penalty * max(0.0, -5000.0 - cash) / 5000.0
        if self.happy_penalty:
            # Unpaid staff (happiness near 0) trigger the write-off penalty; RL drove wages to ~0.02.
            happy = float(self.world._state.retailers[LEARNER].employee_happiness)
            reward -= self.happy_penalty * max(0.0, 0.5 - happy)
        if self.prime_weight:
            shares =np.asarray(self._obs[self.agents_[LEARNER]])[self._prime_idx]
            reward += self.prime_weight * float(shares @ self._prime_w)
        rank = int((self.cum > self.cum[LEARNER]).sum()) + 1
        return self._learner_obs(), reward, False, self.t >= self.ticks, {"rank": rank}

    @staticmethod
    def _buffer_view(me: Any, n_new: int) -> Any:
        """The learner's retailer as it was before opening ``n_new`` stores this round."""
        from types import SimpleNamespace

        before = list(me.stores_per_region)
        return SimpleNamespace(
            cash=float(me.cash) + CAPEX * n_new, last_profit=float(me.last_profit),
            stores_per_region=tuple(before),
        )

    def _with_research(self, action: np.ndarray) -> np.ndarray:
        action = np.array(action, dtype=np.float32)
        action[self._research_at] = 0.4
        return action

    def _rank_shaping(self) -> float:
        if self.t < self.rank_warmup:
            return 0.0
        mine = self.cum[LEARNER]
        rivals = np.delete(self.cum, LEARNER)
        rank = int((rivals > mine).sum()) + 1
        shaping = -self.neg_penalty if mine < 0 else 0.0  # losing money overall
        if rank == 1:
            shaping += self.first_bonus
        if rank > 2:
            return shaping - self.penalty
        if rank == 1 and mine > 0 and mine - rivals.max() > 0.5 * mine:
            return shaping + self.bonus
        return shaping

    def _lead_shaping(self) -> float:
        if not self.lead_shaping:
            return 0.0
        fade = 1.0
        if self.shaping_fade:
            fade = max(0.0, 1.0 - self.episodes / self.shaping_fade)
        scale = self.raw_profit_scale or 1e-3
        phi = (self.cum[LEARNER] - np.delete(self.cum, LEARNER).max()) * scale
        shaped = self.lead_shaping * fade * (self.shaping_gamma * phi - self._lead_phi)
        self._lead_phi = phi
        return float(shaped)

    def _learner_obs(self) -> np.ndarray:
        return (self._obs[self.agents_[LEARNER]] / self.scale).astype(np.float32)


def make_one(seed: int, obs_scale: np.ndarray, args: argparse.Namespace) -> Monitor:
    """One env copy. A module-level function so worker processes can unpickle it by reference."""
    return Monitor(
        LeagueEnv(
            seed,
            obs_scale,
            args.reward_weights,
            args.bonus,
            args.penalty,
            args.rank_warmup,
            neg_penalty=args.neg_penalty,
            any_home=args.any_home,
            prime_weight=args.prime_weight,
            first_bonus=args.first_bonus,
            open_bonus=args.open_bonus,
            happy_penalty=args.happy_penalty,
            research_pulse=args.research_pulse,
            open_value=args.open_value,
            debt_penalty=args.debt_penalty,
            opening_bonus=args.opening_bonus,
            crowd_rule=args.crowd_rule,
            owner_rule=args.owner_rule,
            score_rule=args.score_rule,
            retaliate=args.retaliate,
            open_early_mult=args.open_early_mult,
            research_cap=args.research_cap,
            raw_profit_scale=args.raw_profit_scale,
            lead_shaping=args.lead_shaping,
            shaping_fade_episodes=args.shaping_fade_episodes,
            shaping_gamma=args.gamma or 0.99,
            pool=args.pool,
        ),
        info_keywords=("rank",),
    )


def make_env_fns(args: argparse.Namespace, obs_scale: np.ndarray) -> list[Any]:
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    return [partial(make_one, s, obs_scale, args) for s in seeds]


def train(args: argparse.Namespace) -> TQC:
    set_random_seed(args.seed)
    obs_scale = np.load(args.resume.parent / "obs_scale.npy")
    env_fns = make_env_fns(args, obs_scale)
    venv = SubprocVecEnv(env_fns) if args.subproc else DummyVecEnv(env_fns)
    venv.action_space.seed(args.seed)
    # The saved action space (research cap included) must match the env's, so it is passed in.
    learner = AnchoredTQC if args.anchor_data else TQC
    model = learner.load(
        args.resume,
        env=venv,
        custom_objects={"action_space": capped_action_space(args.research_cap)},
    )
    if args.anchor_data:
        demo_obs, demo_act = load_demos(
            args.anchor_data, args.anchor_max_rank, args.anchor_wage_floor, args.expand_boost,
            args.anchor_early_wage,
            args.frugal_opening,
        )
        model.set_demos(demo_obs, demo_act, args.anchor_alpha)
        print(f"anchoring the actor to {len(demo_obs)} teacher samples (alpha {args.anchor_alpha})")
    if args.gamma is not None:
        model.gamma = args.gamma  # a longer horizon: the default discount looks ~100 rounds ahead
    done = int(model.num_timesteps)
    remaining = args.cap - done
    if remaining <= 0:
        raise SystemExit(f"snapshot is already at {done} steps, cap is {args.cap}")
    print(f"resumed {args.resume} at {done} steps; training {remaining} more to the cap {args.cap}")
    # The replay buffer is not part of a snapshot: refill it before updating, or the first
    # updates would overfit a handful of transitions.
    model.learning_starts = done + args.learning_starts
    sample_action = None
    if args.from_actor:
        # Use the cloned actor from step one instead of SB3's random warm-up actions, and a small
        # learning rate so the first updates (critics still stale) do not undo the clone.
        from stable_baselines3.common.utils import get_schedule_fn

        model.lr_schedule = get_schedule_fn(args.lr)
        sample_action = model._sample_action
        model._sample_action = lambda ls, action_noise=None, n_envs=1: sample_action(
            0, action_noise, n_envs
        )
    update_lr = None
    if args.actor_freeze > 0:
        # Critic warm-up: for the first ``actor_freeze`` steps only the critics learn (the actor's
        # learning rate is 0), so they can fit the cloned policy's returns before they steer it.
        # Afterwards the actor learns at ``actor_lr``.
        update_lr = model._update_learning_rate

        def gated(optimizers: Any) -> None:
            update_lr(optimizers)
            frozen = model.num_timesteps < done + args.actor_freeze
            for group in model.actor.optimizer.param_groups:
                group["lr"] = 0.0 if frozen else args.actor_lr

        model._update_learning_rate = gated
    # The temporary patches above are closures over the model and the worker processes. They must
    # never be pickled into a snapshot (which would also be custom code inside an entry), so they
    # are excluded from every ``model.save``.
    patched = [n for n in ("_sample_action", "_update_learning_rate") if n in model.__dict__]
    if patched:
        excluded = model._excluded_save_params
        model._excluded_save_params = lambda: [*excluded(), *patched, "_excluded_save_params"]
    ckpt_dir = Path("runs") / args.label / "league_ckpt"
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="jacky_t_league",
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "obs_scale.npy", obs_scale)
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    if sample_action is not None:
        del model._sample_action  # the lambda must not be pickled into the saved zip
    if update_lr is not None:
        del model._update_learning_rate  # nor the gate
    if patched:
        del model._excluded_save_params
    venv.close()

    # Fold the obs scale in, checking the network is unchanged on observations of this env.
    env = LeagueEnv(args.seed + 1, obs_scale, args.reward_weights, 0.0, 0.0, 0, research_cap=args.research_cap)
    scaled, _ = env.reset()
    samples = [scaled]
    for _ in range(40):
        scaled, *_ = env.step(env.action_space.sample())
        samples.append(scaled)
    scaled_samples = np.stack(samples)
    before = jacky.network_outputs(model, scaled_samples)
    jacky.fold_obs_scale(model, obs_scale)
    after = jacky.network_outputs(model, scaled_samples * obs_scale)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-4:
        raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
    print(f"obs scale folded into the network (max prediction drift {drift:.1e})")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jacky-T league stage (TQC vs teammates' agents).")
    parser.add_argument("label")
    parser.add_argument("--resume", type=Path, required=True, help="snapshot zip from train.py")
    parser.add_argument("--cap", type=int, default=100_000, help="TOTAL steps, snapshot included")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=4_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--bonus", type=float, default=0.25, help="per tick, lead > half own score")
    parser.add_argument("--penalty", type=float, default=0.25, help="per tick, outside the top 2")
    parser.add_argument("--rank-warmup", type=int, default=50, help="ticks before rank terms apply")
    parser.add_argument(
        "--neg-penalty", type=float, default=0.0, help="per tick while cumulative profit is below 0"
    )
    parser.add_argument(
        "--from-actor",
        action="store_true",
        help="the snapshot is a behaviour-cloned actor: act with it from the first step (no random "
        "warm-up), keep its low noise and use a small learning rate",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--prime-weight",
        type=float,
        default=0.0,
        help="per-tick reward x the demand-weighted market share in METRO and AFFLUENT regions",
    )
    parser.add_argument(
        "--any-home",
        type=float,
        default=0.0,
        help="probability that an episode draws its six homes from all 24 enterable regions "
        "(otherwise the six published metro homes, shuffled over the seats)",
    )
    parser.add_argument(
        "--anchor-data",
        type=Path,
        nargs="+",
        help="teacher recordings (mimic.py collect): anchors the actor update to them (TD3+BC)",
    )
    parser.add_argument("--anchor-alpha", type=float, default=2.5, help="RL weight vs the anchor")
    parser.add_argument("--anchor-max-rank", type=int, default=2)
    parser.add_argument("--anchor-wage-floor", type=float, default=0.5)
    parser.add_argument("--expand-boost", type=int, default=1)
    parser.add_argument("--open-bonus", type=float, default=0.0, help="reward per newly opened region")
    parser.add_argument("--happy-penalty", type=float, default=0.0)
    parser.add_argument("--research-pulse", action="store_true")
    parser.add_argument("--open-value", action="store_true")
    parser.add_argument("--anchor-early-wage", type=float, default=0.0)
    parser.add_argument("--debt-penalty", type=float, default=0.0)
    parser.add_argument("--frugal-opening", action="store_true")
    parser.add_argument("--retaliate", type=float, default=0.0, help="bonus for opening where an intruder is")
    parser.add_argument("--score-rule", action="store_true", help="one profit score ranks all regions")
    parser.add_argument("--owner-rule", action="store_true", help="open the highest-class unowned region")
    parser.add_argument("--crowd-rule", type=int, default=0, help="max rivals already in a region to open it")
    parser.add_argument("--opening-bonus", type=float, default=0.0)
    parser.add_argument("--open-early-mult", type=float, default=1.0)
    parser.add_argument(
        "--pool", nargs="+", default=None, help="opponent names to draw from (default: the full league)"
    )
    parser.add_argument("--first-bonus", type=float, default=0.0, help="per tick while first")
    parser.add_argument(
        "--raw-profit-scale",
        type=float,
        default=None,
        help="train on raw profit x this (e.g. 1e-3) instead of the self-normalized weighted reward",
    )
    parser.add_argument("--lead-shaping", type=float, default=0.0, help="potential-based lead term")
    parser.add_argument(
        "--shaping-fade-episodes", type=int, default=0, help="per env copy; 0 = never fade"
    )
    parser.add_argument(
        "--research-cap", type=float, default=None, help="upper bound of the research lever"
    )
    parser.add_argument("--gamma", type=float, default=None, help="discount; default: the snapshot's")
    parser.add_argument("--actor-freeze", type=int, default=0, help="steps with the actor frozen")
    parser.add_argument("--actor-lr", type=float, default=3e-5, help="actor lr after the freeze")
    parser.add_argument("--subproc", action="store_true", help="one worker process per env copy")
    parser.add_argument(
        "--reward-weights",
        type=lambda text: tuple(float(part) for part in text.split(",")),
        default=(0.3, 0.3, 0.3, 0.1),
    )
    parser.add_argument("--out", type=Path, default=Path("agents/misho/jacky-t/checkpoint.zip"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jacky.stock_schedule(model)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
