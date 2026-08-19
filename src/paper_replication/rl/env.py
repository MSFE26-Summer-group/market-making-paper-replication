"""Market-making RL environment (paper III, IV-C): ties market data, the
simulator, an action space, and the reward function into a step/reset loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

import numpy as np
import numpy.typing as npt

from paper_replication.rl.action_space import (
    ContinuousActionSpace,
    DiscreteActionSpace,
    Quote,
)
from paper_replication.rl.config import RLConfig
from paper_replication.rl.market_data import MarketDataset
from paper_replication.rl.reward import hybrid_reward
from paper_replication.rl.simulator import MarketSimulator
from paper_replication.rl.state import agent_state_vector

FloatArray: TypeAlias = npt.NDArray[np.float64]
ActionSpaceKind = Literal["continuous", "discrete"]
Action: TypeAlias = Union[FloatArray, int]


@dataclass
class Observation:
    """One step's observation, matching paper Fig. 1's three concatenated inputs."""

    lob_state: FloatArray  # (window_T, 4*n_levels)
    dynamic_state: FloatArray  # (n_dynamic_features,)
    agent_state: FloatArray  # (2,) inventory, elapsed_frac


@dataclass
class StepResult:
    observation: Observation | None
    reward: float
    done: bool
    info: dict[str, object]


class MarketMakingEnv:
    """Episodic market-making environment over a `MarketDataset` (paper IV-C2).

    Each episode is a contiguous, non-overlapping block of
    `config.episode_length` steps. The simulator's value starts at
    `config.initial_cash` every episode and the agent's position is forced
    closed with a market order at the episode's end -- both mirroring the
    paper's own episode structure ("the agent's value is initialized to 0
    ... at the end of each episode, the agent will close positions with
    market orders").

    Fills happen against the *next* step's best bid/ask (see `simulator.py`'s
    module docstring for why), so the episode's last step only ever closes
    the position -- there's no "next" snapshot left within the episode to
    fill a fresh quote against.
    """

    def __init__(
        self,
        dataset: MarketDataset,
        config: RLConfig,
        action_space_kind: ActionSpaceKind = "continuous",
    ) -> None:
        if len(dataset) < config.episode_length + 1:
            raise ValueError(
                f"dataset has {len(dataset)} steps, need at least "
                f"{config.episode_length + 1} for one episode plus a final fill"
            )
        self.dataset = dataset
        self.config = config
        self.action_space_kind: ActionSpaceKind = action_space_kind
        self.continuous_action_space = ContinuousActionSpace(
            max_bias=config.max_bias, max_spread=config.max_spread
        )
        self.discrete_action_space = DiscreteActionSpace(
            max_bias=config.max_bias, max_spread=config.max_spread
        )
        self.simulator = MarketSimulator(
            minimum_trade_unit=config.minimum_trade_unit,
            max_inventory=config.max_inventory,
        )

        self.n_episodes = (len(dataset) - 1) // config.episode_length

        self._episode_index = -1
        self._step_in_episode = 0
        self._episode_start = 0
        self._value_prev = 0.0

    @property
    def current_row(self) -> int:
        """Row index of the most recently returned observation.

        After `reset()`, this is the episode's first row. After `step()`,
        it's the row the just-returned (or, if `done`, the just-consumed)
        observation corresponds to -- lets training code that keeps its own
        cache aligned with `MarketDataset`'s rows (e.g.
        `rl.feature_cache.precompute_lob_features`'s output) look up the
        matching cached values without re-deriving this bookkeeping itself.
        """
        return self._episode_start + self._step_in_episode

    def _observation(self, row: int) -> Observation:
        return Observation(
            lob_state=self.dataset.lob_state[row],
            dynamic_state=self.dataset.dynamic_state[row],
            agent_state=agent_state_vector(
                self.simulator.inventory,
                self.config.max_inventory,
                self._step_in_episode / self.config.episode_length,
            ),
        )

    def reset(self, episode_index: int | None = None) -> Observation:
        """Starts a new episode; resets cash/inventory. Returns the first observation.

        `episode_index=None` advances sequentially through episodes
        (wrapping around) -- pass an explicit index to replay a specific one.
        """
        if episode_index is None:
            self._episode_index = (self._episode_index + 1) % self.n_episodes
        else:
            if not 0 <= episode_index < self.n_episodes:
                raise ValueError(f"episode_index must be in [0, {self.n_episodes})")
            self._episode_index = episode_index

        self._episode_start = self._episode_index * self.config.episode_length
        self._step_in_episode = 0
        self.simulator.reset(cash=self.config.initial_cash)
        self._value_prev = self.simulator.value(
            self.dataset.mid_price[self._episode_start]
        )
        return self._observation(self._episode_start)

    def step(self, action: Action) -> StepResult:
        """Advances one step.

        `action`: array of 2 floats (continuous) or int (discrete).
        """
        row = self._episode_start + self._step_in_episode
        next_row = row + 1
        is_last_step = self._step_in_episode == self.config.episode_length - 1

        mid_price = self.dataset.mid_price[row]
        next_best_bid = self.dataset.best_bid[next_row]
        next_best_ask = self.dataset.best_ask[next_row]

        quote: Quote | None = None
        if is_last_step:
            fills = self.simulator.close_position(next_best_bid, next_best_ask)
        else:
            quote = self._decode_action(action, mid_price)
            if quote.close_position:
                fills = self.simulator.close_position(next_best_bid, next_best_ask)
            else:
                fills = self.simulator.step(
                    quote.bid_price, quote.ask_price, next_best_bid, next_best_ask
                )

        next_mid_price = self.dataset.mid_price[next_row]
        value = self.simulator.value(next_mid_price)
        pnl_delta = value - self._value_prev
        reward = hybrid_reward(
            pnl_delta=pnl_delta,
            fills=fills,
            mid_price=next_mid_price,
            inventory=self.simulator.inventory,
            eta=self.config.eta,
            zeta=self.config.zeta,
        )
        self._value_prev = value
        self._step_in_episode += 1

        info: dict[str, object] = {
            "fills": fills,
            "cash": self.simulator.cash,
            "inventory": self.simulator.inventory,
            "value": value,
            "quote": quote,
        }
        observation = None if is_last_step else self._observation(next_row)
        return StepResult(
            observation=observation, reward=reward, done=is_last_step, info=info
        )

    def _decode_action(self, action: Action, mid_price: float) -> Quote:
        inventory = self.simulator.inventory
        if self.action_space_kind == "continuous":
            assert isinstance(action, np.ndarray)
            return self.continuous_action_space.decode(action, mid_price, inventory)
        assert isinstance(action, (int, np.integer))
        return self.discrete_action_space.decode(int(action), mid_price, inventory)
