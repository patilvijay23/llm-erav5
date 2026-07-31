"""Deterministic two-source schedules for the P7 transition ablation."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Iterable

import numpy as np


FINEWEB = 0
INDICCORP = 1
SOURCE_NAMES = {FINEWEB: "fineweb_edu", INDICCORP: "indiccorp_v2"}


@dataclass(frozen=True)
class Phase:
    name: str
    fineweb_share: Fraction

    @property
    def indic_share(self) -> Fraction:
        return Fraction(1, 1) - self.fineweb_share


@dataclass(frozen=True)
class ScheduleConfig:
    total_tokens: int = 20_000_000_000
    shift_tokens: int = 10_000_000_000
    block_size: int = 640
    pre_fineweb_share: Fraction = Fraction(1, 2)
    post_fineweb_share: Fraction = Fraction(4, 5)
    seed: int = 1234

    def validate(self, global_batch_sequences: int) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not 0 < self.shift_tokens < self.total_tokens:
            raise ValueError("shift_tokens must lie strictly inside total_tokens")
        if self.total_tokens % self.block_size:
            raise ValueError(
                f"total_tokens={self.total_tokens:,} is not divisible by "
                f"block_size={self.block_size}."
            )
        if self.shift_tokens % self.block_size:
            raise ValueError(
                f"shift_tokens={self.shift_tokens:,} is not divisible by "
                f"block_size={self.block_size}."
            )
        if global_batch_sequences <= 0:
            raise ValueError("global_batch_sequences must be positive")

        total_sequences = self.total_tokens // self.block_size
        shift_sequences = self.shift_tokens // self.block_size
        if total_sequences % global_batch_sequences:
            raise ValueError(
                "The total run must end on an optimizer-step boundary. "
                f"total_sequences={total_sequences:,} is not divisible by "
                f"global_batch_sequences={global_batch_sequences:,}."
            )
        if shift_sequences % global_batch_sequences:
            raise ValueError(
                "The 10B transition centre must lie on an optimizer-step boundary. "
                f"shift_sequences={shift_sequences:,} is not divisible by "
                f"global_batch_sequences={global_batch_sequences:,}."
            )

        for label, share in (
            ("pre", self.pre_fineweb_share),
            ("post", self.post_fineweb_share),
        ):
            expected = share * global_batch_sequences
            if expected.denominator != 1:
                raise ValueError(
                    f"The {label}-transition FineWeb share {float(share):.3f} "
                    f"cannot be represented exactly by a global batch of "
                    f"{global_batch_sequences} sequences."
                )


class MixtureSchedule:
    """Exact per-step source counts with deterministic source permutations."""

    mode: str

    def __init__(
        self,
        config: ScheduleConfig,
        global_batch_sequences: int,
        fineweb_counts: np.ndarray,
        *,
        mode: str,
        transition_start_step: int,
        transition_end_step: int,
    ) -> None:
        config.validate(global_batch_sequences)
        self.config = config
        self.global_batch_sequences = global_batch_sequences
        self.total_steps = (
            config.total_tokens // config.block_size // global_batch_sequences
        )
        self.shift_step = (
            config.shift_tokens // config.block_size // global_batch_sequences
        )
        if fineweb_counts.shape != (self.total_steps,):
            raise ValueError("fineweb_counts has the wrong length")
        if np.any(fineweb_counts < 0) or np.any(
            fineweb_counts > global_batch_sequences
        ):
            raise ValueError("fineweb_counts contains an invalid batch count")
        self._fineweb_counts = fineweb_counts.astype(np.int64, copy=False)
        self._indic_counts = global_batch_sequences - self._fineweb_counts
        self._fw_prefix = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self._fineweb_counts)]
        )
        self._indic_prefix = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(self._indic_counts)]
        )
        self.mode = mode
        self.transition_start_step = transition_start_step
        self.transition_end_step = transition_end_step

    @property
    def transition_steps(self) -> int:
        return self.transition_end_step - self.transition_start_step

    def phase_for_step(self, step: int) -> Phase:
        if not 0 <= step < self.total_steps:
            raise IndexError(f"step {step} outside [0, {self.total_steps})")
        count = int(self._fineweb_counts[step])
        share = Fraction(count, self.global_batch_sequences)
        if self.mode == "hard":
            name = "pre_shift_50_50" if step < self.shift_step else "post_shift_80_20"
        elif step < self.transition_start_step:
            name = "pre_transition_50_50"
        elif step < self.transition_end_step:
            name = "linear_transition_50_50_to_80_20"
        else:
            name = "post_transition_80_20"
        return Phase(name, share)

    def counts_for_step(self, step: int) -> tuple[int, int]:
        if not 0 <= step < self.total_steps:
            raise IndexError(f"step {step} outside [0, {self.total_steps})")
        return int(self._fineweb_counts[step]), int(self._indic_counts[step])

    def pattern_for_step(self, step: int) -> np.ndarray:
        """Return an exact, deterministically shuffled source vector."""
        fw, indic = self.counts_for_step(step)
        pattern = np.empty(self.global_batch_sequences, dtype=np.int8)
        pattern[:fw] = FINEWEB
        pattern[fw : fw + indic] = INDICCORP
        seed_seq = np.random.SeedSequence([self.config.seed, step, 0x5037])
        rng = np.random.default_rng(seed_seq)
        rng.shuffle(pattern)
        return pattern

    def draw_bases(self, step: int) -> tuple[int, int]:
        """Return source draws completed before optimizer-step index ``step``."""
        if not 0 <= step <= self.total_steps:
            raise IndexError(f"step {step} outside [0, {self.total_steps}]")
        return int(self._fw_prefix[step]), int(self._indic_prefix[step])

    def local_draws(
        self,
        *,
        step: int,
        micro_step: int,
        rank: int,
        world_size: int,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
    ) -> list[tuple[int, int]]:
        """Return ``(source_id, source_draw_ordinal)`` for one rank/micro-step."""
        expected_global = (
            world_size * per_device_batch_size * gradient_accumulation_steps
        )
        if expected_global != self.global_batch_sequences:
            raise ValueError(
                f"Runtime global batch {expected_global} != schedule global batch "
                f"{self.global_batch_sequences}."
            )
        if not 0 <= micro_step < gradient_accumulation_steps:
            raise IndexError("micro_step outside gradient accumulation window")

        pattern = self.pattern_for_step(step)
        fw_base, indic_base = self.draw_bases(step)
        start = (
            micro_step * world_size * per_device_batch_size
            + rank * per_device_batch_size
        )
        end = start + per_device_batch_size
        fw_prefix = np.cumsum(pattern == FINEWEB, dtype=np.int64)
        indic_prefix = np.cumsum(pattern == INDICCORP, dtype=np.int64)
        draws: list[tuple[int, int]] = []
        for position in range(start, end):
            source = int(pattern[position])
            if source == FINEWEB:
                draws.append((source, fw_base + int(fw_prefix[position] - 1)))
            else:
                draws.append((source, indic_base + int(indic_prefix[position] - 1)))
        return draws

    def expected_totals(self) -> dict[str, int]:
        fw_sequences = int(self._fw_prefix[-1])
        indic_sequences = int(self._indic_prefix[-1])
        return {
            "fineweb_sequences": fw_sequences,
            "indic_sequences": indic_sequences,
            "fineweb_tokens": fw_sequences * self.config.block_size,
            "indic_tokens": indic_sequences * self.config.block_size,
            "total_tokens": (fw_sequences + indic_sequences)
            * self.config.block_size,
        }


class HardShiftSchedule(MixtureSchedule):
    """A single hard 50:50 -> 80:20 transition at ``shift_tokens``."""

    def __init__(self, config: ScheduleConfig, global_batch_sequences: int):
        config.validate(global_batch_sequences)
        total_steps = config.total_tokens // config.block_size // global_batch_sequences
        shift_step = config.shift_tokens // config.block_size // global_batch_sequences
        pre_fw = int(config.pre_fineweb_share * global_batch_sequences)
        post_fw = int(config.post_fineweb_share * global_batch_sequences)
        counts = np.full(total_steps, post_fw, dtype=np.int64)
        counts[:shift_step] = pre_fw
        super().__init__(
            config,
            global_batch_sequences,
            counts,
            mode="hard",
            transition_start_step=shift_step,
            transition_end_step=shift_step,
        )


class LinearTransitionSchedule(MixtureSchedule):
    """A centred linear transition with the same full-run source totals as P7.

    The transition is symmetric around the 10B boundary. Its midpoint-sampled
    shares average 65%, exactly matching a hard schedule that spends half of the
    same window at 50% FineWeb and half at 80% FineWeb. Consequently the hard and
    linear arms both consume exactly 13B FineWeb and 7B IndicCorp tokens.
    """

    def __init__(
        self,
        config: ScheduleConfig,
        global_batch_sequences: int,
        *,
        transition_tokens: int,
    ) -> None:
        config.validate(global_batch_sequences)
        global_batch_tokens = global_batch_sequences * config.block_size
        if transition_tokens <= 0:
            raise ValueError("transition_tokens must be positive for linear mode")
        if transition_tokens % global_batch_tokens:
            raise ValueError(
                f"transition_tokens={transition_tokens:,} must be divisible by "
                f"global_batch_tokens={global_batch_tokens:,}."
            )
        transition_steps = transition_tokens // global_batch_tokens
        if transition_steps % 2:
            raise ValueError(
                "A centred linear transition requires an even number of optimizer steps."
            )

        total_steps = config.total_tokens // global_batch_tokens
        shift_step = config.shift_tokens // global_batch_tokens
        start = shift_step - transition_steps // 2
        end = shift_step + transition_steps // 2
        if start < 0 or end > total_steps:
            raise ValueError("The centred transition does not fit inside the run")

        pre_fw = int(config.pre_fineweb_share * global_batch_sequences)
        post_fw = int(config.post_fineweb_share * global_batch_sequences)
        counts = np.full(total_steps, post_fw, dtype=np.int64)
        counts[:start] = pre_fw

        # Cumulative rounding distributes fractional examples across steps while
        # preserving the exact transition total. Midpoint shares preserve the
        # same 13B/7B complete-run accounting as the hard arm.
        cumulative = Fraction(0, 1)
        allocated = 0
        delta = config.post_fineweb_share - config.pre_fineweb_share
        for offset in range(transition_steps):
            midpoint = Fraction(2 * offset + 1, 2 * transition_steps)
            desired_share = config.pre_fineweb_share + delta * midpoint
            cumulative += desired_share * global_batch_sequences
            rounded = (2 * cumulative.numerator + cumulative.denominator) // (
                2 * cumulative.denominator
            )
            counts[start + offset] = rounded - allocated
            allocated = rounded

        expected_transition_fw = int(
            Fraction(
                config.pre_fineweb_share + config.post_fineweb_share, 2
            )
            * global_batch_sequences
            * transition_steps
        )
        if int(counts[start:end].sum()) != expected_transition_fw:
            raise RuntimeError("Linear transition integer allocation did not close")

        super().__init__(
            config,
            global_batch_sequences,
            counts,
            mode="linear",
            transition_start_step=start,
            transition_end_step=end,
        )


def build_schedule(
    *,
    mode: str,
    config: ScheduleConfig,
    global_batch_sequences: int,
    transition_tokens: int,
) -> MixtureSchedule:
    if mode == "hard":
        return HardShiftSchedule(config, global_batch_sequences)
    if mode == "linear":
        return LinearTransitionSchedule(
            config,
            global_batch_sequences,
            transition_tokens=transition_tokens,
        )
    raise ValueError("mode must be 'hard' or 'linear'")


def affine_permutation_index(
    draw_ordinal: int,
    pool_size: int,
    seed: int,
) -> tuple[int, int]:
    """Map a draw ordinal to a no-replacement index within each pool pass."""
    if draw_ordinal < 0:
        raise ValueError("draw_ordinal must be non-negative")
    if pool_size <= 1:
        raise ValueError("pool_size must be greater than one")

    pass_number, position = divmod(draw_ordinal, pool_size)
    ss = np.random.SeedSequence([seed, pass_number, 0xA771])
    state = ss.generate_state(2, dtype=np.uint64)
    a = int(state[0] % pool_size)
    if a == 0:
        a = 1
    while gcd(a, pool_size) != 1:
        a = (a + 1) % pool_size
        if a == 0:
            a = 1
    b = int(state[1] % pool_size)
    return (a * position + b) % pool_size, pass_number


def count_sources(pattern: Iterable[int]) -> dict[str, int]:
    fw = 0
    indic = 0
    for source in pattern:
        if source == FINEWEB:
            fw += 1
        elif source == INDICCORP:
            indic += 1
        else:
            raise ValueError(f"unknown source id: {source}")
    return {"fineweb_edu": fw, "indiccorp_v2": indic}
