from fractions import Fraction

from p7.schedule import (
    FINEWEB,
    INDICCORP,
    HardShiftSchedule,
    LinearTransitionSchedule,
    ScheduleConfig,
    affine_permutation_index,
    count_sources,
)


def test_default_hard_schedule_exact_counts():
    schedule = HardShiftSchedule(ScheduleConfig(), global_batch_sequences=200)
    assert schedule.shift_step == 78_125
    assert schedule.total_steps == 156_250
    assert schedule.transition_start_step == schedule.shift_step
    assert schedule.transition_end_step == schedule.shift_step
    assert schedule.counts_for_step(0) == (100, 100)
    assert schedule.counts_for_step(schedule.shift_step - 1) == (100, 100)
    assert schedule.counts_for_step(schedule.shift_step) == (160, 40)
    assert schedule.expected_totals() == {
        "fineweb_sequences": 20_312_500,
        "indic_sequences": 10_937_500,
        "fineweb_tokens": 13_000_000_000,
        "indic_tokens": 7_000_000_000,
        "total_tokens": 20_000_000_000,
    }


def test_linear_control_preserves_complete_run_totals():
    schedule = LinearTransitionSchedule(
        ScheduleConfig(),
        global_batch_sequences=200,
        transition_tokens=256_000_000,
    )
    assert schedule.transition_steps == 2_000
    assert schedule.transition_start_step == 77_125
    assert schedule.transition_end_step == 79_125
    assert schedule.counts_for_step(schedule.transition_start_step - 1) == (100, 100)
    assert schedule.counts_for_step(schedule.transition_end_step) == (160, 40)
    assert schedule.expected_totals() == {
        "fineweb_sequences": 20_312_500,
        "indic_sequences": 10_937_500,
        "fineweb_tokens": 13_000_000_000,
        "indic_tokens": 7_000_000_000,
        "total_tokens": 20_000_000_000,
    }
    transition_fw = sum(
        schedule.counts_for_step(step)[0]
        for step in range(schedule.transition_start_step, schedule.transition_end_step)
    )
    assert transition_fw == 130 * schedule.transition_steps


def test_patterns_have_exact_ratios():
    schedule = HardShiftSchedule(ScheduleConfig(), global_batch_sequences=200)
    assert count_sources(schedule.pattern_for_step(10)) == {
        "fineweb_edu": 100,
        "indiccorp_v2": 100,
    }
    assert count_sources(schedule.pattern_for_step(schedule.shift_step)) == {
        "fineweb_edu": 160,
        "indiccorp_v2": 40,
    }


def test_local_draws_are_unique_across_one_step():
    schedule = HardShiftSchedule(ScheduleConfig(), global_batch_sequences=200)
    seen = []
    for micro in range(25):
        for rank in range(8):
            seen.extend(
                schedule.local_draws(
                    step=0,
                    micro_step=micro,
                    rank=rank,
                    world_size=8,
                    per_device_batch_size=1,
                    gradient_accumulation_steps=25,
                )
            )
    assert len(seen) == 200
    fw = [ordinal for source, ordinal in seen if source == FINEWEB]
    indic = [ordinal for source, ordinal in seen if source == INDICCORP]
    assert sorted(fw) == list(range(100))
    assert sorted(indic) == list(range(100))


def test_affine_permutation_has_no_duplicates_within_pass():
    size = 101
    first = [affine_permutation_index(i, size, 42)[0] for i in range(size)]
    second = [affine_permutation_index(size + i, size, 42)[0] for i in range(size)]
    assert len(set(first)) == size
    assert len(set(second)) == size
    assert first != second
    assert affine_permutation_index(size, size, 42)[1] == 1


def test_custom_small_hard_and_linear_schedules_match_totals():
    config = ScheduleConfig(
        total_tokens=20_000,
        shift_tokens=10_000,
        block_size=10,
        pre_fineweb_share=Fraction(1, 2),
        post_fineweb_share=Fraction(4, 5),
    )
    hard = HardShiftSchedule(config, global_batch_sequences=100)
    linear = LinearTransitionSchedule(
        config,
        global_batch_sequences=100,
        transition_tokens=4_000,
    )
    assert hard.shift_step == 10
    assert hard.total_steps == 20
    assert hard.expected_totals()["fineweb_tokens"] == 13_000
    assert hard.expected_totals()["indic_tokens"] == 7_000
    assert linear.expected_totals() == hard.expected_totals()
    assert linear.transition_start_step == 8
    assert linear.transition_end_step == 12
