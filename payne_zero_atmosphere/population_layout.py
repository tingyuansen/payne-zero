"""Packed ion-stage population layout and fill schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationJob:
    """One packed population-fill request."""

    code: float
    mode: int
    start_slot: int
    output_slots: int

    @property
    def target(self) -> str:
        return (
            "ion_stage_populations_by_packed_slot"
            if self.mode == 12
            else "partition_normalized_populations_by_packed_slot"
        )


def ion_stage_count_for_atomic_number(atomic_number: int) -> int:
    """Return the supported ion-stage count for one element."""

    z = int(atomic_number)
    if z == 1:
        return 2
    if z == 2:
        return 3
    if z in (3, 4, 5):
        return 4
    if 6 <= z <= 16:
        return 6
    if 17 <= z <= 28:
        return 5
    if z in (29, 30):
        return 3
    if z >= 31:
        return 3
    return 5


def atomic_population_slot_start(atomic_number: int) -> int:
    """Return the zero-based packed population slot where element Z begins."""

    z = int(atomic_number)
    if z <= 30:
        start_1based = 1 + ((z - 1) * (z + 2)) // 2
        return start_1based - 1
    return 496 + (z - 31) * 5 - 1


def decode_population_code(code: float) -> tuple[int, int]:
    """Decode a packed population code into atomic number and stage count."""

    atomic_number = int(code)
    fractional_code = float(code) - float(atomic_number)
    ion_count = int(fractional_code * 100.0 + 1.5)
    return atomic_number, max(1, ion_count)


def output_slots_from_code(code: float) -> int:
    """Return the number of output slots encoded by a population code."""

    return decode_population_code(code)[1]


def population_job_schedule(*, include_molecules: bool) -> list[PopulationJob]:
    """Return the validated packed population-fill sequence."""

    mode12_calls = [
        (1.01, 1),
        (2.02, 3),
        (3.03, 6),
        (4.03, 10),
        (5.03, 15),
        (6.05, 21),
        (7.05, 28),
        (8.05, 36),
        (9.05, 45),
        (10.05, 55),
        (11.05, 66),
        (12.05, 78),
        (13.05, 91),
        (14.05, 105),
        (15.05, 120),
        (16.05, 136),
        (17.04, 153),
        (18.04, 171),
        (19.04, 190),
        (20.04, 210),
        (21.04, 231),
        (22.04, 253),
        (23.04, 276),
        (24.04, 300),
        (25.04, 325),
        (26.04, 351),
        (27.04, 378),
        (28.04, 406),
        (29.02, 435),
        (30.02, 465),
    ]
    mode11_calls = [
        (1.01, 1),
        (2.02, 3),
        (3.03, 6),
        (4.03, 10),
        (5.03, 15),
        (6.05, 21),
        (7.05, 28),
        (8.05, 36),
        (9.05, 45),
        (10.05, 55),
        (11.05, 66),
        (12.05, 78),
        (13.05, 91),
        (14.05, 105),
        (15.05, 120),
        (16.05, 136),
        (17.05, 153),
        (18.04, 171),
        (19.05, 190),
        (20.09, 210),
        (21.09, 231),
        (22.09, 253),
        (23.09, 276),
        (24.09, 300),
        (25.09, 325),
        (26.09, 351),
        (27.09, 378),
        (28.09, 406),
        (29.02, 435),
        (30.02, 465),
    ]

    jobs: list[PopulationJob] = []
    for code, start_1based in mode12_calls:
        jobs.append(
            PopulationJob(
                code=code,
                mode=12,
                start_slot=start_1based - 1,
                output_slots=output_slots_from_code(code),
            )
        )
    for code, start_1based in mode11_calls:
        jobs.append(
            PopulationJob(
                code=code,
                mode=11,
                start_slot=start_1based - 1,
                output_slots=output_slots_from_code(code),
            )
        )

    for atomic_number in range(31, 100):
        code = float(atomic_number) + 0.02
        start_slot = 495 + (atomic_number - 31) * 5
        output_slots = output_slots_from_code(code)
        jobs.append(
            PopulationJob(
                code=code, mode=11, start_slot=start_slot, output_slots=output_slots
            )
        )
        jobs.append(
            PopulationJob(
                code=code, mode=12, start_slot=start_slot, output_slots=output_slots
            )
        )

    if not include_molecules:
        return jobs

    molecular_targets = [
        (101.00, 841),
        (106.00, 846),
        (107.00, 847),
        (108.00, 848),
        (112.00, 851),
        (114.00, 853),
        (120.00, 858),
        (124.00, 862),
        (126.00, 864),
        (606.00, 868),
        (607.00, 869),
        (608.00, 870),
        (814.00, 889),
        (822.00, 895),
        (823.00, 896),
        (10108.00, 940),
    ]
    for code, slot_1based in molecular_targets:
        start_slot = slot_1based - 1
        jobs.append(
            PopulationJob(code=code, mode=1, start_slot=start_slot, output_slots=1)
        )
        jobs.append(
            PopulationJob(code=code, mode=11, start_slot=start_slot, output_slots=1)
        )
    return jobs
