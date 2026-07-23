"""Torch molecular equilibrium for cool-star synthesis.

Reads `molecular_equilibrium_synthesis.npz` (the synthesis catalog, 190
molecules — deliberately distinct from the atmosphere-stage catalog) and
solves the dissociation network for molecular number densities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .constants import (
    BOLTZMANN_ERG_PER_K,
    BOLTZMANN_EV_PER_K,
    REFERENCE_SAHA_COEFFICIENT,
)
from .device import DEFAULT_DTYPE, device as _device
from . import paths as engine_paths

MAX_MOLECULES = 200
MAX_MOLECULAR_EQUATIONS = 30
MAX_MOLECULAR_COMPONENTS = 3 * MAX_MOLECULES


@dataclass(frozen=True)
class MoleculeTable:
    """Decoded molecule table with fixed-size buffers used by the solver."""

    molecule_count: int
    molecule_codes: np.ndarray
    equilibrium_coefficients: np.ndarray
    component_start_indices: np.ndarray
    component_equation_indices: np.ndarray
    equation_species_codes: np.ndarray
    equation_count: int


def _default_molecule_table() -> Path:
    return engine_paths.source_catalog_path(
        "lines", "molecular_equilibrium_synthesis.npz"
    )


def read_molecule_table(molecules_path: Path) -> MoleculeTable:
    """Read the molecular-equilibrium species table into fixed solver buffers.

    Canonical form: an ``.npz`` holding the decoded buffers. The legacy text
    catalog is still parsed for provenance tooling.
    """
    molecules_path = Path(molecules_path)
    if molecules_path.suffix == ".npz":
        with np.load(molecules_path, allow_pickle=False) as arrays:
            return MoleculeTable(
                molecule_count=int(arrays["molecule_count"]),
                molecule_codes=np.asarray(arrays["molecule_codes"]),
                equilibrium_coefficients=np.asarray(arrays["equilibrium_coefficients"]),
                component_start_indices=np.asarray(arrays["component_start_indices"]),
                component_equation_indices=np.asarray(
                    arrays["component_equation_indices"]
                ),
                equation_species_codes=np.asarray(arrays["equation_species_codes"]),
                equation_count=int(arrays["equation_count"]),
            )
    molecule_codes = np.zeros(MAX_MOLECULES, dtype=np.float64)
    equilibrium_coefficients = np.zeros((7, MAX_MOLECULES), dtype=np.float64)
    component_start_indices = np.zeros(MAX_MOLECULES + 1, dtype=np.int32)
    component_indices = np.zeros(MAX_MOLECULAR_COMPONENTS, dtype=np.int32)
    equation_species_codes = np.zeros(MAX_MOLECULAR_EQUATIONS, dtype=np.int32)
    element_is_used = np.zeros(102, dtype=np.int32)

    base100_place_values = np.array(
        [1e14, 1e12, 1e10, 1e8, 1e6, 1e4, 1e2, 1e0],
        dtype=np.float64,
    )

    component_cursor = 0
    component_start_indices[0] = 0
    molecule_count = 0

    table_path = (
        Path(molecules_path)
        if molecules_path is not None
        else _default_molecule_table()
    )
    for raw in table_path.read_text().splitlines():
        line = raw.rstrip("\n\r")
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("C")
            or stripped.startswith("c")
            or stripped.startswith("#")
        ):
            continue
        molecule_code_text = line[0 : min(18, len(line))].strip()
        if not molecule_code_text:
            continue
        try:
            molecule_code = float(molecule_code_text)
        except ValueError:
            continue
        coefficient_columns = [
            (18, 25),
            (25, 36),
            (36, 47),
            (47, 58),
            (58, 69),
            (69, 80),
            (80, 91),
        ]
        coefficient_values = [0.0] * 7
        for coefficient_index, (start, stop) in enumerate(coefficient_columns):
            if len(line) >= stop:
                coefficient_text = line[start:stop].strip()
                if coefficient_text:
                    coefficient_values[coefficient_index] = float(coefficient_text)
        if molecule_code == 0.0 or abs(molecule_code) < 1e-12:
            continue

        first_active_place = 0
        for place_index, place_value in enumerate(base100_place_values):
            if molecule_code >= place_value:
                first_active_place = place_index
                break
        remaining_code = molecule_code
        for place_index in range(first_active_place, base100_place_values.size):
            element_id = int(remaining_code / base100_place_values[place_index])
            remaining_code -= float(element_id) * base100_place_values[place_index]
            if element_id == 0:
                element_id = 100  # electron
            element_is_used[element_id] = 1
            component_indices[component_cursor] = element_id
            component_cursor += 1
        ion_charge = int(remaining_code * 100.0 + 0.5)
        if ion_charge >= 1:
            element_is_used[100] = 1
            element_is_used[101] = 1
            for inverse_electron_index in range(ion_charge):
                component_indices[component_cursor] = 101  # inverse electron
                component_cursor += 1

        component_start_indices[molecule_count + 1] = component_cursor
        molecule_codes[molecule_count] = molecule_code
        for coefficient_index in range(7):
            equilibrium_coefficients[coefficient_index, molecule_count] = (
                coefficient_values[coefficient_index]
            )
        molecule_count += 1

    component_count = component_cursor

    equation_count = 1
    for element_id in range(1, 101):
        if element_is_used[element_id] == 1:
            equation_count += 1
            element_is_used[element_id] = equation_count
            equation_species_codes[equation_count - 1] = element_id
    element_is_used[101] = equation_count + 1

    for component_index in range(component_count):
        component_indices[component_index] = (
            element_is_used[component_indices[component_index]] - 1
        )

    return MoleculeTable(
        molecule_count=molecule_count,
        molecule_codes=molecule_codes,
        equilibrium_coefficients=equilibrium_coefficients,
        component_start_indices=component_start_indices,
        component_equation_indices=component_indices,
        equation_species_codes=equation_species_codes,
        equation_count=equation_count,
    )


def polynomial_formation_constants(temperature, molecule_table: MoleculeTable):
    """Return host-fp64 formation constants shaped ``(depth, molecule)``."""
    temperature = np.asarray(temperature, dtype=np.float64)
    n_depths = temperature.shape[0]
    molecule_count = molecule_table.molecule_count
    molecule_codes = molecule_table.molecule_codes
    equilibrium_coefficients = molecule_table.equilibrium_coefficients
    component_start_indices = molecule_table.component_start_indices
    thermal_energy_ev = temperature * BOLTZMANN_EV_PER_K
    natural_log_temperature = np.log(temperature)
    formation_constants = np.zeros((n_depths, molecule_count), dtype=np.float64)

    component_counts = (
        component_start_indices[1 : molecule_count + 1]
        - component_start_indices[:molecule_count]
    ).astype(np.float64)
    integer_molecule_codes = (
        molecule_codes[:molecule_count].astype(np.int64).astype(np.float64)
    )
    ion_charges = np.rint(
        (molecule_codes[:molecule_count] - integer_molecule_codes) * 100.0
    ).astype(np.float64)

    is_polynomial_molecule = equilibrium_coefficients[0, :molecule_count] != 0.0
    is_hminus = np.abs(molecule_codes[:molecule_count] - 101.0) < 1e-9
    cool = temperature <= 10000.0

    for molecule_index in range(molecule_count):
        if not is_polynomial_molecule[molecule_index]:
            continue
        if is_hminus[molecule_index]:
            hminus_log_formation = (
                4.478 / thermal_energy_ev
                - 46.4584
                + (
                    1.63660e-3
                    + (
                        -4.93992e-7
                        + (
                            1.11822e-10
                            + (
                                -1.49567e-14
                                + (1.06206e-18 - 3.08720e-23 * temperature)
                                * temperature
                            )
                            * temperature
                        )
                        * temperature
                    )
                    * temperature
                )
                * temperature
                - 1.5 * natural_log_temperature
            )
            formation_constants[:, molecule_index] = np.where(
                cool, np.exp(hminus_log_formation), 0.0
            )
            continue
        (
            inverse_temperature_coefficient,
            constant_coefficient,
            linear_coefficient,
            quadratic_coefficient,
            cubic_coefficient,
            quartic_coefficient,
            quintic_coefficient,
        ) = (
            equilibrium_coefficients[coefficient_index, molecule_index]
            for coefficient_index in range(7)
        )
        polynomial = (
            inverse_temperature_coefficient / thermal_energy_ev
            - constant_coefficient
            + (
                linear_coefficient
                + (
                    -quadratic_coefficient
                    + (
                        cubic_coefficient
                        + (-quartic_coefficient + quintic_coefficient * temperature)
                        * temperature
                    )
                    * temperature
                )
                * temperature
            )
            * temperature
        )
        log_temperature_term = (
            -1.5
            * (
                component_counts[molecule_index]
                - 2.0 * ion_charges[molecule_index]
                - 1.0
            )
            * natural_log_temperature
        )
        formation_constants[:, molecule_index] = np.where(
            cool, np.exp(polynomial + log_temperature_term), 0.0
        )
    return formation_constants


@dataclass
class MolecularStructure:
    """Branchless encoding of every molecule's component multiset.

    From the ragged molecule table we precompute multiplicity matrices and
    electron powers, so each molecular term is one vectorized product.
    """

    equation_count: int
    electron_equation_index: int
    component_multiplicity: torch.Tensor
    inverse_electron_power: torch.Tensor
    negative_ion_flag: torch.Tensor
    active_molecule_mask: torch.Tensor
    full_component_multiplicity: torch.Tensor
    full_inverse_electron_power: torch.Tensor

    @classmethod
    def build(cls, molecule_table: MoleculeTable, *, device, dtype):
        molecule_count = molecule_table.molecule_count
        equation_count = molecule_table.equation_count
        component_start_indices = molecule_table.component_start_indices
        component_equation_indices = molecule_table.component_equation_indices
        equation_species_codes = molecule_table.equation_species_codes

        electron_index = (
            equation_count - 1
            if equation_species_codes[equation_count - 1] == 100
            else -1
        )
        all_component_counts = np.zeros(
            (molecule_count, equation_count), dtype=np.float64
        )
        all_inverse_electron_powers = np.zeros(molecule_count, dtype=np.float64)
        negative_ion_flag = np.zeros(molecule_count, dtype=np.float64)
        active_molecule_mask = np.zeros(molecule_count, dtype=np.float64)
        for molecule_index in range(molecule_count):
            start = int(component_start_indices[molecule_index])
            stop = int(component_start_indices[molecule_index + 1])
            if stop - start > 1:
                active_molecule_mask[molecule_index] = 1.0
                if (
                    int(component_equation_indices[stop - 1]) == equation_count - 1
                    and equation_species_codes[equation_count - 1] == 100
                ):
                    negative_ion_flag[molecule_index] = 1.0
            for component_index in range(start, stop):
                equation_index = int(component_equation_indices[component_index])
                if equation_index >= equation_count:
                    all_inverse_electron_powers[molecule_index] += 1.0
                else:
                    all_component_counts[molecule_index, equation_index] += 1.0
        active_component_counts = all_component_counts * active_molecule_mask[:, None]
        active_inverse_electron_powers = (
            all_inverse_electron_powers * active_molecule_mask
        )

        def as_device_tensor(array):
            return torch.as_tensor(array, dtype=dtype, device=device)

        return cls(
            equation_count=equation_count,
            electron_equation_index=electron_index,
            component_multiplicity=as_device_tensor(active_component_counts),
            inverse_electron_power=as_device_tensor(active_inverse_electron_powers),
            negative_ion_flag=as_device_tensor(negative_ion_flag),
            active_molecule_mask=as_device_tensor(active_molecule_mask),
            full_component_multiplicity=as_device_tensor(all_component_counts),
            full_inverse_electron_power=as_device_tensor(all_inverse_electron_powers),
        )


def _safe_log(value):
    """log(x) with a dtype-safe positive floor.

    The molecular product is built in log space; a tiny/zero density must give a
    very-negative-but-FINITE log so that ``count * log`` is 0 where count is 0
    (0 * (-inf) = NaN otherwise).  fp32's smallest normal is ~1.2e-38, so a fixed
    1e-300 floor underflows to 0 in fp32 -> we floor with finfo(dtype).tiny.
    """
    tiny = torch.finfo(value.dtype).tiny
    return torch.log(value.clamp_min(tiny))


def _residual(
    densities,
    natural_log_formation_constants,
    equation_abundance,
    total_particle_density,
    structure: MolecularStructure,
):
    """Coupled molecular-equilibrium residual for one depth."""
    electron_index = structure.electron_equation_index

    # Atomic conservation, total-particle, and charge rows.
    residual = densities - equation_abundance * densities[0]
    total_row = densities[1:].sum() - total_particle_density
    total_density_selector = torch.zeros_like(densities)
    total_density_selector[0] = 1.0
    residual = (
        residual
        - total_density_selector * residual
        + total_density_selector * total_row
    )
    if electron_index >= 0:
        onehot_e = torch.zeros_like(densities)
        onehot_e[electron_index] = 1.0
        residual = (
            residual - onehot_e * residual + onehot_e * (-densities[electron_index])
        )

    # Molecular products are built in log space so fp32 never stores huge
    # intermediate density products directly.
    log_densities = _safe_log(densities)
    log_term = natural_log_formation_constants + (
        structure.component_multiplicity * log_densities
    ).sum(dim=1)
    if electron_index >= 0:
        log_term = (
            log_term - structure.inverse_electron_power * log_densities[electron_index]
        )
    term = torch.exp(log_term) * structure.active_molecule_mask
    term = torch.where(torch.isfinite(term), term, torch.zeros_like(term))

    residual = residual + total_density_selector * term.sum()
    residual = residual + (structure.component_multiplicity * term.unsqueeze(1)).sum(
        dim=0
    )
    if electron_index >= 0:
        # inverse-electron sentinel adds +term to the electron row; negative-ion
        # correction subtracts 2*term from it.
        residual = residual + onehot_e * (structure.inverse_electron_power * term).sum()
        residual = (
            residual + onehot_e * (-2.0 * structure.negative_ion_flag * term).sum()
        )
    return residual


def _newton_step(jacobian, residual, densities):
    """Solve the Newton step with column scaling for conditioning.

    The raw Jacobian spans ~25 orders of magnitude (atomic rows are O(1); the
    molecular rows are enormous), so a plain solve is ill-conditioned in fp32.
    Solving for fractional updates equilibrates the columns without changing the
    nonlinear equations.
    """
    column_scale = densities.abs().clamp_min(torch.finfo(jacobian.dtype).tiny)
    scaled_jacobian = jacobian * column_scale.unsqueeze(0)
    fractional_step = torch.linalg.solve(scaled_jacobian, residual)
    return column_scale * fractional_step


def _molecular_densities(
    densities,
    natural_log_formation_constants,
    structure: MolecularStructure,
):
    """Final molecular number densities.

    Same product as the residual term but without the active mask: single
    atoms/ions report their own number density, matching the molecular population
    table assembled after convergence.
    """
    electron_index = structure.electron_equation_index
    log_densities = _safe_log(densities)
    natural_log_molecular_density = natural_log_formation_constants + (
        structure.full_component_multiplicity * log_densities
    ).sum(dim=1)
    if electron_index >= 0:
        natural_log_molecular_density = (
            natural_log_molecular_density
            - structure.full_inverse_electron_power * log_densities[electron_index]
        )
    return torch.exp(natural_log_molecular_density)


def solve_molecular_equilibrium(
    temperature,
    gas_pressure,
    electron_density,
    elemental_abundances,
    ion_formation_constants,
    *,
    molecules_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    max_iter: int = 200,
    tol: float = 1e-3,
    chain_length: Optional[int] = None,
    return_diagnostics: bool = False,
):
    """Solve molecular equilibrium and return densities for every atmosphere layer."""
    if device is None:
        device = _device()
    # MPS has no fp64; fall back to the requested dtype only where the device allows.
    if dtype == torch.float64 and device.type == "mps":
        dtype = torch.float32

    temperature_array = np.asarray(temperature, dtype=np.float64)
    gas_pressure_array = np.asarray(gas_pressure, dtype=np.float64)
    electron_density_array = np.asarray(electron_density, dtype=np.float64)
    elemental_abundances = np.asarray(elemental_abundances, dtype=np.float64)
    ion_formation_constants = np.asarray(ion_formation_constants, dtype=np.float64)
    n_depths = temperature_array.shape[0]

    if molecules_path is None:
        molecules_path = _default_molecule_table()
    molecule_table = read_molecule_table(molecules_path)
    molecule_count = molecule_table.molecule_count
    equation_count = molecule_table.equation_count
    equilibrium_coefficients = molecule_table.equilibrium_coefficients
    component_start_indices = molecule_table.component_start_indices
    equation_species_codes = molecule_table.equation_species_codes

    # Formation constants span far beyond fp32, so the Newton solve receives
    # their logarithms. Zero/inactive constants use a finite negative sentinel.
    formation_constants = polynomial_formation_constants(
        temperature_array, molecule_table
    )
    ion_mask = equilibrium_coefficients[0, :molecule_count] == 0.0
    component_counts = (
        component_start_indices[1 : molecule_count + 1]
        - component_start_indices[:molecule_count]
    )
    single_mask = ion_mask & (component_counts == 1)
    ion_ratio_mask = ion_mask & (component_counts > 1)
    # Bare atoms/ions have unit formation constants; multi-component atomic ions
    # receive the Saha ratio supplied by the EOS.
    formation_constants[:, single_mask] = 1.0
    formation_constants[:, ion_ratio_mask] = ion_formation_constants[
        :, :molecule_count
    ][:, ion_ratio_mask]
    log_zero_sentinel = -700.0
    with np.errstate(divide="ignore"):
        natural_log_formation_constants = np.where(
            formation_constants[:, :molecule_count] > 0.0,
            np.log(np.maximum(formation_constants[:, :molecule_count], 1e-300)),
            log_zero_sentinel,
        )

    structure = MolecularStructure.build(molecule_table, device=device, dtype=dtype)
    electron_index = structure.electron_equation_index

    equation_abundance = np.zeros(equation_count, dtype=np.float64)
    for equation_index in range(1, equation_count):
        element_id = equation_species_codes[equation_index]
        if element_id < 100:
            equation_abundance[equation_index] = max(
                elemental_abundances[element_id - 1], 1e-20
            )

    thermal_energy_erg = temperature_array * BOLTZMANN_ERG_PER_K
    total_particle_density = gas_pressure_array / thermal_energy_erg

    def as_device_tensor(array):
        return torch.as_tensor(array, dtype=dtype, device=device)

    log_formation_constants_t = as_device_tensor(natural_log_formation_constants)
    equation_abundance_t = as_device_tensor(equation_abundance)
    total_particle_density_t = as_device_tensor(total_particle_density)

    # Autodiff gives the exact Jacobian of the residual actually being solved.
    from torch.func import jacrev

    def resid_one(density_row, natural_log_formation_row, total_density_row):
        return _residual(
            density_row,
            natural_log_formation_row,
            equation_abundance_t,
            total_density_row,
            structure,
        )

    jac_one = jacrev(resid_one, argnums=0)

    # Each layer solves independently, but its initial guess follows the
    # pressure-scaled solution from the previous layer.
    converged_iters = np.zeros(n_depths, dtype=np.int64)
    converged_density_rows = []
    chain_len = n_depths if chain_length is None else max(1, int(chain_length))
    previous_converged_density = None
    for depth_index in range(n_depths):
        chain_start = (depth_index % chain_len) == 0
        if chain_start:
            initial_total_density = total_particle_density[depth_index] / 2.0
            if temperature_array[depth_index] < 4000.0:
                initial_total_density = total_particle_density[depth_index]
            base_density = initial_total_density / 10.0
            seed = np.zeros(equation_count, dtype=np.float64)
            seed[0] = initial_total_density
            seed[1:equation_count] = base_density * equation_abundance[1:equation_count]
            if electron_index >= 0:
                seed[electron_index] = base_density
            current_density = as_device_tensor(seed)
        else:
            pressure_ratio = float(
                gas_pressure_array[depth_index] / gas_pressure_array[depth_index - 1]
            )
            current_density = previous_converged_density * pressure_ratio
        natural_log_formation_row = log_formation_constants_t[depth_index]
        total_density_row = total_particle_density_t[depth_index]
        previous_update = torch.zeros(equation_count, dtype=dtype, device=device)

        for iteration_index in range(max_iter):
            residual = resid_one(
                current_density, natural_log_formation_row, total_density_row
            )
            jacobian = jac_one(
                current_density, natural_log_formation_row, total_density_row
            )
            delta = _newton_step(jacobian, residual, current_density)

            # Test convergence before damping; changing that order changes parity.
            relative_update = delta.abs() / current_density.abs().clamp_min(
                torch.finfo(dtype).tiny
            )
            needs_more_iterations = bool((relative_update > tol).any().item())

            # Relax sign-flipping updates and keep densities positive.
            sign_change = ((previous_update > 0) & (delta < 0)) | (
                (previous_update < 0) & (delta > 0)
            )
            delta = torch.where(sign_change, delta * 0.69, delta)

            candidate_density = current_density - delta
            too_small = candidate_density < (current_density / 100.0)
            current_density = torch.where(
                too_small, current_density / 100.0, candidate_density
            )
            previous_update = delta

            if not needs_more_iterations:
                converged_iters[depth_index] = iteration_index + 1
                break
        else:
            converged_iters[depth_index] = max_iter

        previous_converged_density = current_density.detach()
        converged_density_rows.append(previous_converged_density)

    final_densities = torch.stack(converged_density_rows, dim=0)

    heavy_nucleus_density = final_densities[:, 0]
    electron = (
        final_densities[:, electron_index]
        if electron_index >= 0
        else as_device_tensor(electron_density_array)
    )
    from torch.func import vmap

    batched_moldens = vmap(
        lambda density_row, natural_log_formation_row: _molecular_densities(
            density_row, natural_log_formation_row, structure
        ),
        in_dims=(0, 0),
    )
    active_molecular_populations = batched_moldens(
        final_densities, log_formation_constants_t
    )

    equation_densities = torch.zeros(
        (n_depths, MAX_MOLECULAR_EQUATIONS), dtype=dtype, device=device
    )
    equation_densities[:, :equation_count] = final_densities
    molecular_populations = torch.zeros(
        (n_depths, MAX_MOLECULES), dtype=dtype, device=device
    )
    molecular_populations[:, :molecule_count] = active_molecular_populations

    if return_diagnostics:
        diag = {
            "molecule_count": molecule_count,
            "equation_count": equation_count,
            "equation_species_codes": equation_species_codes[:equation_count].copy(),
            "molecule_codes": molecule_table.molecule_codes[:molecule_count].copy(),
            "iterations_completed": converged_iters,
            "natural_log_formation_constants": log_formation_constants_t,
            "structure": structure,
        }
        return (
            heavy_nucleus_density,
            molecular_populations,
            equation_densities,
            electron,
            diag,
        )
    return heavy_nucleus_density, molecular_populations, equation_densities, electron


_ATOMIC_MASSES_FOR_MOLECULES = np.array(
    [
        1.008,
        4.003,
        6.939,
        9.013,
        10.81,
        12.01,
        14.01,
        16.00,
        19.00,
        20.18,
        22.99,
        24.31,
        26.98,
        28.09,
        30.98,
        32.07,
        35.45,
        39.95,
        39.10,
        40.08,
        44.96,
        47.90,
        50.94,
        52.00,
        54.94,
        55.85,
        58.94,
        58.71,
        63.55,
        65.37,
        69.72,
        72.60,
        74.92,
        78.96,
        79.91,
        83.80,
        85.48,
        87.63,
        88.91,
        91.22,
        92.91,
        95.95,
        99.00,
        101.1,
        102.9,
        106.4,
        107.9,
        112.4,
        114.8,
        118.7,
        121.8,
        127.6,
        126.9,
        131.3,
        132.9,
        137.4,
        138.9,
        140.1,
        140.9,
        144.3,
        147.0,
        150.4,
        152.0,
        157.3,
        158.9,
        162.5,
        164.9,
        167.3,
        168.9,
        173.0,
        175.0,
        178.5,
        181.0,
        183.9,
        186.3,
        190.2,
        192.2,
        195.1,
        197.0,
        200.6,
        204.4,
        207.2,
        209.0,
        210.0,
        211.0,
        222.0,
        223.0,
        226.1,
        227.1,
        232.0,
        231.0,
        238.0,
        237.0,
        244.0,
        243.0,
        247.0,
        247.0,
        251.0,
        254.0,
    ],
    dtype=np.float64,
)


def molecular_line_populations(
    *,
    temperature,
    equation_densities,
    neutral_partition,
    codes,
    molecules_path: Optional[Path] = None,
) -> np.ndarray:
    """Return molecular line populations ``N_mol / U_mol`` for requested species codes."""
    if molecules_path is None:
        molecules_path = _default_molecule_table()
    molecule_table = read_molecule_table(molecules_path)
    molecule_count = molecule_table.molecule_count
    molecule_codes = molecule_table.molecule_codes
    equilibrium_coefficients = molecule_table.equilibrium_coefficients
    component_start_indices = molecule_table.component_start_indices
    component_equation_indices = molecule_table.component_equation_indices
    equation_species_codes = molecule_table.equation_species_codes
    equation_count = molecule_table.equation_count

    temperature_array = np.asarray(temperature, dtype=np.float64)
    n_depths = temperature_array.shape[0]
    transformed_densities = np.array(equation_densities, dtype=np.float64, copy=True)
    neutral_partition = np.asarray(neutral_partition, dtype=np.float64)
    n_partition_columns = neutral_partition.shape[1]
    sqrt_temperature = np.sqrt(np.maximum(temperature_array, 1e-300))

    # Convert equation densities to the line-population normalization.
    for equation_index in range(1, equation_count):
        element_id = int(equation_species_codes[equation_index])
        if element_id <= 0:
            continue
        if element_id == 100:
            transformed_densities[:, equation_index] = transformed_densities[
                :, equation_index
            ] / (
                2.0 * REFERENCE_SAHA_COEFFICIENT * temperature_array * sqrt_temperature
            )
            continue
        atomic_mass = (
            float(_ATOMIC_MASSES_FOR_MOLECULES[element_id - 1])
            if 1 <= element_id <= _ATOMIC_MASSES_FOR_MOLECULES.size
            else float(element_id)
        )
        partition = (
            neutral_partition[:, element_id - 1]
            if 1 <= element_id <= n_partition_columns
            else np.ones(n_depths)
        )
        denominator = np.maximum(
            partition
            * 1.8786e20
            * np.sqrt(np.maximum((atomic_mass * temperature_array) ** 3, 1e-300)),
            1e-300,
        )
        transformed_densities[:, equation_index] = (
            transformed_densities[:, equation_index] / denominator
        )

    all_line_populations = np.zeros((n_depths, molecule_count), dtype=np.float64)
    thermal_energy_ev = temperature_array / 11604.5
    for molecule_index in range(molecule_count):
        leading_coefficient = float(equilibrium_coefficients[0, molecule_index])
        if leading_coefficient == 0.0:
            continue  # non-polynomial (atomic-ion) — none of the _MOL_SLOTS species hit this
        component_start = int(component_start_indices[molecule_index])
        component_stop = int(component_start_indices[molecule_index + 1])
        molecular_mass = 0.0
        for component_index in range(component_start, component_stop):
            equation_index = int(component_equation_indices[component_index])
            if equation_index >= equation_count:
                continue  # inverse-electron sentinel — no atomic mass contribution
            element_id = int(equation_species_codes[equation_index])
            if 1 <= element_id <= _ATOMIC_MASSES_FOR_MOLECULES.size:
                molecular_mass += float(_ATOMIC_MASSES_FOR_MOLECULES[element_id - 1])
        line_population = np.exp(
            leading_coefficient / np.maximum(thermal_energy_ev, 1e-300)
        )
        for component_index in range(component_start, component_stop):
            equation_index = int(component_equation_indices[component_index])
            if equation_index >= equation_count:
                line_population = line_population / np.maximum(
                    transformed_densities[:, equation_count - 1], 1e-300
                )
            else:
                line_population = (
                    line_population * transformed_densities[:, equation_index]
                )
        line_population = (
            line_population
            * 1.8786e20
            * np.sqrt(np.maximum((molecular_mass * temperature_array) ** 3, 1e-300))
        )
        all_line_populations[:, molecule_index] = line_population

    selected_populations = np.zeros((n_depths, len(codes)), dtype=np.float64)
    for column_index, code in enumerate(codes):
        hits = np.where(np.abs(molecule_codes[:molecule_count] - float(code)) < 1e-3)[0]
        if hits.size:
            selected_populations[:, column_index] = all_line_populations[
                :, int(hits[0])
            ]
    return selected_populations


def all_molecular_line_populations(
    *,
    temperature,
    equation_densities,
    molecular_populations,
    neutral_partition,
    partition_functions=None,
    molecules_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return species codes and molecular line populations for every molecule entry."""
    if molecules_path is None:
        molecules_path = _default_molecule_table()
    molecule_table = read_molecule_table(molecules_path)
    molecule_count = molecule_table.molecule_count
    molecule_codes = molecule_table.molecule_codes
    equilibrium_coefficients = molecule_table.equilibrium_coefficients
    component_start_indices = molecule_table.component_start_indices
    component_equation_indices = molecule_table.component_equation_indices
    equation_species_codes = molecule_table.equation_species_codes
    equation_count = molecule_table.equation_count

    temperature_array = np.asarray(temperature, dtype=np.float64)
    n_depths = temperature_array.shape[0]
    transformed_densities = np.array(equation_densities, dtype=np.float64, copy=True)
    molecular_populations = np.asarray(molecular_populations, dtype=np.float64)
    neutral_partition = np.asarray(neutral_partition, dtype=np.float64)
    partition_function_cube = (
        None
        if partition_functions is None
        else np.asarray(partition_functions, dtype=np.float64)
    )
    n_partition_columns = neutral_partition.shape[1]
    sqrt_temperature = np.sqrt(np.maximum(temperature_array, 1e-300))

    for equation_index in range(1, equation_count):
        element_id = int(equation_species_codes[equation_index])
        if element_id <= 0:
            continue
        if element_id == 100:
            transformed_densities[:, equation_index] = transformed_densities[
                :, equation_index
            ] / (
                2.0 * REFERENCE_SAHA_COEFFICIENT * temperature_array * sqrt_temperature
            )
            continue
        atomic_mass = (
            float(_ATOMIC_MASSES_FOR_MOLECULES[element_id - 1])
            if 1 <= element_id <= _ATOMIC_MASSES_FOR_MOLECULES.size
            else float(element_id)
        )
        partition = (
            neutral_partition[:, element_id - 1]
            if 1 <= element_id <= n_partition_columns
            else np.ones(n_depths)
        )
        denominator = np.maximum(
            partition
            * 1.8786e20
            * np.sqrt(np.maximum((atomic_mass * temperature_array) ** 3, 1e-300)),
            1e-300,
        )
        transformed_densities[:, equation_index] = (
            transformed_densities[:, equation_index] / denominator
        )

    all_line_populations = np.zeros((n_depths, molecule_count), dtype=np.float64)
    thermal_energy_ev = temperature_array / 11604.5
    for molecule_index in range(molecule_count):
        leading_coefficient = float(equilibrium_coefficients[0, molecule_index])
        if leading_coefficient == 0.0:
            atomic_number = int(molecule_codes[molecule_index])
            component_start = int(component_start_indices[molecule_index])
            component_stop = int(component_start_indices[molecule_index + 1])
            ion_stage = max(1, component_stop - component_start)
            if (
                partition_function_cube is not None
                and partition_function_cube.ndim == 3
                and 1 <= atomic_number <= partition_function_cube.shape[1]
                and 1 <= ion_stage <= partition_function_cube.shape[2]
            ):
                # For atomic-ion rows, use the partition function of the ion
                # stage represented by this table entry.
                partition = partition_function_cube[:, atomic_number - 1, ion_stage - 1]
            else:
                partition = (
                    neutral_partition[:, atomic_number - 1]
                    if 1 <= atomic_number <= n_partition_columns
                    else np.ones(n_depths)
                )
            all_line_populations[:, molecule_index] = molecular_populations[
                :, molecule_index
            ] / np.maximum(partition, 1e-300)
            continue

        component_start = int(component_start_indices[molecule_index])
        component_stop = int(component_start_indices[molecule_index + 1])
        molecular_mass = 0.0
        for component_index in range(component_start, component_stop):
            equation_index = int(component_equation_indices[component_index])
            if equation_index >= equation_count:
                continue
            element_id = int(equation_species_codes[equation_index])
            if 1 <= element_id <= _ATOMIC_MASSES_FOR_MOLECULES.size:
                molecular_mass += float(_ATOMIC_MASSES_FOR_MOLECULES[element_id - 1])
        line_population = np.exp(
            leading_coefficient / np.maximum(thermal_energy_ev, 1e-300)
        )
        for component_index in range(component_start, component_stop):
            equation_index = int(component_equation_indices[component_index])
            if equation_index >= equation_count:
                line_population = line_population / np.maximum(
                    transformed_densities[:, equation_count - 1], 1e-300
                )
            else:
                line_population = (
                    line_population * transformed_densities[:, equation_index]
                )
        line_population = (
            line_population
            * 1.8786e20
            * np.sqrt(np.maximum((molecular_mass * temperature_array) ** 3, 1e-300))
        )
        all_line_populations[:, molecule_index] = line_population

    return np.asarray(molecule_codes[:molecule_count], np.float64), all_line_populations


# Molecular line-list species id -> molecules.dat species-code mapping.  The
# line catalog uses species ids; the molecular-equilibrium table uses codes.
_SPECIES_CODE_TO_MOLECULE_CODES: dict[int, tuple[float, ...]] = {
    240: (101.0,),
    246: (106.0,),
    252: (107.0,),
    258: (108.0,),
    264: (606.0,),
    270: (607.0,),
    276: (608.0,),
    282: (707.0,),
    288: (708.0,),
    294: (808.0,),
    300: (112.0,),
    306: (113.0,),
    312: (114.0,),
    318: (812.0,),
    324: (813.0,),
    330: (814.0,),
    336: (116.0,),
    342: (120.0,),
    348: (816.0,),
    354: (820.0,),
    360: (821.0,),
    366: (822.0,),
    372: (823.0,),
    378: (103.0,),
    384: (104.0,),
    390: (105.0,),
    396: (109.0,),
    402: (115.0,),
    408: (117.0,),
    414: (121.0,),
    420: (122.0,),
    426: (123.0,),
    432: (124.0,),
    438: (125.0,),
    444: (126.0,),
    492: (111.0,),
    498: (119.0,),
    510: (817.0,),
    516: (824.0,),
    522: (825.0,),
    528: (826.0,),
    534: (10108.0,),
    540: (60808.0,),
    546: (10106.0,),
    552: (60606.0,),
    558: (127.0,),
    564: (128.0,),
    570: (129.0,),
    576: (827.0,),
    582: (828.0,),
    588: (829.0,),
    780: (839.0,),
    786: (840.0,),
    792: (857.0,),
}


def supported_molecular_species_codes() -> np.ndarray:
    """Molecular line-list species codes supported by this equilibrium bridge."""
    return np.asarray(sorted(_SPECIES_CODE_TO_MOLECULE_CODES), dtype=np.int64)


def molecular_line_populations_by_species_code(
    *,
    temperature,
    equation_densities,
    neutral_partition,
    species_codes,
    molecules_path: Optional[Path] = None,
) -> dict[int, np.ndarray]:
    """Return molecular line populations keyed by molecular line-list species code."""
    requested = sorted(
        {int(code) for code in np.asarray(species_codes, np.int64).ravel()}
    )
    code_order: list[float] = []
    for species_code in requested:
        for code in _SPECIES_CODE_TO_MOLECULE_CODES.get(species_code, ()):
            if code not in code_order:
                code_order.append(float(code))
    if not code_order:
        return {}
    line_populations = molecular_line_populations(
        temperature=temperature,
        equation_densities=equation_densities,
        neutral_partition=neutral_partition,
        codes=code_order,
        molecules_path=molecules_path,
    )
    code_position = {code: index for index, code in enumerate(code_order)}
    population_by_species_code: dict[int, np.ndarray] = {}
    for species_code in requested:
        population_columns = [
            code_position[float(molecule_code)]
            for molecule_code in _SPECIES_CODE_TO_MOLECULE_CODES.get(species_code, ())
            if float(molecule_code) in code_position
        ]
        if population_columns:
            population_by_species_code[species_code] = np.sum(
                line_populations[:, population_columns], axis=1
            )
    return population_by_species_code


@dataclass
class MolecularMetadata:
    """Cached molecule table fields needed by EOS coupling."""

    molecule_count: int
    molecule_codes: np.ndarray
    equilibrium_coefficients: np.ndarray
    component_start_indices: np.ndarray
    component_equation_indices: np.ndarray
    equation_species_codes: np.ndarray
    equation_count: int
    ion_formation_rows: list


_MOLMETA_CACHE: dict = {}


def molecular_equilibrium_metadata(
    molecules_path: Optional[Path] = None,
) -> MolecularMetadata:
    """Load and cache the molecule structure used by EOS coupling."""
    if molecules_path is None:
        molecules_path = _default_molecule_table()
    key = str(molecules_path)
    cached = _MOLMETA_CACHE.get(key)
    if cached is not None:
        return cached
    molecule_table = read_molecule_table(molecules_path)
    molecule_count = molecule_table.molecule_count
    molecule_codes = molecule_table.molecule_codes
    equilibrium_coefficients = molecule_table.equilibrium_coefficients
    component_start_indices = molecule_table.component_start_indices
    component_equation_indices = molecule_table.component_equation_indices
    equation_species_codes = molecule_table.equation_species_codes
    equation_count = molecule_table.equation_count
    ion_formation_rows = []
    for molecule_index in range(molecule_count):
        if equilibrium_coefficients[0, molecule_index] != 0.0:
            continue
        component_start = int(component_start_indices[molecule_index])
        component_stop = int(component_start_indices[molecule_index + 1])
        component_count = component_stop - component_start
        if component_count <= 1:
            continue
        atomic_number = int(round(float(molecule_codes[molecule_index])))
        ion_stage = component_count - 1
        if 1 <= atomic_number <= 99:
            ion_formation_rows.append((molecule_index, atomic_number, ion_stage))
    meta = MolecularMetadata(
        molecule_count=molecule_count,
        molecule_codes=molecule_codes[:molecule_count].copy(),
        equilibrium_coefficients=equilibrium_coefficients,
        component_start_indices=component_start_indices,
        component_equation_indices=component_equation_indices,
        equation_species_codes=equation_species_codes,
        equation_count=equation_count,
        ion_formation_rows=ion_formation_rows,
    )
    _MOLMETA_CACHE[key] = meta
    return meta


def ion_formation_constants_from_saha(
    metadata: MolecularMetadata,
    ion_stage_fraction: np.ndarray,
    electron_density: np.ndarray,
) -> np.ndarray:
    """Build the atomic-ion molecular formation constants from the EOS Saha output.

    Each atomic-ion row uses ``F_ion/F_neutral * n_e**ion`` from the EOS
    ion-stage fractions. Bare atoms and polynomial molecules remain zero here;
    they are filled inside `solve_molecular_equilibrium`.
    """
    n_depths = ion_stage_fraction.shape[0]
    ion_formation_constants = np.zeros(
        (n_depths, metadata.molecule_count), dtype=np.float64
    )
    electron_density_safe = np.maximum(np.asarray(electron_density, np.float64), 1e-300)
    for molecule_index, atomic_number, ion_stage in metadata.ion_formation_rows:
        if ion_stage >= ion_stage_fraction.shape[2]:
            continue
        neutral_fraction = ion_stage_fraction[:, atomic_number - 1, 0]
        ion_fraction = ion_stage_fraction[:, atomic_number - 1, ion_stage]
        valid = neutral_fraction > 0.0
        ion_formation_constants[valid, molecule_index] = (
            ion_fraction[valid]
            / neutral_fraction[valid]
            * electron_density_safe[valid] ** ion_stage
        )
    return ion_formation_constants
