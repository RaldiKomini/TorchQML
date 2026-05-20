"""Small helpers for unitary synthesis experiments."""

from .metrics import unitary_distance, unitary_fidelity
from .targets import TARGETS, target_bell, target_ghz, target_t, target_toffoli
from .actions import CircuitAction, apply_action, build_actions
from .baselines import BASELINES, BaselineCircuit, build_baseline_unitary
from .unitary_compilation import (
    Architecture,
    all_architectures,
    apply_student,
    apply_teacher,
    smoke_unitary_distillation,
    train_architecture,
)


__all__ = [
    "TARGETS",
    "target_t",
    "target_bell",
    "target_ghz",
    "target_toffoli",
    "unitary_distance",
    "unitary_fidelity",
    "CircuitAction",
    "apply_action",
    "build_actions",
    "BASELINES",
    "BaselineCircuit",
    "build_baseline_unitary",
    "Architecture",
    "all_architectures",
    "apply_student",
    "apply_teacher",
    "smoke_unitary_distillation",
    "train_architecture",
]
