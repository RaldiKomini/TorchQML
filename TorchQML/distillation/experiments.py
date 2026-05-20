from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import numpy as np
import torch

from TorchQML.core.config import DEVICE
from TorchQML.distillation.datasets import DatasetBundle, DatasetConfig, load_dataset_bundle
from TorchQML.distillation.loops import TrainLoopConfig, TrainResult, fit_model, run_epoch
from TorchQML.distillation.search_spaces import Cell, RepeatedCellSearchSpace
from TorchQML.distillation.surrogate import select_ucb
from TorchQML.distillation.teachers import TeacherBundle, TeacherConfig, attach_kd_loaders, fit_xgb_teacher
from TorchQML.kernels.quantum import fidelity_kernel_matrix, fit_precomputed_svc
from TorchQML.models.pauli_head import PauliHeadModel
from TorchQML.models.sumz import SumZModel
from TorchQML.models.vectorz import VectorZLinear


@dataclass(frozen=True)
class QSVMConfig:
    C: float = 1.0
    block_a: int = 256
    block_b: int = 256


@dataclass(frozen=True)
class MFSearchConfig:
    ncomp: int = 16
    num_qubits: int = 12
    n_repeats: int = 6
    initial_low: int = 60
    initial_high: int = 10
    budget_high: int = 30
    beta: float = 1.0
    seed: int = 0
    low_fidelity: TrainLoopConfig = field(
        default_factory=lambda: TrainLoopConfig(epochs=2, patience=2)
    )
    high_fidelity: TrainLoopConfig = field(
        default_factory=lambda: TrainLoopConfig(epochs=30, patience=6)
    )


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    search: MFSearchConfig = field(default_factory=MFSearchConfig)
    qsvm: QSVMConfig = field(default_factory=QSVMConfig)


@dataclass
class SearchResult:
    best_item: Hashable
    best_score: float
    low_db: dict[Hashable, float]
    high_db: dict[Hashable, float]
    history: list[dict[str, Any]]

    @property
    def best_cell(self):
        return self.best_item


def make_readout_model(circ, spec, *, readout: str = "vector_z", amp_enc=False):
    if readout == "vector_z":
        return VectorZLinear(circ, spec, amp_enc=amp_enc)
    if readout == "sum_z":
        return SumZModel(circ, spec, amp_enc=amp_enc)
    if readout == "pauli_head":
        return PauliHeadModel(circ, spec, amp_enc=amp_enc)
    raise ValueError(f"unknown readout: {readout}")


def evaluate_qsvm_accuracy(model, dataset: DatasetBundle, qsvm_config: QSVMConfig, *, split: str = "test") -> float:
    model.eval()
    theta = model.theta.detach()
    amp_enc = getattr(model, "amp_enc", None)
    X_train = dataset.X_train.to(DEVICE)
    y_train = dataset.y_train.detach().cpu().numpy().astype(np.int64)
    if split == "val":
        X_eval = dataset.X_val.to(DEVICE)
        y_eval = dataset.y_val.detach().cpu().numpy().astype(np.int64)
    elif split == "test":
        X_eval = dataset.X_test.to(DEVICE)
        y_eval = dataset.y_test.detach().cpu().numpy().astype(np.int64)
    else:
        raise ValueError(f"unknown split: {split}")

    K_train = fidelity_kernel_matrix(
        X_train,
        X_train,
        theta,
        model.circ,
        amp_enc=amp_enc,
        blockA=qsvm_config.block_a,
        blockB=qsvm_config.block_b,
        symmetric=True,
    )
    K_eval = fidelity_kernel_matrix(
        X_eval,
        X_train,
        theta,
        model.circ,
        amp_enc=amp_enc,
        blockA=qsvm_config.block_a,
        blockB=qsvm_config.block_b,
    )
    classifier, pred = fit_precomputed_svc(K_train, y_train, K_eval, C=qsvm_config.C)
    return float((pred == y_eval).mean())


class MultiFidelityDistillationExperiment:
    """Reusable multi-fidelity KD search over any hashable architecture items."""

    def __init__(
        self,
        *,
        items: list[Hashable],
        make_model: Callable[[Hashable], torch.nn.Module],
        dataset: DatasetBundle,
        low_config: TrainLoopConfig,
        high_config: TrainLoopConfig,
        initial_low: int,
        initial_high: int,
        budget_high: int,
        beta: float = 1.0,
        seed: int = 0,
        encoder=None,
        device: torch.device = DEVICE,
    ):
        self.items = items
        self.make_model = make_model
        self.dataset = dataset
        self.low_config = low_config
        self.high_config = high_config
        self.initial_low = initial_low
        self.initial_high = initial_high
        self.budget_high = budget_high
        self.beta = beta
        self.seed = seed
        self.encoder = encoder
        self.device = device
        self.low_results: dict[Hashable, TrainResult] = {}
        self.high_results: dict[Hashable, TrainResult] = {}

    def run_search(self) -> SearchResult:
        if self.initial_high > self.initial_low:
            raise ValueError("initial_high must be <= initial_low")
        if self.budget_high < self.initial_high:
            raise ValueError("budget_high must be >= initial_high")

        rng = np.random.default_rng(self.seed)
        items = [self.items[idx] for idx in rng.permutation(len(self.items))]
        low_items = items[: self.initial_low]
        high_items = low_items[: self.initial_high]
        low_db: dict[Hashable, float] = {}
        high_db: dict[Hashable, float] = {}
        history: list[dict[str, Any]] = []

        for item in low_items:
            low_db[item] = self.evaluate_low_fidelity(item)

        for item in high_items:
            high_db[item] = self.evaluate_high_fidelity(item)
            history.append({"mode": "init_hf", "item": item, "high_score": high_db[item]})

        while len(high_db) < self.budget_high:
            item, mu, sd, acq, rho = select_ucb(
                self.items,
                low_db,
                high_db,
                beta=self.beta,
                encoder=self.encoder,
            )
            if item not in low_db:
                low_db[item] = self.evaluate_low_fidelity(item)
            high_db[item] = self.evaluate_high_fidelity(item)
            history.append(
                {
                    "mode": "ucb_hf",
                    "item": item,
                    "pred_mu": mu,
                    "pred_sd": sd,
                    "acq": acq,
                    "rho": rho,
                    "high_score": high_db[item],
                }
            )

        best_item = max(high_db, key=high_db.get)
        return SearchResult(best_item, high_db[best_item], low_db, high_db, history)

    def evaluate_low_fidelity(self, item: Hashable) -> float:
        if item not in self.low_results:
            self.low_results[item] = fit_model(
                self.make_model(item),
                self.dataset.train_loader_kd,
                self.dataset.val_loader_kd,
                self.device,
                "kd",
                self.low_config,
            )
        return self.low_results[item].best_val_acc

    def evaluate_high_fidelity(self, item: Hashable) -> float:
        if item not in self.high_results:
            self.high_results[item] = fit_model(
                self.make_model(item),
                self.dataset.train_loader_plain,
                self.dataset.val_loader_plain,
                self.device,
                "task",
                self.high_config,
            )
        return self.high_results[item].best_val_acc


class RepeatedCellKDExperiment:
    """Convenience wrapper for the original repeated-cell KD search."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.space = RepeatedCellSearchSpace(
            num_qubits=config.search.num_qubits,
            xlen=config.search.ncomp,
            n_repeats=config.search.n_repeats,
        )
        self.data = load_dataset_bundle(config.dataset)
        self.teacher = fit_xgb_teacher(self.data, config.teacher)
        attach_kd_loaders(self.data, self.teacher, config.dataset)
        self.searcher = MultiFidelityDistillationExperiment(
            items=self.space.all_cells(),
            make_model=self.make_student,
            dataset=self.data,
            low_config=config.search.low_fidelity,
            high_config=config.search.high_fidelity,
            initial_low=config.search.initial_low,
            initial_high=config.search.initial_high,
            budget_high=config.search.budget_high,
            beta=config.search.beta,
            seed=config.search.seed,
        )

    @property
    def low_results(self):
        return self.searcher.low_results

    @property
    def high_results(self):
        return self.searcher.high_results

    def make_student(self, cell: Cell):
        spec = self.space.make_spec(cell)
        circ = self.space.build_circuit(cell)
        return VectorZLinear(circ, spec).to(DEVICE)

    def run_search(self) -> SearchResult:
        return self.searcher.run_search()

    def evaluate_best_vqc_accuracy(self, cell: Cell) -> float:
        if cell not in self.high_results:
            self.searcher.evaluate_high_fidelity(cell)
        model = self.make_student(cell)
        model.load_state_dict(self.high_results[cell].best_state)
        metrics = run_epoch(
            model,
            self.data.test_loader_plain,
            DEVICE,
            "task",
            None,
            self.config.search.high_fidelity.alpha,
            self.config.search.high_fidelity.temperature,
        )
        return metrics["acc"]

    def evaluate_best_qsvm_accuracy(self, cell: Cell) -> float:
        if cell not in self.high_results:
            self.searcher.evaluate_high_fidelity(cell)
        model = self.make_student(cell)
        model.load_state_dict(self.high_results[cell].best_state)
        return evaluate_qsvm_accuracy(model, self.data, self.config.qsvm)


__all__ = [
    "ExperimentConfig",
    "MFSearchConfig",
    "MultiFidelityDistillationExperiment",
    "QSVMConfig",
    "RepeatedCellKDExperiment",
    "SearchResult",
    "evaluate_qsvm_accuracy",
    "make_readout_model",
]
