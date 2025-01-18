import datetime
import signal
import sys
import time

import numpy as np
import torch

try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # type: ignore # noqa
except ImportError:
    pass

import optuna
from optuna.storages import RDBStorage


class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch


class CUDAArch(Arch):
    """shared with CUDA and ROCm: NVIDIA + AMD"""

    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device("cuda:0")

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(self.device())

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()


def get_accelerator_arch():
    if torch.cuda.is_available():
        return CUDAArch()

    raise ValueError("Currently only cuda, rocm, hpu and metal are supported")


# Benchmark of a basic GEMM
def benchmark_mm(arch, m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]
    elapsed_time = np.amin(times) / 1000  # want the fastest
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)
    return tflops


def make_objective(
        arch, dtype, device, m_range, n_range, k_range,
        num_iterations, num_warmup_iterations
):
    def objective(trial):
        M = trial.suggest_int("M", m_range[0], m_range[1], step=m_range[2])
        N = trial.suggest_int("N", n_range[0], n_range[1], step=n_range[2])
        K = trial.suggest_int("K", k_range[0], k_range[1], step=k_range[2])

        tflops = benchmark_mm(
            arch, M, N, K, dtype, device, num_iterations, num_warmup_iterations
        )
        return tflops
    return objective


def main():
    dtype = sys.argv[1]
    m_range = [1024, 20480, 64]
    n_range = [1024, 20480, 64]
    k_range = [1024, 20480, 64]
    num_iterations = 100
    num_warmup_iterations = 50
    n_trials = 1000
    study_name = "benchmark_flops"

    if dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError("dtype must be one of 'fp32', 'fp16', or 'bf16'")
    arch = get_accelerator_arch()
    device = arch.device()

    # Create a SQLite storage
    storage = RDBStorage(
        url="sqlite:///optuna.db", engine_kwargs={"connect_args": {"timeout": 30}}
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    start_time = time.time()

    def sigkill_handler(signum, frame):
        finish()
        sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    def finish():
        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        best_trial = study.best_trial
        best_tflops = best_trial.value
        best_config = f"{best_trial.params['M']}x{best_trial.params['N']}x{best_trial.params['K']} (MxNxK)"
        print(
            f"The best outcome was {best_tflops:.1f}TFLOPS @ {best_config} (tried {len(study.trials)} shapes)"
        )
        print(f"Elapsed time: {time_str}")

    def print_progress(study, trial):
        print(
            f"Trial {trial.number:>6} | {trial.value:6.1f} TFLOPS @ {trial.params['M']}x{
                trial.params['N']}x{trial.params['K']:<20} | best: {study.best_value:6.1f} TFLOPS",
            end="\r",
        )

    # Add known best shapes to the study
    known_best_shapes = [
        (6912, 16384, 2048),  # NVIDIA A100 SXM
        (2304, 5120, 1536),  # NVIDIA A100 PCIe
        (6144, 17920, 2816),  # NVIDIA H100 SXM
        (14336, 4096, 4096),  # NVIDIA RTX 4090
        (4352, 13568, 3840),  # AMD MI300X
    ]

    for m, n, k in known_best_shapes:
        study.enqueue_trial({"M": m, "N": n, "K": k})

    objective = make_objective(
        arch, dtype, device, m_range, n_range, k_range,
        num_iterations, num_warmup_iterations)
    study.optimize(objective, n_trials=n_trials, callbacks=[print_progress])
    finish()


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
