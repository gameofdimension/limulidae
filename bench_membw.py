import os
import sys
import time

import torch
import torch.distributed as dist

try:
    import torch_npu  # type: ignore # noqa
    from torch_npu.contrib import transfer_to_npu  # type: ignore # noqa
except ImportError:
    pass


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl")
    dist.barrier()
    return rank, world_size


def prepare_op(method, nglobal):
    device = 'cuda'
    local_tensor = torch.rand(nglobal, device=device)
    if method == 'exp':
        return lambda: torch.exp(local_tensor)
    if method == 'sigmoid':
        sigmoid = torch.nn.Sigmoid()
        return lambda: sigmoid(local_tensor)
    if method == 'slow_sigmoid':
        return lambda: 1 / (1 + torch.exp(-local_tensor))
    assert False


def main():
    method = sys.argv[1]
    multiplier = 1
    volume_list = [
        0.10, 0.12, 0.15, 0.20, 0.32, 0.40, 0.50, 0.64, 0.80, 1.00, 1.25, 1.50, 2.00, 3.16, 4.00, 5.00, 6.40, 8.00,
        10.0, 12.5, 15.0, 20.0, 31.6, 40.0, 50.0, 64.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 316.0, 400.0, 500.0, 640.0, 800.0,
        1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3160.0, 4000.0, 5000.0, 6400.0, 8000.0]
    print(" size(MB)    tavg(usec)     tmin(usec)     tmax(usec)  avgbw(GB/sec)  maxbw(GB/sec)  minbw(GB/sec)")
    for n_mb in volume_list:

        if n_mb < 10.0:
            maxiter = 100 * multiplier
        elif n_mb < 512.0:
            maxiter = 20 * multiplier
        elif n_mb < 2000.0:
            maxiter = 10 * multiplier
        else:
            maxiter = 5 * multiplier

        nglobal = int(n_mb * 1.0e6 / 4.0)
        pointwise_op = prepare_op(method, nglobal)
        torch.cuda.synchronize()

        # launch two calls outside the timing loop
        pointwise_op()
        torch.cuda.synchronize()
        pointwise_op()
        torch.cuda.synchronize()

        tbeg = time.perf_counter()
        tmin = 1.0e30
        tmax = 0.0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(maxiter):
            start.record()
            pointwise_op()
            end.record()
            torch.cuda.synchronize()
            span = start.elapsed_time(end) / 1000
            tmin = min(tmin, span)
            tmax = max(tmax, span)

        torch.cuda.synchronize()
        tend = time.perf_counter()

        torch.cuda.synchronize()

        elapsed = tend - tbeg
        tavg = elapsed / maxiter

        volume_in_gb = 4.0e-9 * nglobal
        avgbw = volume_in_gb / tavg
        maxbw = volume_in_gb / tmin
        minbw = volume_in_gb / tmax

        print(
            "{:8.2f}".format(volume_in_gb * 1000), "  ",
            "{:8.1f}".format(tavg * 1.0e6), "      ",
            "{:8.1f}".format(tmin * 1.0e6), "      ",
            "{:8.1f}".format(tmax * 1.0e6), "     ",
            "{:7.2f}".format(avgbw), "      ",
            "{:7.2f}".format(maxbw), "      ",
            "{:7.2f}".format(minbw)
        )


if __name__ == "__main__":
    main()
