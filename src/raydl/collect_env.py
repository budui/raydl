import time

import torch.cuda
import torch.distributed as dist
import torch.utils.collect_env

__all__ = ["collect_env"]

# Composer environment information string output format
_DISTRIBUTED_CONFIGURE_FORMAT = """
Backend: {backend}
World size: {world_size}
Rank: {rank}
""".strip()


def collect_distributed_configure() -> dict:
    if not dist.is_initialized():
        return dict(backend=None, world_size=1, rank=0)

    return {
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
    }


def collect_running_env() -> str:
    out = "local GPU count: {cuda_device_count}\n".format(
        cuda_device_count=torch.cuda.device_count(),
    )

    out += _DISTRIBUTED_CONFIGURE_FORMAT.format(**collect_distributed_configure())

    return out


def collect_env() -> str:
    # Creation timestamp for report
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time.time()))
    report_hdr = (
        "---------------------------------\n"
        + "System Environment Report        \n"
        + f"Created: {creation_time}\n"
        + "---------------------------------\n"
    )

    # Torch section
    out = report_hdr
    out += "\nPyTorch information\n"
    out += "-------------------\n"
    out += torch.utils.collect_env.get_pretty_env_info()

    out += "\n\nRunning information\n"
    out += "--------------------\n"
    out += collect_running_env()

    return out
