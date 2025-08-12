from model import *
from optimizer import AdamW
import torch
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
import time

def benchmark(func, warmup=5, rep=10, *args, **kwargs):
    start_time = time.time()
    total_time = 0
    with torch.autograd.profiler.emit_nvtx():
        profiler.start()
        for i in range(rep+warmup):
            func(*args, **kwargs)
            torch.cuda.synchronize()
            if i >= warmup:
                duration = time.time() - start_time
                print(f"num {i}, duration: {duration * 1000}ms")
                total_time += duration
            start_time = time.time()
        profiler.stop()

    print(f"Total time {total_time * 1000}ms, average {total_time * 1000 / rep}ms")

def benchmark_model(model, config: ModelConfig, name, warmup=5, rep=10):
    data = torch.randint(config.vocab_size, [16, config.block_size+1], device="cuda:1")
    model = model.to("cuda:1")
    x = data[:, :-1]
    y = data[:, 1:]
    loss_func = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def compute_model():
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with nvtx.range("forward"):
            logits = model(x)
        loss = loss_func(logits.flatten(end_dim=-2), y.flatten())
        optimizer.zero_grad()
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer"):
            optimizer.step()

    print(f"Benchmarking {name} model ...")
    benchmark(compute_model, warmup, rep)

if __name__ == "__main__":
    small_config = ModelConfig(768, 2048, 12, 10000, 12, 12)
    medium_config = ModelConfig(1024, 2688, 24, 10000, 16, 16)
    large_config = ModelConfig(1280, 3392, 36, 10000, 20, 20)
    xl_config = ModelConfig(1600, 4224, 48, 10000, 25, 25)
    b3_config = ModelConfig(2560, 6784, 32, 10000, 32, 32)
    config_list = {"small": small_config, "medium": medium_config, "large": large_config, "xl": xl_config, "3b": b3_config}

    #model_list = {k: Transformer(v) for k, v in config_list.items()}
    #model_list = {"small": Transformer(config_list["small"])}
    name = "small"
    benchmark_model(Transformer(config_list[name]), config_list[name], name, warmup=5, rep=10)
