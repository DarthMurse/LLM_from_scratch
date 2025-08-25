from model import *
from optimizer import AdamW
import torch
import torch.cuda.nvtx as nvtx
import torch.cuda.profiler as profiler
import time

def benchmark(func, warmup=5, rep=10, record_memory=False, *args, **kwargs):
    for i in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    if record_memory:
        torch.cuda.memory._record_memory_history()
    start_time = time.time()
    total_time = 0
    with torch.autograd.profiler.emit_nvtx():
        profiler.start()
        for i in range(rep):
            func(*args, **kwargs)
            torch.cuda.synchronize()
            duration = time.time() - start_time
            print(f"num {i}, duration: {duration * 1000}ms")
            total_time += duration
            start_time = time.time()
        profiler.stop()
    if record_memory:
        torch.cuda.memory._dump_snapshot("memory.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Total time {total_time * 1000}ms, average {total_time * 1000 / rep}ms")

def benchmark_model(model, config: ModelConfig, name, warmup=5, rep=10):
    data = torch.randint(config.vocab_size, [1, 1025], device="cuda:1")
    model = model.to("cuda:1")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Actual model parameter count: {total_params / 1e9:.2f} Billion")
    x = data[:, :-1]
    y = data[:, 1:]
    loss_func = CrossEntropyLoss()
    #optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    def compute_model():
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        optimizer.zero_grad()
        with nvtx.range("forward"):
            logits = model(x)
        loss = loss_func(logits.flatten(end_dim=-2), y.flatten())
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
    model = Transformer(config_list[name])
    benchmark_model(model, config_list[name], name, warmup=5, rep=10)
