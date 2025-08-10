from model import *
import torch
import time

def benchmark(func, warmup=1, rep=5, *args, **kwargs):
    for i in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(rep):
        func(*args, **kwargs)
    duration = time.time() - start_time
    print(f"Total time {duration * 1000 / rep}ms, average {duration * 1000 / rep}ms")

def benchmark_model(model, config: ModelConfig, name, warmup=1, rep=5):
    data = torch.randint(config.vocab_size, [16, config.block_size+1], device="cuda")
    model = model.to("cuda")
    x = data[:, :-1]
    y = data[:, 1:]
    loss_func = CrossEntropyLoss()
    def compute_model():
        logits = model(x)
        loss = loss_func(logits, y)
        model.zero_grad()
        loss.backward()
    print(f"Benchmarking {name} model ...")
    benchmark(compute_model, warmup, rep)

if __name__ == "__main__":
    small_config = ModelConfig(768, 2048, 12, 10000, 12, 12)
    medium_config = ModelConfig(1024, 2688, 24, 10000, 16, 16)
    large_config = ModelConfig(1280, 3392, 36, 10000, 20, 20)
    xl_config = ModelConfig(1600, 4224, 48, 10000, 25, 25)
    3b_config = ModelConfig(2560, 6784, 32, 10000, 32, 32)
    config_list = {"small": small_config, "medium": medium_config, "large": large_config, "xl": xl_config, "3b": 3b_config}

    model_list = {k: Transformer(v) for k, v in config_list.items()}
    benchmark_model(model_list["small"], config_list["small"], "small")
