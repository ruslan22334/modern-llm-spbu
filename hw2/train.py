"""run: python -m hw2.train --world-size 2 --device cpu"""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F
from emugpu import World

from hw1.model import LlamaConfig, LlamaLM
from hw2.fsdp import FSDPLlama


def read_text(data, config_dir):
    """read inline text or a file relative to the input config"""
    if ("text" in data) == ("text_file" in data):
        raise ValueError("data must contain exactly one of text and text_file")
    repeat = data.get("repeat", 1)
    if type(repeat) is not int or repeat < 1:
        raise ValueError("data.repeat must be a positive integer")
    text = (config_dir / data["text_file"]).read_text(encoding="utf-8") if "text_file" in data else data["text"]
    return text * repeat


def make_dataset(text, seq_len, device):
    """split UTF-8 bytes into overlapping next-byte examples"""
    tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    starts = range(0, len(tokens) - seq_len, max(1, seq_len // 2))
    inputs = torch.stack([tokens[i : i + seq_len] for i in starts]).to(device)
    targets = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return inputs, targets


@torch.no_grad()
def evaluate(model, inputs, targets, comm):
    """each rank evaluates its part of the same fixed dataset"""
    x, y = inputs[comm.rank :: comm.size], targets[comm.rank :: comm.size]
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum")
    correct = (logits.argmax(-1) == y).sum()
    totals = comm.all_reduce(torch.tensor(
        [loss.item(), correct.item(), y.numel()], dtype=torch.float64, device=comm.device
    ))
    return (totals[0] / totals[2]).item(), (totals[1] / totals[2]).item()


def worker(ctx, settings, config_dict, text):
    config = LlamaConfig(**config_dict)
    torch.manual_seed(settings["seed"] + ctx.rank)
    comm = ctx.mesh((ctx.size,), ("fsdp",))["fsdp"].get_group()
    model = FSDPLlama(LlamaLM(config).to(ctx.device), comm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    inputs, targets = make_dataset(text, settings["seq_len"], ctx.device)
    counts = settings["local_batches"]
    start = sum(counts[:ctx.rank])
    global_batch = sum(counts)
    global_tokens = int(comm.all_reduce(torch.tensor(
        [counts[ctx.rank] * settings["seq_len"]], device=ctx.device
    )).item())
    generator = torch.Generator().manual_seed(settings["seed"] + 1)
    history = []
    for step in range(settings["steps"] + 1):
        if step > 0:
            indices = torch.randint(len(inputs), (global_batch,), generator=generator)
            local_indices = indices[start : start + counts[ctx.rank]].to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs[local_indices])
            loss = F.cross_entropy(
                logits.flatten(0, 1), targets[local_indices].flatten(), reduction="sum"
            ) / global_tokens
            loss.backward()
            optimizer.step()
        if step % 10 == 0 or step == settings["steps"]:
            loss, accuracy = evaluate(model, inputs, targets, comm)
            history.append((step, loss, accuracy))
            if ctx.rank == 0:
                print(f"step {step}: loss={loss:.4f}, accuracy={accuracy:.1%}", flush=True)
    states = [state for state in optimizer.state.values()]
    return {
        "rank": ctx.rank, "device": str(ctx.device), "examples": len(inputs),
        "local_batch": counts[ctx.rank], "units": model.sharding_info(),
        "local_parameter_numel": sum(p.numel() for p in model.parameters()),
        "adam_moment_numel": sum(s[key].numel() for s in states for key in ("exp_avg", "exp_avg_sq")),
        "history": history if ctx.rank == 0 else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--local-batches", type=int, nargs="+")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("hw2/results"))
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    experiment = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in experiment["training"].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    config = LlamaConfig(**experiment["model"])
    counts = args.local_batches
    if args.world_size < 1 or len(counts) != args.world_size or min(counts) < 0 or sum(counts) == 0:
        parser.error("provide one nonnegative local batch per rank and a positive global batch")
    if args.steps < 1 or args.lr <= 0 or args.weight_decay < 0 or not 1 <= args.seq_len <= config.max_seq_len:
        parser.error("steps and lr must be positive; weight-decay nonnegative; seq-len must fit the model")
    if config.vocab_size < 256:
        parser.error("UTF-8 byte data needs vocab_size >= 256")
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else read_text(experiment["data"], args.config.parent)
    if len(text.encode("utf-8")) <= args.seq_len:
        parser.error("text must be longer than seq-len bytes")
    settings = {"steps": args.steps, "seq_len": args.seq_len, "local_batches": counts,
                "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed}
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    world = World(args.world_size, device=device, timeout=120, run_timeout=1200)
    results = world.run(worker, kwargs={"settings": settings, "config_dict": asdict(config), "text": text})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "model": asdict(config),
        "training": {key: getattr(args, key) for key in experiment["training"]},
        "data": {"text": text, "repeat": 1},
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    history = results[0].pop("history")
    for result in results[1:]:
        result.pop("history")
    with (output / "loss.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "dataset_loss", "next_byte_accuracy"))
        writer.writerows(history)
    run = {**settings, "world_size": args.world_size, "device": device,
           "torch_version": torch.__version__, "input_config": args.config.name,
           "text_file": args.text_file.as_posix() if args.text_file else experiment["data"].get("text_file", "inline text from input config"),
           "gradient_normalization": "local token SUM / global token count, then SUM reduce_scatter",
           "ranks": results,
           "collectives_per_rank": [dict(Counter(e["op"] for e in world.trace if e["rank"] == rank))
                                    for rank in range(args.world_size)]}
    for name, value in (("run_config.json", run), ("model_config.json", asdict(config))):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.trace:
        world.save_trace(args.trace)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llama_hw_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    steps, losses, _ = zip(*history)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o", markersize=3)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy on the fixed dataset")
    plt.title(f"Llama FSDP, {args.world_size} ranks")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "training_loss.png", dpi=160)
    plt.close()
    print(f"saved results in {output}", flush=True)


if __name__ == "__main__":
    main()
