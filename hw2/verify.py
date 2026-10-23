"""compare sharded Llama gradients and AdamW updates with dense training"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from emugpu import World

from hw1.model import LlamaConfig, LlamaLM
from hw2.fsdp import FSDPLlama


def reference_units(model):
    return [model.token_embedding, *model.blocks, nn.Sequential(model.final_norm, model.lm_head)]


def worker(ctx, steps=3):
    config = LlamaConfig(vocab_size=31, d_model=16, n_layers=2, n_heads=4,
                         n_kv_heads=2, ffn_dim=35, max_seq_len=16)
    torch.manual_seed(123)
    reference = LlamaLM(config).to(device=ctx.device, dtype=torch.float64)
    torch.manual_seed(123 + ctx.rank)
    comm = ctx.mesh((ctx.size,), ("fsdp",))["fsdp"].get_group()
    model = FSDPLlama(LlamaLM(config).to(device=ctx.device, dtype=torch.float64), comm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0.0)
    maxima = dict(logits=0.0, gradients=0.0, parameters=0.0, adam_moments=0.0)
    batches = []

    def compare(actual, expected, kind):
        if actual.numel():
            maxima[kind] = max(maxima[kind], (actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)

    for step in range(steps):
        counts = [rank + 1 for rank in range(ctx.size)]
        if step % 2 and ctx.size > 1:
            counts[0] = 0
        batches.append(counts)
        total = sum(counts)
        start = sum(counts[:ctx.rank])
        end = start + counts[ctx.rank]
        generator = torch.Generator().manual_seed(500 + step)
        x = torch.randint(config.vocab_size, (total, 7), generator=generator).to(ctx.device)
        y = torch.randint(config.vocab_size, (total, 7), generator=generator).to(ctx.device)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected_logits = reference(x)
        logits = model(x[start:end])
        compare(logits, expected_logits[start:end], "logits")
        loss = F.cross_entropy(logits.flatten(0, 1), y[start:end].flatten(), reduction="sum") / y.numel()
        loss.backward()
        F.cross_entropy(expected_logits.flatten(0, 1), y.flatten()).backward()
        for unit, dense_unit in zip(model.units(), reference_units(reference)):
            expected = torch.cat([p.grad.flatten() for p in dense_unit.parameters()])
            compare(unit.full_parameters(gradient=True), expected, "gradients")
            for tensor in (unit.shard, unit.shard.grad):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            assert all(p.is_meta for p in unit.template.parameters())
        optimizer.step()
        reference_optimizer.step()
        for unit, dense_unit in zip(model.units(), reference_units(reference)):
            expected = torch.cat([p.detach().flatten() for p in dense_unit.parameters()])
            compare(unit.full_parameters(), expected, "parameters")
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state[unit.shard][key]
                assert state.numel() == unit.shard_size
                actual = comm.all_gather(state)[:unit.numel]
                expected = torch.cat([reference_optimizer.state[p][key].flatten()
                                      for p in dense_unit.parameters()])
                compare(actual, expected, "adam_moments")
    expected_parameters = sum(unit.shard_size for unit in model.units())
    assert sum(p.numel() for p in model.parameters()) == expected_parameters
    assert len(list(model.parameters())) == len(model.units())
    reduce_scatters = sum(e["op"] == "reduce_scatter" for e in ctx.trace)
    assert reduce_scatters == steps * len(model.units())
    return {"status": "PASS", "rank": ctx.rank, "pid": os.getpid(), "device": str(ctx.device),
            "steps": steps, "batches": batches, "max_abs_error": maxima,
            "local_parameter_numel": expected_parameters, "units": model.sharding_info(),
            "reduce_scatter_calls": reduce_scatters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("hw2/verification.json"))
    args = parser.parse_args()
    results = []
    for size in args.world_sizes:
        world = World(size, device=args.device, timeout=120, run_timeout=300)
        ranks = world.run(worker)
        results.append({"world_size": size, "ranks": ranks})
        print(f"PASS: world_size={size}, device={args.device}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torch_version": torch.__version__, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
