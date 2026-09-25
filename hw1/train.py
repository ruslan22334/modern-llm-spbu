"""
run: python train.py
the text is encoded as UTF-8 bytes, token IDs are always in 0..255
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llama_hw_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from model import LlamaConfig, LlamaLM


TRAIN_TEXT = (
    "мама мыла раму.\n"
    "кот сидит у окна.\n"
    "лампа светит ярко.\n"
    "модель учится предсказывать следующий токен.\n"
) * 16


def make_dataset(text: str, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """make a finite set of overlapping next-byte prediction examples"""
    tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    if len(tokens) <= seq_len:
        raise ValueError("text must be longer than seq-len in UTF-8 bytes")
    starts = range(0, len(tokens) - seq_len, max(1, seq_len // 2))
    inputs = torch.stack([tokens[i : i + seq_len] for i in starts]).to(device)
    targets = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return inputs, targets


@torch.no_grad()
def dataset_metrics(
    model: LlamaLM, inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    """loss and next-byte accuracy across the same fixed dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for start in range(0, len(inputs), 16):
        x, y = inputs[start : start + 16], targets[start : start + 16]
        logits = model(x)
        total_loss += F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size), y.reshape(-1), reduction="sum"
        ).item()
        total_tokens += y.numel()
        correct += (logits.argmax(dim=-1) == y).sum().item()
    model.train()
    return total_loss / total_tokens, correct / total_tokens


@torch.no_grad()
def generate_text(
    model: LlamaLM, prompt: str, new_bytes: int = 100, context_len: int | None = None
) -> str:
    """greedily continue a UTF-8 byte prompt, recomputing attention each step."""
    model.eval()
    device = next(model.parameters()).device
    token_ids = torch.tensor([list(prompt.encode("utf-8"))], device=device)
    context_len = context_len or model.config.max_seq_len
    for _ in range(new_bytes):
        context = token_ids[:, -context_len:]
        next_token = model(context)[:, -1].argmax(dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return bytes(token_ids[0].tolist()).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="кот ")
    args = parser.parse_args()

    config = LlamaConfig()
    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.seq_len <= config.max_seq_len:
        parser.error("steps and batch size must be positive; seq-len must fit the model")
    if args.lr <= 0:
        parser.error("lr must be positive")
    if not args.prompt:
        parser.error("prompt must not be empty")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaLM(config).to(device)
    dataset_text = args.text_file.read_text(encoding="utf-8") if args.text_file else TRAIN_TEXT
    inputs, targets = make_dataset(dataset_text, args.seq_len, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device}, parameters={parameters:,}, examples={len(inputs)}", flush=True)
    history = [(0, *dataset_metrics(model, inputs, targets))]
    print(f"step 0: loss={history[-1][1]:.4f}, accuracy={history[-1][2]:.1%}", flush=True)

    for step in range(1, args.steps + 1):
        indices = torch.randint(len(inputs), (args.batch_size,), device=device)
        x, y = inputs[indices], targets[indices]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == args.steps:
            full_loss, accuracy = dataset_metrics(model, inputs, targets)
            history.append((step, full_loss, accuracy))
            print(f"step {step}: loss={full_loss:.4f}, accuracy={accuracy:.1%}", flush=True)

    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "learning_rate": args.lr,
                "seed": 42,
                "device": str(device),
                "torch_version": torch.__version__,
                "dataset": args.text_file.name if args.text_file else "four fixed UTF-8 lines repeated 16 times",
                "examples": len(inputs),
                "sample_prompt": args.prompt,
                "sample_new_bytes": 100,
                "sample_context_len": args.seq_len,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "dataset_loss", "next_byte_accuracy"])
        writer.writerows(history)

    steps, losses, _ = zip(*history)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, marker="o", markersize=3)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy on the fixed dataset")
    plt.title("Small Llama overfitting check")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()
    sample = generate_text(model, args.prompt, context_len=args.seq_len)
    (output_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    print("saved sample.txt, graph, metrics, and config", flush=True)


if __name__ == "__main__":
    main()
