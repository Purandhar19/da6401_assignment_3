"""
train.py - Training, evaluation, checkpointing, and experiments.
"""

import glob
import os
from typing import Optional

import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import get_dataloaders
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_value = self.smoothing / max(1, self.vocab_size - 1)
        smooth_dist = torch.full_like(log_probs, smooth_value)
        confidence = 1.0 - self.smoothing
        smooth_dist.scatter_(1, target.unsqueeze(1), confidence)
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0.0
        token_loss = -(smooth_dist * log_probs).sum(dim=-1)
        valid_mask = ~pad_mask
        if valid_mask.any():
            return token_loss[valid_mask].mean()
        return token_loss.mean() * 0.0


def _detokenize_text(text: str) -> str:
    replacements = {
        " .": ".",
        " ,": ",",
        " !": "!",
        " ?": "?",
        " :": ":",
        " ;": ";",
        " n't": "n't",
        " 's": "'s",
        " 're": "'re",
        " 've": "'ve",
        " 'm": "'m",
        " 'll": "'ll",
        " 'd": "'d",
        "( ": "(",
        " )": ")",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train(is_train)
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(data_iter, desc=f"{'Train' if is_train else 'Eval'} Epoch {epoch_num + 1}", leave=False)

    if not hasattr(run_epoch, "global_step"):
        run_epoch.global_step = 0

    for src, tgt in progress:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=model.pad_idx).to(device)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_gold.reshape(-1)
            loss = loss_fn(logits_flat, tgt_flat)

            if is_train and optimizer is not None:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                run_epoch.global_step += 1

                if wandb.run is not None:
                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"]
                    )
                    with torch.no_grad():
                        probs = F.softmax(logits_flat, dim=-1)
                        valid_mask = tgt_flat != model.pad_idx
                        if valid_mask.any():
                            correct_prob = probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)[valid_mask].mean().item()
                        else:
                            correct_prob = 0.0

                    log_payload = {
                        "train_loss": float(loss.item()),
                        "lr": float(current_lr),
                        "step": run_epoch.global_step,
                        "grad_clip_norm": float(grad_norm),
                        "pred_confidence": float(correct_prob),
                    }
                    if run_epoch.global_step <= 1000:
                        for name, param in model.named_parameters():
                            if ("W_q" in name or "W_k" in name) and param.grad is not None:
                                log_payload[f"grad_norm/{name}"] = float(param.grad.norm().item())
                    if run_epoch.global_step % 50 == 0:
                        wandb.log(log_payload)

        total_loss += float(loss.item())
        total_steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, total_steps)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=model.pad_idx).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_word = torch.argmax(out[:, -1, :], dim=-1).item()
        ys = torch.cat(
            [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_word == end_symbol:
            break
    return ys


def _ids_to_sentence(token_ids, vocab) -> str:
    specials = {"<sos>", "<eos>", "<pad>"}
    tokens = []
    for idx in token_ids:
        token = vocab.lookup_token(int(idx))
        if token in specials:
            continue
        tokens.append(token)
    return _detokenize_text(" ".join(tokens))


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    metric = evaluate.load("sacrebleu")
    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(test_dataloader, desc="BLEU", leave=False):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            for i in range(src_batch.size(0)):
                src = src_batch[i : i + 1]
                tgt = tgt_batch[i]
                src_mask = make_src_mask(src, pad_idx=model.pad_idx).to(device)
                pred_ids = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len=max_len,
                    start_symbol=tgt_vocab.stoi["<sos>"],
                    end_symbol=tgt_vocab.stoi["<eos>"],
                    device=device,
                ).squeeze(0)
                predictions.append(_ids_to_sentence(pred_ids.tolist(), tgt_vocab))
                references.append([_ids_to_sentence(tgt.tolist(), tgt_vocab)])

    result = metric.compute(predictions=predictions, references=references, force=True)
    return float(result["score"])


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "vocab_metadata": {
            "src_stoi": getattr(model, "src_stoi", {}),
            "tgt_stoi": getattr(model, "tgt_stoi", {}),
            "src_itos": getattr(model, "src_itos", {}),
            "tgt_itos": getattr(model, "tgt_itos", {}),
        },
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout_rate,
            "pad_idx": model.pad_idx,
            "use_learned_positional_encoding": model.use_learned_positional_encoding,
            "attention_use_scaling": model.attention_use_scaling,
            "max_len": model.max_len,
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    vocab_metadata = checkpoint.get("vocab_metadata", {})
    model.src_stoi = vocab_metadata.get("src_stoi", getattr(model, "src_stoi", {}))
    model.tgt_stoi = vocab_metadata.get("tgt_stoi", getattr(model, "tgt_stoi", {}))
    model.src_itos = vocab_metadata.get("src_itos", getattr(model, "src_itos", {}))
    model.tgt_itos = vocab_metadata.get("tgt_itos", getattr(model, "tgt_itos", {}))
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])


def _maybe_upload_to_drive(local_path: str) -> None:
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "")
    if not folder_id:
        return
    try:
        import __main__

        if hasattr(__main__, "upload_to_gdrive"):
            __main__.upload_to_gdrive(local_path, folder_id)
    except Exception as exc:
        print(f"Drive upload skipped: {exc}")


def _sanitize_run_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()


def _run_single_experiment(
    run_name: str,
    base_config: dict,
    train_loader,
    val_loader,
    test_loader,
    src_vocab,
    tgt_vocab,
    src_tokenizer,
    device: str,
    use_noam: bool = True,
    fixed_lr: float = 1.0,
    use_scaling: bool = True,
    use_learned_positional_encoding: bool = False,
    smoothing: float = 0.1,
    resume_main_run: bool = False,
    save_primary_best: bool = False,
    num_epochs: Optional[int] = None,
) -> float:
    config = dict(base_config)
    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    run_id_base = os.environ.get("WANDB_RUN_ID", "da6401-a3-run")
    run_id = run_id_base if resume_main_run else f"{run_id_base}-{_sanitize_run_name(run_name)}"
    wandb_project = os.environ.get("WANDB_PROJECT", "da6401-a3-submission")
    wandb.init(
        project=wandb_project,
        id=run_id,
        resume="allow",
        name=run_name,
        config={
            **config,
            "use_noam": use_noam,
            "fixed_lr": fixed_lr if not use_noam else None,
            "attention_scaling": use_scaling,
            "use_learned_positional_encoding": use_learned_positional_encoding,
            "label_smoothing": smoothing,
        },
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        pad_idx=1,
        src_itos=src_vocab.itos,
        tgt_itos=tgt_vocab.itos,
        tgt_stoi=tgt_vocab.stoi,
        src_stoi=src_vocab.stoi,
        src_tokenizer=src_tokenizer,
        use_learned_positional_encoding=use_learned_positional_encoding,
        attention_use_scaling=use_scaling,
        max_len=config["max_len"],
    ).to(device)

    optimizer_lr = 1.0 if use_noam else fixed_lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
        if use_noam
        else None
    )
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=1, smoothing=smoothing)

    safe_name = _sanitize_run_name(run_name)
    checkpoint_pattern = f"checkpoint_{safe_name}_epoch_*.pt"
    start_epoch = 0
    checkpoints = sorted(glob.glob(checkpoint_pattern))
    if checkpoints:
        latest = checkpoints[-1]
        start_epoch = load_checkpoint(latest, model, optimizer, scheduler) + 1
        print(f"Resumed from {latest}, starting at epoch {start_epoch + 1}")

    best_bleu = -1.0
    best_path = f"best_model_{safe_name}.pt"

    for epoch in range(start_epoch, config["num_epochs"]):
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )
        val_bleu = evaluate_bleu(
            model,
            val_loader,
            tgt_vocab=tgt_vocab,
            device=device,
            max_len=config["max_len"],
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_epoch_loss": train_loss,
                "val_loss": val_loss,
                "val_bleu": val_bleu,
            }
        )

        epoch_ckpt = f"checkpoint_{safe_name}_epoch_{epoch + 1:03d}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_ckpt)

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            if save_primary_best:
                save_checkpoint(model, optimizer, scheduler, epoch, "best_model.pt")

    load_checkpoint(best_path, model)
    if save_primary_best and os.path.exists("best_model.pt"):
        _maybe_upload_to_drive("best_model.pt")
    test_bleu = evaluate_bleu(
        model,
        test_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_len=config["max_len"],
    )
    wandb.log({"best_val_bleu": best_bleu, "final_test_bleu": test_bleu})
    wandb.finish()
    return test_bleu


def run_training_experiment() -> None:
    CONFIG = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 25,
        "warmup_steps": 4000,
        "label_smoothing": 0.1,
        "max_len": 100,
        "clip": 1.0,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de = get_dataloaders(
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
    )

    baseline_bleu = _run_single_experiment(
        run_name="noam-baseline",
        base_config=CONFIG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=spacy_de,
        device=device,
        use_noam=True,
        use_scaling=True,
        use_learned_positional_encoding=False,
        smoothing=CONFIG["label_smoothing"],
        resume_main_run=True,
        save_primary_best=True,
    )

    ablation_epochs = int(os.environ.get("ABLATION_EPOCHS", "10"))
    _run_single_experiment(
        run_name="fixed-lr-1e4",
        base_config=CONFIG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=spacy_de,
        device=device,
        use_noam=False,
        fixed_lr=1e-4,
        smoothing=CONFIG["label_smoothing"],
        num_epochs=ablation_epochs,
    )
    _run_single_experiment(
        run_name="no-scaling",
        base_config=CONFIG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=spacy_de,
        device=device,
        use_noam=True,
        use_scaling=False,
        smoothing=CONFIG["label_smoothing"],
        num_epochs=ablation_epochs,
    )
    _run_single_experiment(
        run_name="learned-positional-encoding",
        base_config=CONFIG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=spacy_de,
        device=device,
        use_noam=True,
        use_scaling=True,
        use_learned_positional_encoding=True,
        smoothing=CONFIG["label_smoothing"],
        num_epochs=ablation_epochs,
    )
    _run_single_experiment(
        run_name="no-label-smoothing",
        base_config=CONFIG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=spacy_de,
        device=device,
        use_noam=True,
        use_scaling=True,
        smoothing=0.0,
        num_epochs=ablation_epochs,
    )

    print(f"Baseline test BLEU: {baseline_bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()
