#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# PEFT for LoRA
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm


# ============== CLI ==============
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train BERT on SST-2 with LoRA or Head-Tuning; configurable via CLI arguments."
    )
    # Training hyperparameters
    parser.add_argument("--tuning", type=str, default="head", choices=["lora", "head"],
                        help="Tuning method: 'lora' or 'head' (freeze backbone, train classifier head only)")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",
                        help="Hugging Face model name or local checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' to select automatically")
    parser.add_argument("--log_every", type=int, default=10, help="Print log every N batches")

    # Data and saving
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--save_dir", type=str, default="./outputs",
                        help="Directory to save plots and best weights")
    parser.add_argument("--save_plot", type=str, default="acc.png",
                        help="Filename for training/validation accuracy plot")
    parser.add_argument("--save_best", type=str, default=None,
                        help="Filename to save best model state_dict (default auto-name)")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    return parser


# ============== Utilities ==============
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(arg: str):
    """Choose device according to argument."""
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def print_trainable_parameters(model):
    """Print the number of trainable vs total parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || Total params: {all_param} || "
        f"Trainable ratio: {100 * trainable_params / all_param:.2f}%"
    )


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    """Evaluate the model on a dataset."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


# ============== Main Training Script ==============
def main():
    args = build_argparser().parse_args()

    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Tuning method: {args.tuning.upper()}")
    print(f"Using device: {device}")
    print("Loading tokenizer and dataset...")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("wics/strategy-qa")['test'].to_pandas()


    train,val = train_test_split(dataset, test_size=0.2, random_state=42)
    val,test = train_test_split(val, test_size=0.5, random_state=42)



    dataset = DatasetDict({
    "train": Dataset.from_pandas(train),
    "validation": Dataset.from_pandas(val),
    "test": Dataset.from_pandas(test),
    }   )

    def tokenize_function(examples):

        prompts = [
            f"Question: {q}\nFact: {f}\n Please answer the question based on the fact."
            for q, f in zip(examples["question"], examples["facts"])
        ]
        labels = [0 if a == False else 1 for a in examples["answer"]]
        data = {
            "input_ids": tokenizer(prompts, padding="max_length", truncation=True, max_length=args.max_length)["input_ids"],
            "attention_mask": tokenizer(prompts, padding="max_length", truncation=True, max_length=args.max_length)["attention_mask"],
            "labels": labels,
        }
        return data

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask","labels"]  # 加 token_type_ids 如果有
    )


    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=args.batch_size)

    print("Data processing finished.")

    # Load model
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    # Apply tuning method
    if args.tuning == "lora":
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            # task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["layers.21.attn.Wo"],

            bias="none",
        )
        model = get_peft_model(model, lora_config)
    elif args.tuning == "head":
        print("Applying Head-Tuning (freeze backbone, train classifier head only)...")
        if hasattr(model, "bert"):
            for p in model.bert.parameters():
                p.requires_grad = False
        else:
            # Fallback: freeze everything except classifier
            clf_names = {"classifier", "score", "scores"}
            for n, p in model.named_parameters():
                p.requires_grad = any(x in n for x in clf_names)
    else:
        raise ValueError("Invalid --tuning. Choose 'lora' or 'head'.")

    model.to(device)
    print_trainable_parameters(model)
    # head: 1538
    # lora: 1538

    # Optimizer and loss
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    print("Start training...")
    history = {"train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        train_preds, train_labels = [], []

        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch["labels"]

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            if (i + 1) % args.log_every == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs}, Step {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        train_acc = accuracy_score(train_labels, train_preds)
        val_acc, val_loss = evaluate(model, eval_dataloader, device, loss_fn)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} Summary:")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"  Saved new best model with Val Acc: {best_val_acc:.4f}")
        print("-" * 50)

    print("Training finished.")

    # Plot accuracy curve
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, history["train_acc"], "o-", label="Train Accuracy")
    plt.plot(epochs_range, history["val_acc"], "o-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, f"{args.tuning}.png"), bbox_inches="tight")

    print(f"Accuracy curve saved at: {args.save_dir}/{args.tuning}.png")

    # Evaluate best model on test set


    test_accuracy, _ = evaluate(model, test_dataloader, device, loss_fn)

    # Save best weights
    save_best = args.save_best or (f"best_{args.tuning}.pt")
    best_path = os.path.join(args.save_dir, save_best)
    # torch.save(best_model_state, best_path)
    print(f"Best model state_dict saved at: {best_path}")

    # Final results
    print("\n--- Final Results ---")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("---------------------")


if __name__ == "__main__":
    main()
