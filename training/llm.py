"""Fine-tune a 7B chat model as a 3-class sentiment classifier on a free T4 GPU (QLoRA).

The base model is loaded in 4-bit (bitsandbytes NF4) and kept frozen; small LoRA adapters on every
attention and MLP projection, plus a new 3-class head, are trained. The frozen protocol is the same
as for the smaller models: train on the train split (or a given frame), keep the epoch with the
best validation macro-F1, never look at test. Only the adapter and head are saved (about 100 MB).
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from training.protocol import LABELS, LABEL_TO_ID, load_prepared
from training.run import scores, token_batches, write_json

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base(base, revision, torch):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)  # T4 has no bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        base, revision=revision, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
        quantization_config=quant, device_map={"": 0}, dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def predict_logits(model, tokenizer, frame, torch, batch_size=32):
    model.eval()
    out = []
    with torch.inference_mode():
        for inputs, _ in token_batches(frame, tokenizer, batch_size, torch, "cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def train_llm(data_dir, output_dir, *, base, revision, epochs=3, batch_size=8, accumulation=2, learning_rate=1e-4,
              seed=1, lora_r=16, train_frame=None, train_name="train split", eval_batch_size=32, **_):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_linear_schedule_with_warmup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    frame, manifest = load_prepared(data_dir)
    train, val = frame[frame.split == "train"], frame[frame.split == "validation"]
    if train_frame is not None:
        train = train_frame[["text", "label_id"]].reset_index(drop=True)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output already contains artifacts; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_base(base, revision, torch)
    commit = getattr(model.config, "_commit_hash", None)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=lora_r, lora_alpha=2 * lora_r,
                                             lora_dropout=0.05, target_modules=TARGETS))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    n_batches = math.ceil(len(train) / batch_size)
    steps = math.ceil(n_batches / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    config = {"base_model": base, "base_revision": commit or revision, "method": "QLoRA (4-bit NF4 base, LoRA adapters)",
              "lora_r": lora_r, "train_data": train_name, "train_rows": len(train), "split_sha256": manifest["split_sha256"],
              "epochs": epochs, "batch_size": batch_size, "accumulation": accumulation, "learning_rate": learning_rate,
              "max_length": 128, "seed": seed, "device": "cuda", "selection": "validation macro-F1 only",
              "trainable_parameters": sum(p.numel() for p in trainable)}
    best, history, selected_epoch, best_val = -1, [], None, None
    write_json(output / "training.json", {**config, "status": "running", "history": history})
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start, total = time.time(), 0.0
        for step, (inputs, labels) in enumerate(token_batches(train, tokenizer, batch_size, torch, "cuda", shuffle=True), 1):
            group_size = min(accumulation, n_batches - ((step - 1) // accumulation) * accumulation)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            total += float(loss.detach())
            scaler.scale(loss / group_size).backward()
            if step % accumulation == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "batches": n_batches, "loss": round(total / step, 4),
                                  "elapsed_seconds": round(time.time() - start)}), flush=True)
        pred = predict_logits(model, tokenizer, val, torch, eval_batch_size).argmax(1)
        metrics = scores(val.label_id, pred)
        record = {"epoch": epoch, "train_loss": total / n_batches, "validation": metrics, "seconds": round(time.time() - start, 2)}
        history.append(record)
        if not np.isfinite(record["train_loss"]):
            raise FloatingPointError("Training loss is not finite; 16-bit overflow on this GPU.")
        if metrics["macro_f1"] > best:
            best, selected_epoch, best_val = metrics["macro_f1"], epoch, metrics
            model.save_pretrained(output / "model")
            tokenizer.save_pretrained(output / "model")
            write_json(output / "model" / "base.json", {"base": base, "revision": revision})
        write_json(output / "training.json", {**config, "status": "running", "selected_epoch": selected_epoch, "history": history})
        print(json.dumps({k: record[k] for k in ["epoch", "train_loss", "seconds"]} |
                         {"val_macro_f1": metrics["macro_f1"]}), flush=True)
    write_json(output / "training.json", {**config, "status": "complete", "selected_epoch": selected_epoch,
                                          "selected_validation": {"accuracy": best_val["accuracy"], "macro_f1": best_val["macro_f1"]},
                                          "history": history})
    del model
    torch.cuda.empty_cache()


def saved_logits(model_dir, frames):
    """Logits from a saved adapter: reload the 4-bit base it was trained on, then the adapter."""
    import torch
    from peft import PeftModel
    spec = json.loads((Path(model_dir) / "base.json").read_text())
    tokenizer, base = load_base(spec["base"], spec["revision"], torch)
    model = PeftModel.from_pretrained(base, model_dir)
    result = {k: predict_logits(model, tokenizer, f, torch) for k, f in frames.items()}
    del model, base
    torch.cuda.empty_cache()
    return result
