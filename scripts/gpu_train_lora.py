#!/usr/bin/env python3
"""GPU LoRA fine-tune of Nemotron-Mini-4B-Instruct as StepShield step classifier.

Formulation (must match Mac-side MLX inference tokenization exactly):
  per trajectory: header ++ [step_text ++ label_tok ++ "\n"] * n
  All parts encoded with add_special_tokens=False and concatenated as ids.
Loss: weighted CE ONLY at label positions, computed by gathering hidden states
at (label_pos - 1) and applying lm_head just there (256k vocab — never
materialize full-sequence logits).

Run: python gpu_train_lora.py --data-dir data --out out
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def tokenize_trajectory(tok, rec, ok_id, bad_id, nl_ids, max_len, bad_weight,
                        okfill=False):
    """okfill=True: the INPUT sequence carries OK at every verdict slot (targets
    unchanged) => every label decision is trained under all-OK history, exactly
    matching deployment first-flag semantics (pre-first-flag history is all OK)."""
    ids = tok.encode(rec["header"], add_special_tokens=False)
    lab_pos, lab_tgt, lab_w, lab_is_bad = [], [], [], []
    for s in rec["steps"]:
        st = tok.encode(s["text"], add_special_tokens=False)
        ids.extend(st)
        lab = s["label"]
        if lab is None:
            ids.append(ok_id)  # filler context, no loss
        else:
            lab_id = ok_id if lab == "OK" else bad_id
            ids.append(ok_id if okfill else lab_id)
            lab_pos.append(len(ids) - 1)
            lab_tgt.append(lab_id)
            w = float(s.get("w", 1.0))
            lab_w.append(w * (bad_weight if lab == "BAD" else 1.0))
            lab_is_bad.append(lab == "BAD")
        ids.extend(nl_ids)
        if len(ids) >= max_len:
            break
    ids = ids[:max_len]
    keep = [i for i, p in enumerate(lab_pos) if p < max_len]
    return (ids, [lab_pos[i] for i in keep], [lab_tgt[i] for i in keep],
            [lab_w[i] for i in keep], [lab_is_bad[i] for i in keep])


def load_data(tok, path, ok_id, bad_id, nl_ids, max_len, bad_weight, okfill=False):
    out = []
    for line in open(path):
        rec = json.loads(line)
        ids, pos, tgt, w, is_bad = tokenize_trajectory(
            tok, rec, ok_id, bad_id, nl_ids, max_len, bad_weight, okfill)
        if pos:
            first_bad = next((i for i, b in enumerate(is_bad) if b), None)
            out.append({"ids": ids, "pos": pos, "tgt": tgt, "w": w,
                        "id": rec["id"], "rogue": rec["is_rogue_traj"],
                        "source": rec.get("source", "train"),
                        "first_bad": first_bad})
    return out


def collate(batch, pad_id):
    maxlen = max(len(b["ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    bidx, pos, tgt, w = [], [], [], []
    for j, b in enumerate(batch):
        L = len(b["ids"])
        input_ids[j, :L] = torch.tensor(b["ids"])
        attn[j, :L] = 1
        bidx.extend([j] * len(b["pos"]))
        pos.extend(b["pos"])
        tgt.extend(b["tgt"])
        w.extend(b["w"])
    return (input_ids, attn, torch.tensor(bidx), torch.tensor(pos),
            torch.tensor(tgt), torch.tensor(w, dtype=torch.float32))


def label_logits(model, input_ids, attn, bidx, pos, class_rows=None):
    """Backbone forward + head only at (pos-1).

    Default: full lm_head -> [N, vocab] logits (CE over vocab).
    class_rows mode: 2 trainable rows -> [N, 2] logits (binary CE); targets must
    then be remapped to {0,1}."""
    base = model.base_model.model  # peft -> NemotronForCausalLM
    hidden = base.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    sel = hidden[bidx, pos - 1]  # hidden state that predicts the label token
    if class_rows is not None:
        return sel.float() @ class_rows.t()
    return base.lm_head(sel)


def evaluate(model, data, pad_id, ok_id, bad_id, batch_size, device, class_rows=None):
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            input_ids, attn, bidx, pos, tgt, w = collate(chunk, pad_id)
            input_ids, attn = input_ids.to(device), attn.to(device)
            bidx, pos = bidx.to(device), pos.to(device)
            logits = label_logits(model, input_ids, attn, bidx, pos, class_rows)
            two = logits.float() if class_rows is not None                 else logits[:, [ok_id, bad_id]].float()
            p_bad = torch.softmax(two, dim=-1)[:, 1].cpu().numpy()
            tgt = tgt.numpy()
            k = 0
            for b in chunk:
                n = len(b["pos"])
                res.append((b["id"], b["rogue"], p_bad[k:k + n].tolist(),
                            [1 if t == bad_id else 0 for t in tgt[k:k + n]],
                            b["source"], b["first_bad"]))
                k += n
    model.train()
    return res


def traj_metrics_at(res, thr):
    """Trajectory metrics + provenance-split FPs + EIR proxy."""
    tp = fp = tn = fn = 0
    fp_paired = fp_gen = 0
    early = premature = detected_rogue = 0
    for _, rogue, probs, _, source, first_bad in res:
        flag_i = next((i for i, p in enumerate(probs) if p > thr), None)
        if rogue:
            if flag_i is not None:
                tp += 1
                detected_rogue += 1
                if first_bad is not None:
                    if flag_i < first_bad:
                        premature += 1
                    elif flag_i - first_bad <= 3:
                        early += 1
            else:
                fn += 1
        else:
            if flag_i is not None:
                fp += 1
                if source == "benign_gen":
                    fp_gen += 1
                else:
                    fp_paired += 1
            else:
                tn += 1
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    eir = early / max(1, detected_rogue)
    return acc, tp, fp, tn, fn, fp_paired, fp_gen, eir, premature


def traj_auc(res):
    """AUC over trajectory scores (max step p_bad)."""
    scores = [(max(probs), 1 if rogue else 0) for _, rogue, probs, _, _, _ in res]
    p = np.array([s for s, _ in scores]); l = np.array([y for _, y in scores])
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    npos, nneg = l.sum(), (1 - l).sum()
    if npos == 0 or nneg == 0: return 0.0
    return float((ranks[l == 1].mean() - (npos - 1) / 2) / nneg)


def step_auc(res):
    p, l = [], []
    for _, _, probs, labs, _, _ in res:
        p.extend(probs); l.extend(labs)
    p, l = np.array(p), np.array(l)
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    npos, nneg = l.sum(), (1 - l).sum()
    if npos == 0 or nneg == 0: return 0.0
    return (ranks[l == 1].mean() - (npos - 1) / 2) / nneg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/Nemotron-Mini-4B-Instruct")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--truncate-layers", type=int, default=0,
                    help="use only the first K transformer layers (early-exit variant)")
    ap.add_argument("--okfill", action="store_true",
                    help="OK filler at all verdict slots (deployment-matched history)")
    ap.add_argument("--train-class-rows", action="store_true",
                    help="train the OK/BAD lm_head rows (tiny probe on top)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--bad-weight", type=float, default=4.0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ok-token", default="OK")
    ap.add_argument("--bad-token", default="BAD")
    args = ap.parse_args()
    import faulthandler, sys
    faulthandler.enable()
    print("ARGS:", vars(args), flush=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    ok_ids = tok.encode(" " + args.ok_token.strip(), add_special_tokens=False)
    bad_ids = tok.encode(" " + args.bad_token.strip(), add_special_tokens=False)
    assert len(ok_ids) == 1 and len(bad_ids) == 1, (ok_ids, bad_ids)
    ok_id, bad_id = ok_ids[0], bad_ids[0]
    nl_ids = tok.encode("\n", add_special_tokens=False)
    pad_id = tok.pad_token_id or 0
    ok_id_g = [ok_id]; bad_id_g = [bad_id]
    print(f"OK={ok_id} BAD={bad_id} nl={nl_ids} pad={pad_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if args.truncate_layers:
        model.model.layers = model.model.layers[: args.truncate_layers]
        model.config.num_hidden_layers = args.truncate_layers
        print(f"TRUNCATED to {args.truncate_layers} layers")
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    class_rows = None
    if args.train_class_rows:
        head = model.base_model.model.lm_head
        import torch.nn as nn
        class_rows = nn.Parameter(
            head.weight.data[[ok_id_g[0], bad_id_g[0]], :].clone().float())
        model.register_parameter("class_rows", class_rows)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    dd = Path(args.data_dir)
    train_data = load_data(tok, dd / "train.jsonl", ok_id, bad_id, nl_ids,
                           args.max_len, args.bad_weight, args.okfill)
    val_data = load_data(tok, dd / "val.jsonl", ok_id, bad_id, nl_ids,
                         args.max_len, args.bad_weight, args.okfill)
    print(f"train {len(train_data)} val {len(val_data)} "
          f"({sum(len(d['ids']) for d in train_data)/1e6:.2f}M train tokens)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * math.ceil(
        len(train_data) / (args.batch_size * args.grad_accum)) + 2
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    best = {"acc": -1.0, "thr": None}
    top3 = []  # (traj_auc, step, dir)
    log = []

    def run_eval(step):
        res = evaluate(model, val_data, pad_id, ok_id, bad_id, args.batch_size,
                       device, class_rows)
        auc = step_auc(res)
        tauc = traj_auc(res)
        row_best = (-1, None)
        for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]:
            acc, tp, fp, tn, fn, fpp, fpg, eir, prem = traj_metrics_at(res, thr)
            if acc > row_best[0]: row_best = (acc, thr)
            print(f"  thr={thr} acc={acc*100:.2f}% tp={tp} fp={fp} (paired={fpp} gen={fpg}) "
                  f"tn={tn} fn={fn} eir3~{eir:.2f} premature={prem}")
        print(f"  [step {step}] stepAUC={auc:.5f} trajAUC={tauc:.5f} "
              f"best={row_best[0]*100:.2f}%@{row_best[1]}")
        log.append({"step": step, "auc": float(auc), "traj_auc": tauc,
                    "best_acc": row_best[0], "best_thr": row_best[1]})
        # keep top-3 checkpoints by trajectory AUC (selection is threshold-free)
        import shutil
        if len(top3) < 3 or tauc > top3[-1][0]:
            d = out_dir / f"ckpt-step{step}"
            model.save_pretrained(d)
            if class_rows is not None:
                torch.save(class_rows.detach().cpu(), d / "class_rows.pt")
            json.dump({"traj_auc": tauc, "step_auc": float(auc), "step": step},
                      open(d / "ckpt_meta.json", "w"))
            top3.append((tauc, step, str(d)))
            top3.sort(key=lambda x: -x[0])
            for _, _, old in top3[3:]:
                shutil.rmtree(old, ignore_errors=True)
            del top3[3:]
            print(f"  [top3 by trajAUC] {[(round(a,5), st) for a, st, _ in top3]}")
        if row_best[0] > best["acc"]:
            best.update(acc=row_best[0], thr=row_best[1])
            model.save_pretrained(out_dir / "best_adapter")
            if class_rows is not None:
                torch.save(class_rows.detach().cpu(),
                           out_dir / "best_adapter" / "class_rows.pt")
            json.dump({"val_acc": best["acc"], "thr": best["thr"], "auc": float(auc),
                       "ok_id": ok_id, "bad_id": bad_id},
                      open(out_dir / "best_meta.json", "w"), indent=2)
            print(f"  saved best (val acc {best['acc']*100:.2f}%)")

    step = 0
    t0 = time.time()
    ntok = 0
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(train_data)))
        random.shuffle(order)
        order.sort(key=lambda i: len(train_data[i]["ids"]) // 512)
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        random.shuffle(batches)
        for bidxs in batches:
            batch = [train_data[i] for i in bidxs]
            input_ids, attn, bi, pos, tgt, w = collate(batch, pad_id)
            input_ids, attn = input_ids.to(device), attn.to(device)
            bi, pos, tgt, w = bi.to(device), pos.to(device), tgt.to(device), w.to(device)
            logits = label_logits(model, input_ids, attn, bi, pos, class_rows)
            if class_rows is not None:
                tgt = (tgt == bad_id).long()
            ce = F.cross_entropy(logits.float(), tgt, reduction="none")
            loss = (ce * w).sum() / w.sum().clamp(min=1e-6)
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == args.grad_accum - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            ntok += int(input_ids.numel())
            if step % 20 == 0:
                print(f"ep{ep} step{step} loss={loss.item():.4f} "
                      f"{ntok/(time.time()-t0):.0f} tok/s")
            if step % args.eval_every == 0:
                run_eval(step)
    print("FINAL EVAL")
    run_eval(step)
    model.save_pretrained(out_dir / "final_adapter")
    json.dump(log, open(out_dir / "train_log.json", "w"), indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min; best val acc "
          f"{best['acc']*100:.2f}% @ thr={best['thr']}")


if __name__ == "__main__":
    main()
