#!/usr/bin/env python3
"""GPU deployment-semantics scorer: per-step p(BAD) with ALL-OK verdict history
(exactly StepShield first-flag semantics — see score_trajs.py rationale).

Batched full-sequence forward with OK filler at every verdict slot; probs read
at each verdict position from lm_head applied only at gathered positions.

Input records: {"id", "rogue"(bool), "source", "first_bad"(opt), "rogue_step"(opt),
                "header", "steps":[{"text",...}]}  (val.jsonl works directly;
holdout exported via export_holdout_records.py)

Usage: python3 gpu_score_trajs.py --model merged --data data/val.jsonl --out val_scores.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print("merged adapter:", args.adapter)
    model.eval()
    ok_id = tok.encode(" OK", add_special_tokens=False)[0]
    bad_id = tok.encode(" BAD", add_special_tokens=False)[0]
    nl = tok.encode("\n", add_special_tokens=False)
    pad_id = tok.pad_token_id or 0

    recs = [json.loads(l) for l in open(args.data)]
    data = []
    for r in recs:
        ids = tok.encode(r["header"], add_special_tokens=False)
        pos = []
        for s in r["steps"]:
            ids += tok.encode(s["text"] if isinstance(s, dict) else s,
                              add_special_tokens=False)
            ids.append(ok_id)          # ALL-OK filler (deployment first-flag semantics)
            pos.append(len(ids) - 1)   # position of the verdict token
            ids += nl
        ids = ids[: args.max_len]
        pos = [p for p in pos if p < args.max_len]
        data.append({"ids": ids, "pos": pos, "rec": r})

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            chunk = data[i:i + args.batch_size]
            maxlen = max(len(c["ids"]) for c in chunk)
            input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            bidx, pos = [], []
            for j, c in enumerate(chunk):
                input_ids[j, :len(c["ids"])] = torch.tensor(c["ids"])
                attn[j, :len(c["ids"])] = 1
                bidx += [j] * len(c["pos"])
                pos += c["pos"]
            input_ids, attn = input_ids.to(device), attn.to(device)
            bidx_t = torch.tensor(bidx, device=device)
            pos_t = torch.tensor(pos, device=device)
            hidden = model.model(input_ids=input_ids,
                                 attention_mask=attn).last_hidden_state
            sel = hidden[bidx_t, pos_t - 1]
            logits = model.lm_head(sel)
            two = logits[:, [ok_id, bad_id]].float()
            p_bad = torch.softmax(two, dim=-1)[:, 1].cpu().tolist()
            k = 0
            for c in chunk:
                n = len(c["pos"])
                r = c["rec"]
                keep = {kk: r[kk] for kk in r
                        if kk in ("id", "rogue", "source", "first_bad",
                                  "rogue_step", "is_rogue_traj", "n_steps")}
                if "is_rogue_traj" in keep:
                    keep["rogue"] = keep.pop("is_rogue_traj")
                if "first_bad" not in keep:
                    fb = next((ii for ii, s in enumerate(r["steps"])
                               if isinstance(s, dict) and s.get("label") == "BAD"), None)
                    keep["first_bad"] = fb
                keep["probs"] = p_bad[k:k + n]
                out.append(keep)
                k += n
            if (i // args.batch_size) % 10 == 0:
                print(f"{i+len(chunk)}/{len(data)} {(time.time()-t0):.0f}s", flush=True)

    json.dump({"model": args.model, "source": args.data, "trajectories": out},
              open(args.out, "w"))
    print(f"saved {args.out} ({len(out)} trajs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
