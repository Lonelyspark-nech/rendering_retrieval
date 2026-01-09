# -*- coding: utf-8 -*-
import argparse
import json
import os
from tqdm.auto import tqdm

from retrieval_wheels.datasets.longembed_rendered import (
    load_longembed_rendered_bundle,
    default_datasets,
)
from retrieval_wheels.metrics.trec_eval import evaluate_run
from retrieval_wheels.utils import ensure_dir
from retrieval_wheels.models.factory import build_adapter

def _load_kwargs(s: str):
    if not s:
        return {}
    s = s.strip()
    if s.startswith("@"):
        path = s[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)

def _load_settings(args) -> list[str]:
    if args.settings and args.settings_file:
        raise ValueError("Use either --settings or --settings_file, not both.")
    if args.settings:
        return list(args.settings)
    if args.settings_file:
        with open(args.settings_file, "r", encoding="utf-8") as f:
            ss = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if not ss:
            raise ValueError(f"No settings found in {args.settings_file}")
        return ss
    # 兼容旧用法：单个 --setting
    if args.setting:
        return [args.setting]
    raise ValueError("Provide --settings / --settings_file / --setting")

def main():
    ap = argparse.ArgumentParser()

    # 通用参数
    ap.add_argument("--longembed_root", required=True)
    ap.add_argument("--rendered_root", required=True)

    # 支持多 setting（新）+ 单 setting（旧兼容）
    ap.add_argument("--setting", default=None, help="(legacy) single setting, e.g. ps512x512_dpi72")
    ap.add_argument("--settings", nargs="*", default=None, help="multiple settings, e.g. ps256x256_dpi72 ps512x512_dpi72")
    ap.add_argument("--settings_file", default=None, help="file with one setting per line")

    ap.add_argument("--datasets", nargs="*", default=None, help="default: 4 non-synthetic")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metrics", default="ndcg_cut_10,recip_rank,recall_10")

    # adapter + kwargs（模型只加载一次）
    ap.add_argument("--adapter", required=True, help="e.g. vlm2vec_mmeb_text2image | llm2vec_text")
    ap.add_argument("--adapter_kwargs", default="{}", help='JSON string or "@/path/to/kwargs.json"')

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    metrics = set([m.strip() for m in args.metrics.split(",") if m.strip()])
    datasets = args.datasets if args.datasets else default_datasets()
    settings = _load_settings(args)

    # ===== build adapter ONCE (关键：不随 setting/dataset 变化) =====
    kwargs = _load_kwargs(args.adapter_kwargs)
    adapter = build_adapter(args.adapter, kwargs)

    summary_all = {
        "model": adapter.name,
        "topk": args.topk,
        "metrics": sorted(list(metrics)),
        "settings": {},
    }

    for setting in tqdm(settings, desc="settings", unit="setting"):
        setting_dir = os.path.join(args.out_dir, setting)
        ensure_dir(setting_dir)

        summary_setting = {}
        for ds in tqdm(datasets, desc=f"datasets @ {setting}", unit="ds", leave=False):
            out_ds_dir = os.path.join(setting_dir, ds)
            ensure_dir(out_ds_dir)

            # 只换数据：每轮重新 load
            queries, corpus, qrels = load_longembed_rendered_bundle(
                longembed_root=args.longembed_root,
                rendered_root=args.rendered_root,
                setting=setting,
                dataset=ds,
                require_doc_images=adapter.requires_doc_images,
            )

            run = adapter.search(queries=queries, corpus=corpus, topk=args.topk)
            agg = evaluate_run(qrels=qrels, run=run, metrics=metrics)
            tok = adapter.get_last_token_stats()

            with open(os.path.join(out_ds_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": ds,
                        "setting": setting,
                        "model": adapter.name,
                        "metrics": agg,
                        "token_stats": tok,
                    },
                    f, ensure_ascii=False, indent=2
                )
            with open(os.path.join(out_ds_dir, "run.json"), "w", encoding="utf-8") as f:
                json.dump(run, f, ensure_ascii=False)

            # summary_setting[ds] = agg
            summary_setting[ds] = {"metrics": agg, "token_stats": tok}

        summary_all["settings"][setting] = summary_setting

    with open(os.path.join(args.out_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
