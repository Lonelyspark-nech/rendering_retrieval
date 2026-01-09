# -*- coding: utf-8 -*-
from typing import Dict, Iterable, Set
from tqdm.auto import tqdm
import numpy as np
import pytrec_eval

def evaluate_run(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    metrics: Set[str],
) -> Dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    # pytrec_eval 内部很快；这里用 tqdm 给一个阶段感
    per_q = evaluator.evaluate(run)

    agg = {}
    for m in tqdm(sorted(list(metrics)), desc="pytrec_eval aggregate", unit="metric"):
        vals = [per_q[qid].get(m, 0.0) for qid in per_q]
        agg[m] = float(np.mean(vals)) if vals else 0.0
    return agg
