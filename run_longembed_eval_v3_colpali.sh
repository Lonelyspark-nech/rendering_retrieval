# 让 python 能找到 retrieval_wheels/
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# ColPali (run in the ocr env)
/data3/sunbo/miniconda3/envs/ocr/bin/python run_longembed_eval_v3.py \
  --adapter colpali \
  --adapter_kwargs '{
    "model_name": "/data3/sunbo/models/vidore/colpali-v1.2",
    "device_map": "cuda:0",
    "dtype": "bfloat16",
    "qry_bs": 8,
    "doc_bs": 8,
    "rerank_doc_bs": 8,
    "preselect_k": 200,
    "use_faiss": true,
    "faiss_use_gpu": false,
    "faiss_normalize": true
  }' \
  --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
  --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --out_dir ./outputs_v2/colpali
