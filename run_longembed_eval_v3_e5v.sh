# 让 python 能找到 retrieval_wheels/
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

python run_longembed_eval_v3.py \
  --adapter e5_v \
  --adapter_kwargs '{
    "model_name": "/data3/sunbo/models/royokong/e5-v",
    "device": "cuda:0",
    "dtype": "float16",
    "qry_bs": 8,
    "doc_bs": 4,
    "use_faiss": true,
    "faiss_use_gpu": true,
    "faiss_gpu_device": 0,
    "faiss_normalize": true
  }' \
  --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
  --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --out_dir ./outputs_v2/e5_v