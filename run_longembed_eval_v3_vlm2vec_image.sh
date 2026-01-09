# # 让 python 能找到 retrieval_wheels/
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# # 让 python 能找到 VLM2Vec/src/ 里的包（因为 adapter 里 import src.model 等）
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/VLM2Vec:${PYTHONPATH:-}"

# /data3/sunbo/miniconda3/envs/vlm2vec/bin/python run_longembed_eval_v3.py \
#   --adapter vlm2vec_mmeb_text2image \
#   --adapter_kwargs '{"doc_image_prompt":"Represent the given image.","doc_bs":8,"qry_bs":16}' \
#   --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
#   --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
#   --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi48 ps768x768_dpi60 ps768x768_dpi72 ps768x768_dpi84 ps768x768_dpi96 ps1024x1024_dpi48 ps1024x1024_dpi60 ps1024x1024_dpi72 ps1024x1024_dpi84 ps1024x1024_dpi96 \
#   --topk 50 \
#   --out_dir ./outputs_v2/vlm2vec_manysettings


# MLDR
# 让 python 能找到 retrieval_wheels/
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# 让 python 能找到 VLM2Vec/src/ 里的包（因为 adapter 里 import src.model 等）
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/VLM2Vec:${PYTHONPATH:-}"

/data3/sunbo/miniconda3/envs/vlm2vec/bin/python run_longembed_eval_v3.py \
  --adapter vlm2vec_mmeb_text2image \
  --adapter_kwargs '{"doc_image_prompt":"Represent the given image.","doc_bs":8,"qry_bs":16}' \
  --longembed_root /data1/sunbo/datasets/MLDR \
  --rendered_root /data1/sunbo/datasets_rendered/MLDR \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --datasets mldr-v1.0-en \
  --out_dir ./outputs_v2/vlm2vec_manysettings
