# # 让 python 能找到 retrieval_wheels/
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# # 让 python 能找到 llm2vec 里的包（因为 adapter 里 import src.model 等）
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/llm2vec:${PYTHONPATH:-}"

# /data3/sunbo/miniconda3/envs/llm2vec/bin/python run_longembed_eval_v3.py \
#   --adapter llm2vec_text \
#   --adapter_kwargs '{
#     "model_dir":"/data3/sunbo/models/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
#     "instruction":"Represent this query for retrieving relevant documents:",
#     "max_length":8192,
#     "encode_bs":32,
#     "pooling_mode":"mean",
#     "dtype":"bf16"
#   }'  \
#   --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
#   --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
#   --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi48 ps768x768_dpi60 ps768x768_dpi72 ps768x768_dpi84 ps768x768_dpi96 ps1024x1024_dpi48 ps1024x1024_dpi60 ps1024x1024_dpi72 ps1024x1024_dpi84 ps1024x1024_dpi96 \
#   --topk 50 \
#   --out_dir ./outputs_v2/llm2vec_manysettings


# 让 python 能找到 retrieval_wheels/
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# 让 python 能找到 llm2vec 里的包（因为 adapter 里 import src.model 等）
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/llm2vec:${PYTHONPATH:-}"

/data3/sunbo/miniconda3/envs/llm2vec/bin/python run_longembed_eval_v3.py \
  --adapter llm2vec_text \
  --adapter_kwargs '{
    "model_dir":"/data3/sunbo/models/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    "instruction":"Represent this query for retrieving relevant documents:",
    "max_length":8192,
    "encode_bs":16,
    "pooling_mode":"mean",
    "dtype":"bf16"
  }'  \
  --longembed_root /data1/sunbo/datasets/MLDR \
  --rendered_root /data1/sunbo/datasets_rendered/MLDR \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --datasets mldr-v1.0-en \
  --out_dir ./outputs_v2/llm2vec_manysettings