# # 让 python 能找到 retrieval_wheels/
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# # （可选）如果你的 retrieval_wheels 里还会 import llm2vec 的 src.*，保留这句；否则可以删
# export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/ocr:${PYTHONPATH:-}"

# # 用你当前跑 LLM2Vec 的 env（里面应当有 transformers>=4.51）
# /data3/sunbo/miniconda3/envs/ocr/bin/python run_longembed_eval_v3.py \
#   --adapter qwen3_embedding_8b \
#   --adapter_kwargs '{
#     "model_name":"/data3/sunbo/models/Qwen/Qwen3-Embedding-8B",
#     "instruction":"Given a web search query, retrieve relevant passages that answer the query.",
#     "max_length":8192,
#     "encode_bs":8,
#     "dtype":"bf16"
#   }' \
#   --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
#   --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
#   --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
#   --topk 50 \
#   --out_dir ./outputs_v2/qwen3emb8b_manysettings



# 让 python 能找到 retrieval_wheels/
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"

# （可选）如果你的 retrieval_wheels 里还会 import llm2vec 的 src.*，保留这句；否则可以删
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/ocr:${PYTHONPATH:-}"

# 用你当前跑 LLM2Vec 的 env（里面应当有 transformers>=4.51）
/data3/sunbo/miniconda3/envs/ocr/bin/python run_longembed_eval_v3.py \
  --adapter qwen3_embedding_8b \
  --adapter_kwargs '{
    "model_name":"/data3/sunbo/models/Qwen/Qwen3-Embedding-8B",
    "instruction":"Given a web search query, retrieve relevant passages that answer the query.",
    "max_length":8192,
    "encode_bs":8,
    "dtype":"bf16"
  }' \
  --longembed_root /data1/sunbo/datasets/MLDR \
  --rendered_root /data1/sunbo/datasets_rendered/MLDR \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --datasets mldr-v1.0-en \
  --out_dir ./outputs_v2/qwen3emb8b_manysettings