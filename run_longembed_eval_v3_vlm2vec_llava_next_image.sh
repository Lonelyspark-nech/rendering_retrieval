export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/retrieval_wheels:${PYTHONPATH:-}"
export PYTHONPATH="/data3/sunbo/ocr2/rendering_glyph/VLM2Vec:${PYTHONPATH:-}"

python run_longembed_eval_v3.py \
  --adapter vlm2vec_llava_next_text2image \
  --adapter_kwargs '{
    "model_name": "/data3/sunbo/models/TIGER-Lab/VLM2Vec-LLaVa-Next",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "doc_image_question": "What is in the image",
    "qry_bs": 16,
    "doc_bs": 8,
    "use_faiss": true,
    "faiss_use_gpu": false,
    "faiss_normalize": false
  }' \
  --longembed_root /data3/sunbo/ocr2/datasets/dwzhu/LongEmbed \
  --rendered_root /data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v5 \
  --settings ps256x256_dpi72 ps512x512_dpi72 ps768x768_dpi72 ps1024x1024_dpi72 \
  --topk 50 \
  --out_dir ./outputs_v2/vlm2vec_llava_next
