"""
CUDA_VISIBLE_DEVICES=2 python3.10 -m transformers_openai.main \
--host 0.0.0.0 --port 7088 \
--model-type transformers_openai.models.WhisperForConditionalGeneration \
--processor-type transformers_openai.models.WhisperFeatureExtractor \
--serving-type whisper --hf-model openai/whisper-large-v3 \
--tokenizer-use-fast false \
--static-cache true \
--static-cache-encoder-max-length 1500 --static-cache-decoder-max-length 446 \
--continuous-batching-batch-size 2
"""