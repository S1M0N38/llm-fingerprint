################################################################################
# Generate
################################################################################

timestamp := `date +%Y%m%dT%H%M%S`
model := ""

llama_models := "llama-3.2-1b llama-3.2-3b llama-3.1-8b"
mistral_models := "ministral-8b mistral-nemo-12b mistral-7b"
qwen_models := "qwen-2.5-0.5b qwen-2.5-1.5b qwen-2.5-3b qwen-2.5-7b qwen-2.5-14b"
gemma_models := "gemma-3-1b gemma-3-4b gemma-3-12b"
phi_models := "phi-4 phi-4-mini"
smollm_models := "smollm-2-135m smollm-2-360m smollm-2-1.7b"


generate-samples:
    llm-fingerprint generate \
      --language-model {{model}} \
      --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
      --samples-path "./data/samples/{{timestamp}}.jsonl" \
      --samples-num 16 \
      --concurrent-requests 6

generate-samples-for-all-models:
    llm-fingerprint generate \
      --language-model {{llama_models}} {{mistral_models}} {{qwen_models}} {{gemma_models}} {{phi_models}} {{smollm_models}} \
      --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
      --samples-path "./data/samples/{{timestamp}}.jsonl" \
      --samples-num 16 \
      --concurrent-requests 6

generate-samples-1b-models:
    llm-fingerprint generate \
      --language-model "llama-3.2-1b qwen-2.5-1.5b gemma-3-1b smollm-2-1.7b" \
      --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
      --samples-path "./data/samples/{{timestamp}}.jsonl" \
      --samples-num 16 \
      --concurrent-requests 6

################################################################################
# Test
################################################################################

test-generate-models-swapping:
    llm-fingerprint generate \
      --language-model {{llama_models}} {{mistral_models}} {{qwen_models}} {{gemma_models}} {{phi_models}} {{smollm_models}} \
      --prompts-path "./data/prompts/prompts_single_v1.jsonl" \
      --samples-path "/tmp/llm_samples_{{timestamp}}.jsonl" \
      --samples-num 1

################################################################################
# ChromaDB
################################################################################

chroma-run:
    #!/usr/bin/env bash
    mkdir -p ./data/chroma
    nohup chroma run --path ./data/chroma --log-path ./data/chroma/chroma.log --port 1236 > /dev/null 2>&1 &
    echo $! > ./data/chroma/.chroma.pid
    echo "ChromaDB started in background (PID: $(cat ./data/chroma/.chroma.pid))"

chroma-stop:
    #!/usr/bin/env bash
    if [ -f ./data/chroma/.chroma.pid ]; then
        pid=$(cat ./data/chroma/.chroma.pid)
        kill $pid
        rm ./data/chroma/.chroma.pid
        echo "ChromaDB stopped (PID: $pid)"
    else
        echo "ChromaDB is not running or PID file not found"
    fi
