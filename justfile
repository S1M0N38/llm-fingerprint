################################################################################
# Generate
################################################################################

meta_models := "llama-3.2-1b llama-3.2-3b llama-3.1-8b"
mistral_models := "ministral-8b mistral-nemo-12b mistral-7b"
qwen_models := "qwen-2.5-0.5b qwen-2.5-1.5b qwen-2.5-3b qwen-2.5-14b"
google_models := "gemma-3-1b gemma-3-4b gemma-3-12b"

timestamp := `date +%Y%m%dT%H%M%S`

generate-samples-for-all-models:
    llm-fingerprint generate \
      --language-model {{meta_models}} {{mistral_models}} {{qwen_models}} {{google_models}} \
      --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
      --samples-path "./data/samples/{{timestamp}}.jsonl" \
      --samples-num 4

################################################################################
# ChromaDB
################################################################################

chroma-run:
    #!/usr/bin/env bash
    mkdir -p ./data/chroma
    nohup chroma run --path ./data/chroma --log-path ./data/chroma/chroma.log --port 1235 > /dev/null 2>&1 &
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
