################################################################################
# Generate
################################################################################

models := "phi-4 meta-llama-3.1-8b-instruct granite-3.2-8b-instruct gemma-3-4b-it"
timestamp := `date +%Y%m%dT%H%M%S`

generate-samples-for-all-models:
    llm-fingerprint generate \
      --language-model {{models}} \
      --prompts-path "./data/prompts/prompts_general_v1.jsonl" \
      --samples-path "./data/samples/{{timestamp}}.jsonl" \
      --samples-num 3

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
