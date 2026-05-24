#!/usr/bin/env bash


#!/usr/bin/env bash

# Colors for scannable terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# The model evaluation lineup
MODELS=(
    "Qwen3.6-35B-A3B-UD-Q5_K_M"
    "gemma-4-26B-A4B-it-Q8_0"
    "Qwopus3.6-27B-v2-MTP-Q4_K_S"
    "DeepSeek-R1-Distill-Qwen-32B-Q5_K_M"
)

TARGET_FILE="hello.txt"
declare -A RESULTS

echo -e "${CYAN}==================================================${NC}"
echo -e "${CYAN}  STARTING LOCAL AGENT TOOL-CALLING BENCHMARK     ${NC}"
echo -e "${CYAN}==================================================${NC}"

for MODEL in "${MODELS[@]}"; do
    echo -e "\n${CYAN}► Testing Model:${NC} ${MODEL}"

    # 1. Rigorous Clean Room Setup
    if [ -f "$TARGET_FILE" ]; then
        rm "$TARGET_FILE"
    fi

    # 2. Unload current model to flush VRAM context
    echo "  Cleaning VRAM allocation..."
    lls unload > /dev/null 2>&1
    sleep 2

    # 3. Load next model variant
    echo "  Loading model weights..."
    if ! lls load "$MODEL"; then
        echo -e "  ${RED}❌ Failed to load model weights.${NC}"
        RESULTS["$MODEL"]="LOAD_FAILED"
        continue
    fi
    sleep 3 # Give llama.cpp/LlamaStudio a moment to stabilize the memory map

    # 4. Fire the One-Shot Tool Call Test
    echo "  Executing one-shot prompt tool instruction..."
    lls oneshot "Write a hello.txt in the current directory and say hi" > /dev/null 2>&1

    # 5. Verify physical file system mutation
    if [ -f "$TARGET_FILE" ]; then
        # Double check it's not empty and actually contains data
        if [ -s "$TARGET_FILE" ]; then
            echo -e "  ${GREEN}✓ Success! File written onto NVMe drive.${NC}"
            RESULTS["$MODEL"]="${GREEN}PASS${NC}"
        else
            echo -e "  ${RED}✗ Partial Fail: File created but left empty.${NC}"
            RESULTS["$MODEL"]="${RED}FAIL (Empty File)${NC}"
        fi
    else
        echo -e "  ${RED}✗ Failure: No file written to disk (Tooling blind).${NC}"
        RESULTS["$MODEL"]="${RED}FAIL (No File)${NC}"
    fi
done

# Cleanup final test artifact
if [ -f "$TARGET_FILE" ]; then rm "$TARGET_FILE"; fi

# Unload the last model to leave the workstation in a clean state
lls unload > /dev/null 2>&1

# ==================================================
#                  FINAL REPORT CARD
# ==================================================
echo -e "\n\n${CYAN}==================================================${NC}"
echo -e "${CYAN}               BENCHMARK REPORT CARD              ${NC}"
echo -e "${CYAN}==================================================${NC}"

# Format output as a clean table
printf "%-40s | %s\n" "MODEL RUNNER IDENTIFIER" "TOOL-CALL STATUS"
echo "------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    printf "%-40s | %b\n" "$MODEL" "${RESULTS[$MODEL]}"
done

echo -e "${CYAN}==================================================${NC}"


