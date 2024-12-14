#!/bin/bash

# Usage function to display help for the hapless user
usage() {
    echo "Usage: $0 --script SCRIPTS --dataset DATASETS --model MODELS --positive POSITIVES --importance METHOD --granularity GRANULARITY"
    echo "Example: $0 --script my_python_script.py --dataset BIOS10,IMDB --model distilbert,t5,llama --positive True,False --importance grad_only,grad_input --granularity whole,sentence-part"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --script) IFS=',' read -r -a script <<< "$2"; shift ;;
        --dataset) IFS=',' read -r -a datasets <<< "$2"; shift ;;
        --model) IFS=',' read -r -a models <<< "$2"; shift ;;
        --positive) IFS=',' read -r -a positives <<< "$2"; shift ;;
        --importance) IFS=',' read -r -a importance <<< "$2"; shift ;;
        --granularity) IFS=',' read -r -a granularity <<< "$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if all required parameters are provided
if [[ -z "${script[*]}" || -z "${datasets[*]}" || -z "${models[*]}" || -z "${positives[*]}" || -z "${importance[*]}" || -z "${granularity[*]}" ]]; then
    echo "Error: Missing required parameters."
    usage
fi

# Iterate over all combinations and execute the Python script
for script in "${script[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for positive in "${positives[@]}"; do
                if [[ "$model" == "llama" && "$positive" == "True" ]]; then
                    continue
                fi
                for gran in "${granularity[@]}"; do
                    for imp in "${importance[@]}"; do
                        cmd="python scripts/$script --dataset $dataset --model $model --granularity $gran"
                        if [[ "$positive" == "True" ]]; then
                            cmd="$cmd --positive"
                        fi
                        if [[ "$imp" == "grad_input" ]]; then
                            cmd="$cmd --gradxinput"
                        fi
                        # if [[ "$script" == "make_GPT4_prompts.py" ]]; then
                        #     cmd="$cmd --force"
                        # fi
                        echo $cmd
                        $cmd
                    done
                done
            done
        done
    done
done
