#!/usr/bin/env bash
# Run likelihood computations (means_nlls + tasks_nlls) across all configurations.
# Each configuration is run in a separate Python subprocess to avoid memory accumulation.
#
# Usage (from project root):
#   bash tests/test_likelihoods_configurations.sh
#
# Variables iterated (matching the notebook):
#   sth  (shared_task_hps)       : true / false
#   sch  (shared_cluster_hps)    : true / false
#   chit (cluster_hps_in_tasks)  : true / false
#   I    (input_dim)             : 1 / 2
#   soh  (shared_output_hps)     : true / false
#   siit (shared_inputs_in_tasks): true / false
#
# Fixed: fh=false, siif=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_likelihoods.py"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.15

ok=0
ko=0
total=0

for sth in true false; do
    for sch in true false; do
        for chit in true false; do
            for I in 1 2; do
                for soh in true false; do
                    for siit in true false; do
                        total=$((total + 1))
                        output=$(python "$PYTHON_SCRIPT" \
                            --sth "$sth" \
                            --sch "$sch" \
                            --chit "$chit" \
                            --I "$I" \
                            --soh "$soh" \
                            --siit "$siit" 2>&1)
                        exit_code=$?
                        echo "$output"
                        if [ $exit_code -eq 0 ]; then
                            ok=$((ok + 1))
                        else
                            ko=$((ko + 1))
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "Results: $ok/$total OK, $ko/$total KO"