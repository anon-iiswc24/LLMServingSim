#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath $0)")
#./build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
BINARY="${SCRIPT_DIR:?}"/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
#./inputs/custom_workload/llm
WORKLOAD="${SCRIPT_DIR:?}"/inputs/custom_workload/opt-30b_b1_s64_orca_n64/llm
#./inputs/system/sample_fully_connected_sys.txt
SYSTEM="${SCRIPT_DIR:?}"/inputs/system/sample_fully_connected_sys.txt
#./inputs/network/analytical/fully_connected.json
NETWORK="${SCRIPT_DIR:?}"/inputs/network/analytical/fully_connected_5d_64.json
#./inputs/remote_memory/analytical/per_npu_memory_expansion.json
MEMORY="${SCRIPT_DIR:?}"/inputs/remote_memory/analytical/per_npu_memory_expansion.json

# running version
"${BINARY}" \
  --workload-configuration="${WORKLOAD}" \
  --system-configuration="${SYSTEM}" \
  --network-configuration="${NETWORK}"\
  --remote-memory-configuration="${MEMORY}"

# debugging run using gdb
# gdb --args "${BINARY}" \
#   --workload-configuration="${WORKLOAD}" \
#   --system-configuration="${SYSTEM}" \
#   --network-configuration="${NETWORK}"\
#   --remote-memory-configuration="${MEMORY}"
