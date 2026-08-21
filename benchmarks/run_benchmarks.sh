#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$DIR/.."

cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT_DIR/build" --target benchmark_cpp >/dev/null

SIZES=(1000 5000 10000 25000 50000)
if [ "$#" -gt 0 ]; then
  SIZES=("$@")
fi

printf "lang\tn\tac1_ms\tac2_ms\n"
for N in "${SIZES[@]}"; do
  "$ROOT_DIR/build/benchmark_cpp" "$N" 5
  julia --banner=no --project="$DIR" "$DIR/benchmark_jl.jl" "$N" 5 2>/dev/null
done
