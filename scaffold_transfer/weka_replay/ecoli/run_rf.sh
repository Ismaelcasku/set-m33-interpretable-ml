#!/usr/bin/env bash
set -euo pipefail
usage() {
    echo 'Usage: bash run_rf.sh inspect WEKA_CLASSPATH MODELS_ROOT'
    echo '       bash run_rf.sh run WEKA_CLASSPATH MODELS_ROOT AUDIT_CSV SCALE_PLUS4 SCALE_PLUS5 SCALE_PLUS6'
}
mode="${1:-}"
case "$mode:$#" in inspect:3|run:7) ;; *) usage >&2; exit 2;; esac
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
classpath="$2"
models="$(realpath -- "$3")"
py="${PYTHON_BIN:-python3}"
mic="$root/tetramers_MIC_no_exact_training_overlap.csv"
if [[ "$mode" == run ]]; then
    audit="$(realpath -- "$4")"
    scales=("$5" "$6" "$7")
    for s in "${scales[@]}"; do
        case "$s" in raw_uM|log10_uM) ;; *) usage >&2; exit 2;; esac
    done
    "$py" -c 'import numpy, scipy'
    [[ -s "$audit" ]]
fi
command -v java >/dev/null && command -v javac >/dev/null || {
    echo "Carga el entorno Java/JDK que usas con WEKA (java y javac)." >&2
    exit 1
}
mkdir -p "$root/results"
out="$(mktemp -d "$root/results/ecoli_charge_${mode}_XXXXXX")"
printf 'Results: %s\n' "$out"
mkdir "$out/classes"
javac -cp "$classpath" -d "$out/classes" "$root/FrozenRF.java"
java_cp="$out/classes:$classpath"
modelnames=(ecol10_batch4.model batch5ecol10.model batch6ecol10.model)
poolnames=(ecoli_training_pool_10.csv ecoli5_training_pool_10.csv ecoli6_training_pool_10.csv)
sha256sum -- "$mic" "$root/data/tetramers_qsar.csv" "$root/pilot.py" "$root/FrozenRF.java" > "$out/input_sha256.txt"
for i in 0 1 2; do
    q=$((i+4))
    d="$out/charge_$q"
    mkdir "$d"
    model="$models/Training/WEKA_models/ecoli/batch $q/${modelnames[i]}"
    pool="$models/Training/Subsets_pool_descriptors/ecoli/subset$q/${poolnames[i]}"
    sha256sum -- "$model" "$pool" >> "$out/input_sha256.txt"
    if ! java -cp "$java_cp" FrozenRF header "$model" > "$d/model_header.arff" 2> "$d/target.txt"; then
        cat "$d/target.txt" >&2
        exit 1
    fi
    target="$(sed -n 's/^WEKA_CLASS=//p' "$d/target.txt")"
    [[ -n "$target" ]]
    "$py" "$root/pilot.py" prepare --mic "$mic" --species E_coli --charge "$q" \
        --out "$d" --header "$d/model_header.arff" --target "$target" --pool "$pool" > "$d/class_index.txt"
    cat "$d/pool_check.json"
done
if [[ "$mode" == run ]]; then
    printf 'scale_plus4=%s\nscale_plus5=%s\nscale_plus6=%s\n' "${scales[@]}" > "$out/scales.txt"
    sha256sum -- "$audit" >> "$out/input_sha256.txt"
    for i in 0 1 2; do
        q=$((i+4)); d="$out/charge_$q"
        model="$models/Training/WEKA_models/ecoli/batch $q/${modelnames[i]}"
        java -cp "$java_cp" FrozenRF predict "$model" "$d/test.arff" > "$d/weka_predictions.csv"
    done
    "$py" "$root/pilot.py" evaluate --mic "$mic" --out "$out" --species E_coli \
        --audit "$audit" --matched-dir "$out" --scales "${scales[@]}"
fi
