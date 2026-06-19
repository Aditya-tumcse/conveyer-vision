#!/usr/bin/env bash
set -euo pipefail

GDRIVE_FILE_ID="1EPeMx8w3NBVmJz1cKLztCbOE9v25WPP3"
INPUT_DIR="/workspace/input"

echo "=== Conveyor Vision setup ==="

if [ ! -d "$INPUT_DIR" ]; then
    echo "Downloading conveyor-sim.zip from Google Drive..."
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O /tmp/conveyor-sim.zip

    echo "Unzipping..."
    unzip -o -q /tmp/conveyor-sim.zip -d /tmp/conveyor-sim
    rm /tmp/conveyor-sim.zip

    echo "Copying data into input/..."
    mkdir -p "$INPUT_DIR/meshes"
    mkdir -p "$INPUT_DIR/images"
    mkdir -p "$INPUT_DIR/metadata"
    cp -r /tmp/conveyor-sim/conveyor-sim/meshes/. "$INPUT_DIR/meshes/"
    cp -r /tmp/conveyor-sim/conveyor-sim/output/frames/. "$INPUT_DIR/images/"
    cp -r /tmp/conveyor-sim/conveyor-sim/output/metadata/. "$INPUT_DIR/metadata/"
    rm -rf /tmp/conveyor-sim

    echo "Data ready at $INPUT_DIR"
else
    echo "Input data already present — skipping download."
fi

echo ""
echo "=== Ready. Run the pipeline: ==="
echo "  python -m pipeline.precompute_gallery --meshes input/meshes --config sim_config.yaml --out artifacts/gallery.pkl"
echo "  python -m pipeline.run_pipeline --data input/images --config sim_config.yaml --gallery artifacts/gallery.pkl --out artifacts/predictions.json"
echo "  python -m pipeline.evaluate --predictions artifacts/predictions.json --ground-truth input/metadata/objects.json --meshes input/meshes"