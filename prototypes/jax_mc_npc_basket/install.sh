#!/bin/bash
# Put the JAX sampler next to the published NPC_Basket scripts:
# clones integrativemodeling/NPC_Basket at the tested commit (or uses an
# existing clone), copies bench/ into it, and makes the patched modeling
# scripts in bench/patched/ from the clone's own scripts and the patches
# (modeling_basket.patch, modeling_mbasket.patch).
#   bash install.sh [path/to/NPC_Basket]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:-$PWD/NPC_Basket}
if [ ! -d "$DEST/.git" ]; then
    git clone -q https://github.com/integrativemodeling/NPC_Basket.git "$DEST"
    git -C "$DEST" checkout -q 7511ffd
fi
if [ -e "$DEST/bench" ]; then
    echo "$DEST/bench exists; remove it or pick another clone" >&2
    exit 1
fi
cp -r "$HERE/bench" "$DEST/"
P=$DEST/bench/patched
mkdir -p "$P/Yeast_src" "$P/Mouse_src"
cp "$DEST/Yeast_NPC_Basket/src/modeling_basket.py" "$P/Yeast_src/"
patch -s "$P/Yeast_src/modeling_basket.py" < "$P/modeling_basket.patch"
for f in __init__.py cc_tpr_coconut.py StructuralEquivalenceRestraint.py \
         pdbclass.py modeling_mbasket.py; do
    cp "$DEST/Mouse_NPC_Basket/src/$f" "$P/Mouse_src/"
done
patch -s "$P/Mouse_src/modeling_mbasket.py" < "$P/modeling_mbasket.patch"
echo "installed into $DEST/bench; run from $DEST/bench/jax"
