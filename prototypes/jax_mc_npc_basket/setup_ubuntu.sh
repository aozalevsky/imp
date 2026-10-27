#!/bin/bash
# Optional: everything on a fresh Ubuntu machine with an NVIDIA GPU
# (needs sudo): the IMP nightly .deb, a Python venv with JAX 0.11.2 for
# CUDA, then install.sh. Skip this if you already have IMP nightly and JAX.
#   bash setup_ubuntu.sh [workdir]      (default ~/npcjax)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${1:-$HOME/npcjax}
mkdir -p "$W" && cd "$W"
sudo apt-get -o DPkg::Lock::Timeout=600 update -qq
sudo apt-get -o DPkg::Lock::Timeout=600 install -y -qq wget git python3-venv python3-pip >/dev/null
# newest IMP nightly for this Ubuntu release
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
BASE=https://integrativemodeling.org/nightly/download
DEB=$(wget -qO- "$BASE/$CODENAME/" | grep -o 'href="[^"]*imp_[^"]*\.deb"' \
      | sed 's/href="//;s/"$//' | sort | tail -1)
echo "IMP nightly: $DEB"
wget -q "$BASE/$CODENAME/$DEB" -O imp.deb
sudo apt-get -o DPkg::Lock::Timeout=600 install -y -qq ./imp.deb >/dev/null
# JAX: CUDA 13 wheels need driver >= 580, otherwise CUDA 12
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CU=cuda12; [ "${DRV%%.*}" -ge 580 ] && CU=cuda13
python3 -m venv --system-site-packages venv      # sees the system IMP
. venv/bin/activate
pip install -q "jax[$CU]==0.11.2" matplotlib scipy ihm
bash "$HERE/install.sh" "$W/NPC_Basket"
python -c "import IMP, IMP.bayesianem, IMP.npc, IMP.pmi, jax; \
print('IMP', IMP.get_module_version(), 'jax', jax.__version__, jax.devices())"
echo "next: . $W/venv/bin/activate; cd $W/NPC_Basket/bench/jax"
