#!/bin/bash
# ============================================================
# MARL4DRP - Conda 環境セットアップスクリプト
# 動作確認環境: Python 3.9, PyTorch 2.7.0, CUDA 12.8 (RTX 5080)
#
# 使い方:
#   bash setup_conda.sh          # デフォルト: CUDA 12.8
#   bash setup_conda.sh cpu      # CPU のみ（GPU なし）
#   bash setup_conda.sh cu121    # CUDA 12.1 の場合
# ============================================================

set -e  # エラーが発生したら即終了

ENV_NAME="drp_new"
PYTHON_VERSION="3.9"
CUDA_TAG="${1:-cu128}"  # デフォルトは CUDA 12.8

# ---- カラー出力 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERR]${NC}  $1"; exit 1; }

# ---- スクリプトのあるディレクトリに移動 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "作業ディレクトリ: $SCRIPT_DIR"

# ============================================================
# 1. conda の確認
# ============================================================
info "conda を確認中..."
if ! command -v conda &>/dev/null; then
    error "conda が見つかりません。Miniconda/Anaconda をインストールしてください。"
fi
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
success "conda: $(conda --version)"

# ============================================================
# 2. conda 環境の作成
# ============================================================
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "conda 環境 '${ENV_NAME}' は既に存在します。スキップします。"
    warn "  再作成する場合: conda env remove -n ${ENV_NAME}"
else
    info "conda 環境 '${ENV_NAME}' を作成中 (Python ${PYTHON_VERSION})..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    success "conda 環境を作成しました。"
fi

conda activate "$ENV_NAME"
info "conda 環境を有効化: $CONDA_DEFAULT_ENV"

# ============================================================
# 3. PyTorch のインストール
# ============================================================
info "PyTorch をインストール中 (${CUDA_TAG})..."

if [ "$CUDA_TAG" = "cpu" ]; then
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
else
    TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
    pip install \
        torch==2.7.0+${CUDA_TAG} \
        torchvision==0.22.0+${CUDA_TAG} \
        torchaudio==2.7.0+${CUDA_TAG} \
        --index-url "$TORCH_INDEX"
fi
success "PyTorch をインストールしました。"

# ============================================================
# 4. gym 0.21.0 のインストール（特定コミット）
# ============================================================
info "gym 0.21.0 をインストール中..."
# gym 0.21 の setup.py には2つの問題がある:
#   1. extras_require の形式が setuptools>=67 で拒否される
#   2. opencv-python>=3. という不正なバージョン指定子
# → クローンしてパッチを当ててからインストールする
GYM_TMP="/tmp/gym_install_$$"
git clone --quiet https://github.com/openai/gym.git "$GYM_TMP"
git -C "$GYM_TMP" checkout --quiet c755d5c35a25ab118746e2ba885894ff66fb8c43
sed -i 's/opencv-python>=3\./opencv-python>=3.0/g' "$GYM_TMP/setup.py"
pip install "setuptools<67" --quiet
pip install --no-build-isolation --no-deps "$GYM_TMP"
pip install --upgrade setuptools --quiet
rm -rf "$GYM_TMP"
success "gym をインストールしました。"

# ============================================================
# 5. 依存パッケージのインストール（requirements.txt）
# ============================================================
info "依存パッケージをインストール中 (requirements.txt)..."
pip install -r requirements.txt
success "依存パッケージをインストールしました。"

# ============================================================
# 6. SMAC のインストール（オプション）
# ============================================================
read -r -p "$(echo -e "${YELLOW}[Q]${NC} SMAC (StarCraft II) をインストールしますか？ [y/N]: ")" install_smac
if [[ "$install_smac" =~ ^[Yy]$ ]]; then
    info "SMAC をインストール中..."
    pip install "SMAC @ git+https://github.com/oxwhirl/smac.git@d6aab33f76abc3849c50463a8592a84f59a5ef84"
    success "SMAC をインストールしました。"
else
    info "SMAC のインストールをスキップしました。"
fi

# ============================================================
# 7. drp_env パッケージのローカルインストール
# ============================================================
info "drp_env パッケージをローカルインストール中 (pip install -e .)..."
pip install -e .
success "drp_env をインストールしました。"

# ============================================================
# 8. PAC アルゴリズム依存パッケージ（オプション）
# ============================================================
if [ -f "epymarl/pac_requirements.txt" ]; then
    read -r -p "$(echo -e "${YELLOW}[Q]${NC} PAC アルゴリズムの依存パッケージをインストールしますか？ [y/N]: ")" install_pac
    if [[ "$install_pac" =~ ^[Yy]$ ]]; then
        info "PAC 依存パッケージをインストール中..."
        # torch_scatter は torch を検出するため --no-build-isolation が必要
        pip install einops
        pip install --no-build-isolation torch_scatter
        success "PAC 依存パッケージをインストールしました。"
    else
        info "PAC のインストールをスキップしました。"
    fi
fi

# ============================================================
# 完了メッセージ
# ============================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} セットアップ完了！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  conda 環境を有効化:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  実行例 (QMIX):"
echo "    conda run -n ${ENV_NAME} python3 epymarl/src/main.py \\"
echo "        --config=qmix --env-config=gymma \\"
echo "        with env_args.key=\"drp_env:drp-5agent_map_10x10-v2\" \\"
echo "             env_args.state_repre_flag=\"onehot\""
echo ""
