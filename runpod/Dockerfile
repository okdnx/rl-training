# Runpod公式ベース：CUDA 13.2 / Ubuntu 24.04
# 確認できたdigestに固定
FROM runpod/base:2.0.0-cuda1320-ubuntu2404@sha256:9359a93cff068a583f2663b8b2eef439cc3199ae54a029ec291ae5213e44e7b2

# uvもバージョンとdigestを固定
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b \
    /uv /uvx /usr/local/bin/

# Python本体と仮想環境は、Volumeとは別の場所に配置
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

ENV PATH="/opt/venv/bin:${PATH}"

# モデルキャッシュは、実行時の永続ストレージ側へ
ENV HF_HOME=/workspace/.cache/huggingface

# 環境定義はイメージ内へ保存
WORKDIR /opt/rl-environment

COPY pyproject.toml uv.lock .python-version ./

# ロック済みのPython依存関係をインストール
# ダウンロードキャッシュは完成イメージへ含めない
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

WORKDIR /workspace

# CMDは変更しない。
# Runpod公式ベースの/start.shを引き継ぎ、SSH起動処理を残す。