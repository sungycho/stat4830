#!/usr/bin/env bash
set -euo pipefail

# Prime Intellect VM bootstrap for this repo.
# Installs system prerequisites, uv, project dependencies, and optional
# LOZO + dry-run preflight checks for week9 notebook workflow.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_SYNC=1
CLONE_LOZO=1
RUN_PREFLIGHT=1
INSTALL_PYTHON=1
PYTHON_VERSION="3.13"
LOZO_REF="5d1ade5bf41846471d33ba526f877829b9a19856"
REQUIRE_DATA_PREFLIGHT=0

usage() {
  cat <<'EOF'
Usage: bash environments/bootstrap_prime_vm.sh [options]

Options:
  --repo-root PATH        Override repo root (default: inferred from script path)
  --python-version VER    Python version for uv-managed interpreter (default: 3.13)
  --no-sync               Skip "uv sync"
  --no-lozo-clone         Skip cloning external/LOZO
  --no-preflight          Skip dry-run preflight runner checks
  --no-python-install     Skip "uv python install"
  --lozo-ref REF          Pin LOZO checkout to git ref/commit (default: ${LOZO_REF})
  --require-data-preflight
                          Require k-shot data dirs during preflight checks
  -h, --help              Show this help text

Examples:
  bash environments/bootstrap_prime_vm.sh
  bash environments/bootstrap_prime_vm.sh --no-preflight
  bash environments/bootstrap_prime_vm.sh --python-version 3.12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --no-sync)
      RUN_SYNC=0
      shift
      ;;
    --no-lozo-clone)
      CLONE_LOZO=0
      shift
      ;;
    --no-preflight)
      RUN_PREFLIGHT=0
      shift
      ;;
    --no-python-install)
      INSTALL_PYTHON=0
      shift
      ;;
    --lozo-ref)
      LOZO_REF="$2"
      shift 2
      ;;
    --require-data-preflight)
      REQUIRE_DATA_PREFLIGHT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[error] Repo root not found: ${REPO_ROOT}"
  exit 1
fi

cd "${REPO_ROOT}"
echo "[info] Repo root: ${REPO_ROOT}"

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

echo "[step] Installing OS prerequisites (curl, git, jq, build tools)..."
${SUDO} apt-get update
${SUDO} apt-get install -y \
  curl \
  ca-certificates \
  git \
  jq \
  build-essential \
  pkg-config \
  python3 \
  python3-venv

if ! command -v uv >/dev/null 2>&1; then
  echo "[step] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    ${SUDO} ln -sf "${HOME}/.local/bin/uv" /usr/local/bin/uv || true
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv still not found on PATH."
  echo "        Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
fi

echo "[info] uv version: $(uv --version)"

if [[ "${INSTALL_PYTHON}" -eq 1 ]]; then
  echo "[step] Installing Python ${PYTHON_VERSION} through uv (idempotent)..."
  uv python install "${PYTHON_VERSION}"
fi

if [[ "${RUN_SYNC}" -eq 1 ]]; then
  echo "[step] Running uv sync..."
  uv sync
fi

if [[ "${CLONE_LOZO}" -eq 1 ]]; then
  echo "[step] Ensuring external/LOZO exists..."
  mkdir -p external
  if [[ ! -d "external/LOZO/.git" ]]; then
    git clone https://github.com/optsuite/LOZO "external/LOZO"
  else
    echo "[info] external/LOZO already present."
  fi

  if [[ -n "${LOZO_REF}" ]]; then
    echo "[step] Enforcing LOZO pinned ref: ${LOZO_REF}"
    git -C "external/LOZO" fetch --all --tags --prune
    git -C "external/LOZO" checkout --detach "${LOZO_REF}"
  fi

  echo "[info] LOZO current SHA: $(git -C external/LOZO rev-parse HEAD)"
fi

if [[ "${RUN_PREFLIGHT}" -eq 1 ]]; then
  if [[ -f "src/scripts/run_lozo_medium_suite.py" && -d "external/LOZO" ]]; then
    echo "[step] Running LOZO environment preflight checks..."
    PREFLIGHT_ARGS=(
      --lozo-root external/LOZO
      --tasks SST-2,RTE
      --k-values 16
      --seeds 42
    )
    if [[ "${REQUIRE_DATA_PREFLIGHT}" -eq 1 ]]; then
      PREFLIGHT_ARGS+=(--require-data)
    fi
    uv run python -m src.scripts.check_lozo_medium_env "${PREFLIGHT_ARGS[@]}"

    echo "[step] Running suite dry-run (smoke/full)..."
    uv run python -m src.scripts.run_lozo_medium_suite \
      --lozo-root external/LOZO \
      --run-root results/preflight_smoke \
      --profile smoke \
      --dry-run

    uv run python -m src.scripts.run_lozo_medium_suite \
      --lozo-root external/LOZO \
      --run-root results/preflight_full \
      --profile full \
      --dry-run
  else
    echo "[warn] Skipping preflight: LOZO clone or runner script missing."
  fi
fi

echo "[step] GPU visibility check..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "[warn] nvidia-smi not found."
fi

echo "[done] Bootstrap complete."
echo "       Next: uv run jupyter lab --allow-root --no-browser --ip 127.0.0.1 --port 8888"
