# Source this: the Panther-EM environment for the SVD steps (search, site scoring, autotrace).
#
# The SVD steps run in Panther-EM's own venv; the Leopard-EM steps in a separate Python
# (LEOPARD_PY in run_svd_pipeline.sh). Every setting can be overridden from the caller:
#   PANTHER_ROOT  Panther-EM checkout with a .venv (default: Panther-EM next to this
#                 Leopard-EM checkout). The pipeline needs the emit_entries search
#                 (branch jd/mt-correlation-table, on top of mdg-seed-local-stage1).
#   PANTHER_VENV  its venv (default: $PANTHER_ROOT/.venv)
#   CUDA_HOME     CUDA toolkit whose nvcc builds the fused-kernel JIT; it must match the
#                 CUDA of the venv's torch (default: /usr/local/cuda-12.8 if present)
# Or set PANTHER_ENV to a script of your own; the pipeline sources that instead.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PANTHER_ROOT=${PANTHER_ROOT:-$(cd "$_here/../../.." && pwd)/Panther-EM}
PANTHER_VENV=${PANTHER_VENV:-$PANTHER_ROOT/.venv}
if [ ! -x "$PANTHER_VENV/bin/python" ]; then
    echo "panther_env.sh: no Python in $PANTHER_VENV (set PANTHER_ROOT or PANTHER_VENV)" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.8 ]; then
    CUDA_HOME=/usr/local/cuda-12.8
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME
    export PATH=$PANTHER_VENV/bin:$CUDA_HOME/bin:$PATH
else
    export PATH=$PANTHER_VENV/bin:$PATH
fi
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions/panther-em}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}
unset _here
