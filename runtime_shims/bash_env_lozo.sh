# shellcheck shell=bash
#
# Keep legacy LOZO shell env vars from accidentally enabling distributed mode
# in modern transformers/accelerate stacks.
unset LOCAL_RANK
unset WORLD_SIZE
unset MASTER_ADDR
unset MASTER_PORT
unset RANK
