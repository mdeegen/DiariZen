#!/bin/bash
gcc_dir=$1

recipe_root=/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc

# >>> micromamba setup (EXPLIZIT) <<<
export MAMBA_EXE='/homes/eva/q/qdeegen/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/homes/eva/q/qdeegen/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from micromamba activate
fi
unset __mamba_setup

micromamba activate diarizen && python "$recipe_root/precomputation/merge_index.py" --out_dir $gcc_dir