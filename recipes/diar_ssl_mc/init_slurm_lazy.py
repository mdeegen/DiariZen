from paderbox.io.new_subdir import NameGenerator
import padertorch as pt
from pathlib import Path


BATCHFILE_TEMPLATE_EVAL = """#!/bin/bash
#SBATCH -t 50:00:00          # time limit hrs:min:sec
#SBATCH -A hpc-prf-nt2
#SBATCH -p gpu
#SBATCH --output {nickname}_eval_%j.out
#SBATCH --error {nickname}_eval_%j.err
#SBATCH -J {nickname}_eval
#SBATCH --gres=gpu:a100:4            # 4 GPU
#SBATCH --mem=450G               # entspricht mem_free
#SBATCH --ntasks=1          # slurm und accelerate get into conflict => slurm =1

#SBATCH --cpus-per-task=16          # 4 CPUs per GPU


cd /scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc
srun bash run_stage_noctua2_lazy.sh diarizen_shared_wavlm_counting_pruned_att
"""

def init(_run, experiment_dir=None):
    #
    # pt.io.dump_config(
    #     copy.deepcopy(_config),
    #     Path(experiment_dir) / 'config.json'
    # )
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_storage_dir(
            'DiariZen',
            id_naming=NameGenerator(('adjectives', 'colors', 'animals')),
        )
    batchfile_path = Path(experiment_dir) / "run_slurm.sh"
    print(f'Creating batchfile at {batchfile_path}')
    batchfile_path.write_text(
        BATCHFILE_TEMPLATE_EVAL.format(
            main_python_path=pt.configurable.resolve_main_python_path(),
            nickname=experiment_dir.name,
        )
    )
    print(f"cd {experiment_dir} \n sbatch run_slurm.sh")


if __name__ == '__main__':
    import argparse
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=False,
        help='Path to the experiment directory where the batchfile will be created.',
    )
    args = parser.parse_args()

    init(_run=None, experiment_dir=args.experiment_dir)