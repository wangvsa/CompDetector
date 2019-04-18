#!/bin/bash
#SBATCH --job-name="restart_sedov"
#SBATCH --output="log_sedov_0_190.log"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 24:00:00

conda deactivate
conda activate flash_env
module purge
module load gnutools
module load gnu openmpi_ib
module load hdf5
export PYTHONPATH=/home/wangvsa/anaconda2/envs/flash_env/lib

# Sedov
cd /oasis/scratch/comet/wangvsa/temp_project/Flash/Sedov/
cp /home/wangvsa/softwares/CompDetector/tools/restart_comet.py ./
cp /home/wangvsa/softwares/CompDetector/create_dataset.py ./
python ./restart_comet.py sedov_hdf5_chk_ 0 90 &
python ./restart_comet.py sedov_hdf5_chk_ 90 190
