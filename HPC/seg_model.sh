#!/bin/bash
#SBATCH --partition=gpu_windfall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --export=NONE
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=suxingliu@arizona.edu

cd $SLURM_SUBMIT_DIR

cd /groups/bucksch

singularity exec --nv docker://computationalplantscience/ai_u2net_color_clustering python3 /opt/code/ai_color_cluster_seg.py -p /groups/bucksch/test/30stops_ori/ -ft jpg -o /groups/bucksch/test/30stops_seg/ -s lab -c 2 -min 500 -max 1000000 -pl 0

singularity exec --nv docker://computationalplantscience/dirt3d-reconstruction python3 /opt/code/pipeline.py -i /groups/bucksch/test/30stops_seg/ -o /groups/bucksch/test/30stops_seg//model/ -g 1 -d COLMAP






