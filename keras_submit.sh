#!/bin/bash
#
# CompecTA (c) 2018
#
#
# TODO:
#   - Set name of the job below changing "Keras" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch keras_pulsar_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=ae
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --partition=long
#SBATCH --output=%j-deep.out

# #SBATCH --mail-user=your_mail@foo.com

################################################################################
#source /etc/profile.d/lmod.sh
################################################################################

# MODULES LOAD...
echo "Load Anaconda-3..."
module load anaconda/3

################################################################################

echo ""
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo ""
echo ""

echo "Running Keras-Tensorflow command..."
echo "===================================================================================="
RET=$?

python AE2_xy.py --num_windows 32 \
 --seq_window_lengths 3 4 8 \
 --smi_window_lengths 4 6 8 \
 --batch_size 256 \
 --num_epoch 100 \
 --max_num_words 20000 \
 --max_seq_len 100 \
 --max_smi_len 100 \
 --dataset_path data/davis/ \
 --prot_embedding_path data/emb/gensim.prot.l3.w20.100d.txt \
 --drug_embedding_path data/emb/drug.rdkit.can.4M.l8.ws20.txt \
 --idf_path data/emb/idfs_rdk_l8_ws20.txt \
 --problem_type 1 \
 --bind_threshold 7 \
 --is_ligand_centric 1 \
 --log_dir logs/
RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET
