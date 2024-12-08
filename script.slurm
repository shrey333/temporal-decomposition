#!/bin/bash
#SBATCH --job-name=llama_finetune_gsm8k     # Job name
#SBATCH -q public
#SBATCH --partition=general                     # Partition (queue) name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (use 1 task for single process)
#SBATCH --cpus-per-task=8                  # Number of CPU cores per task
#SBATCH --gpus=a100:1                       # Number of A100 GPUs
#SBATCH -C a100_80
#SBATCH --mem=16G                           # Total memory for the job (64GB)
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --output=llama_finetune_%j.log      # Standard output log (%j will be replaced by job ID)
#SBATCH --error=llama_finetune_%j.err       # Error log (%j will be replaced by job ID)
#SBATCH --mail-type=END,FAIL                # Send email when the job ends or fails
#SBATCH --mail-user=sbhadiya@asu.edu        # Your email address for notifications

# Load Mamba and Initialize
module load mamba/latest

# Load CUDA
module load cuda-12.6.1-gcc-12.1.0

module load gcc-12.1.0-gcc-11.2.0 

Create the Environment
mamba env create -f timeset.yaml -y

if [ $? -ne 0 ]; then
    echo "Mamba environment creation failed."
    exit 1
fi

# Activate the Environment
source activate timeset

if [ $? -ne 0 ]; then
    echo "Failed to activate the timeset environment."
    exit 1
fi

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.version.cuda)"


# Verify Activation
echo "Activated environment: $(mamba info --envs | grep '*' | awk '{print $1}')"

cd timeset

# Install Spacy Model
python -m spacy download en_core_web_sm

if [ $? -ne 0 ]; then
    echo "Failed to download Spacy model."
    exit 1
fi

# Execute Preprocessing Script
python preprocess_torque.py

if [ $? -ne 0 ]; then
    echo "Preprocessing script failed."
    exit 1
fi

python finetune.py

if [ $? -ne 0 ]; then
    echo "Finetune script failed."
    exit 1
fi

# Execute Benchmarking Script
python inference.py

if [ $? -ne 0 ]; then
    echo "Inference script failed."
    exit 1
fi

# Deactivate the Environment
source deactivate

echo "Job completed successfully."
