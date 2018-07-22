#PBS -N ColCntSeq
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# This experiment is expected to run in less than 220 minutes, using a single CPU core.

# Load modules
# We compile using nvcc because we use CUDA data structures.
# However, the program does not use the GPU.
module load gcc
module load cuda-toolkit/9.0.176

# Compile different methods
nvcc -O3 -DMETHOD=3 collision.cu -o prog3;

# Execute each method multiple times
for i in $(seq 100); do
	./prog3 >> prog3_output.txt
done


