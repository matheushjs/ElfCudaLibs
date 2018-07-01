#PBS -N ColCntSeqLin
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# Load modules
# We compile using nvcc because we use CUDA data structures.
# However, the program does not use the GPU.
module load gcc
module load cuda-toolkit/9.0.176

DIR=/home/mathjs/collision_count
cd $DIR

nvcc -O3 -DMETHOD=4 collision.cu -o prog_seq_linear;
echo "Compiled."

time {
	psizeBegin=256
	psizeInc=256
	psizeEnd=65536

	for problemSize in $(seq $psizeBegin $psizeInc $psizeEnd); do
		for executionId in $(seq 5); do
			# Echo progress onto stderr
			echo $progName $problemSize $executionId 1>&2

			# Run code
			output=$( { time ./prog_seq_linear $problemSize | grep Elapsed; } 2>&1 )
			
			# Ouput as csv
			echo -n ${executionId},
			echo -n ${problemSize},
			echo $output | cut -f2,5,7,9 -d' ' | sed -e "s/ /,/g"
		done;
	done >> prog_seq_linear.out;
}

