#PBS -N ColCntSeqQuad
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# Load modules
# We compile using nvcc because we use CUDA data structures.
# However, the program does not use the GPU.
module load gcc
module load cuda-toolkit/9.0.176

DIR=/home/mathjs/ElfColCnt
cd $DIR

binary=seq_quad

make $binary

time {
	psizeBegin=1024
	psizeInc=32768
	psizeEnd=983040

	for problemSize in $(seq $psizeBegin $psizeInc $psizeEnd); do
		for executionId in $(seq 10); do
			# Echo progress onto stderr
			echo $binary $problemSize $executionId 1>&2

			# Run code
			output=$( { time ./$binary $problemSize | grep Elapsed; } 2>&1 )
			
			# Ouput as csv
			echo -n ${executionId},
			echo -n ${problemSize},
			echo $output | cut -f2,5,7,9 -d' ' | sed -e "s/ /,/g"
		done;
	done >> $binary.out;
}
