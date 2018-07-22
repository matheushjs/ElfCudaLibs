#PBS -N ColCntSeqSmall
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc
module load cuda-toolkit/9.0.176

DIR=/home/mathjs/collision_count
cd $DIR

# Here we will experiment with the multiple-blocks approaches
# We want to plot a graph [problem size]x[execution time]


# Compile different methods
nvcc -O3 -DMETHOD=3 collision.cu -o prog_quad_batch_2K; # Sequential
echo "Compiled."
nvcc -O3 -DMETHOD=4 collision.cu -o prog_lin_batch_2K; # Sequential Linear
echo "Compiled."

time {
	for progName in prog_quad_batch_2K prog_lin_batch_2K; do
		echo "id,psize,elapsed,real,user,system" > $progName.out

		psizeBegin=128
		psizeInc=$((256))
		psizeEnd=$((2048+128))

		for problemSize in $(seq $psizeBegin $psizeInc $psizeEnd); do
			for executionId in $(seq 10); do
				# Echo progress onto stderr
				echo $progName $problemSize $executionId 1>&2

				# Run code
				output=$( { time ./$progName $problemSize 60000 | grep Elapsed; } 2>&1 )
				
				# Ouput as csv
				echo -n ${executionId},
				echo -n ${problemSize},
				echo $output | cut -f2,5,7,9 -d' ' | sed -e "s/ /,/g"
			done;
		done >> $progName.out;
	done;
}
