#PBS -N ColCntGpuBig
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc
module load cuda-toolkit/9.0.176

DIR=/home/mathjs/ElfColCnt
cd $DIR

# Here we will experiment with the multiple-blocks approaches
# We want to plot a graph [problem size]x[execution time]

binaries="ns_sr hs_sr ns_mr"

for bin in $binaries; do
	make $bin;
done;

time {
	for progName in $binaries; do
		echo "id,psize,elapsed,real,user,system" > $progName.out

		psizeBegin=1024
		psizeInc=32768
		psizeEnd=983040

		for problemSize in $(seq $psizeBegin $psizeInc $psizeEnd); do
			for executionId in $(seq 10); do
				# Echo progress onto stderr
				echo $progName $problemSize $executionId 1>&2

				# Run code
				output=$( { time ./$progName $problemSize | grep Elapsed; } 2>&1 )
				
				# Ouput as csv
				echo -n ${executionId},
				echo -n ${problemSize},
				echo $output | cut -f2,5,7,9 -d' ' | sed -e "s/ /,/g"
			done;
		done >> $progName.out;
	done;
}
