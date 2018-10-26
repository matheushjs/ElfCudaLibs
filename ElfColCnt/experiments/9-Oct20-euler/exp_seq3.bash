#PBS -N ColCntSeq3
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc/7.1.0;

DIR=/home/mathjs/ElfColCnt;
cd $DIR;

binaries="seq_lin";

make $binaries;

for progName in $binaries; do
	echo "psize,execid,elapsed" > $progName.out;

	beg=64;
	inc=64;
	end=$((1024*5));
	outerIters=100; # program executions
	intraIters=1000; # iterations within the program

	echo "Begins at $beg and ends at $end" 1>&2;

	for problemSize in $(seq $beg $inc $end); do
		# Do some warmup runs
		./$progName $problemSize 5 &> /dev/null;

		for i in $(seq $outerIters); do
			# Echo progress onto stderr
			echo $profName $problemSize $executionId 1>&2;

			# Run code
			output=$( ./$progName $problemSize $intraIters | grep Elapsed );

			# Ouput as csv
			echo -n ${problemSize},;
			echo -n ${i},;
			echo $output | cut -f2 -d' ';
		done;
	done >> second_$progName.out;
done;
