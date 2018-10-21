#PBS -N ColCntSeq
#PBS -l ncpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc;

DIR=/home/mathjs/ElfColCnt;
cd $DIR;

binaries="seq_quad seq_lin";

make $binaries;

for progName in $binaries; do
	echo "psize,elapsed" > $progName.out;

	beg=64;
	inc=64;
	end=$((1024*8));
	iterations=1000;

	echo "Begins at $beg and ends at $end" 1>&2;

	for problemSize in $(seq $beg $inc $end); do
		# Do some warmup runs
		./$progName $problemSize 5 &> /dev/null;

		# Echo progress onto stderr
		echo $profName $problemSize $executionId 1>&2;

		# Run code
		output=$( ./$progName $problemSize $iterations | grep Elapsed );

		# Ouput as csv
		echo -n ${problemSize},;
		echo $output | cut -f2 -d' ';
	done >> $progName.out;
done;
