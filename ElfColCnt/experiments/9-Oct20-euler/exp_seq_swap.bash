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
	beg=1024;
	inc=1024;
	end=$((1024*2));
	outerIters=5; # program executions
	intraIters=1000; # iterations within the program

	for problemSize in $(seq $beg $inc $end); do
		# Do some warmup runs
		./$progName $problemSize 5 &> /dev/null;

		for i in $(seq $outerIters); do
			# Run code
			/usr/bin/time -v ./$progName $problemSize $intraIters > /dev/null;
		done;
	done &> swap_$progName.out;
done;
