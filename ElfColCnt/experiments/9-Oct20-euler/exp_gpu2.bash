#PBS -N ColCntGpu2
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc;
module load cuda-toolkit/9.0.176;

DIR=/home/mathjs/ElfColCnt;
cd $DIR;

binaries="cuda_n";

for bin in $binaries; do
	make $bin;
done;

for progName in $binaries; do
	echo "psize,execid,real,user,system" > second_$progName.out;

	beg=1024;
	inc=65536;
	end=$((beg + 24*inc));
	outerIters=100;
	innerIters=100;

	echo "Begins at $beg and ends at $end" 1>&2;

	for problemSize in $(seq $beg $inc $end); do
		# Do some warmup runs
		./$progName $problemSize 5 2>&1 > /dev/null;

		for i in $(seq $outerIters); do
			# Echo progress onto stderr
			echo $progName $problemSize $i 1>&2;

			# Run code
			output=$( { time ./$progName $problemSize $innerIters &> /dev/null; } 2>&1 );
			
			# Ouput as csv
			echo -n ${problemSize},;
			echo -n ${i},;
			echo $output | cut -f2,4,6 -d' ' | sed -e "s/ /,/g";
		done;
	done >> second_$progName.out;
done;
