#PBS -N ColCntGpu
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -M matheus.saldanha@usp.br

# load modules
module load gcc;
module load cuda-toolkit/9.0.176;

DIR=/home/mathjs/ElfColCnt;
cd $DIR;

binaries="cuda_n cuda_half";

for bin in $binaries; do
	make $bin;
done;

for progName in $binaries; do
	echo "psize,real,user,system" > $progName.out;

	beg=1024;
	inc=65536;
	end=$((beg + 16*inc));
	iterations=100;

	echo "Begins at $beg and ends at $end" 1>&2;

	for problemSize in $(seq $beg $inc $end); do
		# Do some warmup runs
		./$progName $problemSize 5 2>&1 > /dev/null;

		# Echo progress onto stderr
		echo $progName $problemSize $executionId 1>&2;

		# Run code
		output=$( { /usr/bin/time -p ./$progName $problemSize $iterations > /dev/null; } 2>&1 );
		
		# Ouput as csv
		echo -n ${problemSize},;
		echo $output | cut -f2,4,6 -d' ' | sed -e "s/ /,/g";
	done >> $progName.out;
done;
