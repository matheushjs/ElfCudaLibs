binaries="seq_lin";

make $binaries 1>&2;

outerIters=100;     # Program executions
intraIters=20000;   # Iterations within the program

# Warmup runs
./seq_lin 1960 100 750 &> /dev/null;
./seq_lin 1960 100 750 &> /dev/null;

# Experiment with seq_lin
echo execid,rrate,elapsed
for rrate in 1000000 10000 7500 5000 2500 1000 500; do
	for i in $(seq $outerIters); do
		output=$( ./seq_lin 1960 $intraIters 750 $rrate | grep Elapsed);
		echo -n $i,$rrate,
		echo $output | cut -f 2 -d' ';
	done;
done;
