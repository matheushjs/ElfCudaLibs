binaries="seq_lin";

make $binaries 1>&2;

outerIters=100;     # Program executions
intraIters=13000;   # Iterations within the program

# Warmup runs
./seq_lin 1960 100 750 &> /dev/null;
./seq_lin 1960 100 750 &> /dev/null;

# Experiment with seq_lin
echo execid,rrate,elapsed
for rrate in 500 1000 2500 5000 7500 10000 1000000; do
	for i in $(seq $outerIters); do
		output=$( /usr/bin/time -f "%e" ./seq_lin 1960 $intraIters 750 $rrate 2>&1 | tail -1 );
		echo -n $i,$rrate,
		echo $output;
	done;
done;
