binaries="seq_quad seq_lin";

make $binaries 1>&2;

outerIters=100;    # Program executions
intraIters=1000;  # Iterations within the program

# Warmup runs
./seq_quad 1960 100 100 &> /dev/null;
./seq_quad 1960 100 100 &> /dev/null;

# Experiment with seq_quad
echo "type,execid,std,elapsed";
for std in 50 100 200 500 750 1000 1250 1500 1750 2000; do
	for i in $(seq $outerIters); do
		output=$( ./seq_quad 1960 $intraIters $std | grep Elapsed);
		echo -n quad,$i,$std,
		echo $output | cut -f 2 -d' ';
	done;
done;

# Warmup runs
./seq_lin 1960 100 100 &> /dev/null;
./seq_lin 1960 100 100 &> /dev/null;

# Experiment with seq_lin
for std in 50 100 200 500 750 1000 1250 1500 1750 2000; do
	for i in $(seq $outerIters); do
		output=$( ./seq_lin 1960 $intraIters $std | grep Elapsed);
		echo -n lin,$i,$std,
		echo $output | cut -f 2 -d' ';
	done;
done;
