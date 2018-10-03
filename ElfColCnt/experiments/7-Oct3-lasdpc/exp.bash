
binary="seq_quad seq_lin"

make $binary

time {
	for progName in $binary; do
		echo "id,psize,elapsed,real,user,system" > $progName.out

		psizeBegin=128
		psizeInc=$((128))
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
