binary=seq_quad

make $binary

time {
	echo "id,psize,elapsed,real,user,system" > $binary.out
	
	beg=1024
	inc=32768
	end=$((beg + 32*inc))

	for problemSize in $(seq $beg $inc $end); do
		for executionId in $(seq 10); do
			# Echo progress onto stderr
			echo $binary $problemSize $executionId 1>&2

			# Run code
			output=$( { time ./$binary $problemSize | grep Elapsed; } 2>&1 )
			
			# Ouput as csv
			echo -n ${executionId},
			echo -n ${problemSize},
			echo $output | cut -f2,5,7,9 -d' ' | sed -e "s/ /,/g"
		done;
	done >> $binary.out;
}

