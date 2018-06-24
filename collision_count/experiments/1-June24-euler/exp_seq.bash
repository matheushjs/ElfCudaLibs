

# This experiment is expected to run in less than 220 minutes, using a single GPU.

# Compile different methods
nvcc -O3 -DMETHOD=3 collision.cu -o prog3;

# Execute each method multiple times
for i in $(seq 100); do
	./prog3 >> prog3_output.txt
done


