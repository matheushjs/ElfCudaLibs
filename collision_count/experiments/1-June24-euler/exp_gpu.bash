
# This experiment is expected to run in less than 50 minutes, using a single GPU.

# Compile different methods
nvcc -O3 -DMETHOD=10 collision.cu -o prog10;
nvcc -O3 -DMETHOD=11 collision.cu -o prog11;
nvcc -O3 -DMETHOD=12 collision.cu -o prog12;

# Execute each method multiple times
for i in $(seq 100); do
	./prog10 >> prog10_output.txt
done

for i in $(seq 100); do
	./prog11 >> prog11_output.txt
done

for i in $(seq 100); do
	./prog12 >> prog12_output.txt
done
