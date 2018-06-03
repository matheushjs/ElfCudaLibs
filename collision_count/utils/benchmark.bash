#!/bin/bash

for METHOD in $(seq 0 4); do
	echo Method: $METHOD
	nvcc -O3 -DMETHOD=$METHOD collision.cu -o prog;
	./prog;
	echo;
done
