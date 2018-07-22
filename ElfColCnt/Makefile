METHOD?=2

all:
	nvcc -O3 -DMETHOD=$(METHOD) collision.cu -o prog

help:
	@echo "Usage"
	@echo "====="
	@echo
	@echo "    make all [ARG1=val1 [ARG2=val2]]"
	@echo
	@echo "Args:"
	@echo
	@echo "   METHOD   - Method for collisions counting [0-11]"
