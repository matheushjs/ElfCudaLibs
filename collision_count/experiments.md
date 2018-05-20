Experiments
===


Experimental Conditions
---

- 10K iterations

- Vector of size 1000


N Steps
---

- 21423.849609 ms
	- **warp-efficient strides**
	- **use of shared memory**
	- **fitting the vector to a power of 2**
	- **launch & fetch separation**


- 6183.920410 ms
	- warp-efficient strides
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- **usage of 8 cuda streams, with asynchronous memcpy and kernel launch**
	

Half Steps
---

- 4121.961426 ms
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch

Single Steps
---

- 10023.894531 ms
	- use of shared memory
	- assumption that number of threads per block is a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch
	- usage of multiple blocks
	- reduce happening in the CPU

Sequential
---

- 25222.143000 ms
