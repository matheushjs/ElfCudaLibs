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
	
- Single execution information:
	- Average memcpy time: 2.9520us
	- Average malloc time: 13.027us
	- Average kernel time: 822.37us

Half Steps
---

- 4121.961426 ms
	- use of shared memory
	- fitting the vector to a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch

- Single execution information:
	- Average memcpy time: 2.9780us
	- Average malloc time: 12.804us
	- Average kernel time: 811.25us

Single Steps
---

- 10023.894531 ms
	- use of shared memory
	- assumption that number of threads per block is a power of 2
	- launch & fetch separation
	- usage of 8 cuda streams, with asynchronous memcpy and kernel launch
	- usage of multiple blocks
	- reduce happening in the CPU

- Single execution information:
	- Average memcpy time: 3.2730us
	- Average malloc time: 12.628us
	- Average kernel time: 977.95us

Sequential
---

- 25222.143000 ms

- 4600.959000 ms
	- **With O3**
	- **Volatile trick to prevent loop removal**

- 3816.593000 ms
	- **With O3 only for the collision count function**
	- **No more volatile trick**
	- **Beads positioned along the z-axis, sequentially in the interval [0, size/2], in a circular fashion.**
	- Average collision count time: 0.3816593us

- 1779.082000 ms
	- With O3 only for the collision count function
	- No more volatile trick
	- **Beads positioned randomly along all the space**
	- Average collision count time: 0.1779082us

Sequential Linear
---

- 152.499000 ms 
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects

- 90.898000 ms
	- Using size/2 due to memory allocation problem
	- Randomizing vector to prevent cache effects
	- **With O3**

