#ifndef LINEAR_H
#define LINEAR_H

typedef int int3[3];

#define COORD3D(V, AXIS) COORD(V[0], V[1], V[2], AXIS)
#define COORD(X, Y, Z, AXIS) ( (Z+AXIS/2) * ((long int) AXIS*AXIS) + (Y+AXIS/2) * ((long int) AXIS) + (X+AXIS/2))

#define AXISTYPE char

int count_collisions(int3 *vector, int size, AXISTYPE *space3d);

#endif
