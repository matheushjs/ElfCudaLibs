#ifndef _RNORM_H_
#define _RNORM_H_

/*
 * Code taken from https://rosettacode.org/wiki/Statistics/Normal_distribution#C
 * With little modifications.
 */


/*
 * RosettaCode example: Statistics/Normal distribution in C
 *
 * The random number generator rand() of the standard C library is obsolete
 * and should not be used in more demanding applications. There are plenty
 * libraries with advanced features (eg. GSL) with functions to calculate 
 * the mean, the standard deviation, generating random numbers etc. 
 * However, these features are not the core of the standard C library.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/*
 * Normal random numbers generator - Marsaglia algorithm.
 */
double *rnorm(int n){
    int i;
    int m = n + n % 2;
    double* values = (double*) malloc(m * sizeof(double));
    static int counter = 1;

    srand(time(NULL) + counter);
    counter++;

    if(values){
        for(i = 0; i < m; i += 2){
            double x, y, rsq, f;
            do {
                x = 2.0 * rand() / (double)RAND_MAX - 1.0;
                y = 2.0 * rand() / (double)RAND_MAX - 1.0;
                rsq = x * x + y * y;
            } while(rsq >= 1. || rsq == 0.);
            f = sqrt(-2.0 * log(rsq) / rsq);
            values[i]   = x * f;
            values[i+1] = y * f;
        }
    }
    return values;
}

#endif
