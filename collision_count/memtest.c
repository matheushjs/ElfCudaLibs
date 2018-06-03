#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>

int main(int argc, char *argv[]){
	struct rlimit lim;

	getrlimit(RLIMIT_AS, &lim);
	printf("AS %lu %lu\n", lim.rlim_cur, lim.rlim_max);

	getrlimit(RLIMIT_DATA, &lim);
	printf("DATA %lu %lu\n", lim.rlim_cur, lim.rlim_max);
	

	return 0;
}
