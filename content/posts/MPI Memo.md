---
title: "MPI Memo"
date: 2025-05-28T09:06:19.269Z
draft: false
tags: []
---

# Compile & Run
using a **wrapper** for the C compiler.
```shell
mpicc -g -Wall -o mpi_hello mpi_hello.c
```
mpicc is a wrapper script, that telling c compiler where to find header file and what libraries should be linked.

using mpiexec to run a mpi program
```c
mpiexec -n 4 ./mpi_hello
```

# MPI
## feature
- Distributed Memory


## API
These apis are included by \<mpi.h\>
### MPI_Init
```c
int MPI_Init(int* argc_p, char*** argv_p);
```
the parameter is pointer to argc and argv. If main(void), just pass NULL.

it allocate storage for message buffers, and decide which process gets which rank

### MPI_Finalize
```c
int MPI_Finalize(void);
```
resources allocated for MPI can be freed.

```c
# basic framework for MPI program
#include <mpi.h>
int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	#...
	MPI_Finalize();
	return 0;
}
```

## Communicators and MPI_Comm_size & MPI_Comm_rank
**communicator** is a collection of processes that can send messages to each other. This communicator is called **MPI_COMM_WORLD**.

These apis get information about MPI_COMM_WORLD
```c
# get size 
int MPI_Comm_size(
	MPI_Comm comm,
	int * comm_sz_p; /*a int pointer*/
)
# get rank
int MPI_Comm_rank(
	MPI_Comm comm,
	int * comm_rank_p; /*a int pointer*/
)
```
MPI_Comm typically refer to MPI_COMM_WORLD

## MPI_Send & MPI_Recv
```c
int MPI_Send(
	void* msg_buf_p /*in*/,
	void* msg_size  /*in*/,
	MPI_Datatype msg_type /*in*/,
	int   dest/*in*/,
	int   tag/*in*/,  /*this is used to specify the usage of different send message, for example tag0 for print and tag1 for computation*/
	MPI_Comm communicator/*in*/;
)

int MPI_Recv(
	void* msg_buf_p /*out*/,
	void* buf_size  /*in*/,
	MPI_Datatype buf_type /*in*/,
	int   source/*in*/,
	int   tag/*in*/,  /*this is used to specify the usage of different send message, for example tag0 for print and tag1 for computation*/
	MPI_Comm communicator/*in*/,
	MPI_Status* status_p;  // usually MPI_STATUS_IGNORE
)
```

About first 3 parameter: If recv_type = send_type & recv_buf_sz >= send_buf_sz, the message sent by q can be successfully received by r.

there is a special parameter MPI_ANY_SOURCE, to receive message from any source.
there is a special parameter MPI_ANY_TAG, too. (for receive)
only receiver can use a wildcard argument.

## MPI_Status
The receiver can receive message without knowing: 1. the amount of data in message 2. the sender of the message and 3. the tag of the message. So where to find these message? use MPI_Status
```c
MPI_Status status;
status.MPI_SOURCE;
status.MPI_TAG;
// you need func to get the amount of data that's been received
MPI_Get_count(&status, recv_type, &count); 
int MPI_Get_count(
	MPI_Status* status_p; //in
	MPI_Datatype* type;   //in
	int* count_p;         //out
)
```

## detail of MPI_Send & MPI_Recv
if the size of message is less than the cutoff, it will be buffered(not blocked), if the size of message is greater than the cutoff, MPI_Send will block.
MPI_Recv will block utill matching messages has been received.
The message sent by the same process will follow the order to be received. but it's not guaranteed between different process.

## pitfall
MPI_Recv will block when no correspond message get sent.
MPI_Send also(If not block, but buffered, the message will lost)


## Input & Output
Output: all process can get access to stdout, but usually, we utilize rank 0 to get all the information and print
Input: Only process 0 in MPI_COMM_WORLD get access to stdin.
```c
// a typcial get input
void Get_input(
	int my_rank,
	int comm_sz,
	double* a_p, //out
	double* b_p, //out
	double* n_p, //out
) {
	int dest;
	if(my_rank == 0) {
		printf("Enter a., b. and n\n");
		scanf("%lf %lf %d", a_p, b_p, n_p);
		for (dest = 1; dest < comm_sz; dest++) {
			MPI_Send(a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			MPI_Send(b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
			MPI_Send(n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(n_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}
```


## MPI_Reduce
global-sum function (tree-like) (not only sum). 
**collective communications**: communication functions that involve all the processes in a communicator. MPI_Send and MPI_Recv are called point-to-point communications.

operator list:
- MPI_MAX
- MPI_MIN
- MPI_SUM
- MPI_PROD (product)
- MPI_LAND  (logical and)
- MPI_BAND  (bitwise and)
- MPI_LOR  (logical or)
- MPI_LXOR  (logical exclusive or)
- MPI_BXOR  (bitwise exclusive or)
- MPI_MAXLOC  (maximum and location of maximum)
- MPI_MIXLOC  (minimum and location of minimum)
```c
int MPI_Reduce(
	void* input_data_p,    //in
	void* output_data_p,   //out
	int count,             //in
	MPI_Datatype datatype, //in
	MPI_Op  operator,      //in
	int dest_process       //in
	MPI_Comm  comm         //in
)

MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WROLD);
//or
double local_x[N], sum[N];
MPI_Reduce(local_x, sum, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

```


## Collective vs. point-to-point communications
- all processes in the communicator must call the same collective function. MPI_Reduce will not corresponde to MPI_Recv
- all collective communication must be compatible(like dest) 
- Collective don't use tags. It match solely on the basic of the communicator.
- using the same buffer for both input and output is illegal for MPI_Reduce: MPI_Reduce(&x, &x, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

## MPI_Allreduce
It's not hard to imagine a situation in which all of the processes need the reuslt of a global sum. We can use allreduce. (strucutre like a butterfly. It's butterfly-structured global sum)
```c
int MPI_Allreduce(
	void* input_data_p   /*in*/,
	void* output_data_p  /*out*/,
	void* count          /*in*/,
	MPI_Datatype datatype /*in*/,
	MPI_Op operator      /*in*/,
	MPI_Comm comm        /*in*/,
)
// there is no dest_process compared with reduce.
```


## MPI_Broadcast
It's commonly used to distribute input data. It's a reverse tree-structure of MPI_Reduce

```c
int MPI_Bcast(
	void* data_p,         // in out
	int   count,          // in
	MPI_Datatype datatype //in
	int   source_proc     //in
	MPI_Comm comm         //in
)
//send data_p from source_proce to all processes


//example code
void Get_input(
	int my_rank,
	int comm_sz,
	double* a_p, //out
	double* b_p, //out
	double* n_p, //out
) {
	if(my_rank == 0) {
		printf("Enter a., b. and n\n");
		scanf("%lf %lf %d", a_p, b_p, n_p);
	}
	MPI_Bcast(a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
```

## MPI_Scatter
To distribute partial input data(like partial vector instead of whole vector when doing vector adding).
```c
int MPI_Scatter(
	void* send_buf_p, //in
	int   send_count, //in   // it's the amount of data going to each process, not the amount of data in memory refered by send_buf_p.
	MPI_Datatype send_type, //in
	void* recv_buf_p, //out
	int   recv_count, //in
	MPI_Datatype recv_type, //in
	int   src_proc,   //in
	MPI_Comm comm     //in
)


//exmaple code
if (my_rank == 0) {
	a = malloc(n * sizeof(double));
	printf("Enter the vector %s\n", vec_name);
	for (i = 0; i < n; i++)
		scanf("%lf", &a[i]);
	MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
	free(a)
} else {
	MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, comm);
}
//the out will be local_a
// you can notice that local_n is a given parameter
```

## MPI_Gather
If we ditribute the input data, and do some transformation, then we need to gather the data to a whole vector.
```c
int MPI_Gather(
	void* send_buf_p, //in 
	int send_count, //in 
	MPI_Datatype send_type, //in
	void* recv_buf_p, //out
	int recv_count, //in
	MPI_Datatype recv_type, //in
	int dest_proc, //in
	MPI_Comm
)
//example:
double* b = NULL; //recv buff
int i;
if (my_rank == 0) {
	b = malloc(n * sizeof(double));
	MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
} else {
	MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOULBE, 0, comm);
}
```

## MPI_Allgather
This give every process to get a distributed array to a whole one.
```c
int MPI_Allgather(
	void* send_buf_p, //in 
	int send_count, //in 
	MPI_Datatype send_type, //in
	void* recv_buf_p, //out
	int recv_count, //in
	MPI_Datatype recv_type, //in
	MPI_Comm
)
// no dest here.
// it concatednates each send_buf_p and store in each process's recv_buf_p. recv_count is the amount of data being received from each process. So in most cases, recv_count will be the same as send_count >
```

```c
// code of Mat mult
void Mat_vect_mult(
	doulbe local_A[], /*in*/
	doulbe local_x[], /*in*/
	doulbe local_y[], /*out*/
	int local_m[],    /*in*/
	int n,    /*in*/
	int local_n,    /*in*/
	MPI_Comm comm,    /*in*/
) {
	double* x;
	int local_i, j;
	int local_ok = l;

	//gather x, it's previous y.
	x = malloc(n * sizeof(double)):
	MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);
	
	for (local_i = 0; local_i < local_m; local_i++) {
		local_y[local_i] = 0.0;
		for (j = 0; j < n; j++) {
			local_y[local_i] += local_A[local_i * n + j] * x[j];
		}
	}
	free(x);
}
```


# MPI-derived datatypes
```c
double x[1000];
if (my_rank == 0) {
	for (i == 0; i < 1000; i++) {
		MPI_Send(&x[i], 1, MPI_DOUBLE, 1, 0, comm);
	}
} else {
	for (i == 0; i < 1000; i++) {
		MPI_Recv(&x[i], 1, MPI_DOUBLE, 0, 0, comm, &status);
	}
}


// faster one
if (my_rank == 0) {
	MPI_Send(x, 1000, MPI_DOUBLE, 1, 0, comm);
} else {
	MPI_Recv(x, 1000, MPI_DOUBLE, 0, 0, comm, &status);
}
```

So to consolidate the data, we can use a derived datatypes: MPI_PACK/Unpack. (the count argument is used to group continuous array elements into single message).
**Derived datatype** can be used to represent any collection of data items in memory.

```c
// example code
int MPI_Type_create_struct(
	int count; //in
	int array_of_blocklengths[] //in,   how much each type of element is. for example {1, 1, 1} represents, 1 of first, 1 of second, 1 of third.
	MPI_Aint array_of_displacements[] //in, the displacemnt
	MPI_Datatype array_of_types[]  //in  type array
	MPI_Datatype* new_type_p //out  the output new type
)


//usage
// we want a type consisting of 2 double, 1 int
int array_of_blocklengths[3] = {1, 1, 1};
MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
MPI_Aint a_addr, b_addr, n_addr;
MPI_Get_address(a_p, &a_addr);  //use this func to calculate the array_of_displacement
MPI_Get_address(b_p, &b_addr);  
MPI_Get_address(n_p, &n_addr);  
array_of_displacements[1] = b_addr-a_addr;
array_of_displacements[2] = n_addr-a_addr;
MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, input_mpi_t_p); //MPI_Datatype* input_mpi_t_p is the output type
MPI_Type_commit(input_mpi_t_p);
```

## MPI_Wtime
code:
```c
start = MPI_Wtime();
finish = MPI_Wtime();
printf('time: %e', finish - start);
```

## MPI_Barrier
```c
int MPI_Barrier(MPI_Comm comm);

// a timer using barrier
double local_start, local_finish, local_elapsed, elapsed;
//...
MPI_Barrier(comm):
local_start = MPI_Wtime();
//code to be timed
loca_finished = MPI_Wtime();
local_elapsed = local_finish - local_start;
MPI_Reduce(&local_elapsed, &elapsed, 1, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
if (my_rank == 0)
	prinf("Elapsed time = %e", elapsed);
```

## Timming
$T_{parallel}(n, p) = T_{serial}(n)/p + T_{overhead}$
n for size of task, p for num of processor.

Speed up:
$S(n, p) = \frac{T_{serial}(n)}{T_{parallel}(n,p)}$
The best result for speedup is n

Efficiency:
$E(n, p) = \frac{S(n, p)}{p} = \frac{T_{serial}(n)}{p \times T_{parallel}(n, p)}$
best performance, linear speedup is efficiency equal to 1.

# Scalablity
A parallel program is said to be strongly scalable if its efficiency can be kept constant with increase in number of processors. it's weakly scalable if its efficiency can be kept constant with both increase in number of processors and problem size at the same rate.
# somethign
**Wallclock time** (also called **real time** or **elapsed time**) refers to the actual time taken by a process or task as measured by a clock on the wall (or a stopwatch).
# TODO
- [ ] sort 
- [ ] safety need to do!