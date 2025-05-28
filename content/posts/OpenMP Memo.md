---
title: "OpenMP Memo"
date: 2025-05-28T09:06:20.417Z
draft: false
tags: []
---

# feature
- shared-memory MIMD(Multi-Instruction Multi Data).
- Need support from the compiler
- simple to code (small change to serialized program)

# words
- Directive-based shared memory: allows multiple processing units to access a common memory space using compiler **directives**. It uses **pragmas**(compiler hints). (If pragmas are supported, then it will be parallelized; otherwise, it will still be serialized.)

# Header APIs
```c
# pragma omp parallel num_threads(thread_count)
  Hello()
# You DO NOT have to specify how many threads should be here.
# It will utilize all the cores if you didn't specify the number of cores.

# This comes from omp.h header file
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
```

## Critical
To avoid the race condition, when we aggregate values, we need critical directive
```c
# pragma omp critical
  global_result += my_result;
```

## Reduction clause
use reduction clause to the parallel directive.
syntax of reduction clause:
reduction(\<operator\>: \<variable list\>)

```c
	global _ result = 0.0;
#   pragma omp parallel num_threads(thread_count) \
		reduction(+: global_result)
	global_result += local_trap(double a, double b, int n);
```

This is the supplement of the following code:
```c
# wrong one: this will lead to a serialized runtime.
	global_result = 0.0;
#   pragma omp parallel num_threads(thread_count)
	{
#      progma omp critical  
		global_result += local_trap(double a, double b, int n);
	}

# trivial one:

	global_result = 0.0;
#   pragma omp parallel num_threads(thread_count)
	{
		double my_result = 0.0
		my_result += local_trap(double a, double b, int n);
#.  pragma omp critical
		global_result += my_result. 
	}
```

### subtraction
OpenMP treats `reduction(-:var)` as a **special case** to avoid non-associativity.

### floating
float can be different when reduction.

## private clause
make a default variable private(make a copy each thread)
```c
#   pragma omp parallel for num_threads(tread_count) \
        reduction(+:sum) private(factor)    
```

the print of a private variable is not specified.
```c
pragma omp parallel num_threads(thread_count) private(x)
{
	int my_rank = omp_get_thread_num();
	#no specified since x is not initialized
	printf("Thread %d > before initialization. x = %d\n", my_rank, x);
	x = 2 * my_rank + 2;
	# ok
	printf("Thread %d > after initialization. x = %d\n", my_rank, x);
}
# not specified, since x is private, and we print it after parallel block
printf("After parallel block, x = %d\n", x);
```
## For directive

### feature
- must be after a parallel directive
- the thread created by parallel directive will split the for loop equally to execute.
- the variable i is not share, each thread get their own copy. 

```c
    #pragma omp parallel num_threads(4)  // Create 4 threads
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for  // Split loop iterations among the 4 threads
        for (int i = 0; i < 8; i++) {
            printf("Thread %d processes iteration %d\n", thread_id, i);
        }
    }
    return 0;
	>>> 
	Thread 0 processes iteration 0  
	Thread 0 processes iteration 1  
	Thread 1 processes iteration 2  
	Thread 1 processes iteration 3  
	Thread 2 processes iteration 4  
	Thread 2 processes iteration 5  
	Thread 3 processes iteration 6  
	Thread 3 processes iteration 7  


# example 2
double Trap(double a, double b, int n, int thread_count) {
   double  h, approx;
   int  i;
   # here i is declare outside the parallel directive scope
   # but it will be modifies to a private, to make sure the
   # correctness of parallel for.

   h = (b-a)/n; 
   approx = (f(a) + f(b))/2.0; 
#  pragma omp parallel for num_threads(thread_count) \
      reduction(+: approx)
   for (i = 1; i <= n-1; i++)
     approx += f(a + i*h);
   approx = h*approx; 

   return approx;
}  /* Trap */

```

### Little summary about parallel and for

#### parallel
- When placed **before a block**, it creates a **team of threads**.
- **Every thread executes the entire block** (redundantly, unless work is split explicitly).
```c
#pragma omp parallel  // All threads run this block
{
	printf("Hello from thread %d\n", omp_get_thread_num());
}
#or
#  pragma omp parallel num_threads(thread_count) 
   Trap(a, b, n, &global_result);
```    
#### for
- **Must be nested inside a `parallel` block** (or combined with `parallel for`).
- Splits loop iterations across threads (work-sharing).
```c
#pragma omp parallel  // Team of threads
{
	#pragma omp for  // Loop iterations divided
	for (int i = 0; i < 10; i++) {
		printf("%d by thread %d\n", i, omp_get_thread_num());
	}
}

#pragma omp parallel for  // Team of threads + split loop
for (int i = 0; i < 10; i++) {
	printf("%d by thread %d\n", i, omp_get_thread_num());
```

## Alert
- while or do-while will not be parallelized
- only the for loop with number of iterations can be determined
- no exit point in the for loop, like return or break
- only deal with canonical form: for(index=start: index <:<=:>=:> end; index++:++index:index--:...)
- index in for loop can only be modified by increment expression in for statement (these restrictions allow the run-time system to determine the number of iterations prior to execution of loop)
- no dependencies are allowed between element of array. like: fibo\[i\] = fibo\[i-1\] + fibo\[i-2\];
- OpenMp compiler don't check for dependences among iterations in loop.
- A loop in which the results of one or more iterations depend on other iterations cannot, in general, be correctly parallelized by OpenMP without features like Tasking API.
- the dependency like fibo, is called a loop-carried dependence

## Default(none) clause
If you add this cluase, you need to specify all the variable scope in the block.

## Odd-even transposition sort
We can reuse threads we created this way:
```c
#   pragma omp parallel num_threads(thread_count) \
		default(none) shared(a, n) private(i, tmp, phase)
	for (phase = 0; phase < n; phase++) {
		if (phase % 2 == 0)
			pragma omp for
			for (i = 1; i< n; i += 2) {
				if (a[i-1] > a[i]){
					//swamp
				}
			}
		# barrier here
		else
			pragma omp for
			for(i = 1; i < n - 1; i += 2){
				if (a[i] > a[i+1]) {
					//swap
				}
			}
	}
```

## Schedule clause 
schedule(\<type\>, \[,\<chunksize\>\])

If we want work-load balance, when come across work-load distributes according to index, we can use cylic schedule, i.e. schedule.
```c
sum = 0.9;
#  pragma omp parallel for num_threads(thread_count) \
       reduction(+:sum) schedule(static, 1) 
	for(i = 0; i <= n; i++)
		sum += f(i);
```
- static. assigned to threads before loop is executed
- dynamic or guided. assigned while executing. so after thread completes its current set of iterations, it can request more.
- auto. compiler or run-time determine the schedule
- runtime. schedule is determined at run-time based on an environment variable.

### Comparison
- static is good for each iteration takes roughly the same amount of time; and can improve the speed of memory accesses
- in guided schedule, as chunks are completed, the size of the new chunks decreases. good for when later iterations are more compute-intensive.

## runtime 
takes type from env variable (static, dynamic or guided)

## Atomic Directive
```c
#   pragma omp atomic
	x += expression;
```
can be many thing, like + - * /, ++, etc.

## Comparison between critical, lock and atomic
- atomic is equivalent to using unnamed critical, s.t. it can block different part at the same time. 
- using unnamed critical need to watch out the dependencies between different part, it can be ok to parallel sometimes.
- use named critical can be hard to implement, since you need explicitly name all the critical directive, so you can use lock
example code:
```c
#pragma omp critical(phase1)
{/*code A*/}

#pragma omp critical(phase2)
{/*code A*/}


omp_set_lock(&q_p->lock);
Enqueue(q_p, my_rank, mesg);
omp_unset_lock(&q_p->lock);
# omp_init_lock(lock);
# omp_destroy)lock(lock);
```

## Alert
- different type of mutual exclusion for a single critical section can be bad. like you use atomic and use critical, they can lead to misunderstanding cause they will exclude each other.
- no fairness is guaranteed in a exclusion. Some thread can wait very long time to get access
- nest exclusion in exclusion can lead to deadlock
## Critical(name) and lock
# Trapezoidal rule
the approximation of integration of f from a to b:
$h[f(x_0)/2 + f(x_1) + \cdots + f(x_{n-1})+f(x_{x_n})/2]$
```c
h = (b-a)/n;
approx = (f(a) + f(b))/2.0;
for (i = 1; i <= n - 1; i++) {
	x_i = a + i * h;
	approx += f(x_i);
}
approx = h * approx;
```


```c
# omp one
h = (b-a)/n;
local_n = n / thread_count;
local_a = a + my_rank * local_n * h;
local_b = local_a + local_n * h;
approx = (f(local_a) + f(local_b)) / 2.0;
for (i = 1; i <= local_n - 1; i++) {
	x = local_a + i * h;
	approx += f(x);
}
approx = approx * h;
#critical here
global_approx += approx;
```
# Details
- There's an implicit barrier. Threads in a team will wait for the completion of other members in the team before they terminate.
- To check OPENMP is supported or not, we need:
	```
	#ifdef _OPENMP
	#    include <omp.h>
	#endif
	```

## Some formula
Efficiency $E = \frac{S}{t}$
Speedup $S = (\frac{T_{serial}}{T_{parallel}})$

# TODO 
- [ ] defualt cluase 
- [ ] bubble sort 
- [ ] odd-even transposition sort
- [ ] scheduling loops 
- [ ] 1
- [ ] 1