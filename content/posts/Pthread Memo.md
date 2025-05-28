---
title: "Pthread Memo"
date: 2025-05-28T09:06:21.535Z
draft: false
tags: []
---

# Feature
- shared memory 
- using thread
# Compile
```shell
gcc -g -Wall -o pth_hello pth_hello.c -lpthread
// link the libaray
```

# Introduction of code
```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

//global var, can be accessed by all the threads
int thread_count;

void* Hello(void* rank); //Function started by thread_create. Note this is the prototype, void* func_name(void* ...).

int main(int argc, char* argv[]) {
	long thread;   // use long in case of 64-bit system;
	pthread_t* thread_handles;  // the identifier of each thread, readable by system, not user
	thread_count = strtol(argv[1], NULL, 10); // convert str to long, the last argument is base
	thread_handles = malloc(thread_count*sizeof(pthread_t));
	for (thread = 0; thread < thread_count; thread++) {
		pthread_create(&thread_handles(thread), NULL, Hello, (void*) thread); // 1st arg: p of thread handle, 2nd arg: not used, 3rd arg: func to start, 4th arg: arguments of func
	}
	printf("Hello from the main thread\n");
	for (thread = 0; thread < thread_count; thread++){
		pthread_join(thread_handles[thread], NULL);  // finalize all the thread
	}
	free(thread_handles);
	return 0;
}

void* Hello(void* rank) {
	long my_rank = (long) rank;
	printf("Hello from thread %ld of %d\n", my_rank, thread_count); // thread count is global variable here
	return NULL;
}

```


# API
## pthread_create
```c
int pthread_create(
	pthread_t* thread_p, //out, the thread handle, it will modify
	const pthread_attr_t* attr_p, //in
	void* (*start_routine)(void*) //in, the function to start
	void* args_p, //in the arguments
)
```
## pthread_join
```c
int pthread_join(
	pthread_t thread, //in
	void** ret_val_p //out, returned value from thread
)
```
If not joining the thread, will lead to zombie thread, waste resources(stacks and local variables) and may prevent the creation of new threads. If program does not need to wait for a particular thread to finish, it can be detached with the pthread_detach function.

We can manage thread using thread pool, too. to minimize overhead of creating and eliminating


# Matrix-vector multiplication

```c
for (i = 0; i < m; i++) {
	y[i] = 0.0;
	for (j = 0; j < n; j++) {
		y[i] += A[i][j] * x[j];
		//or
		y[i] += A[i * n + j] * x[j];
	}
}
```

```c
void *Pth_mat_vect(void* rank) {
	long my_rank = (long) rank;
	int i, j;
	int local_m = m / thread_count;
	int my_first_row = my_rank * local_m;
	int my_last_row = (my_rank + 1) * local_m - 1;

	for(i = my_first_row; i <= my_last_row; i++) {
		y[i] = 0.0;
		for (j = 0; j < n; j++)
			y[i] += A[i][j] * x[j];
	}
	return NULL;
}
```

# Critical section
It's a block of code that updates a shared resource that can only be updated by one thread at a time
## mutexes
```c
int pthread_mutex_init(
	pthread_mutex_t* mutex_p, /*out*/
	const pthread_mutexattr_t* attr_p, /*out*/ // not used actually
)
pthread_mutex_lock(&mutex);
//critical section
pthread_mutex_unlock(&mutex);
```


# Producer-consumer synchronization and semaphores
only locks can not complete the task like "send a message to another thread".
It's important to guarantee the order in which threads will execute the code in critical section.


This type of synchronization, when a thread can't proceed until another thread has taken some action, is sometimes called **producer-consumer synchronization**.

## semaphores
syntax of various semaphore:
```c
int sem_init(
	sem_t* semaphore_p,    // out
	int shared,            //in   to control whether the semaphore is shared among threads or processes.
	unsigned initial_val   //in
);

int sem_destroy(sem_t* semaphore_p /*in/out*/)
int sem_post(sem_t* semaphore_p /*in/out*/) //at semaphore
int sem_wait(sem_t* semaphore_p) //wait if semaphore is 0
```

`
```c
// example of message passing
void* Send_msg(void* rank) {
	long my_rank = (long) rank;
	long dest = (my_rank + 1) % thread_count;
	char* my_msg = malloc(MSG_MAX * sizeof(char));

	sprintf(my_msg, "Hello to %ld from %ld", dest, my_rank);
	messages[dest] = my_msg;
	sem_post(&semaphores[dest]);
	sem_wait(&semaphores[my_rank]);
	printf("Thread %ld > %s\n", my_rank, message[my_rank]);
	return NULL;
}
```

## Barrier
3 way to implement:
### busy-waiting with mutex
```c
int counter;
int thread_count;
pthread_mutex_t barrier_mutex;

void* Thread_work() {
	pthread_mutex_lock(&barrier_mutex);
	counter++:
	pthread_mutex_unlock(&barrier_mutex);
	while(counter < thread_count);  //barrier here
}

```
**problem**:
- busy-waiting will waste CPU cycles when threads are in the busy-wait loop, and if we run the program with more threads than cores, we may find that the performance of the program seriously degrades.
- We can not put another barrier by reusing this counter. If we reset the counter in the last thread to enter the loop, some thread may never see the fact that counter == thread_count. If some thread tries to reset the counter after the barrier, some other thread may enter the second barrier before the counter is reset and its increment to the counter will be lost.


### Semaphores

```c
int counter;
sem_t count_sem;
sem_t barrier_sem;

void* Thread_work(...) {
	/*Barrier*/
	sem_wait(&count_sem); //request counter lock
	if (counter == thread_count - 1) {
		counter = 0;
		sem_post(&count_sem);  // lock the counter sem
		//to let all the other thread pass
		for (j = 0; j < thread_count - 1; j++) {
			sem_post(&barrier_sem);
		}
	} else { // accumulate the counter
		counter++;
		sem_post(&count_sem);
		sem_wait(&barrier_sem);
	}
}
```
**problem**:Considering reusing. Suppose we get 2 thread. When the thread 0 comes first and get the sem_wait(&barrier_sem), the other get to the loop of sem_post(&barrier_sem).
We suppose thread 0 stuck there, and thread 1 get to the next barrier. Here the barrier_sem will remain to be 1, since the thread 0 did not decrease it.
And thread 1 will get through the 2nd barrier, because of the barrier_sem remain to be one.
So, thread 0 will stuck there.


## Conditional variables
conditional variable is a data object that allows a thread to suspend execution until a certain even or condition occurs.
```c
// typical condition
lock mutex;
if condition has occurred
	signal thread(s);
else {
	unlock the mutex and block;
	/*when thread is unblocked, mutex is relocked*/
}
unlock mutex;
```

the main point is func: pthread_cond_wait
```c
int pthread_cond_wait(
	pthread_cond_t* cond_var_p, //in and out
	pthread_mutex_t* mutex_p,   //in and out
)
// it's essentially this:
pthread_mutex_unlock(&mutex_p);
wait_on_signal(&cond_var_p);
pthread_mutex_lock(&mutex_p);
```
pthread_cond_wait will unlock the mutex referred to by mutex_p and cause the executing thread to block(the lock is unlocked, and trigged wait_on_signal. Other thread can get access to this lock). Until it is unblocked by another thread's call to pthread_cond_signal or pthread_cond_broadcast(They will terminate the wait_on_signal, lead to mutex_lock(other thread can not access to this lock)). 
```c
//barrier implemented by conditional variables 
int counter = 0;
pthread_mutex_t mutex;
pthread_cond_t cond_var;
//...
void* Thread_work(...) {
	/*barrier*/
	pthread_mutex_lock(&mutex); //usual lock
	counter++;
	if (counter == thread_count) {
		counter = 0;
		pthread_cond_broadcast(&cond_var);
	} else {
		while(pthread_cond_wait(&cond_var, &mutex) != 0); // it will unlock the mutex, and wait for signal, after signal, it will get their previous lock and continue.
	}
	pthread_mutex_unlock(&mutex);
}


int pthread_cond_init(
	pthread_cond_t* cond_p, //out
	const pthread_condattr_t* cond_attr_p, //in NULL for us
)
int pthread_cond_destroy(
	pthread_cond_t* cond_p, //in & out
)
```
### Note
the pthread_cond_wait should usually be placed in a while loop. (it can be unblock because of event other than pthread_cont_broadcast...)



## Read-write locks
In the sorted linked list, we get 1 read(Member) and 2 write(Delete and Insert).
So we need to strict no write can happen when read.

syntax for the read-write locks
```c
int pthread_rwlock_rdlock(pthread_rwlock_t* rwlock_p) //in and out
int pthread_rwlock_wrlock(pthread_rwlock_t* rwlock_p) //in and out
int pthread_rwlock_unlock(pthread_rwlock_t* rwlock_p) //in and out
int pthread_rwlock_init(pthread_rwlock_t* rwlock_p, const pthread_rwlockattr_t* attr_p) 
int pthread_rwlock_destroy(pthread_rwlock_t* rwlock_p);
```
If there is a write locked, we can not get read or write lock.
If there is a read locked, we can get read lock, but we can not get write lock.


# Cache
Considering the matrix multiplication, we know there are 3 different setup: 8,000,000 x 8, 8,000 x 8,000, 8 x 8000, 000 (element size for each dimension)

y is 8,000,000 in the first case, and it can cause more write miss. (write miss happens when a core tries to update a variable that's not in the cache)

x has 8,000,000 element in the second setup, so there will be more read miss.
and we know there are a false sharing. If one is updated in the cache line, the whole cache line will be invalid.
so for y is 8, we know it will lead to false sharing.
considering y\[0\],y\[1\]...y\[7\]. Suppose they are in the same cache line, if y\[i\] is updated by thread i, and thread j need to update y\[j\]. thread j has to reload this since cache line is invalid.
But if we get larger y, most elements of y belong to the same thread, so it will not be invalid for the most time.


# Safety
## TODO
- [x] cache
- [ ] safety