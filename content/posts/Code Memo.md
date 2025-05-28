---
title: "Code Memo"
date: 2025-05-28T09:06:15.768Z
draft: false
tags: []
---

# MPI
broadcast k, and n
```c
MPI_Bcast(&k, 1, MPI_INT, 0, comm); // or MPI_COMM_WORLD
MPI_Bcast(&n, 1, MPI_INT, 0, comm); // or MPI_COMM_WORLD

```

set num_of_reads_local
```c
num_of_reads_local = n / num_process;
```

allocate csr displs and csr offs
```c
int readCSR_displs[MAX_PROCESS+1], readCSR_counts[MAX_PROCESS+1];
int readCSRoffs_displs[MAX_PROCESS+1],readCSRoffs_counts[MAX_PROCESS+1]; 

MPI_Bcast(&max_readCSR_size_local, 1, MPI_INT, 0, comm);
MPI_Bcast(readCSR_counts, num_process, MPI_INT, 0, comm);
MPI_Bcast(readCSRoffs_counts, num_process, MPI_INT, 0, comm);
 ```


copy data
```c
num_of_read_local_p0 = n - num_of_reads_local * (num_process - 1);
readCSRoffs_displs[i] = readCSRoffs_displs[i-1] + n / num_process;
readCSR_displs[i] = reads_CSR_offs[readCSRoffs_displs[i]];
readCSR_counts[i-1] = readCSR_displs[i] - readCSR_displs[i-1];
readCSRoffs_counts[i-1] = readCSRoffs_displs[i] - readCSRoffs_displs[i-1] + 1;

reads_CSR_local = new char[max_readCSR_size_local+1];//
reads_CSR_offs_local = new int[num_of_reads_local+2];//
```



generate universal minimizer
```c
um_lists_local.push_back(generate_universal_minimizer_list(k, reads_CSR_offs_local[i+1] - reads_CSR_offs_local[i], reads_CSR_local + reads_CSR_offs_local[i]));

```



process modified offset array
```c
for (int i=0; i<=num_of_reads_local; i++) reads_CSR_offs_local[i] -= first_offset;

```


recv and send
```c
MPI_Recv(um_lists_CSR[i], num_of_ums_proc[i], MPI_UNSIGNED, i, 10, comm, MPI_STATUS_IGNORE);
MPI_Recv(um_lists_CSR_offs[i], readCSRoffs_counts[i], MPI_INT, i, 20, comm, MPI_STATUS_IGNORE);

MPI_Send(um_lists_CSR_local, num_of_ums_local, MPI_UNSIGNED, 0, 10, comm);
MPI_Send(um_lists_CSR_offs_local, num_of_reads_local+1, MPI_INT, 0, 20, comm);

```


convert and insert
```c
std::vector<std::vector<kmer_t>> um_lists_proc;
CSR2Vector(readCSRoffs_counts[i]-1, um_lists_CSR[i], um_lists_CSR_offs[i], um_lists_proc);
um_lists.insert(um_lists.end(), um_lists_proc.begin(), um_lists_proc.end());

```



# Pthread
set local
```c
pthread_args *args_ptr = (pthread_args *)args;
vector<string>& reads = args_ptr->reads;
vector<vector<kmer_t>>& um_lists = args_ptr->um_lists;
int k = args_ptr->k;
int n = args_ptr->n;
int num_threads = args_ptr->num_threads;
read_range ranges = args_ptr->ranges;
int my_rank = (long)args_ptr->my_rank;

```

generate
```c
for (int i = ranges.start; i < ranges.end; i++) {
	um_lists[i] = generate_universal_minimizer_list(k, reads[i]);
}

```

thread handles
```c
pthread_t* thread_handles;
thread_handles = (pthread_t*)malloc(num_threads * sizeof(pthread_t));

```


k-mer total
```c
int tot_kmers = 0;
for (int i = 0; i < n; i++) {
	tot_kmers += reads[i].length() - k + 1;
}
int target = tot_kmers / num_threads;

```



create and call (args & function)
```c
pthread_args* args = new pthread_args(reads, um_lists, k, n, num_threads, ranges[thread], (void*)thread);
pthread_create(&thread_handles[thread], NULL, gen_um_lists, (void*)args);

```


join
```c
pthread_join(thread_handles[thread], NULL);

```


# CUDA
find read id
```c
offset_inside_read = thread_id - reads_offset[mid];
```

cuda malloc
```c
cudaMalloc((void**)&d_reads_array, total_length * sizeof(char));
cudaMemcpy(d_reads_array, reads_array, total_length * sizeof(char), cudaMemcpyHostToDevice);


```

cuda memcpy
```c
int* d_reads_offset_array;
cudaMalloc((void**)&d_reads_offset_array, (n + 1) * sizeof(int));
cudaMemcpy(d_reads_offset_array, reads_offset_array, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

```

call of kernel
```c
gen_um_lists_kernel<<<gridSize, blockSize>>>(k, n, d_reads_array, d_reads_offset_array, 

```


tid and total num calculate
```c
int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
int total_length = reads_offset[num_reads];
```




condition
```c
if (is_universal_minimizer(kmer))
```