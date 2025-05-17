# Ideas
## $\epsilon$-deficient synopsis
- Condition 1: There is no false negative (true one wil be true)
- The difference between estimated and true is at most $\epsilon$N.(error rate)
- True frequencies less than (s-$\epsilon$)N are classified as infrequent (the most error one are still infrequent)

# Kind
- Sticky Sampling Algorithm
- Lossy Counting Algorithm
- Space-Saving Algorithm
# Sticky Sampling Algorithm
## properties
- using probability
- has confidence parameter $\delta$ (how confident your result is)
- support threshold s and error parameter $\epsilon$
## bucket design
$t = \lceil 1/ \epsilon ln(s^{-1}\sigma^{-1})\rceil$ (decide the size of each bucket)

1st bucket: 1 ~ 2t, size = 2t, r = 1
2nd bucket: 2t+1 ~ 4t, size = 2t, r = 2
3rd bucket: 4t ~ 8t, size = 4t, r = 4 ... (r = 8, size = 8t...)
## algorithm
![[Pasted image 20250516204827.png]]
- 1st bucket, it just absort everything into entries.
- change to 2nd bucket, toss coin to decrease and diminish counter to diminish used memo. (memo change to 2t / 2 = t)
- 2nd bucket, toss coin to add counter (upper bounder of added memo: 2t * 1/2 = t, prev memo = t)
- change to 3rd bucket, memo change to 2t * 1/4 = t / 2
- 3rd bucket, upper bounder of added memo: 4t * 1/4 = t
- Output: $f + \epsilon N \geq sN$ (considering error)  
## feature
- $\epsilon$-deficient synopsis with probability at least $1-\delta$
- at most 2t memo used on average

# Lossy Counting Algorithm
## properties
- using fix-sized bucket
- support threshold and error parameter
bucket size = $\lceil \frac{1}{\epsilon} \rceil$
bucket index: count starting 1. i.e. ($b_\text{current} = \lceil\frac{N}{w}\rceil$)
## algorithm
![[Pasted image 20250516210217.png]]
- 1st bucket, we just insert, with $\Delta = \text{id of bucket} - 1$ 
- changing bucket, remove $f + \Delta \leq \text{id of bucket}$
- 2nd bucket, we insert, new created entry will get new $\Delta$
- changing bucket, we remove...
- output: $f + \epsilon N \geq sN$ (considering error)
## feature
- 100% $\epsilon$-deficient synopsis
- at most $\lceil 1/\epsilon log(\epsilon N) \rceil$ entries (related to N, i.e. data you read)


# Space-saving algorithm
## properties
- just support threshold s and memory parameter M(the greatest number of possible entries stored in the memory)

## algorithm
![[Pasted image 20250516210935.png]]
When memory full, we remove those entries with smallest $p_e = f + \Delta$, and add new insert as $(e, 1, p_e)$

We output: $f + \Delta \geq sN$ (consider error)

## feature
- greatest error: $E \leq 1/M$
- if $E \leq \epsilon$, we make sure $\epsilon$-deficient synopsis
- Memory consumption: M


