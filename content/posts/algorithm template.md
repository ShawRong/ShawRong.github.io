---
title: "algorithm template"
date: 2025-06-01T10:42:13.426Z
draft: false
tags: []
---

# Qicksort

leetcode: https://leetcode.cn/problems/sort-an-array/description/

template:
```cpp
class Solution {
public:
    void quick_sort(vector<int>& q, int l, int r) {
        if (l >= r) return;
        int x = q[l], i = l - 1, j = r + 1;
        while(i < j) {
            do i++; while(q[i] < x);
            do j--; while(q[j] > x);
            if (i < j) swap(q[i], q[j]);
        }
        quick_sort(q, l, j);
        quick_sort(q, j + 1, r);
    }

    vector<int> sortArray(vector<int>& nums) {
        int n = nums.size();
        quick_sort(nums, 0, n - 1);
        return nums;
    }
};

```
thought: 
- You chose a number x, and you make every thing that in its left being smaller than it, and in its right being greater than it.
- One way to do the rearrage, using two pointer, move to the first number the not smaller than x(or vice verse), and swap the number of index i and j. And continue to move, until you find next number not satisfies this condition.
- do the some procedure using recurrence to the both side.
- boundary check: when you use q\[l\] as k, you need j and j+1 to be the boundary. when you use q\[r\] as k, you need i and i + 1 to be the boundary. MOST OF THE TIME, you should use j and j+1

# Merge Sort
```c
void merge_sort(vector<int>& q, int l, int r) {
	if (l >= r) return;
	int mid = (l + r) >> 1;
	merge_sort(q, l, mid), merge_sort(q, mid + 1, r);
	int k = 0, i = l, j = mid + 1;
	while(i <= mid && j <= r)
		if (q[i] < q[j]) tmp[k++] = q[i++];
		else tmp[k++] = q[j++];
	while(i <= mid) tmp[k++] = q[i++];
	while(j <= r) tmp[k++] = q[j++];
	for (i = l, j = 0; i <= r; i++, j++) q[i] = tmp[j]; 
}
```


# Binary Search
There are 2 version of binary search.
```c
int binary_search(int l, int r) {
	while(l < r){
		int mid = l + r >> 1;
		if (check(mid)) r = mid;   // check() to identify if mid satisfy the property. If yes, then we need to include mid too, if not, we need do not include mid.
		else l = mid + 1;
	}
	return l;
}

int binary_search(int l, int r) {
	while(l < r) {
		int mid = l + r + 1 >> 1;
		if (check(mid)) l = mid; // we need to include the right hand side this time. and mid is ok, so we need to include it, too.
		else r = mid - 1;
		return l;
	}
}
```

leetcode list:
[1](https://leetcode.cn/problems/search-a-2d-matrix/?envType=study-plan-v2&envId=top-100-liked)
[2](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/)

# Matrix Indexing
```c
int m;  // row num
int n;  // col num

// 2 dimension to 1 dimension
int i,j // row index, and col index i.e. x[i][j]
matrix[i][j] == matrix[i * n + j];

// 1 dimension to 2 dimension
int index; //index of 1 dimension
matrix[index] = matrix[index / n][index % n];
```
# Float binary search
```c
double x;
cin >> x;
double l = 0, r = x;
while(r - l > 1e-8) {
	double mid = (l + r) / 2;
	if (mid * mid >= x) r = mid;   //condition here.
	else l = mid
}
```