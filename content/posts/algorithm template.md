---
title: "Algorithm Template"
date: 2025-06-07T13:13:49.304Z
draft: false
tags: []
---

# Qicksort

leetcode: https://leetcode.cn/problems/sort-an-array/description/

template:
```cpp
class Solution {
publick:
    void quick_sort(vector<int>& q, int l, int r) {
        if (l >= r) returnkk'k;
        int x = q[l], i = l - 1, j = r + 1;
        while(i < j) {
            do i++; while(q[i] < x);
            do j--; while(q[j] > x);
            if (i < j) swap(q[i], q[j]);
        }
        quick_sort(q, l, j);  // if we use x = q[r], we need i - 1 here.
        quick_sort(q, j + 1, r); // and i here.
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
[1](https://leetcode.cn/problems/sqrtx/?envType=problem-list-v2&envId=binary-search)


# Prefix sum
```c
// https://www.acwing.com/problem/content/797/
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;
int a[N], s[N];

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]); // here we read from index 1
    
    for (int i = 1; i <= n; i++) s[i] = s[i - 1] + a[i]; // here the key is s[i] = s[i - 1] + a[i]

    while(m--) {
        int l, r;
        scanf("%d%d", &l, &r); 
        printf("%d\n", s[r] - s[l - 1]); // here is the sum of [l, r], i.e. s[r] - s[l - 1]
    }

    return 0;
}

// this is 2 dimension thing
#include <iostream>

int n, m, q;

const int N = 1010;

int s[N][N];


int main() {
    scanf("%d%d%d", &n, &m, &q);
    
    for (int i = 1; i <= n; i++) 
        for (int j = 1; j <= m; j++) 
            scanf("%d", &s[i][j]);
    for (int i = 1; i <= n; i++) 
        for (int j = 1; j <= m; j++) 
            s[i][j] += s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1]; // how to calclulate the prefix sum

    while(q -- ) {
        int x1, y1, x2, y2;
        scanf("%d%d%d%d", &x1, &y1, &x2, &y2);
        printf("%d\n", s[x2][y2] - s[x1 - 1][y2] - s[x2][y1 - 1] + s[x1 - 1][y1 - 1]); // how to get the [l, r]
    }

    return 0;
}

```

# Diff

```c
#include <iostream>

using namespace std;

const int N = 100010;

int n, m;

int a[N], b[N];

void insert(int l, int r, int c) { // this is the insert, key is l and r + 1
    b[l] += c;
    b[r + 1] -= c;
}

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
    for (int i = 1; i <= n; i++) insert(i, i, a[i]);
    while(m -- ) {
        int l, r, c;
        scanf("%d%d%d", &l, &r, &c);
        insert(l, r, c);
    }
    for(int i = 1; i <= n; i++) b[i] += b[i - 1]; // this is prefix sum
    for(int i = 1; i <= n; i++) printf("%d ", b[i]);
    return 0;
}


```

```c

// two dimensional diff
#include <iostream>

using namespace std;

const int N = 1010;

int n, m, q;
int a[N][N], b[N][N];

void insert(int x1, int y1,int x2, int y2, int c) { // this is the insert
    b[x1][y1] += c;
    b[x2 + 1][y1] -= c;
    b[x1][y2 + 1] -= c;
    b[x2 + 1][y2 + 1] += c;
}


int main() {
    scanf("%d%d%d", &n, &m, &q);
    for (int i = 1; i <= n; i++) 
        for(int j = 1; j <= m; j++)
            scanf("%d", &a[i][j]);
    for (int i = 1; i <= n; i++) 
        for(int j = 1; j <= m; j++)
            insert(i, j, i, j, a[i][j]);
    while(q--) {
        int x1, y1, x2, y2, c;
        cin >> x1 >> y1 >> x2 >> y2 >> c;
        insert(x1, y1, x2, y2, c);
    }
    for (int i = 1; i <= n; i++) 
        for(int j = 1; j <= m; j++)
            b[i][j] += b[i - 1][j] + b[i][j - 1] - b[i - 1][j - 1]; // prefix sum

    for (int i = 1; i <= n; i++)  {
        for(int j = 1; j <= m; j++) printf("%d", b[i][j]);
        puts("");
    }
    
    return 0;
}

```

[1](https://leetcode.cn/problems/left-and-right-sum-differences/solutions/2134741/ti-jie-qian-zhui-he-si-xiang-di-tui-onsh-b0wb/?envType=problem-list-v2&envId=prefix-sum)

# MAX and MIN
```c
int ans = INT_MAX;
```


# High Precision Addition
```c
vector<int> add(vector<int> &A, vector<int> &B) {
	if (A.size() < B.size) return add(B, A);
	vector<int> C;
	int t = 0;
	for (int i = 0; i < A.size(); i++) {
		t += A[i];
		if (i < B.size()) t += B[i];
		C.push_back(t % 10);
		t /= 10;
	}
	if (t) C.push_back(t);
	return C;
}


// sub
vector<int> sub(vector<int> &A, vector<int> &B) {
	vector<int> C;
	for (int i = 0, t = 0; i < A.size(); i++) {
		t = A[i] - t;
		if (i < B.size()) t -= B[i];
		C.push_back((t + 10) % 10);
		if (t < 10) t = 1;
		else t = 0;
	}
	while(C.size() > 1 && C.back() == 0) C.push_back();
	return C;
}
```


# sort
```c
sort(nums.begin(), nums.end(), [](int a, int b) {
	return a > b;
});

```
[1](https://leetcode.cn/problems/rearrange-array-to-maximize-prefix-score/submissions/634821398/?envType=problem-list-v2&envId=prefix-sum)