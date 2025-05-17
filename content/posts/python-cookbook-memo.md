+++
date = '2025-05-06T16:58:38+08:00'
draft = false 
title = 'Python Cookbook Memo'
+++
# 3.1 round of numbers

```python
round(num, n_digit)
round(1.23, 1)
>> 1.2
round(1.77, 1)
>> 1.8
```
However, something like 1.5 or 2.5, it will return you the nearest even number. Here, it shoulb be 2.
```python
round(1.5, 0)
>>
round(2.5, 0)
>>2.0
```

n_digit can be negative, too. It will handle the digit before the decimal point.

# 3.2 Accurate Decimal Arithmetic
If you want more accurate arithmetic, do this:
```python
from decimal import Decimal

a = Decimal('4.2')
b = Decimal('2.1')
a + b
...
(a + b) == Decimal('6.3')
>>> True
```
And if you want to control the precision of your computation, you can use context like this:
```python
from decimal import localcontext, Decimal
a = Decimal('1.3')
b = Decimal('1.7')
with localcontext() as ctx:
	ctx.prec = 3
	print(a / b)
>>> 0.765
with localcontext() as ctx:
	ctx.prec = 50
	print(a / b)
>>> 0.764705...
```

