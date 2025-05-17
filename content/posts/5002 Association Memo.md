# Basic concept
The support of itemset of different size like: 
- itemset-size1 :{B}
- itemset-size2: {A, B}
supp({B}) = count of apparence of B. ...

Association rule: {A, B} -> C
support of associaiton rule: supp({A, B}->C) = supp({A, B, C})
confidence of association rule: conf({A, B}->C) = supp({A, B, C}) / supp({A, B})
