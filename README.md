# tugas-mednum

Repositori ini merupakan implementasi rumus gradient descent dan steepest descent yang sudah diturunkan 

Output yang didapatkan untuk gradient descent:
```bash
$ python gradient-descent.py
Gradient Descent Result
Final w: 0.1053846153847146
Final loss: 3.3311538461538457
Final equation: y_hat = 0.1053846153847146 * x

First 10 iterations:
{'iteration': 0, 'w': 0.0, 'loss': np.float64(10.55), 'gradient': np.float64(-137.0)}
{'iteration': 1, 'w': np.float64(0.137), 'loss': np.float64(3.9808500000000007), 'gradient': np.float64(41.100000000000016)}
{'iteration': 2, 'w': np.float64(0.09589999999999999), 'loss': np.float64(3.389626500000001), 'gradient': np.float64(-12.330000000000018)}
{'iteration': 3, 'w': np.float64(0.10823), 'loss': np.float64(3.336416385), 'gradient': np.float64(3.6990000000000043)}
{'iteration': 4, 'w': np.float64(0.104531), 'loss': np.float64(3.3316274746500003), 'gradient': np.float64(-1.1097000000000072)}
{'iteration': 5, 'w': np.float64(0.1056407), 'loss': np.float64(3.3311964727185), 'gradient': np.float64(0.3329100000000018)}
{'iteration': 6, 'w': np.float64(0.10530779), 'loss': np.float64(3.331157682544665), 'gradient': np.float64(-0.09987300000000765)}
{'iteration': 7, 'w': np.float64(0.10540766300000001), 'loss': np.float64(3.3311541914290204), 'gradient': np.float64(0.02996190000001242)}
{'iteration': 8, 'w': np.float64(0.1053777011), 'loss': np.float64(3.3311538772286124), 'gradient': np.float64(-0.00898857000000497)}
{'iteration': 9, 'w': np.float64(0.10538668967), 'loss': np.float64(3.3311538489505748), 'gradient': np.float64(0.002696571000005754)}
```
Output yang didapatkan untuk steepest descent:
```bash
$ python steepest-descent.py
Steepest Descent Result
Final w: 0.10538461538461538
Final loss: 3.331153846153846
Final equation: y_hat = 0.10538461538461538 * x

Iterations:
{'iteration': 0, 'w': 0.0, 'loss': np.float64(10.55), 'gradient': np.float64(-137.0), 'alpha': np.float64(0.0007692307692307692)}
{'iteration': 1, 'w': np.float64(0.10538461538461538), 'loss': np.float64(3.331153846153846), 'gradient': np.float64(-7.105427357601002e-15), 'alpha': 0.0}
```