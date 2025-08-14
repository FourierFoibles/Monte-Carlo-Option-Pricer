import numpy as np
import math
'''
x=np.linspace(0,10,11)
print(x)
rng = np.random.default_rng()
y=rng.standard_normal((4,3))
print(y)
z=np.cumsum(y,axis=1)
print(z)
test_1=np.zeros((1,20))
print(test_1)


 knockout code
 
for path in S:
    for idx, val in enumerate(path):
        if val <= 0:
            path[idx:] = [0] * (len(path) - idx)
            break


'''
N = 2
T = 5
T_d=3
vol=0.3
drift=0.2
s_0=50
steps = 20
strike=50


dt = T / steps #small time increment

# simulate GBM paths
t = np.linspace(0, T, steps+1) # equally spaced points in time to calcultae data for
dt = T / steps
Z = np.random.randn(N, steps)
increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z
log_S = np.cumsum(increments, axis=1)
log_S = np.hstack((np.zeros((N,1)), log_S))  # add initial 0 for time 0
S = s_0 * np.exp(log_S)  # exponentiate to get actual paths

print(S)

T_d_target=math.floor(T_d/steps)

for path in S:
    for element in path:
        if


