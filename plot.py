import numpy as np 
import matplotlib.pyplot as plt

x = np.arange(1,2,0.1)
y = np.sin(np.pi * x)

plt.plot(x,y)
plt.show()