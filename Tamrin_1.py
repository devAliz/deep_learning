import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 1000)

y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin, label="sin(x)", color="green", linewidth=2)
plt.plot(x, y_cos, label="cos(x)", color="blue", linewidth=2)
plt.fill_between(x, y_sin, y_cos, color="gray", alpha=0.5)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
