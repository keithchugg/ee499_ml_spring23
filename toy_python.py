# %% [markdown]
# # Toy Python Notebook
# Check to see that you can run this notebook in your environment!

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# Let's define a couple of simple functions that can generate sinusoids

# %%
def simple_cos_wave(t, f):
 return np.cos(2 * np.pi * f * t)

def simple_sin_wave(t, f):
 return np.sin(2 * np.pi * f * t)

# %% [markdown]
# Now, let's use these at a fixed frequency to generate a plot...

# %%
f0 = 10
t = np.arange(0, 3 / f0, 1/(100 * f0))

plt.figure(figsize=(12,8))
plt.plot(t, simple_cos_wave(t, f0), color = 'r', label='cos')
plt.plot(t, simple_sin_wave(t, f0), color = 'b', label='sin')
plt.legend()
plt.grid(linestyle=':')
plt.xlabel("time (sec)")
plt.ylabel("signals")
plt.show()

# %% [markdown]
# 


