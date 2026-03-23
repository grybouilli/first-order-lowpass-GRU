import create_dataset as cd
import matplotlib.pyplot as plt

sample_rate = 44100
sweeps, fsweeps = cd.generate_log_sweep_optimal(sample_rate, 1000, 7500, 3 * 1024, 750)

x, y = cd.generate_pink_noise(1, 48000, 1000)
print("max is ", max(x))
print("min is ", min(x))
N = len(sweeps)
t = cd.np.linspace(0, N / sample_rate, N)
print(t.shape)
print(sweeps.shape)

plt.figure()
plt.plot(t, sweeps)
plt.plot(t, fsweeps)
plt.show()
