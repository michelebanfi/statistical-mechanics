import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000           # Number of particles
m = 1.0            # Mass
gamma = 1          # Friction coefficient
k = 10.0           # Spring constant
kb = 1.0           # Boltzmann Constant
T = 2.0            # Temperature
dt = 0.01          # Time step, with this choice is guarantee that the integral is good
total_time = 100   # Total simulation time

# Initial conditions
x0 = np.ones(N)

# Function to generate Gaussian white noise
def generate_noise(dt, size):
    return np.sqrt(2 * kb * T /dt/m/gamma) * np.random.normal(size=size)

# Initialize arrays, everything to zero
time = np.arange(0, total_time, dt)
x = np.zeros((N, len(time)))

# Set initial conditions
x[:, 0] = x0
xx = x

passageTimes = []

# Langevin equation
for i in range(1, len(time)):

    noise = generate_noise(dt, N)
    xx[:, i] = xx[:, i - 1] - (2 * k * (xx[:, i - 1] ** 2 - 1) * 2) * xx[:, i - 1] / (m * gamma) * dt + noise * dt

    if x.shape[0] > 0:

        noise = generate_noise(dt, x.shape[0])
        x[:, i] = x[:, i - 1] - (2*k*(x[:, i - 1]**2 - 1)*2) * x[:, i - 1]/(m*gamma)*dt + noise*dt

        arePassed = x[:, i] < -0.5
        nParticales = np.sum(arePassed)
        passageTimes.append(i * nParticales)

        x = x[~arePassed]

# mean first passage time
MFPT = np.sum(passageTimes) / N

print(f"Mean First Passage Time: {MFPT * dt}")

def potential(x):
    return k*(x**2 -1)**2

def boltz(x):
    return np.exp(- 1/(kb * T) * potential(x))

x_values = np.linspace(-2., 2., 100)
boltz_values = boltz(x_values)
y_values = potential(x_values)

# Plot the results
plt.figure(figsize=(10, 6))
for j in range(3):
    plt.plot(time, xx[j, :], label=f'Particle {j + 1}')
    plt.ylim((-2, 2))

plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()
plt.close()

final_positions = xx[N-1, :]
num_bins= 30
plt.hist(final_positions, bins=num_bins, density=True, alpha=0.7, color='blue')
plt.plot(x_values, boltz_values, color='red')
plt.xlabel('Final Position')
plt.ylabel('Frequency')
plt.title('Histogram of Final Positions')
plt.show()
plt.close()

mean = np.mean(xx, axis=1)

print("mean: ", mean)

# relaxation time
# tau mean first passage time
# tau of relaxation time


# GOAL = compute the relaxation time and show that is longer than the MFPT.
# from the MFPT we can get the k_(l,r)

# MEAN FIRST RELAXATION TIME