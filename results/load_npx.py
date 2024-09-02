import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('results/sepsis_d=13_a=2_pi=eps-greedy0.1_std=0/approx-neural-gridsearch_epochs=1_m=5_layern=True_buffer=-1_bs=2_lr=0.001_beta=0.1_lambda=0.0001_lambda0=0.1.npz')

# Access the arrays stored in the .npz file
regrets = data['regrets']
errs = data['errs']

# If you want to see all the arrays' names

print(data.files)

# Optionally, convert to a dictionary for easier access (optional)
data_dict = dict(data)
print(data_dict)


print("Regrets shape:", regrets.shape)
print("Errors shape:", errs.shape)

regrets_flat = regrets.flatten()
errs_flat = errs.flatten()

plt.plot(regrets_flat, label='Regrets', color='blue')
plt.title('Flattened Regrets')
plt.xlabel('Index')
plt.ylabel('Regrets')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(errs_flat, label='Errors', color='red')
plt.title('Flattened Errors')
plt.xlabel('Index')
plt.ylabel('Errors')
plt.legend()
plt.grid(True)
plt.show()