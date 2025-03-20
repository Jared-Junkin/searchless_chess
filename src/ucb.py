import numpy as np
import matplotlib.pyplot as plt

def plot_ucb_heatmap(t_max, n_max, c=1, mu_i=0):
    """
    Plot a heatmap of the UCB value as a function of timesteps t and visits to arm i (n_i).

    Args:
    - t_max (int): Maximum number of timesteps.
    - n_max (int): Maximum number of visits to arm i.
    - c (float): Exploration-exploitation trade-off parameter.
    - mu_i (float): Empirical reward of arm i.
    """
    t_values = np.arange(1, t_max + 1)
    n_values = np.arange(1, n_max + 1)
    T, N = np.meshgrid(t_values, n_values)
    
    # Calculate UCB values
    ucb_values = mu_i + c * np.sqrt(np.log(T) / N)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.title("Upper Confidence Bound (UCB) Heatmap (empirical reward = 0)")
    plt.xlabel("Timesteps (t)")
    plt.ylabel("Number of Visits to Arm i (n_i)")
    plt.pcolormesh(T, N, ucb_values, shading='auto', cmap='viridis')
    plt.colorbar(label="UCB Value")
    # plt.show()
    plt.savefig("UCB.png")

# Example usage
plot_ucb_heatmap(t_max=1000, n_max=200, c=1, mu_i=0)
