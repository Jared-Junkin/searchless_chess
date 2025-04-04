# this code contains several functions for generating estimates of compute-optimal ratios of non-embedding parameters and visualizations for them


# from Kaplan et. al (https://arxiv.org/pdf/2001.08361)
# defines the dependence of performance on N (non embedding parameters) and D (tokens in dataset) for a language model
import numpy as np
import matplotlib.pyplot as plt

def ND(N: int, D: int) -> float:
    N_c = 8.8e13 
    D_c = 5.4e13
    alpha_N = 0.076
    alpha_D = 0.095  # Corrected from 0.0095 to 0.095
    
    return ((N_c / N)**(alpha_N / alpha_D) + (D_c / D))**alpha_D


def calculate_ND_grid(N_values:np.ndarray, D_values: np.ndarray)->np.ndarray:
    # Create an empty matrix to store the ND values
    ND_values = np.zeros((len(D_values), len(N_values)))
    
    # Loop through each pair (N, D) and calculate ND
    for i, D in enumerate(D_values):
        for j, N in enumerate(N_values):
            ND_values[i, j] = ND(N, D)
    
    return ND_values

def plot_heatmap(N_values: np.ndarray, D_values: np.ndarray, ND_values:np.ndarray, filename: str="ScalingLaws.png")->None:
    plt.figure(figsize=(10, 8))
    plt.imshow(ND_values, aspect='auto', origin='lower',
               extent=[N_values.min(), N_values.max(), D_values.min(), D_values.max()])
    plt.colorbar(label='ND Value')
    plt.xlabel('N (Number of Parameters)')
    plt.ylabel('D (Number of Training Tokens)')
    plt.title('Heatmap of ND Scaling Law Equation')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(filename)
    # plt.show()

if __name__ == "__main__":
    # Define the range of N and D values
    N_values = np.logspace(start=7, stop=11, num=1000)  # From 10^7 to 10^11
    D_values = np.logspace(start=7, stop=12, num=1000)  # From 10^7 to 10^12

    # Calculate the ND values for each pair (N, D)
    ND_values = calculate_ND_grid(N_values, D_values)

    # Plot the heatmap
    plot_heatmap(N_values, D_values, ND_values)

    ## define these list comprehensions
    ## make ssecond function for L(N,S), 
    ## plot both as heatmaps
    ## 