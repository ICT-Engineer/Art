import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import time

def construct_matrix(N, lambd):
    """
    Construit la matrice tridiagonale représentant l'opérateur de régularisation.
    Pour i = 0 et i = N-1 (bords) :
        A[0, 0] = A[N-1, N-1] = 1 + lambd,
        A[0, 1] ou A[N-1, N-2] = -lambd.
    Pour les indices intérieurs :  
        A[i, i-1] = A[i, i+1] = -lambd, et A[i, i] = 1 + 2*lambd.
    """
    A = np.zeros((N, N))
    # Bord supérieur
    A[0, 0] = 1 + lambd
    if N > 1:
        A[0, 1] = -lambd
    # Lignes intérieures
    for i in range(1, N - 1):
        A[i, i - 1] = -lambd
        A[i, i] = 1 + 2 * lambd
        A[i, i + 1] = -lambd
    # Bord inférieur
    if N > 1:
        A[N - 1, N - 2] = -lambd
        A[N - 1, N - 1] = 1 + lambd
    return A

def solve_system_lu(A, b):
    """
    Résout le système Ax = b via décomposition LU (solution directe).
    """
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x

def solve_system_sor(A, b, omega=1.5, tol=1e-6, max_iter=15):
    """
    Résout le système Ax = b via la méthode SOR. Limite les itérations pour une solution approximative.
    """
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma1 = np.dot(A[i, :i], x[:i])
            sigma2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sigma1 - sigma2)
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"SOR converged after {k} iterations")
            break
    else:
        print(f"SOR n'a pas convergé dans le nombre d'itérations (max_iter={max_iter}).")
    return x

if __name__ == '__main__':
    # Paramètres
    N = 100  # Nombre d'échantillons
    t = np.linspace(0, 2 * np.pi, N)
    
    # Signal original : sinusoïdal
    signal_original = np.sin(t)
    
    # Signal bruité : ajout de bruit gaussien
    np.random.seed(0)
    noise_amplitude = 0.3
    signal_bruite = signal_original + noise_amplitude * np.random.randn(N)
    
    # Paramètre de régularisation (lambda)
    lambd = 1.0
    A = construct_matrix(N, lambd)
    
    # Solution par méthode LU
    signal_lu = solve_system_lu(A, signal_bruite)
    
    # Solution par méthode SOR (nombre d'itérations limité pour observer la différence)
    signal_sor = solve_system_sor(A, signal_bruite, omega=1.5, max_iter=15)
    
    # Tracé de tous les signaux
    plt.figure(figsize=(12, 8))
    plt.plot(t, signal_original, 'g-', label="Signal Original", linewidth=2)
    plt.plot(t, signal_bruite, 'k-', label="Signal Bruité", linewidth=2)
    plt.plot(t, signal_lu, 'b-', label="Filtrage LU", linewidth=2)
    plt.plot(t, signal_sor, 'r-', label="Filtrage SOR", linewidth=2)
    plt.xlabel("Temps")
    plt.ylabel("Amplitude")
    plt.title("Comparaison : Signal Original, Bruité, Filtré (LU et SOR)")
    plt.legend()
    plt.grid(True)
    plt.show()
