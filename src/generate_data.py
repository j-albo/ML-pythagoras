import numpy as np
import pandas as pd
import os

def generate_dataset(n=1000, seed=42):
    np.random.seed(seed)
    leg_a = np.random.uniform(1, 100, n)
    leg_b = np.random.uniform(1, 100, n)
    hypotenuse = np.sqrt(leg_a**2 + leg_b**2)

    df = pd.DataFrame({'leg_a': leg_a, 'leg_b': leg_b, 'hypotenuse': hypotenuse})
    
    os.makedirs('data', exist_ok=True)  # Asegura que la carpeta exista
    df.to_csv('data/triangles.csv', index=False)
    print("Dataset saved to data/triangles.csv")

if __name__ == "__main__":
    generate_dataset()

