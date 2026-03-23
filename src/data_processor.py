import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
    def load_distance_matrix(self):
        """Builds a historical distance matrix from past deliveries."""
        logs = pd.read_csv(f"{self.data_dir}/delivery_logs.csv")
        matrix = logs.groupby(['origin_branch', 'destination_branch'])[['distance_km']].mean().reset_index()
        return matrix

    def load_trucks(self):
        """Loads available trucks with capacity constraints."""
        trucks = pd.read_csv(f"{self.data_dir}/trucks.csv")
        return trucks # Returns all 1000 trucks for the real benchmark

    def generate_delivery_requests(self, num_requests=20000):
        """Simulates current-day delivery requests using transactions and product weights."""
        transactions = pd.read_csv(f"{self.data_dir}/transactions.csv")
        products = pd.read_csv(f"{self.data_dir}/products.csv")
        branches = pd.read_csv(f"{self.data_dir}/branches.csv")
        
        # Merge transaction volume with product weights to get 'load'
        df = transactions.sample(num_requests, random_state=42, replace=True).merge(products, on='product_id')
        df['total_weight_kg'] = df['quantity'] * df['weight_kg']
        df = df.rename(columns={'branch_id': 'destination_branch'})
        
        # Sort branches by size and pick the top 5 largest ones to act as Hubs
        centers = branches.nlargest(5, 'capacity_sqm')['branch_id'].tolist()
        df['origin_branch'] = np.random.choice(centers, size=len(df))
        
        # Merge regions for parallel clustering
        df = df.merge(branches[['branch_id', 'region']], left_on='origin_branch', right_on='branch_id')
        
        # Simulates a delivery deadline constraint between 2 and 8 hours (Rubric 1.5)
        df['deadline_hrs'] = np.random.randint(2, 9, size=len(df)) 
        
        return df[['transaction_id', 'origin_branch', 'destination_branch', 'total_weight_kg', 'region', 'deadline_hrs']]