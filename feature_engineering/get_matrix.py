"""
Matrix Powers Computation for HOGRL Model
==========================================

This file provides utilities for computing matrix powers of adjacency
matrices, which is required by the HOGRL (High-Order Graph Representation
Learning) model.

The HOGRL model uses k-hop neighbor information captured through matrix
powers: A^k gives the k-hop connectivity matrix.

Key Functions:
    - create_adjacency_matrix(): Convert adjacency list to dense matrix
    - block_matrix_multiply(): Memory-efficient blocked matrix multiplication
    - matrix_powers_gpu(): Compute A^k for k=1 to 10 with GPU acceleration

Output:
    Saves pickle files: {prefix}1.pkl, {prefix}2.pkl, ... {prefix}10.pkl
    Each file contains the k-hop connectivity matrix (excluding lower hops)

Note: This is primarily for the HOGRL model, not used by GTAN/RGTAN/STAGN.
"""

import pickle
import os
import torch
import numpy as np


def create_adjacency_matrix(adj_list, n):
    """
    Convert adjacency list to dense adjacency matrix
    
    Args:
        adj_list (dict): Adjacency list {node: set(neighbors)}
        n (int): Number of nodes
        
    Returns:
        ndarray: Dense adjacency matrix [n, n] with 1s for edges
    """
    adj_matrix = np.zeros((n, n))
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    return adj_matrix


def block_matrix_multiply(A, B, block_size, device):
    """
    Blocked matrix multiplication for memory efficiency
    
    For large matrices, computing A @ B directly may exceed GPU memory.
    This function divides matrices into blocks and multiplies block by block.
    
    Algorithm: C[i,j] = Σ_k A[i,k] @ B[k,j]
    
    Args:
        A (Tensor): First matrix [n, n]
        B (Tensor): Second matrix [n, n]
        block_size (int): Size of each block
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        Tensor: Result matrix C = A @ B [n, n]
    """
    n = A.shape[0]
    C = torch.zeros((n, n), device=device)
    
    # Triple nested loop over blocks
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                k_end = min(k + block_size, n)
                
                # Accumulate block products
                C[i:i_end, j:j_end] += torch.matmul(
                    A[i:i_end, k:k_end], 
                    B[k:k_end, j:j_end]
                )
    return C


def matrix_powers_gpu(adj_list, n, block_size, matrix_prefix):
    """
    Compute matrix powers A^1 through A^10 with GPU acceleration
    
    For each k, computes: B^k = A^k - A^(k-1) + I
    This gives exactly the k-hop neighbors (not reachable in fewer hops).
    
    The computation is done in blocks to handle large matrices that
    don't fit in GPU memory.
    
    Args:
        adj_list (dict): Adjacency list representation of graph
        n (int): Number of nodes (must be divisible by block_size)
        block_size (int): Block size for blocked multiplication
        matrix_prefix (str): Prefix for output pickle files
        
    Output Files:
        {matrix_prefix}1.pkl: Adjacency matrix A^1
        {matrix_prefix}2.pkl: 2-hop matrix (A^2 - A^1 + I)
        ...
        {matrix_prefix}10.pkl: 10-hop matrix
    """
    assert n % block_size == 0, "n must be divisible by block_size"
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create adjacency matrix and move to device
    adj_matrix_np = create_adjacency_matrix(adj_list, n)
    adj_matrix = torch.from_numpy(adj_matrix_np).float().to(device)
    
    # Save A^1 (the adjacency matrix itself)
    file_name = f'{matrix_prefix}1.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(adj_matrix_np, f)
    
    # Compute A^k for k = 2 to 10
    for k in range(2, 11):
        result_blocks = []

        # Process block by block
        for i in range(0, n, block_size):
            row_blocks = []

            for j in range(0, n, block_size):
                block_shape = (min(i + block_size, n) - i, min(j + block_size, n) - j)
                
                # Initialize A^k block as identity
                A_k_block = torch.eye(*block_shape, device=device)
                # Initialize A^(k-1) block
                A_k_minus_1_block = torch.eye(*block_shape, device=device) if k > 1 else torch.zeros(*block_shape, device=device)

                adj_block = adj_matrix[i:i + block_size, j:j + block_size]

                # Compute A^k through repeated multiplication
                for _ in range(k):
                    A_k_block = block_matrix_multiply(A_k_block, adj_block, block_size, device)
                # Binarize: any non-zero means connected
                A_k_block[A_k_block != 0] = 1

                # Compute A^(k-1) similarly
                if k > 1:
                    for _ in range(k - 1):
                        A_k_minus_1_block = block_matrix_multiply(A_k_minus_1_block, adj_block, block_size, device)
                    A_k_minus_1_block[A_k_minus_1_block != 0] = 1

                # Extract exactly k-hop neighbors: A^k - A^(k-1)
                result_block = A_k_block - A_k_minus_1_block
                # Clip negative values
                result_block = torch.maximum(result_block, torch.tensor(0))

                # Add self-loop for diagonal blocks
                if i == j:
                    result_block += torch.eye(*block_shape, device=device)

                row_blocks.append(result_block.cpu().numpy())

            result_blocks.append(np.concatenate(row_blocks, axis=1))

        # Combine all blocks into full matrix
        full_result = np.concatenate(result_blocks, axis=0)

        # Save the k-hop matrix
        file_name = f'{matrix_prefix}{k}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(full_result, file)

        # Clear GPU memory for next iteration
        torch.cuda.empty_cache()


if __name__ == '__main__':
    """
    Main script to generate matrix powers for HOGRL model
    
    Reads adjacency lists from pickle files and generates
    k-hop connectivity matrices for each relation type.
    
    Block sizes:
        - Amazon: 1493 (smaller graph)
        - YelpChi/S-FFSD: 7659 (larger graphs)
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "methods/hogrl"))
    from hogrl_utils import filelist, file_matrix_prefix
    
    DATADIR = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "data/")
    
    for filename, matrix_prefix in zip(filelist.values(), file_matrix_prefix.values()):
        print('generating matrix for: ', filename)
        print('matrix prefix: ', matrix_prefix)
        
        filepath = os.path.join(DATADIR, filename)
        matrix_prefix = os.path.join(DATADIR, matrix_prefix)

        with open(filepath, 'rb') as file:
            relation1 = pickle.load(file)
        
        # Use smaller block size for Amazon (smaller dataset)
        block_size = 1493 if filename.startswith('amz') else 7659
        matrix_powers_gpu(relation1, len(relation1), block_size, matrix_prefix)