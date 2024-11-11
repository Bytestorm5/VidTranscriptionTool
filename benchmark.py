import torch
import time
import matplotlib.pyplot as plt

# Set CuDNN benchmark mode for optimized performance
torch.backends.cudnn.benchmark = True

# Benchmarking function for matrix multiplication with TFLOPS calculation
def benchmark_matmul_tflops(sizes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tflops = []
    
    for size in sizes:
        # Create random tensors
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)

        # Warm-up
        for _ in range(10):
            torch.matmul(A, B)
        
        # Measure execution time
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            torch.matmul(A, B)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Average time per operation
        avg_time = (end_time - start_time) / 100
        
        # Calculate FLOPS: 2 * size^3 (for matrix multiplication)
        flops = 2 * (size ** 3)  # Floating-point operations
        tflops_value = (flops / avg_time) / 1e12  # Convert to TFLOPS
        
        tflops.append(tflops_value)
        print(f"Size: {size}x{size}, Avg Time: {avg_time:.6f} seconds, TFLOPS: {tflops_value:.2f}")
    
    return tflops

# Define matrix sizes
sizes = [10000]

# Run benchmark
tflops_results = benchmark_matmul_tflops(sizes)

# Visualize results
plt.plot(sizes, tflops_results, marker='o')
plt.xlabel('Matrix Size')
plt.ylabel('Performance (TFLOPS)')
plt.title('GPU Matrix Multiplication TFLOPS Benchmark')
plt.grid(True)
plt.show()
