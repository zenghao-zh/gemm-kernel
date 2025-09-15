#!/usr/bin/env python3
"""
GEMM Kernel Performance and Accuracy Analysis
=============================================

This script performs comprehensive performance and accuracy analysis for three GEMM kernels:
- matmul_w8a8: 8-bit weights × 8-bit activations
- matmul_w4a4: 4-bit weights × 4-bit activations  
- matmul_w4a8: 4-bit weights × 8-bit activations

Key features:
1. Correct 4-bit data packing according to CUTLASS format
2. Accuracy comparison with PyTorch reference implementations
3. Performance benchmarking with throughput analysis
4. Visualization of scaling behavior
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from typing import Tuple, Dict, List
from gemm_kernel import matmul_w8a8, matmul_w4a4, matmul_w4a8

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_4bit_tensor_correct(shape: Tuple[int, int]) -> torch.Tensor:
    """
    Create a 4-bit tensor with correct CUTLASS packing format.
    
    Args:
        shape: (M, K_packed) where K_packed is the physical packed dimension
               Each K_packed element stores two 4-bit values
        
    Returns:
        torch.Tensor: uint8 tensor with packed 4-bit values
    """
    M, K_packed = shape
    
    # Generate random 4-bit signed values in range [-8, 7]
    low_4bits = torch.randint(-8, 8, (M, K_packed), dtype=torch.int8)
    high_4bits = torch.randint(-8, 8, (M, K_packed), dtype=torch.int8)
    
    # Convert to unsigned [0, 15] for packing
    low_unsigned = (low_4bits + 8).to(torch.uint8)
    high_unsigned = (high_4bits + 8).to(torch.uint8)
    
    # Pack: high nibble | low nibble
    packed = (high_unsigned << 4) | (low_unsigned & 0x0F)
    
    return packed.cuda()

def unpack_4bit_tensor(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack 4-bit tensor for reference computation.
    
    Args:
        packed: uint8 tensor with packed 4-bit values
        
    Returns:
        Tuple of two int8 tensors representing the unpacked values
    """
    # Extract low and high nibbles
    low_unsigned = packed & 0x0F
    high_unsigned = (packed >> 4) & 0x0F
    
    # Convert to signed 4-bit [-8, 7] range correctly
    # 0-7 stays as 0-7, 8-15 becomes -8 to -1
    low_signed = torch.where(low_unsigned < 8, low_unsigned, low_unsigned - 16).to(torch.int8)
    high_signed = torch.where(high_unsigned < 8, high_unsigned, high_unsigned - 16).to(torch.int8)
    
    return low_signed, high_signed

def create_reference_4bit_tensor(packed_tensor: torch.Tensor, logical_K: int) -> torch.Tensor:
    """Create a full int8 tensor for reference computation from packed 4-bit."""
    M, K_packed = packed_tensor.shape
    assert logical_K == K_packed * 2, f"Logical K ({logical_K}) should be 2x packed K ({K_packed})"
    
    low, high = unpack_4bit_tensor(packed_tensor)
    
    # Interleave to get original order
    result = torch.zeros(M, logical_K, dtype=torch.int8, device='cuda')
    result[:, 0::2] = low
    result[:, 1::2] = high
    
    return result

def pytorch_reference_matmul_int8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation for int8 matrix multiplication."""
    # PyTorch doesn't support int32 mm, so use float64 for high precision, then round
    result = torch.mm(A.to(torch.float64), B.T.to(torch.float64))
    return result.round().to(torch.int32)

def pytorch_reference_matmul_4bit(A_packed: torch.Tensor, B_packed: torch.Tensor, 
                                  logical_K: int) -> torch.Tensor:
    """PyTorch reference implementation for 4-bit matrix multiplication."""
    # Convert packed tensors to full int8
    A_full = create_reference_4bit_tensor(A_packed, logical_K)
    B_full = create_reference_4bit_tensor(B_packed, logical_K)
    
    return pytorch_reference_matmul_int8(A_full, B_full)

def benchmark_kernel(func, *args, warmup=3, num_runs=50) -> float:
    """
    Benchmark a GEMM kernel function.
    
    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        try:
            _ = func(*args)
        except:
            return float('inf')
    
    torch.cuda.synchronize()
    
    # Actual timing
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        try:
            _ = func(*args)
        except:
            return float('inf')
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
    return avg_time

def calculate_throughput_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Calculate throughput in GFLOPS."""
    if time_ms == float('inf') or time_ms == 0:
        return 0.0
    
    # Matrix multiplication: 2*M*N*K operations
    flops = 2 * M * N * K
    gflops = flops / (time_ms * 1e6)  # Convert ms to seconds, then to GFLOPS
    return gflops

def calculate_bandwidth_gb_s(M: int, N: int, K: int, time_ms: float, 
                            bits_A: int, bits_B: int) -> float:
    """Calculate memory bandwidth in GB/s."""
    if time_ms == float('inf') or time_ms == 0:
        return 0.0
    
    bytes_A = (M * K * bits_A) // 8
    bytes_B = (N * K * bits_B) // 8
    bytes_C = M * N * 4  # 32-bit output
    
    total_bytes = bytes_A + bytes_B + bytes_C
    gb_per_s = (total_bytes / (1024**3)) / (time_ms / 1000)
    
    return gb_per_s

def accuracy_test(M: int, N: int, K: int) -> Dict[str, float]:
    """
    Test accuracy of all three GEMM kernels against PyTorch reference.
    
    Returns:
        Dictionary with accuracy metrics for each kernel
    """
    results = {}
    
    try:
        # Test matmul_w8a8
        A_8bit = torch.randint(-128, 128, (M, K), dtype=torch.int8, device='cuda')
        B_8bit = torch.randint(-128, 128, (N, K), dtype=torch.int8, device='cuda')
        
        C_kernel = matmul_w8a8(A_8bit, B_8bit)
        C_reference = pytorch_reference_matmul_int8(A_8bit, B_8bit)
        
        # Calculate accuracy metrics
        abs_error = torch.abs(C_kernel - C_reference).float()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        rel_error = (abs_error / (torch.abs(C_reference).float() + 1e-8)).mean().item()
        
        results['w8a8'] = {
            'max_abs_error': max_error,
            'mean_abs_error': mean_error,
            'mean_rel_error': rel_error,
            'matches_reference': max_error == 0
        }
        
    except Exception as e:
        print(f"w8a8 accuracy test failed: {e}")
        results['w8a8'] = {'error': str(e)}
    
    try:
        # Test matmul_w4a4 (ensure K is even)
        K_even = K if K % 2 == 0 else K + 1
        
        A_4bit = create_4bit_tensor_correct((M, K_even // 2))
        B_4bit = create_4bit_tensor_correct((N, K_even // 2))
        
        C_kernel = matmul_w4a4(A_4bit, B_4bit)
        C_reference = pytorch_reference_matmul_4bit(A_4bit, B_4bit, K_even)
        
        abs_error = torch.abs(C_kernel - C_reference).float()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        rel_error = (abs_error / (torch.abs(C_reference).float() + 1e-8)).mean().item()
        
        results['w4a4'] = {
            'max_abs_error': max_error,
            'mean_abs_error': mean_error,
            'mean_rel_error': rel_error,
            'matches_reference': max_error == 0
        }
        
    except Exception as e:
        print(f"w4a4 accuracy test failed: {e}")
        results['w4a4'] = {'error': str(e)}
    
    try:
        # Test matmul_w4a8 (ensure K is even)
        K_even = K if K % 2 == 0 else K + 1
        
        A_8bit_mixed = torch.randint(-128, 128, (M, K_even), dtype=torch.int8, device='cuda')
        B_4bit_mixed = create_4bit_tensor_correct((N, K_even // 2))
        
        C_kernel = matmul_w4a8(A_8bit_mixed, B_4bit_mixed)
        
        # Reference: A (int8) × B (4bit converted to int8)
        B_full = create_reference_4bit_tensor(B_4bit_mixed, K_even)
        C_reference = pytorch_reference_matmul_int8(A_8bit_mixed, B_full)
        
        abs_error = torch.abs(C_kernel - C_reference).float()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        rel_error = (abs_error / (torch.abs(C_reference).float() + 1e-8)).mean().item()
        
        results['w4a8'] = {
            'max_abs_error': max_error,
            'mean_abs_error': mean_error,
            'mean_rel_error': rel_error,
            'matches_reference': max_error == 0
        }
        
    except Exception as e:
        print(f"w4a8 accuracy test failed: {e}")
        results['w4a8'] = {'error': str(e)}
    
    return results

def performance_test(M: int, N: int, K: int) -> Dict[str, Dict]:
    """
    Test performance of all three GEMM kernels.
    
    Returns:
        Dictionary with performance metrics for each kernel
    """
    results = {}
    
    # Test matmul_w8a8
    try:
        A_8bit = torch.randint(-128, 128, (M, K), dtype=torch.int8, device='cuda')
        B_8bit = torch.randint(-128, 128, (N, K), dtype=torch.int8, device='cuda')
        
        time_ms = benchmark_kernel(matmul_w8a8, A_8bit, B_8bit)
        gflops = calculate_throughput_gflops(M, N, K, time_ms)
        bandwidth = calculate_bandwidth_gb_s(M, N, K, time_ms, 8, 8)
        
        results['w8a8'] = {
            'time_ms': time_ms,
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth,
            'memory_mb': (M * K + N * K + M * N * 4) / (1024**2)
        }
        
    except Exception as e:
        print(f"w8a8 performance test failed: {e}")
        results['w8a8'] = {'error': str(e)}
    
    # Test matmul_w4a4 (ensure K is even)
    try:
        K_even = K if K % 2 == 0 else K + 1
        
        A_4bit = create_4bit_tensor_correct((M, K_even // 2))
        B_4bit = create_4bit_tensor_correct((N, K_even // 2))
        
        time_ms = benchmark_kernel(matmul_w4a4, A_4bit, B_4bit)
        gflops = calculate_throughput_gflops(M, N, K_even, time_ms)
        bandwidth = calculate_bandwidth_gb_s(M, N, K_even, time_ms, 4, 4)
        
        results['w4a4'] = {
            'time_ms': time_ms,
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth,
            'memory_mb': (M * K_even * 4 // 8 + N * K_even * 4 // 8 + M * N * 4) / (1024**2)
        }
        
    except Exception as e:
        print(f"w4a4 performance test failed: {e}")
        results['w4a4'] = {'error': str(e)}
    
    # Test matmul_w4a8 (ensure K is even)  
    try:
        K_even = K if K % 2 == 0 else K + 1
        
        A_8bit = torch.randint(-128, 128, (M, K_even), dtype=torch.int8, device='cuda')
        B_4bit = create_4bit_tensor_correct((N, K_even // 2))
        
        time_ms = benchmark_kernel(matmul_w4a8, A_8bit, B_4bit)
        gflops = calculate_throughput_gflops(M, N, K_even, time_ms)
        bandwidth = calculate_bandwidth_gb_s(M, N, K_even, time_ms, 8, 4)
        
        results['w4a8'] = {
            'time_ms': time_ms,
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth,
            'memory_mb': (M * K_even + N * K_even * 4 // 8 + M * N * 4) / (1024**2)
        }
        
    except Exception as e:
        print(f"w4a8 performance test failed: {e}")
        results['w4a8'] = {'error': str(e)}
    
    return results

def run_comprehensive_analysis():
    """Run comprehensive performance and accuracy analysis."""
    
    print("🚀 GEMM Kernel Comprehensive Analysis")
    print("=====================================\n")
    
    # Test matrix sizes
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128), 
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (2560, 2560, 2560),
        (3072, 3072, 3072),
        (4096, 4096, 4096),
    ]
    
    # Store results
    all_performance_results = []
    all_accuracy_results = []
    
    # Header
    print(f"{'Size':<12} {'Kernel':<8} {'Time(ms)':<10} {'GFLOPS':<10} {'BW(GB/s)':<10} {'Max Err':<10} {'Mean Err':<12}")
    print("="*80)
    
    for M, N, K in test_sizes:
        print(f"\n📐 Testing size: {M}×{N}×{K}")
        
        # Performance test
        perf_results = performance_test(M, N, K)
        for kernel, metrics in perf_results.items():
            if 'error' not in metrics:
                all_performance_results.append({
                    'M': M, 'N': N, 'K': K, 
                    'kernel': kernel,
                    **metrics
                })
        
        # Accuracy test (only for smaller sizes to save time)
        if M <= 1024:
            acc_results = accuracy_test(M, N, K)
            for kernel, metrics in acc_results.items():
                if 'error' not in metrics:
                    all_accuracy_results.append({
                        'M': M, 'N': N, 'K': K,
                        'kernel': kernel, 
                        **metrics
                    })
        
        # Print results
        for kernel in ['w8a8', 'w4a4', 'w4a8']:
            perf = perf_results.get(kernel, {})
            if 'error' not in perf:
                # Get accuracy if available
                acc = None
                for a in all_accuracy_results:
                    if (a['M'], a['N'], a['K'], a['kernel']) == (M, N, K, kernel):
                        acc = a
                        break
                
                time_ms = perf.get('time_ms', float('inf'))
                gflops = perf.get('gflops', 0)
                bandwidth = perf.get('bandwidth_gb_s', 0)
                max_err = acc.get('max_abs_error', 'N/A') if acc else 'N/A'
                mean_err = acc.get('mean_abs_error', 'N/A') if acc else 'N/A'
                
                print(f"  {kernel:<8} {time_ms:<10.3f} {gflops:<10.1f} {bandwidth:<10.1f} {max_err:<10} {mean_err:<12}")
            else:
                print(f"  {kernel:<8} FAILED")
    
    # Create visualizations
    create_visualizations(all_performance_results, all_accuracy_results)
    
    # Summary analysis
    print_summary_analysis(all_performance_results, all_accuracy_results)

def create_visualizations(perf_results: List[Dict], acc_results: List[Dict]):
    """Create performance and accuracy visualization plots."""
    
    if not perf_results:
        print("❌ No performance results to plot")
        return
    
    perf_df = pd.DataFrame(perf_results)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GEMM Kernel Analysis: Performance and Scaling', fontsize=16, fontweight='bold')
    
    # 1. GFLOPS vs Matrix Size
    ax1 = axes[0, 0]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = perf_df[perf_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax1.plot(kernel_data['M'], kernel_data['gflops'], 
                    marker='o', linewidth=2, markersize=6, label=kernel)
    
    ax1.set_xlabel('Matrix Size (M)')
    ax1.set_ylabel('Throughput (GFLOPS)')
    ax1.set_title('Throughput vs Matrix Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Memory Bandwidth vs Matrix Size
    ax2 = axes[0, 1]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = perf_df[perf_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax2.plot(kernel_data['M'], kernel_data['bandwidth_gb_s'], 
                    marker='s', linewidth=2, markersize=6, label=kernel)
    
    ax2.set_xlabel('Matrix Size (M)')
    ax2.set_ylabel('Memory Bandwidth (GB/s)')
    ax2.set_title('Memory Bandwidth vs Matrix Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Execution Time vs Matrix Size
    ax3 = axes[0, 2]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = perf_df[perf_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax3.plot(kernel_data['M'], kernel_data['time_ms'], 
                    marker='^', linewidth=2, markersize=6, label=kernel)
    
    ax3.set_xlabel('Matrix Size (M)')
    ax3.set_ylabel('Execution Time (ms)')
    ax3.set_title('Execution Time vs Matrix Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # 4. GFLOPS Efficiency Heatmap
    ax4 = axes[1, 0]
    pivot_gflops = perf_df.pivot(index='M', columns='kernel', values='gflops').fillna(0)
    if not pivot_gflops.empty:
        sns.heatmap(pivot_gflops, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('GFLOPS Heatmap')
        ax4.set_ylabel('Matrix Size (M)')
    
    # 5. Bandwidth Efficiency
    ax5 = axes[1, 1]  
    pivot_bandwidth = perf_df.pivot(index='M', columns='kernel', values='bandwidth_gb_s').fillna(0)
    if not pivot_bandwidth.empty:
        sns.heatmap(pivot_bandwidth, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax5)
        ax5.set_title('Memory Bandwidth Heatmap (GB/s)')
        ax5.set_ylabel('Matrix Size (M)')
    
    # 6. Relative Performance Comparison
    ax6 = axes[1, 2]
    if len(perf_df) > 0:
        # Calculate relative performance (speedup over w8a8)
        relative_perf = []
        for size in perf_df['M'].unique():
            size_data = perf_df[perf_df['M'] == size]
            w8a8_time = size_data[size_data['kernel'] == 'w8a8']['time_ms']
            
            if not w8a8_time.empty:
                baseline_time = w8a8_time.iloc[0]
                
                for kernel in ['w8a8', 'w4a4', 'w4a8']:
                    kernel_time = size_data[size_data['kernel'] == kernel]['time_ms']
                    if not kernel_time.empty:
                        speedup = baseline_time / kernel_time.iloc[0] if kernel_time.iloc[0] > 0 else 0
                        relative_perf.append({
                            'M': size, 
                            'kernel': kernel, 
                            'speedup': speedup
                        })
        
        if relative_perf:
            rel_df = pd.DataFrame(relative_perf)
            for kernel in ['w8a8', 'w4a4', 'w4a8']:
                kernel_rel = rel_df[rel_df['kernel'] == kernel]
                if not kernel_rel.empty:
                    ax6.plot(kernel_rel['M'], kernel_rel['speedup'], 
                            marker='d', linewidth=2, markersize=6, label=kernel)
            
            ax6.set_xlabel('Matrix Size (M)')
            ax6.set_ylabel('Speedup vs w8a8')
            ax6.set_title('Relative Performance (Speedup)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_xscale('log')
            ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('gemm_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Accuracy visualization if available
    if acc_results:
        create_accuracy_plots(acc_results)

def create_accuracy_plots(acc_results: List[Dict]):
    """Create accuracy analysis plots."""
    
    acc_df = pd.DataFrame(acc_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('GEMM Kernel Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Max Absolute Error
    ax1 = axes[0]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = acc_df[acc_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax1.plot(kernel_data['M'], kernel_data['max_abs_error'], 
                    marker='o', linewidth=2, markersize=6, label=kernel)
    
    ax1.set_xlabel('Matrix Size (M)')
    ax1.set_ylabel('Max Absolute Error')
    ax1.set_title('Maximum Absolute Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Mean Absolute Error
    ax2 = axes[1]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = acc_df[acc_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax2.plot(kernel_data['M'], kernel_data['mean_abs_error'], 
                    marker='s', linewidth=2, markersize=6, label=kernel)
    
    ax2.set_xlabel('Matrix Size (M)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Mean Relative Error
    ax3 = axes[2]
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = acc_df[acc_df['kernel'] == kernel]
        if not kernel_data.empty:
            ax3.plot(kernel_data['M'], kernel_data['mean_rel_error'], 
                    marker='^', linewidth=2, markersize=6, label=kernel)
    
    ax3.set_xlabel('Matrix Size (M)')
    ax3.set_ylabel('Mean Relative Error (%)')
    ax3.set_title('Mean Relative Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('gemm_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_analysis(perf_results: List[Dict], acc_results: List[Dict]):
    """Print comprehensive summary analysis."""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    if not perf_results:
        print("❌ No performance results available for analysis")
        return
        
    perf_df = pd.DataFrame(perf_results)
    
    # Performance summary
    print("\n🚀 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    for kernel in ['w8a8', 'w4a4', 'w4a8']:
        kernel_data = perf_df[perf_df['kernel'] == kernel]
        if not kernel_data.empty:
            avg_gflops = kernel_data['gflops'].mean()
            max_gflops = kernel_data['gflops'].max()
            avg_bandwidth = kernel_data['bandwidth_gb_s'].mean()
            
            print(f"{kernel.upper():<6}: Avg GFLOPS={avg_gflops:.1f}, Peak GFLOPS={max_gflops:.1f}, Avg BW={avg_bandwidth:.1f} GB/s")
    
    # Find best performers
    print("\n🏆 BEST PERFORMERS BY SIZE RANGE:")
    print("-" * 40)
    
    size_ranges = [
        ("Small (<512)", perf_df[perf_df['M'] < 512]),
        ("Medium (512-1536)", perf_df[(perf_df['M'] >= 512) & (perf_df['M'] <= 1536)]),
        ("Large (>1536)", perf_df[perf_df['M'] > 1536])
    ]
    
    for range_name, range_data in size_ranges:
        if not range_data.empty:
            best_kernel = range_data.loc[range_data['gflops'].idxmax()]
            print(f"{range_name:<20}: {best_kernel['kernel'].upper()} ({best_kernel['gflops']:.1f} GFLOPS)")
    
    # Accuracy summary
    if acc_results:
        print("\n🎯 ACCURACY SUMMARY:")
        print("-" * 40)
        
        acc_df = pd.DataFrame(acc_results)
        
        for kernel in ['w8a8', 'w4a4', 'w4a8']:
            kernel_data = acc_df[acc_df['kernel'] == kernel]
            if not kernel_data.empty:
                perfect_matches = kernel_data['matches_reference'].sum()
                total_tests = len(kernel_data)
                avg_max_error = kernel_data['max_abs_error'].mean()
                
                print(f"{kernel.upper():<6}: Perfect matches: {perfect_matches}/{total_tests}, Avg max error: {avg_max_error:.2e}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    print("""
📱 Memory-constrained environments:
   → Use w4a4 for minimum memory footprint
   → Use w4a8 for best performance/memory tradeoff

⚡ Latency-critical applications:
   → Small matrices: Check specific benchmarks
   → Large matrices: Typically w8a8 or w4a8

🔥 High-throughput processing:
   → Use the kernel with highest GFLOPS for your matrix size
   → Consider w4a8 as generally well-optimized

🎯 Accuracy-sensitive applications:
   → w8a8 provides highest precision
   → Verify w4a4 and w4a8 meet your accuracy requirements
""")

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)
        
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    run_comprehensive_analysis() 