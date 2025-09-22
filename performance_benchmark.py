#!/usr/bin/env python3
"""
GEMM Kernel Performance Benchmark
=================================

Simple performance benchmarking script for three GEMM kernels:
- matmul_w8a8: 8-bit weights × 8-bit activations
- matmul_w4a4: 4-bit weights × 4-bit activations  
- matmul_w4a8x4: Mixed precision with different 4-bit:8-bit ratios
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from typing import Tuple, Dict, List
from gemm_kernel import matmul_w8a8, matmul_w4a4, matmul_w4a8x4

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def create_4bit_tensor(shape: Tuple[int, int]) -> torch.Tensor:
    """Create a packed 4-bit tensor with CUTLASS format."""
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

def unpack_4bit_tensor(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 4-bit tensor to int8 for reference computation."""
    M, K_packed = packed.shape
    logical_K = K_packed * 2
    
    # Extract low and high nibbles
    low_unsigned = packed & 0x0F
    high_unsigned = (packed >> 4) & 0x0F
    
    # Convert to signed 4-bit [-8, 7] range
    low_signed = torch.where(low_unsigned < 8, low_unsigned, low_unsigned - 16).to(torch.int8)
    high_signed = torch.where(high_unsigned < 8, high_unsigned, high_unsigned - 16).to(torch.int8)
    
    # Interleave to get original order
    result = torch.zeros(M, logical_K, dtype=torch.int8, device='cuda')
    result[:, 0::2] = low_signed
    result[:, 1::2] = high_signed
    
    return result

def benchmark_kernel(func, *args, warmup=5, num_runs=100) -> float:
    """Benchmark a GEMM kernel and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    torch.cuda.synchronize()
    
    # Timing
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = func(*args)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return (end_time - start_time) / num_runs * 1000  # ms

def calculate_gflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Calculate throughput in GFLOPS."""
    if time_ms <= 0:
        return 0.0
    flops = 2 * M * N * K
    return flops / (time_ms * 1e6)

def calculate_bandwidth(M: int, N: int, K: int, time_ms: float, bits_A: int, bits_B: int) -> float:
    """Calculate memory bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    
    bytes_A = (M * K * bits_A) // 8       # Matrix A: M×K
    bytes_B = (K * N * bits_B) // 8       # Matrix B: K×N (corrected from N×K)
    bytes_C = M * N * 4                   # Output C: M×N int32
    
    total_bytes = bytes_A + bytes_B + bytes_C
    return (total_bytes / (1024**3)) / (time_ms / 1000)

def calculate_w4a8x4_bandwidth(M: int, N: int, K1_even: int, K2_even: int, time_ms: float) -> float:
    """Calculate memory bandwidth for w4a8x4 mixed precision kernel in GB/s."""
    if time_ms <= 0:
        return 0.0
    
    # Actual memory accesses for w4a8x4:
    # A_K2_8bit: M × K2_even × 8 bits
    # B_K2_8bit: K2_even × N × 8 bits (unpacked from 4bit)  
    # A_K1_4bit: M × (K1_even/2) × 8 bits (4bit packed, accessed as bytes)
    # B_K1_4bit: (K1_even/2) × N × 8 bits (4bit packed, accessed as bytes)
    # Output C: M × N × 32 bits
    
    bytes_A_K2 = M * K2_even * 1        # 8bits = 1 byte
    bytes_B_K2 = K2_even * N * 1         # 8bits = 1 byte
    bytes_A_K1 = M * (K1_even // 2) * 1 # 4bit packed = 0.5 byte per element, accessed as bytes
    bytes_B_K1 = (K1_even // 2) * N * 1 # 4bit packed = 0.5 byte per element, accessed as bytes  
    bytes_C = M * N * 4                 # int32 output
    
    total_bytes = bytes_A_K2 + bytes_B_K2 + bytes_A_K1 + bytes_B_K1 + bytes_C
    return (total_bytes / (1024**3)) / (time_ms / 1000)

def test_single_kernel(M: int, N: int, K: int, kernel_name: str) -> Dict:
    """Test a single kernel and return metrics."""
    
    if kernel_name == 'w8a8':
        try:
            A_8bit = torch.randint(-128, 128, (M, K), dtype=torch.int8, device='cuda')
            B_8bit = torch.randint(-128, 128, (N, K), dtype=torch.int8, device='cuda')
            
            time_ms = benchmark_kernel(matmul_w8a8, A_8bit, B_8bit)
            gflops = calculate_gflops(M, N, K, time_ms)
            bandwidth = calculate_bandwidth(M, N, K, time_ms, 8, 8)
            
            return {
                'time_ms': time_ms,
                'gflops': gflops,
                'bandwidth_gb_s': bandwidth
            }
        except Exception as e:
            return {'error': str(e)}
    
    elif kernel_name == 'w4a4':
        try:
            K_even = K if K % 2 == 0 else K + 1
            
            A_4bit = create_4bit_tensor((M, K_even // 2))
            B_4bit = create_4bit_tensor((N, K_even // 2))
            
            time_ms = benchmark_kernel(matmul_w4a4, A_4bit, B_4bit)
            gflops = calculate_gflops(M, N, K_even, time_ms)
            bandwidth = calculate_bandwidth(M, N, K_even, time_ms, 4, 4)
            
            return {
                'time_ms': time_ms,
                'gflops': gflops,
                'bandwidth_gb_s': bandwidth
            }
        except Exception as e:
            return {'error': str(e)}

def test_w4a8x4_kernels(M: int, N: int, K: int) -> Dict[str, Dict]:
    """Test w4a8x4 kernel with different ratios."""
    results = {}
    ratios = [1,3,7,15]  # 4bit:8bit ratios
    
    for ratio in ratios:
        try:
            K_even = K if K % 2 == 0 else K + 1
            
            # Split K dimension based on ratio
            K1_even = int(K_even * ratio / (ratio + 1))
            if K1_even % 2 != 0:
                K1_even += 1
            K2_even = K_even - K1_even
            
            if K1_even < 2 or K2_even < 2:
                continue
                
            # Create tensors
            A_K1_4bit = create_4bit_tensor((M, K1_even // 2))
            B_K1_4bit = create_4bit_tensor((N, K1_even // 2))
            A_K2_8bit = torch.randint(-128, 128, (M, K2_even), dtype=torch.int8, device='cuda')
            B_K2_4bit = create_4bit_tensor((N, K2_even // 2))
            B_K2_8bit = unpack_4bit_tensor(B_K2_4bit)
            
            time_ms = benchmark_kernel(matmul_w4a8x4, A_K2_8bit, B_K2_8bit, A_K1_4bit, B_K1_4bit)
            gflops = calculate_gflops(M, N, K_even, time_ms)
            
            # Calculate bandwidth based on actual memory accesses
            bandwidth = calculate_w4a8x4_bandwidth(M, N, K1_even, K2_even, time_ms)
            
            results[f'w4a8x4_{ratio}to1'] = {
                'time_ms': time_ms,
                'gflops': gflops,
                'bandwidth_gb_s': bandwidth,
                'ratio': f'{ratio}:1'
            }
        except Exception as e:
            results[f'w4a8x4_{ratio}to1'] = {'error': str(e)}
    
    return results

def run_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 GEMM Kernel Performance Benchmark")
    print("====================================\n")
    
    # Test matrix sizes - all aligned to 32 elements (meets both int8 and int4 requirements)
    test_sizes = [
        # (256, 256, 256),    # 256 = 32 * 8
        (512, 512, 512),    # 512 = 32 * 16  
        (1024, 1024, 1024), # 1024 = 32 * 32
        (1536, 1536, 1536), # 1536 = 32 * 48
        (2048, 2048, 2048), # 2048 = 32 * 64
        (2560, 2560, 2560), # 2560 = 32 * 80 (替代3072)
        (4096, 4096, 4096), # 4096 = 32 * 128
        (6144, 6144, 6144), # 6144 = 32 * 192 (新增更大尺寸)
        (8192, 8192, 8192), # 8192 = 32 * 256
        (10240, 10240, 10240), # 10240 = 32 * 320
        (12288, 12288, 12288), # 12288 = 32 * 384
        (16384, 16384, 16384), # 16384 = 32 * 512
        (24576, 24576, 24576), # 24576 = 32 * 768
        (32768, 32768, 32768), # 32768 = 32 * 1024
    ]
    
    all_results = []
    
    # Print header
    print(f"{'Size':<15} {'Kernel':<12} {'Ratio':<8} {'Time(ms)':<10} {'GFLOPS':<10} {'BW(GB/s)':<10}")
    print("="*75)
    
    for M, N, K in test_sizes:
        print(f"\n📐 Testing {M}×{N}×{K}")
        
        # Test w8a8
        result = test_single_kernel(M, N, K, 'w8a8')
        if 'error' not in result:
            all_results.append({'M': M, 'N': N, 'K': K, 'kernel': 'w8a8', **result})
            print(f"  {'w8a8':<12} {'N/A':<8} {result['time_ms']:<10.3f} {result['gflops']:<10.1f} {result['bandwidth_gb_s']:<10.1f}")
        else:
            print(f"  {'w8a8':<12} {'N/A':<8} FAILED: {result['error']}")
        
        # Test w4a4
        result = test_single_kernel(M, N, K, 'w4a4')
        if 'error' not in result:
            all_results.append({'M': M, 'N': N, 'K': K, 'kernel': 'w4a4', **result})
            print(f"  {'w4a4':<12} {'N/A':<8} {result['time_ms']:<10.3f} {result['gflops']:<10.1f} {result['bandwidth_gb_s']:<10.1f}")
        else:
            print(f"  {'w4a4':<12} {'N/A':<8} FAILED: {result['error']}")
        
        # Test w4a8x4 with different ratios
        w4a8x4_results = test_w4a8x4_kernels(M, N, K)
        for kernel, metrics in w4a8x4_results.items():
            if 'error' not in metrics:
                all_results.append({'M': M, 'N': N, 'K': K, 'kernel': kernel, **metrics})
                ratio = metrics['ratio']
                print(f"  {'w4a8x4':<12} {ratio:<8} {metrics['time_ms']:<10.3f} {metrics['gflops']:<10.1f} {metrics['bandwidth_gb_s']:<10.1f}")
            else:
                print(f"  {'w4a8x4':<12} {'N/A':<8} FAILED: {metrics['error']}")
    
    # Create visualization and summary
    if all_results:
        create_plots(all_results)
        print_summary(all_results)
    else:
        print("❌ No successful results to analyze")

def create_plots(results: List[Dict]):
    """Create performance visualization plots with optimized colors and styles."""
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('GEMM Kernel Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # Define kernel styles with distinct colors, linestyles, and markers
    def get_kernel_style(kernel_name):
        if kernel_name == 'w8a8':
            return {
                'color': '#2E86C1',      # 深蓝色 - 8位基准
                'linestyle': '-',        # 实线
                'marker': 'o',           # 圆形
                'markersize': 7,
                'linewidth': 2.5,
                'label': 'W8A8 (8-bit)'
            }
        elif kernel_name == 'w4a4':
            return {
                'color': '#28B463',      # 深绿色 - 4位基准  
                'linestyle': '--',       # 虚线
                'marker': 's',           # 方形
                'markersize': 6,
                'linewidth': 2.5,
                'label': 'W4A4 (4-bit)'
            }
        elif 'w4a8x4' in kernel_name:
            # w4a8x4 混合精度系列使用暖色调
            ratio_colors = {
                'w4a8x4_1to1': {'color': '#E74C3C', 'label': 'W4A8×4 (1:1)'},    # 红色
                'w4a8x4_3to1': {'color': '#F39C12', 'label': 'W4A8×4 (1:3)'},    # 橙色
                'w4a8x4_7to1': {'color': '#9B59B6', 'label': 'W4A8×4 (1:7)'},    # 紫色
                'w4a8x4_15to1': {'color': '#E67E22', 'label': 'W4A8×4 (1:15)'},  # 橙色
            }
            base_style = {
                'linestyle': ':',        # 点线
                'marker': '^',           # 三角形
                'markersize': 7,
                'linewidth': 2.5,
            }
            if kernel_name in ratio_colors:
                base_style.update(ratio_colors[kernel_name])
            else:
                # 备用样式
                base_style.update({'color': '#D35400', 'label': kernel_name})
            return base_style
        else:
            # 默认样式
            return {
                'color': '#85929E',
                'linestyle': '-',
                'marker': 'D',
                'markersize': 6,
                'linewidth': 2,
                'label': kernel_name
            }
    
    # Get unique kernels
    unique_kernels = df['kernel'].unique()
    
    # Create plots with enhanced styling
    plot_configs = [
        {
            'ax': axes[0],
            'y_col': 'gflops',
            'title': 'Computational Throughput',
            'ylabel': 'Performance (GFLOPS)',
            'marker_scale': 1.0
        },
        {
            'ax': axes[1], 
            'y_col': 'bandwidth_gb_s',
            'title': 'Memory Bandwidth Utilization',
            'ylabel': 'Bandwidth (GB/s)',
            'marker_scale': 1.0
        },
        {
            'ax': axes[2],
            'y_col': 'time_ms',
            'title': 'Execution Latency',
            'ylabel': 'Time (ms)',
            'marker_scale': 1.0,
            'log_y': True
        }
    ]
    
    for config in plot_configs:
        ax = config['ax']
        
        for kernel in unique_kernels:
            kernel_data = df[df['kernel'] == kernel]
            if not kernel_data.empty:
                style = get_kernel_style(kernel)
                
                ax.plot(kernel_data['M'], kernel_data[config['y_col']], 
                       color=style['color'],
                       linestyle=style['linestyle'], 
                       marker=style['marker'],
                       markersize=style['markersize'] * config['marker_scale'],
                       linewidth=style['linewidth'],
                       label=style['label'],
                       alpha=0.85,
                       markeredgewidth=0.5,
                       markeredgecolor='white')
        
        # Styling
        ax.set_xlabel('Matrix Size (M)', fontsize=12, fontweight='semibold')
        ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='semibold') 
        ax.set_title(config['title'], fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xscale('log')
        if config.get('log_y'):
            ax.set_yscale('log')
            
        # Beautify ticks
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    plt.savefig('gemm_performance_benchmark.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def print_summary(results: List[Dict]):
    """Print performance summary."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("📊 PERFORMANCE SUMMARY")
    print("="*70)
    
    # Base kernels summary
    print("\n🔧 Base Kernels:")
    for kernel in ['w8a8', 'w4a4']:
        kernel_data = df[df['kernel'] == kernel]
        if not kernel_data.empty:
            avg_gflops = kernel_data['gflops'].mean()
            max_gflops = kernel_data['gflops'].max()
            print(f"  {kernel.upper():<8}: Avg {avg_gflops:6.1f} GFLOPS, Peak {max_gflops:6.1f} GFLOPS")
    
    # Mixed precision kernels summary
    print("\n🔀 Mixed Precision Kernels:")
    w4a8x4_kernels = sorted([k for k in df['kernel'].unique() if 'w4a8x4' in k])
    for kernel in w4a8x4_kernels:
        kernel_data = df[df['kernel'] == kernel]
        if not kernel_data.empty:
            avg_gflops = kernel_data['gflops'].mean()
            max_gflops = kernel_data['gflops'].max()
            ratio = kernel_data.iloc[0].get('ratio', 'N/A')
            print(f"  Ratio {ratio:<6}: Avg {avg_gflops:6.1f} GFLOPS, Peak {max_gflops:6.1f} GFLOPS")
    
    # Best performers
    print(f"\n🏆 Best Overall Performance:")
    best_overall = df.loc[df['gflops'].idxmax()]
    if 'ratio' in best_overall and pd.notna(best_overall['ratio']):
        print(f"  w4a8x4 (ratio {best_overall['ratio']}) with {best_overall['gflops']:.1f} GFLOPS")
    else:
        print(f"  {best_overall['kernel'].upper()} with {best_overall['gflops']:.1f} GFLOPS")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    run_benchmark()