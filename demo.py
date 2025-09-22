import torch
from typing import Tuple
from gemm_kernel import matmul_w8a8, matmul_w4a4, matmul_w4a8x4


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


def pack_4bit_tensor(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor to 4-bit format."""
    M, logical_K = unpacked.shape
    assert logical_K % 2 == 0, "logical_K must be even for packing"
    K_packed = logical_K // 2
    
    # Extract low and high elements (reverse of interleaving)
    low_signed = unpacked[:, 0::2]  # even indices
    high_signed = unpacked[:, 1::2]  # odd indices
    
    # Convert signed 4-bit [-8, 7] to unsigned [0, 15]
    # This matches the inverse of the unpack logic:
    # unpack: unsigned < 8 ? unsigned : unsigned - 16
    # pack: signed >= 0 ? signed : signed + 16  
    low_unsigned = torch.where(low_signed >= 0, low_signed, low_signed + 16).to(torch.uint8)
    high_unsigned = torch.where(high_signed >= 0, high_signed, high_signed + 16).to(torch.uint8)
    
    # Pack: high nibble | low nibble  
    packed = (high_unsigned << 4) | (low_unsigned & 0x0F)
    
    return packed


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

if __name__ == "__main__":
    M, K_packed = 4, 32  # Use smaller size for easier verification
    
    print("=== Testing 4-bit tensor packing/unpacking ===")
    
    # Test 1: Create random 4-bit tensor and unpack
    print("\n1. Create and unpack 4-bit tensor:")
    A_4bit_original = create_4bit_tensor((M, K_packed))
    A_8bit = unpack_4bit_tensor(A_4bit_original)
    int4result = matmul_w4a4(A_4bit_original, A_4bit_original)
    int8result = matmul_w8a8(A_8bit, A_8bit)
    print(f"Int4 result:\n{int4result}")
    print(f"Int8 result:\n{int8result}")
    print(f"Packed 4-bit shape: {A_4bit_original.shape}")
    print(f"Unpacked 8-bit shape: {A_8bit.shape}")
    print(f"packed 4-bit tensor:\n{A_4bit_original}")
    print(f"Unpacked 8-bit tensor:\n{A_8bit}")
    
    # Test 2: Pack the unpacked tensor and verify round-trip
    print("\n2. Pack the unpacked tensor (round-trip test):")
    A_4bit_repacked = pack_4bit_tensor(A_8bit)
    print(f"Repacked 4-bit shape: {A_4bit_repacked.shape}")
    print(f"Repacked 4-bit tensor:\n{A_4bit_repacked}")
    
    # Verify they match
    match = torch.equal(A_4bit_original, A_4bit_repacked)
    print(f"Round-trip test passed: {match}")
    
    # Test 3: Create a simple known int8 tensor and pack it
    print("\n3. Pack a simple int8 tensor:")
    simple_8bit = torch.tensor([[-8, 7, -1, 0], 
                               [1, -2, 3, -4], 
                               [5, -6, 7, -8],
                               [0, 1, -1, 2]], dtype=torch.int8, device='cuda')
    simple_4bit = pack_4bit_tensor(simple_8bit)
    simple_8bit_unpacked = unpack_4bit_tensor(simple_4bit)
    
    print(f"Original int8:\n{simple_8bit}")
    print(f"Packed 4-bit:\n{simple_4bit}")
    print(f"Unpacked back to int8:\n{simple_8bit_unpacked}")
    print(f"Round-trip match: {torch.equal(simple_8bit, simple_8bit_unpacked)}")



