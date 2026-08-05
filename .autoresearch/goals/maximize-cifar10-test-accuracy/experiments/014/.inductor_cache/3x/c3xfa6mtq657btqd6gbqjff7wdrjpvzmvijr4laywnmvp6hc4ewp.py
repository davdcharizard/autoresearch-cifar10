# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/n4/cn4tfxgqx6eul32cxhpqah7eajzsfc7n7obn3zn5etq6djazvfke.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "bf16[512, 10][10, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %mul_72 : Tensor "bf16[512, 10][10, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, 0.125), kwargs = {})
#   return %mul_72
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/3g/c3gav5azb3q4476syqv2nggt3z33lbbljhmun6wyzhrzxbohtred.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_72 : Tensor "bf16[512, 10][10, 1]cuda:0" = PlaceHolder[target=mul_72]
#   %constant_pad_nd_default : Tensor "bf16[512, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%mul_72, [0, 6, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_mm_1 = async_compile.triton('triton_poi_fused_mm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 10, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 10*x1), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ir/cirxgojxqyyaz3vlrluzrxgf6rypdsyteloleufnyaa5eb57itt6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %permute_3 : Tensor "bf16[10, 512][512, 1]cuda:0" = PlaceHolder[target=permute_3]
#   %constant_pad_nd_default_1 : Tensor "bf16[16, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_3, [0, 0, 0, 6]), kwargs = {})
#   return %constant_pad_nd_default_1
triton_poi_fused_mm_2 = async_compile.triton('triton_poi_fused_mm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 10, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/5v/c5vf2b3eog5c2ckyvikudf4yrhaev2wzmojgymhsbh5qo5nxpoqr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_1 : Tensor "bf16[10, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %convert_element_type_40 : Tensor "f32[10, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
#   return %convert_element_type_40
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/hx/chxeafqo66posniizszwow25t66jlfp3gpjym7tihjcyxgf6zc5k.py
# Topologically Sorted Source Nodes: [max_pool2d_3, input_34, input_35], Original ATen: [aten.view, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_34 => add_51, convert_element_type_32, mul_64, mul_70, sub_9, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_35 => relu_9
#   max_pool2d_3 => _low_memory_max_pool_offsets_to_indices_3
# Graph fragment:
#   %getitem_27 : Tensor "i8[512, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_27]
#   %mm_default : Tensor "bf16[512, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %convolution_10 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %getitem_25 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_25]
#   %rsqrt_9 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %primals_62 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_62]
#   %primals_63 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_63]
#   %max_pool2d_with_indices_backward : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward]
#   %view_1 : Tensor "bf16[512, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_default, [512, 512, 1, 1]), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices_3 : Tensor "i64[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_27, [4, 4], [4, 4], [4, 4], [0, 0], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%view_1, %add_52, [4, 4], [4, 4], [0, 0], [1, 1], False, %_low_memory_max_pool_offsets_to_indices_3), kwargs = {})
#   %sub_9 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %getitem_25), kwargs = {})
#   %mul_64 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_9), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_62, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_70 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_63, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_51 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %unsqueeze_39), kwargs = {})
#   %convert_element_type_32 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_51, torch.bfloat16), kwargs = {})
#   %relu_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_32,), kwargs = {})
#   %le : Tensor "b8[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %max_pool2d_with_indices_backward), kwargs = {})
#   %convert_element_type_41 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   return %max_pool2d_with_indices_backward,%convert_element_type_41
triton_poi_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_view_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x2 = xindex // 8192
    x1 = ((xindex // 512) % 16)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 16), "index out of bounds: 0 <= tmp4 < 16")
    tmp7 = tmp4
    tmp8 = x1
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 - tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tmp24 <= tmp10
    tmp26 = tl.where(tmp25, tmp10, tmp11)
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp11, None)
    tl.store(out_ptr1 + (x4), tmp27, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/sm/csm2hmxt7tsyxozkcltw6wvzknvpnq6y5xkx62hgvsszd2447hv4.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_34 => convert_element_type_31, squeeze_27
# Graph fragment:
#   %convert_element_type_41 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convert_element_type_41]
#   %convolution_10 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %getitem_25 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_25]
#   %squeeze_27 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_25, [0, 2, 3]), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_27, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %sum_1 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_10 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_31, %unsqueeze_42), kwargs = {})
#   %mul_73 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_41, %sub_10), kwargs = {})
#   %sum_2 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_73, [0, 2, 3]), kwargs = {})
#   return %buf8,%buf10
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_5 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25692160, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/vl/cvli332brcfep7m7yllifwwkuwnmqxqgmqsjk6prndsqap2bquwe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf8 : Tensor "f32[512, 64][1, 512]cuda:0" = PlaceHolder[target=buf8]
#   %sum_1 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {})
#   return %sum_1
triton_per_fused_native_batch_norm_backward_6 = async_compile.triton('triton_per_fused_native_batch_norm_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 135168, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_batch_norm_backward_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/sb/csbxsrpqgln2xay3hogdv4arm4apajjivgm7dn44gawmktoojqsn.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_34 => convert_element_type_31, squeeze_27, squeeze_28
# Graph fragment:
#   %buf10 : Tensor "f32[512, 64][1, 512]cuda:0" = PlaceHolder[target=buf10]
#   %sum_2 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_2]
#   %rsqrt_9 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %squeeze_27 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_25, [0, 2, 3]), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_27, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_10 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_31, %unsqueeze_42), kwargs = {})
#   %mul_73 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_41, %sub_10), kwargs = {})
#   %sum_2 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_73, [0, 2, 3]), kwargs = {})
#   %squeeze_28 : Tensor "f32[512][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_9, [0, 2, 3]), kwargs = {})
#   %mul_81 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %squeeze_28), kwargs = {})
#   return %sum_2,%mul_81
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 141312, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/dd/cddff6mz32xcoxhfuipwhjmg3hlpfwmahjv6wh6rqtc2e5yhyni4.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_34 => convert_element_type_31, squeeze_27, squeeze_28
# Graph fragment:
#   %convert_element_type_41 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convert_element_type_41]
#   %convolution_10 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %getitem_25 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_25]
#   %sum_2 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_2]
#   %rsqrt_9 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %sum_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_1]
#   %primals_62 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_62]
#   %squeeze_27 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_25, [0, 2, 3]), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_27, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %convert_element_type_31 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_10 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_31, %unsqueeze_42), kwargs = {})
#   %mul_74 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 0.0001220703125), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_74, 0), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_43, 2), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, 3), kwargs = {})
#   %mul_75 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 0.0001220703125), kwargs = {})
#   %squeeze_28 : Tensor "f32[512][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_9, [0, 2, 3]), kwargs = {})
#   %mul_76 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_28, %squeeze_28), kwargs = {})
#   %mul_77 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %mul_76), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_77, 0), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, 2), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_47, 3), kwargs = {})
#   %mul_78 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_28, %primals_62), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_78, 0), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_49, 2), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, 3), kwargs = {})
#   %mul_79 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_48), kwargs = {})
#   %sub_12 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_41, %mul_79), kwargs = {})
#   %sub_13 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %unsqueeze_45), kwargs = {})
#   %mul_80 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_51), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_80, torch.bfloat16), kwargs = {})
#   %convolution_backward : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_43, %relu_8, %convert_element_type_30, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf13
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41953280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 0.0001220703125
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ok/coklviyzpk67vwp2ueftpbfuqqzyr7wmaea7ckzonby6qlwbfx2m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_29 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=getitem_29]
#   %convert_element_type_44 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_29, torch.float32), kwargs = {})
#   return %convert_element_type_44
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ef/cef26mfksfzgel3pwcdjpfbeacewjetf4dbskfjxtolypfz2kvgq.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_31 => convert_element_type_28
# Graph fragment:
#   %relu_8 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=relu_8]
#   %getitem_28 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_28]
#   %convolution_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_9]
#   %unsqueeze_54 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_54]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : Tensor "b8[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_1 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_28), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %sum_3 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_45, [0, 2, 3]), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %sub_14 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_28, %unsqueeze_54), kwargs = {})
#   %mul_82 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_45, %sub_14), kwargs = {})
#   %sum_4 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_82, [0, 2, 3]), kwargs = {})
#   return %buf18,%buf20
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25692160, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/7h/c7hagop4dba7zkjnnnqoqvxcjf3jiwmyzcihlfdk6r4mv2bgxwec.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_31 => convert_element_type_28
# Graph fragment:
#   %relu_8 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=relu_8]
#   %getitem_28 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_28]
#   %convolution_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_9]
#   %unsqueeze_54 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_54]
#   %sum_4 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_4]
#   %squeeze_25 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=squeeze_25]
#   %sum_3 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_3]
#   %primals_56 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_56]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : Tensor "b8[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_1 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_28), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %sub_14 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_28, %unsqueeze_54), kwargs = {})
#   %mul_83 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_3, 0.0001220703125), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_83, 0), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_55, 2), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, 3), kwargs = {})
#   %mul_84 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, 0.0001220703125), kwargs = {})
#   %mul_85 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_25, %squeeze_25), kwargs = {})
#   %mul_86 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %mul_85), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_86, 0), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, 2), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_59, 3), kwargs = {})
#   %mul_87 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_25, %primals_56), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_87, 0), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_61, 2), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, 3), kwargs = {})
#   %mul_88 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_60), kwargs = {})
#   %sub_16 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_45, %mul_88), kwargs = {})
#   %sub_17 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %unsqueeze_57), kwargs = {})
#   %mul_89 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_63), kwargs = {})
#   %convert_element_type_47 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_89, torch.bfloat16), kwargs = {})
#   %convolution_backward_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_47, %getitem_20, %convert_element_type_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf23
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41953280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0001220703125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/nx/cnx2kpywnbe5lugrb4fruyvjz7ezeq75suat3atslbydmy2cg7eb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %max_pool2d_with_indices_backward : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward]
#   %getitem_31 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_31]
#   %add_53 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_pool2d_with_indices_backward, %getitem_31), kwargs = {})
#   return %add_53
triton_poi_fused_add_12 = async_compile.triton('triton_poi_fused_add_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33554432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/vi/cvikzuhnbo2jifja4ciu3i6ti7db765qnqqvwzs2kl4rejryoy3n.py
# Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   input_29 => _low_memory_max_pool_offsets_to_indices_2
# Graph fragment:
#   %getitem_21 : Tensor "i8[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_21]
#   %add_53 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=add_53]
#   %add_53 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_pool2d_with_indices_backward, %getitem_31), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices_2 : Tensor "i64[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_21, [2, 2], [8, 8], [2, 2], [0, 0], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward_1 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_53, %relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool_offsets_to_indices_2), kwargs = {})
#   return %max_pool2d_with_indices_backward_1
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_13 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 8)
    x2 = ((xindex // 4096) % 8)
    x3 = xindex // 32768
    x4 = ((xindex // 512) % 64)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2048*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 512*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2048*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 8192*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp7 = tmp4 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (4)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 6*(tmp4 // 2) + 16*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4))))) + ((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) * (((-1) + ((4) * ((4) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (4)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tl.store(out_ptr0 + (x5), tmp11, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/3e/c3ez6wspuz3cp6mwcueqvvvxi2oo2mg6ycinyi5yvu2prvu5n3bs.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_27 => convert_element_type_25
# Graph fragment:
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=relu_7]
#   %max_pool2d_with_indices_backward_1 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_1]
#   %convolution_8 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_8]
#   %unsqueeze_66 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_66]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_2 : Tensor "b8[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
#   %where_2 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %max_pool2d_with_indices_backward_1), kwargs = {})
#   %convert_element_type_49 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_2, torch.float32), kwargs = {})
#   %sum_5 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_49, [0, 2, 3]), kwargs = {})
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %sub_18 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_25, %unsqueeze_66), kwargs = {})
#   %mul_91 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, %sub_18), kwargs = {})
#   %sum_6 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_91, [0, 2, 3]), kwargs = {})
#   return %buf30,%buf32
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 101713920, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fw/cfwlwvwgs5jx5q5aqvwwlmrfvndvnqjzlqa5qbos3757xbybburz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf30 : Tensor "f32[512, 128][1, 512]cuda:0" = PlaceHolder[target=buf30]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_2 : Tensor "b8[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
#   %where_2 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %max_pool2d_with_indices_backward_1), kwargs = {})
#   %convert_element_type_49 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_2, torch.float32), kwargs = {})
#   %sum_5 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_49, [0, 2, 3]), kwargs = {})
#   return %sum_5
triton_red_fused_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 266240, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/vt/cvta2bcwgqkilyudafmeylzmyr5alztdpu6hltwdzcvvroh2qwqe.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_27 => convert_element_type_25
# Graph fragment:
#   %buf32 : Tensor "f32[512, 128][1, 512]cuda:0" = PlaceHolder[target=buf32]
#   %sum_6 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_6]
#   %squeeze_22 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=squeeze_22]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_2 : Tensor "b8[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
#   %where_2 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %max_pool2d_with_indices_backward_1), kwargs = {})
#   %convert_element_type_49 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_2, torch.float32), kwargs = {})
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %sub_18 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_25, %unsqueeze_66), kwargs = {})
#   %mul_91 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, %sub_18), kwargs = {})
#   %sum_6 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_91, [0, 2, 3]), kwargs = {})
#   %mul_99 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, %squeeze_22), kwargs = {})
#   return %sum_6,%mul_99
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 272384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/5t/c5tyue2guhe2imx3hzuxas5lzfjpqclbhepfuusu52axtemmnpy7.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_27 => convert_element_type_25
# Graph fragment:
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=relu_7]
#   %max_pool2d_with_indices_backward_1 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_1]
#   %convolution_8 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_8]
#   %unsqueeze_66 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_66]
#   %sum_6 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_6]
#   %squeeze_22 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=squeeze_22]
#   %sum_5 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=sum_5]
#   %primals_50 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_50]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_2 : Tensor "b8[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
#   %where_2 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %max_pool2d_with_indices_backward_1), kwargs = {})
#   %convert_element_type_49 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_2, torch.float32), kwargs = {})
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %sub_18 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_25, %unsqueeze_66), kwargs = {})
#   %mul_92 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, 3.0517578125e-05), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_92, 0), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_67, 2), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, 3), kwargs = {})
#   %mul_93 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, 3.0517578125e-05), kwargs = {})
#   %mul_94 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_22, %squeeze_22), kwargs = {})
#   %mul_95 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %mul_94), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_95, 0), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, 2), kwargs = {})
#   %unsqueeze_72 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_71, 3), kwargs = {})
#   %mul_96 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_22, %primals_50), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_96, 0), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_73, 2), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, 3), kwargs = {})
#   %mul_97 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_72), kwargs = {})
#   %sub_20 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_49, %mul_97), kwargs = {})
#   %sub_21 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_20, %unsqueeze_69), kwargs = {})
#   %mul_98 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_75), kwargs = {})
#   %convert_element_type_51 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_98, torch.bfloat16), kwargs = {})
#   %convolution_backward_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_51, %convert_element_type_24, %convert_element_type_23, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf35
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167782400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.0517578125e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/t5/ct5cb5uvrdfvtezbx4pf3nrwyoynqxlotplotqqyxknriyuo6vv2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_35 : Tensor "bf16[512, 320, 3, 3][2880, 1, 960, 320]cuda:0" = PlaceHolder[target=getitem_35]
#   %convert_element_type_53 : Tensor "f32[512, 320, 3, 3][2880, 1, 960, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_35, torch.float32), kwargs = {})
#   return %convert_element_type_53
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14745600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1474560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/4m/c4mlzpnvl6ycq3vydgvk53fm2fgaee5utzib2225xuzp65v3fztl.py
# Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_23 => add_35, convert_element_type_22, mul_42, mul_48, sub_6, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27
#   input_24 => relu_6
# Graph fragment:
#   %convolution_7 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_7]
#   %getitem_17 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=getitem_17]
#   %rsqrt_6 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %primals_44 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=primals_44]
#   %primals_45 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=primals_45]
#   %sub_6 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %getitem_17), kwargs = {})
#   %mul_42 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_44, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %mul_48 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %unsqueeze_25), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_45, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %add_35 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %unsqueeze_27), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.bfloat16), kwargs = {})
#   %relu_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_22,), kwargs = {})
#   return %relu_6
triton_poi_fused__native_batch_norm_legit_functional_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62919680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/nz/cnz4tszbfb2k453v65wpe7xj723lr3lt577lz46gmxfalu457dpl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_34 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_34]
#   %relu_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=relu_6]
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_101 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %relu_6), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_101, [0, 1, 2, 3], True), kwargs = {dtype: torch.float32})
#   return %buf42
triton_per_fused__to_copy_mul_sum_20 = async_compile.triton('triton_per_fused__to_copy_mul_sum_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_20(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 81920
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (320*((r0_2 % 64)) + 20480*((r0_2 + 128*x0 + 32768*x1) // 20480) + ((((r0_2 + 128*x0 + 32768*x1) // 64) % 320))), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (320*((r0_2 % 64)) + 20480*((r0_2 + 128*x0 + 32768*x1) // 20480) + ((((r0_2 + 128*x0 + 32768*x1) // 64) % 320))), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/gu/cguawvl6bdmf56tifgx3cr2r2onnt2v3pubqcoo2vb33f6fr2t6z.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf42 : Tensor "f32[1, 1, 1, 1, 320, 256][81920, 81920, 81920, 81920, 256, 1]cuda:0" = PlaceHolder[target=buf42]
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_101 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %relu_6), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_101, [0, 1, 2, 3], True), kwargs = {dtype: torch.float32})
#   return %buf43
triton_per_fused__to_copy_mul_sum_21 = async_compile.triton('triton_per_fused__to_copy_mul_sum_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2560, 'r0_': 327680}}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_21(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/e3/ce3gd2xzrl2gn7xex3shaajejapfldmgb7z7sfncsi6lfnwfv573.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf43 : Tensor "f32[1, 1, 1, 1, 320][320, 320, 320, 320, 1]cuda:0" = PlaceHolder[target=buf43]
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_101 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %relu_6), kwargs = {})
#   %sum_7 : Tensor "f32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_101, [0, 1, 2, 3], True), kwargs = {dtype: torch.float32})
#   return %sum_7
triton_per_fused__to_copy_mul_sum_22 = async_compile.triton('triton_per_fused__to_copy_mul_sum_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 1280}}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_22(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 320
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/jj/cjjs6x6iao2en64zgpko4i3n5ayshcrcwoclyi2f2nakuwskt3fx.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_23 => convert_element_type_21, squeeze_18
# Graph fragment:
#   %relu_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=relu_6]
#   %getitem_34 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_34]
#   %primals_33 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_33]
#   %convolution_7 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_7]
#   %getitem_17 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=getitem_17]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %primals_33), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %le_3 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_3 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %convert_element_type_55), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %squeeze_18 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_17, [0, 2, 3]), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_18, 0), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, 2), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_77, 3), kwargs = {})
#   %sum_8 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_56, [0, 2, 3]), kwargs = {})
#   %convert_element_type_21 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %sub_22 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_21, %unsqueeze_78), kwargs = {})
#   %mul_102 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_56, %sub_22), kwargs = {})
#   %sum_9 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_102, [0, 2, 3]), kwargs = {})
#   return %buf45,%buf47
triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64226560, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 81920
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 320)
    x1 = xindex // 320
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr3 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.where(tmp2, tmp1, tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(r0_mask, tmp21, _tmp20)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fv/cfv2kdomitlxdijykuvyhmk4jgafvvy47kemqwevzatzdfuboyay.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf45 : Tensor "f32[320, 256][1, 320]cuda:0" = PlaceHolder[target=buf45]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %primals_33), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %le_3 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_3 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %convert_element_type_55), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %sum_8 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_56, [0, 2, 3]), kwargs = {})
#   return %sum_8
triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 330240, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/un/cunb4hjpqnkbpnaze7jt5po44ze2jqd4mxdzwyid7th6xezpltw2.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_23 => convert_element_type_21, squeeze_18, squeeze_19
# Graph fragment:
#   %buf47 : Tensor "f32[320, 256][1, 320]cuda:0" = PlaceHolder[target=buf47]
#   %sum_9 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_9]
#   %rsqrt_6 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %primals_33), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %le_3 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_3 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %convert_element_type_55), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %squeeze_18 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_17, [0, 2, 3]), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_18, 0), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, 2), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_77, 3), kwargs = {})
#   %convert_element_type_21 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %sub_22 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_21, %unsqueeze_78), kwargs = {})
#   %mul_102 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_56, %sub_22), kwargs = {})
#   %sum_9 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_102, [0, 2, 3]), kwargs = {})
#   %squeeze_19 : Tensor "f32[320][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_6, [0, 2, 3]), kwargs = {})
#   %mul_110 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_9, %squeeze_19), kwargs = {})
#   return %sum_9,%mul_110
triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 334080, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ik/cikblg3hx6lzgca3or2vx7pj47oblcgpwqhsifuhltu37bcy3smc.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_23 => convert_element_type_21, squeeze_18, squeeze_19
# Graph fragment:
#   %relu_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=relu_6]
#   %getitem_34 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_34]
#   %primals_33 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_33]
#   %convolution_7 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_7]
#   %getitem_17 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=getitem_17]
#   %sum_9 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_9]
#   %rsqrt_6 : Tensor "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %sum_8 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_8]
#   %primals_44 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=primals_44]
#   %mul_109 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=mul_109]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %mul_100 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %primals_33), kwargs = {})
#   %convert_element_type_55 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_100, torch.bfloat16), kwargs = {})
#   %le_3 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_3 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %convert_element_type_55), kwargs = {})
#   %convert_element_type_56 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %squeeze_18 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_17, [0, 2, 3]), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_18, 0), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, 2), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_77, 3), kwargs = {})
#   %convert_element_type_21 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %sub_22 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_21, %unsqueeze_78), kwargs = {})
#   %mul_103 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_8, 3.0517578125e-05), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_103, 0), kwargs = {})
#   %unsqueeze_80 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_79, 2), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, 3), kwargs = {})
#   %mul_104 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_9, 3.0517578125e-05), kwargs = {})
#   %squeeze_19 : Tensor "f32[320][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_6, [0, 2, 3]), kwargs = {})
#   %mul_105 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_19, %squeeze_19), kwargs = {})
#   %mul_106 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %mul_105), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_106, 0), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, 2), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_83, 3), kwargs = {})
#   %mul_107 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_19, %primals_44), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_107, 0), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_85, 2), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, 3), kwargs = {})
#   %mul_108 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_84), kwargs = {})
#   %sub_24 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_56, %mul_108), kwargs = {})
#   %sub_25 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_24, %unsqueeze_81), kwargs = {})
#   %mul_109 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_87), kwargs = {})
#   %convert_element_type_58 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_109, torch.bfloat16), kwargs = {})
#   %convolution_backward_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_58, %relu_5, %convert_element_type_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %mul_109,%buf51
triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_backward_mul_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_backward_mul_native_batch_norm_backward_threshold_backward_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_backward_mul_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 104864000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_backward_mul_native_batch_norm_backward_threshold_backward_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.where(tmp2, tmp1, tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 - tmp13
    tmp16 = 3.0517578125e-05
    tmp17 = tmp15 * tmp16
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 * tmp20
    tmp22 = tmp10 - tmp21
    tmp24 = tmp23 * tmp16
    tmp25 = tmp22 - tmp24
    tmp27 = tmp18 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/da/cdatpoh3pkri6ijg372o42brtabkdvkdoyhoog3zmnnntkeshzmy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_38 : Tensor "bf16[320, 320, 3, 3][2880, 1, 960, 320]cuda:0" = PlaceHolder[target=getitem_38]
#   %convert_element_type_59 : Tensor "f32[320, 320, 3, 3][2880, 1, 960, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_38, torch.float32), kwargs = {})
#   return %convert_element_type_59
triton_poi_fused__to_copy_27 = async_compile.triton('triton_poi_fused__to_copy_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/pk/cpkh3crbefnedzqbrx45ebvbh54qxoitd7tfynim7udezhfyrsqz.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_20 => convert_element_type_18
# Graph fragment:
#   %relu_5 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=relu_5]
#   %getitem_37 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_37]
#   %convolution_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_6]
#   %unsqueeze_90 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_90]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_4 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
#   %where_4 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default, %getitem_37), kwargs = {})
#   %convert_element_type_60 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_4, torch.float32), kwargs = {})
#   %sum_10 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_60, [0, 2, 3]), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %sub_26 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_18, %unsqueeze_90), kwargs = {})
#   %mul_111 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_60, %sub_26), kwargs = {})
#   %sum_11 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [0, 2, 3]), kwargs = {})
#   return %buf56,%buf58
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64226560, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 81920
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 320)
    x1 = xindex // 320
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 320*r0_2 + 40960*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/j6/cj6peziscavjul43s7k6zjbca7gxlurd2iaoxnrqps4otxcor6vj.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_20 => convert_element_type_18
# Graph fragment:
#   %relu_5 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=relu_5]
#   %getitem_37 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_37]
#   %convolution_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_6]
#   %unsqueeze_90 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_90]
#   %sum_11 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_11]
#   %squeeze_16 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=squeeze_16]
#   %sum_10 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_10]
#   %primals_38 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=primals_38]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_4 : Tensor "b8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
#   %where_4 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_4, %full_default, %getitem_37), kwargs = {})
#   %convert_element_type_60 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_4, torch.float32), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %sub_26 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_18, %unsqueeze_90), kwargs = {})
#   %mul_112 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_10, 3.0517578125e-05), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_112, 0), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_91, 2), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, 3), kwargs = {})
#   %mul_113 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_11, 3.0517578125e-05), kwargs = {})
#   %mul_114 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_16, %squeeze_16), kwargs = {})
#   %mul_115 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %mul_114), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_115, 0), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, 2), kwargs = {})
#   %unsqueeze_96 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_95, 3), kwargs = {})
#   %mul_116 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_16, %primals_38), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_116, 0), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_97, 2), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, 3), kwargs = {})
#   %mul_117 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_96), kwargs = {})
#   %sub_28 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_60, %mul_117), kwargs = {})
#   %sub_29 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_28, %unsqueeze_93), kwargs = {})
#   %mul_118 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_99), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_118, torch.bfloat16), kwargs = {})
#   %convolution_backward_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_62, %getitem_12, %convert_element_type_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf61
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 104864000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.0517578125e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/xw/cxwttgpwww32ugng2uh7wqqwtjsyjrees5gen36k6jo6437vqsjf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_34 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_34]
#   %getitem_40 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_40]
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_52, torch.bfloat16), kwargs = {})
#   %add_54 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_54, %getitem_40), kwargs = {})
#   return %add_54
triton_poi_fused__to_copy_add_30 = async_compile.triton('triton_poi_fused__to_copy_add_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83886080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/bq/cbqcszlpwkb6njt5xx7432pgok6fkpaxoqskhvqwfui2vohrtu7f.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy, aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   input_18 => _low_memory_max_pool_offsets_to_indices_1
# Graph fragment:
#   %getitem_13 : Tensor "i8[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_13]
#   %add_54 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=add_54]
#   %convert_element_type_52 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_34, torch.float32), kwargs = {})
#   %convert_element_type_54 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_52, torch.bfloat16), kwargs = {})
#   %add_54 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_54, %getitem_40), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices_1 : Tensor "i64[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_13, [2, 2], [16, 16], [2, 2], [0, 0], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward_2 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_54, %relu_4, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool_offsets_to_indices_1), kwargs = {})
#   return %max_pool2d_with_indices_backward_2
triton_poi_fused__to_copy_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_31 = async_compile.triton('triton_poi_fused__to_copy_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 320)
    x1 = ((xindex // 320) % 16)
    x2 = ((xindex // 5120) % 16)
    x3 = xindex // 81920
    x4 = ((xindex // 320) % 256)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 320*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2560*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 20480*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 320*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2560*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 20480*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp7 = tmp4 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (8)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 14*(tmp4 // 2) + 32*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8))))) + ((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) * (((-1) + ((8) * ((8) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (8)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tl.store(out_ptr0 + (x5), tmp11, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/z4/cz4x34dvxedyno2vunlmqrbk2sb72wanrucuacqo4eepk6s6fkaf.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_15
# Graph fragment:
#   %relu_4 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=relu_4]
#   %max_pool2d_with_indices_backward_2 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_2]
#   %convolution_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=convolution_5]
#   %unsqueeze_102 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_102]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_5 : Tensor "b8[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %where_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default, %max_pool2d_with_indices_backward_2), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_5, torch.float32), kwargs = {})
#   %sum_12 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_64, [0, 2, 3]), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %sub_30 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_15, %unsqueeze_102), kwargs = {})
#   %mul_120 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %sub_30), kwargs = {})
#   %sum_13 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_120, [0, 2, 3]), kwargs = {})
#   return %buf68,%buf70
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 253062400, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 66560
    r0_numel = 631
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 320
    x0 = (xindex % 320)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 631*x1
        tmp1 = tl.full([1, 1], 131072, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 320*(((r0_2 + 631*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + 320*(((r0_2 + 631*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (x0 + 320*(((r0_2 + 631*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fw/cfwofjf7ama4p4tczfjmlunnrj3fa3jmcl4w6dk5fohweouxb342.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf68 : Tensor "f32[320, 208][1, 320]cuda:0" = PlaceHolder[target=buf68]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_5 : Tensor "b8[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %where_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default, %max_pool2d_with_indices_backward_2), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_5, torch.float32), kwargs = {})
#   %sum_12 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_64, [0, 2, 3]), kwargs = {})
#   return %sum_12
triton_red_fused_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 268800, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_33(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/dg/cdgxsfbh2bga5kp4i2sgoyt67ah2hvzxjuqkpc5klh432bs6cayo.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_15
# Graph fragment:
#   %buf70 : Tensor "f32[320, 208][1, 320]cuda:0" = PlaceHolder[target=buf70]
#   %sum_13 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_13]
#   %squeeze_13 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=squeeze_13]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_5 : Tensor "b8[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %where_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default, %max_pool2d_with_indices_backward_2), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_5, torch.float32), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %sub_30 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_15, %unsqueeze_102), kwargs = {})
#   %mul_120 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %sub_30), kwargs = {})
#   %sum_13 : Tensor "f32[320][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_120, [0, 2, 3]), kwargs = {})
#   %mul_128 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_13, %squeeze_13), kwargs = {})
#   return %sum_13,%mul_128
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 272640, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_34(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 320*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/o5/co5k6my2zbx766fbossyhjyd7rpooyjiwwlsx5idf6q6ut5jto2d.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_15
# Graph fragment:
#   %relu_4 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=relu_4]
#   %max_pool2d_with_indices_backward_2 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_2]
#   %convolution_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=convolution_5]
#   %unsqueeze_102 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_102]
#   %sum_13 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_13]
#   %squeeze_13 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=squeeze_13]
#   %sum_12 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=sum_12]
#   %primals_31 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=primals_31]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_5 : Tensor "b8[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
#   %where_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_5, %full_default, %max_pool2d_with_indices_backward_2), kwargs = {})
#   %convert_element_type_64 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_5, torch.float32), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %sub_30 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_15, %unsqueeze_102), kwargs = {})
#   %mul_121 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_12, 7.62939453125e-06), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_121, 0), kwargs = {})
#   %unsqueeze_104 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_103, 2), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_104, 3), kwargs = {})
#   %mul_122 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_13, 7.62939453125e-06), kwargs = {})
#   %mul_123 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_13, %squeeze_13), kwargs = {})
#   %mul_124 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_122, %mul_123), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_124, 0), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, 2), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_107, 3), kwargs = {})
#   %mul_125 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_13, %primals_31), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[1, 320][320, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_125, 0), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[1, 320, 1][320, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_109, 2), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_110, 3), kwargs = {})
#   %mul_126 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_108), kwargs = {})
#   %sub_32 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_64, %mul_126), kwargs = {})
#   %sub_33 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_32, %unsqueeze_105), kwargs = {})
#   %mul_127 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_111), kwargs = {})
#   %convert_element_type_66 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_127, torch.bfloat16), kwargs = {})
#   %convolution_backward_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_66, %add_20, %convert_element_type_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf73
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 419436800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 7.62939453125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/db/cdbyluifrfakd3j7amk6gvynmwutuzvu3ltwystsqeodvdl6nueg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_44 : Tensor "bf16[320, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=getitem_44]
#   %convert_element_type_67 : Tensor "f32[320, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_44, torch.float32), kwargs = {})
#   return %convert_element_type_67
triton_poi_fused__to_copy_36 = async_compile.triton('triton_poi_fused__to_copy_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3686400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/mn/cmnkzcjecocmxnp6llfrl6xsf3gowwraiynp5wfnm7frsligbtar.py
# Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_12 => add_19, convert_element_type_13, mul_21, mul_27, sub_3, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15
#   input_13 => relu_3
# Graph fragment:
#   %convolution_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_4]
#   %getitem_9 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_25 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_25]
#   %primals_26 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_26]
#   %getitem_43 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_43]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_3 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %getitem_9), kwargs = {})
#   %mul_21 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_25, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_27 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_26, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_19 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_15), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
#   %relu_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_13,), kwargs = {})
#   %le_6 : Tensor "b8[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %where_6 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default, %getitem_43), kwargs = {})
#   %convert_element_type_68 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_6, torch.float32), kwargs = {})
#   return %convert_element_type_68
triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_threshold_backward_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_threshold_backward_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_threshold_backward_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201328640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_threshold_backward_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp13 = 0.0
    tmp14 = tmp12 <= tmp13
    tmp16 = tl.where(tmp14, tmp13, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/wk/cwkoxzsluk52wusas4ywenlvhfujkpbj5qjp5jvokofc2thdzbyk.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_12, squeeze_9
# Graph fragment:
#   %convert_element_type_68 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convert_element_type_68]
#   %convolution_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_4]
#   %getitem_9 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %squeeze_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2, 3]), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, 2), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_113, 3), kwargs = {})
#   %sum_14 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_68, [0, 2, 3]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %sub_34 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %unsqueeze_114), kwargs = {})
#   %mul_129 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, %sub_34), kwargs = {})
#   %sum_15 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_129, [0, 2, 3]), kwargs = {})
#   return %buf79,%buf81
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_38 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 101712384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/i4/ci4by6l2lhnkpektersttwlcuxwtlwhhoocmnjp5mftpm6kcqogs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf79 : Tensor "f32[128, 512][1, 128]cuda:0" = PlaceHolder[target=buf79]
#   %sum_14 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_68, [0, 2, 3]), kwargs = {})
#   return %sum_14
triton_red_fused_native_batch_norm_backward_39 = async_compile.triton('triton_red_fused_native_batch_norm_backward_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 263168, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_39(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/5k/c5kcemdvbw7khbaor3oayc27zefcq4cxbklk52fddcbiap4nll6n.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_12, squeeze_10, squeeze_9
# Graph fragment:
#   %buf81 : Tensor "f32[128, 512][1, 128]cuda:0" = PlaceHolder[target=buf81]
#   %sum_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_15]
#   %rsqrt_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %squeeze_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2, 3]), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, 2), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_113, 3), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %sub_34 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %unsqueeze_114), kwargs = {})
#   %mul_129 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_68, %sub_34), kwargs = {})
#   %sum_15 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_129, [0, 2, 3]), kwargs = {})
#   %squeeze_10 : Tensor "f32[128][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_3, [0, 2, 3]), kwargs = {})
#   %mul_137 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_15, %squeeze_10), kwargs = {})
#   return %sum_15,%mul_137
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 264704, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/26/c26ng2f72vr3ihpcfyqmumq3q5a552gtgo2gs2kacqlmaexeurep.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_12 => convert_element_type_12, squeeze_10, squeeze_9
# Graph fragment:
#   %convert_element_type_68 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convert_element_type_68]
#   %convolution_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_4]
#   %getitem_9 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %sum_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_15]
#   %rsqrt_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %sum_14 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_14]
#   %primals_25 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_25]
#   %squeeze_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2, 3]), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, 2), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_113, 3), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %sub_34 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %unsqueeze_114), kwargs = {})
#   %mul_130 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_14, 7.62939453125e-06), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_130, 0), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_115, 2), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, 3), kwargs = {})
#   %mul_131 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_15, 7.62939453125e-06), kwargs = {})
#   %squeeze_10 : Tensor "f32[128][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_3, [0, 2, 3]), kwargs = {})
#   %mul_132 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %squeeze_10), kwargs = {})
#   %mul_133 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %mul_132), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_133, 0), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, 2), kwargs = {})
#   %unsqueeze_120 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_119, 3), kwargs = {})
#   %mul_134 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %primals_25), kwargs = {})
#   %unsqueeze_121 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_134, 0), kwargs = {})
#   %unsqueeze_122 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_121, 2), kwargs = {})
#   %unsqueeze_123 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_122, 3), kwargs = {})
#   %mul_135 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_120), kwargs = {})
#   %sub_36 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_68, %mul_135), kwargs = {})
#   %sub_37 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_36, %unsqueeze_117), kwargs = {})
#   %mul_136 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_123), kwargs = {})
#   %convert_element_type_70 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_136, torch.bfloat16), kwargs = {})
#   %convolution_backward_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_70, %relu_2, %convert_element_type_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf84
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167774720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 7.62939453125e-06
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/wu/cwuu4tiw7hl6viiutrjdz3aijywpyj4yunfsmkbn6nwk6vcmco2q.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_47 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=getitem_47]
#   %convert_element_type_71 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_47, torch.float32), kwargs = {})
#   return %convert_element_type_71
triton_poi_fused__to_copy_42 = async_compile.triton('triton_poi_fused__to_copy_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/nt/cntijqot37ijj7raa6upcdyhjgtwhitiijwmeu5hr3yirs5t3jlm.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_9
# Graph fragment:
#   %relu_2 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=relu_2]
#   %getitem_46 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_46]
#   %convolution_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_3]
#   %unsqueeze_126 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_126]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_7 : Tensor "b8[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
#   %where_7 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default, %getitem_46), kwargs = {})
#   %convert_element_type_72 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_7, torch.float32), kwargs = {})
#   %sum_16 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_72, [0, 2, 3]), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %sub_38 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %unsqueeze_126), kwargs = {})
#   %mul_138 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_72, %sub_38), kwargs = {})
#   %sum_17 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_138, [0, 2, 3]), kwargs = {})
#   return %buf89,%buf91
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 101712384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/7s/c7szouwnbcghvl7hscw5psu7selotjjafxjduoxb3nyg2o5x37bq.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_9
# Graph fragment:
#   %relu_2 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=relu_2]
#   %getitem_46 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_46]
#   %convolution_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_3]
#   %unsqueeze_126 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_126]
#   %sum_17 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_17]
#   %squeeze_7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=squeeze_7]
#   %sum_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_16]
#   %primals_19 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_19]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_7 : Tensor "b8[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
#   %where_7 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default, %getitem_46), kwargs = {})
#   %convert_element_type_72 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_7, torch.float32), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %sub_38 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_9, %unsqueeze_126), kwargs = {})
#   %mul_139 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_16, 7.62939453125e-06), kwargs = {})
#   %unsqueeze_127 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_139, 0), kwargs = {})
#   %unsqueeze_128 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_127, 2), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, 3), kwargs = {})
#   %mul_140 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_17, 7.62939453125e-06), kwargs = {})
#   %mul_141 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_7, %squeeze_7), kwargs = {})
#   %mul_142 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_140, %mul_141), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_142, 0), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, 2), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_131, 3), kwargs = {})
#   %mul_143 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_7, %primals_19), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_143, 0), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_133, 2), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, 3), kwargs = {})
#   %mul_144 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_132), kwargs = {})
#   %sub_40 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_72, %mul_144), kwargs = {})
#   %sub_41 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_40, %unsqueeze_129), kwargs = {})
#   %mul_145 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_135), kwargs = {})
#   %convert_element_type_74 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_145, torch.bfloat16), kwargs = {})
#   %convolution_backward_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_74, %getitem_4, %convert_element_type_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf94
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167774720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 7.62939453125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/55/c55p3qmkn4t5afh4hhrg7dueasihqyez2c5abyhcqv2yqaezgcdn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_43 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_43]
#   %getitem_49 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_49]
#   %add_55 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_43, %getitem_49), kwargs = {})
#   return %add_55
triton_poi_fused_add_45 = async_compile.triton('triton_poi_fused_add_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134217728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_45(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/id/cid5zq4jbu4ryfimbntwnwkf32tqzssxenfwsfseuquoapyolhs7.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   input_7 => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %getitem_5 : Tensor "i8[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %add_55 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=add_55]
#   %add_55 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_43, %getitem_49), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices : Tensor "i64[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_5, [2, 2], [32, 32], [2, 2], [0, 0], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward_3 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_55, %relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
#   return %max_pool2d_with_indices_backward_3
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_46 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_46(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32)
    x2 = ((xindex // 4096) % 32)
    x3 = xindex // 131072
    x4 = ((xindex // 128) % 1024)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2048*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 32768*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 128*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 2048*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 32768*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp7 = tmp4 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x1 // 2))) + (1 + (x1 // 2)) * ((1 + (x1 // 2)) < (16)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 30*(tmp4 // 2) + 64*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16))))) + ((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) * (((-1) + ((16) * ((16) <= (1 + (x2 // 2))) + (1 + (x2 // 2)) * ((1 + (x2 // 2)) < (16)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tl.store(out_ptr0 + (x5), tmp11, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/57/c574egya4xp37icif3ea6lthqjspmoibf6f7747hium4c4h5egmq.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_6
# Graph fragment:
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=relu_1]
#   %max_pool2d_with_indices_backward_3 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_3]
#   %convolution_2 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %unsqueeze_138 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_138]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : Tensor "b8[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %where_8 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %max_pool2d_with_indices_backward_3), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %sum_18 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_76, [0, 2, 3]), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %sub_42 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %unsqueeze_138), kwargs = {})
#   %mul_147 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %sub_42), kwargs = {})
#   %sum_19 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_147, [0, 2, 3]), kwargs = {})
#   return %buf101,%buf103
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 403972096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 79616
    r0_numel = 843
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 128
    x0 = (xindex % 128)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 843*x1
        tmp1 = tl.full([1, 1], 524288, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 128*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + 128*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (x0 + 128*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ki/ckimiahf2b36rodscgeq2lcbvy2klq533isqrr2bptdhkqrvngs2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf101 : Tensor "f32[128, 622][1, 128]cuda:0" = PlaceHolder[target=buf101]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : Tensor "b8[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %where_8 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %max_pool2d_with_indices_backward_3), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %sum_18 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_76, [0, 2, 3]), kwargs = {})
#   return %sum_18
triton_red_fused_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 319488, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_48(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 622
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/3w/c3wsalhpwxm6rk4qt7dfkjmtvihbuvpz45cyea263ttzwdhq2d4a.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_6
# Graph fragment:
#   %buf103 : Tensor "f32[128, 622][1, 128]cuda:0" = PlaceHolder[target=buf103]
#   %sum_19 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_19]
#   %squeeze_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=squeeze_4]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : Tensor "b8[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %where_8 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %max_pool2d_with_indices_backward_3), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %sub_42 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %unsqueeze_138), kwargs = {})
#   %mul_147 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %sub_42), kwargs = {})
#   %sum_19 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_147, [0, 2, 3]), kwargs = {})
#   %mul_155 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_19, %squeeze_4), kwargs = {})
#   return %sum_19,%mul_155
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 321024, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_49(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 622
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ee/ceevozx75whum7yltv7dnibomttbwzpisexxqicqdsm6kxq3mtc4.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_6
# Graph fragment:
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=relu_1]
#   %max_pool2d_with_indices_backward_3 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=max_pool2d_with_indices_backward_3]
#   %convolution_2 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %unsqueeze_138 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_138]
#   %sum_19 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_19]
#   %squeeze_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=squeeze_4]
#   %sum_18 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=sum_18]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : Tensor "b8[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
#   %where_8 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %max_pool2d_with_indices_backward_3), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %sub_42 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %unsqueeze_138), kwargs = {})
#   %mul_148 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_18, 1.9073486328125e-06), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_148, 0), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_139, 2), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, 3), kwargs = {})
#   %mul_149 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_19, 1.9073486328125e-06), kwargs = {})
#   %mul_150 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, %squeeze_4), kwargs = {})
#   %mul_151 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %mul_150), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_151, 0), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, 2), kwargs = {})
#   %unsqueeze_144 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_143, 3), kwargs = {})
#   %mul_152 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, %primals_13), kwargs = {})
#   %unsqueeze_145 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_152, 0), kwargs = {})
#   %unsqueeze_146 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_145, 2), kwargs = {})
#   %unsqueeze_147 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_146, 3), kwargs = {})
#   %mul_153 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_144), kwargs = {})
#   %sub_44 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_76, %mul_153), kwargs = {})
#   %sub_45 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_44, %unsqueeze_141), kwargs = {})
#   %mul_154 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_147), kwargs = {})
#   %convert_element_type_78 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_154, torch.bfloat16), kwargs = {})
#   %convolution_backward_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_78, %relu, %convert_element_type_5, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   return %buf106
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 671091200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.9073486328125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ol/colafynwqrt4sonehat2yi6fn7t2s2bw2u72q7k72wsyos52etfk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_53 : Tensor "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=getitem_53]
#   %convert_element_type_79 : Tensor "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_53, torch.float32), kwargs = {})
#   return %convert_element_type_79
triton_poi_fused__to_copy_51 = async_compile.triton('triton_poi_fused__to_copy_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 737280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fp/cfp3pycfxsppzqrcosril4hwsjetvf23hnkr5pa5dp3s7cjiaklv.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_3
# Graph fragment:
#   %relu : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=relu]
#   %getitem_52 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=getitem_52]
#   %convolution_1 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %unsqueeze_150 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_150]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_9 : Tensor "b8[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_9 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %getitem_52), kwargs = {})
#   %convert_element_type_80 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %sum_20 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_80, [0, 2, 3]), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %sub_46 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %unsqueeze_150), kwargs = {})
#   %mul_156 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_80, %sub_46), kwargs = {})
#   %sum_21 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_156, [0, 2, 3]), kwargs = {})
#   return %buf111,%buf113
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201986048, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 39808
    r0_numel = 843
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 64
    x0 = (xindex % 64)
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 843*x1
        tmp1 = tl.full([1, 1], 524288, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 64*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + 64*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.full(tmp8.shape, 0, tmp8.dtype)
        tmp10 = tl.where(tmp2, tmp8, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tl.load(in_ptr2 + (x0 + 64*(((r0_2 + 843*x1) % 524288))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fv/cfvy4zbr77nh6vqnucfqsq4k5zel3mhz3bawlsuwok2hyditua7y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf111 : Tensor "f32[64, 622][1, 64]cuda:0" = PlaceHolder[target=buf111]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_9 : Tensor "b8[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_9 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %getitem_52), kwargs = {})
#   %convert_element_type_80 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %sum_20 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_80, [0, 2, 3]), kwargs = {})
#   return %sum_20
triton_red_fused_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 159744, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_53(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 622
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/5w/c5wltjflc232p7ppmzgzrfpwhl4okpafbstry32vztovorz6ac72.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_3
# Graph fragment:
#   %buf113 : Tensor "f32[64, 622][1, 64]cuda:0" = PlaceHolder[target=buf113]
#   %sum_21 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_21]
#   %squeeze_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=squeeze_1]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_9 : Tensor "b8[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_9 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %getitem_52), kwargs = {})
#   %convert_element_type_80 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %sub_46 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %unsqueeze_150), kwargs = {})
#   %mul_156 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_80, %sub_46), kwargs = {})
#   %sum_21 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_156, [0, 2, 3]), kwargs = {})
#   %mul_164 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_21, %squeeze_1), kwargs = {})
#   return %sum_21,%mul_164
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 160512, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_54(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 622
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/lw/clw4stdmhsfsirmbbzchturdxurs3qsuw5vuzd4y4uu7v6rqril5.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_3
# Graph fragment:
#   %relu : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=relu]
#   %getitem_52 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=getitem_52]
#   %convolution_1 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %unsqueeze_150 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_150]
#   %sum_21 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_21]
#   %squeeze_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=squeeze_1]
#   %sum_20 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_20]
#   %primals_7 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_7]
#   %full_default : Tensor "bf16[][]cuda:0"[num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_9 : Tensor "b8[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_9 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %getitem_52), kwargs = {})
#   %convert_element_type_80 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %sub_46 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, %unsqueeze_150), kwargs = {})
#   %mul_157 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_20, 1.9073486328125e-06), kwargs = {})
#   %unsqueeze_151 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_157, 0), kwargs = {})
#   %unsqueeze_152 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_151, 2), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, 3), kwargs = {})
#   %mul_158 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_21, 1.9073486328125e-06), kwargs = {})
#   %mul_159 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %squeeze_1), kwargs = {})
#   %mul_160 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %mul_159), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_160, 0), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, 2), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_155, 3), kwargs = {})
#   %mul_161 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %primals_7), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_161, 0), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_157, 2), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, 3), kwargs = {})
#   %mul_162 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_156), kwargs = {})
#   %sub_48 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_80, %mul_162), kwargs = {})
#   %sub_49 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_48, %unsqueeze_153), kwargs = {})
#   %mul_163 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_159), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_163, torch.bfloat16), kwargs = {})
#   %convolution_backward_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_82, %convolution, %convert_element_type_2, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]), kwargs = {})
#   return %buf116
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 335545600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.9073486328125e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ni/cnidslqb6q7lvyqj6yvxrs7vov3ql5cmfvsjemyfkx35gmtj6547.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem_56 : Tensor "bf16[64, 54, 3, 3][486, 1, 162, 54]cuda:0" = PlaceHolder[target=getitem_56]
#   %convert_element_type_83 : Tensor "f32[64, 54, 3, 3][486, 1, 162, 54]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_56, torch.float32), kwargs = {})
#   return %convert_element_type_83
triton_poi_fused__to_copy_56 = async_compile.triton('triton_poi_fused__to_copy_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 311040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_7, primals_13, primals_19, primals_25, primals_26, primals_31, primals_33, primals_38, primals_44, primals_45, primals_50, primals_56, primals_62, primals_63, convolution, convert_element_type_2, convolution_1, squeeze_1, relu, convert_element_type_5, convolution_2, squeeze_4, relu_1, getitem_4, getitem_5, convert_element_type_8, convolution_3, squeeze_7, relu_2, convert_element_type_11, convolution_4, getitem_9, rsqrt_3, add_20, convert_element_type_14, convolution_5, squeeze_13, relu_4, getitem_12, getitem_13, convert_element_type_17, convolution_6, squeeze_16, relu_5, convert_element_type_20, convolution_7, getitem_17, rsqrt_6, convert_element_type_23, convert_element_type_24, convolution_8, squeeze_22, relu_7, getitem_20, getitem_21, convert_element_type_27, convolution_9, squeeze_25, relu_8, convert_element_type_30, convolution_10, getitem_25, rsqrt_9, add_52, getitem_27, view, permute_3, unsqueeze_54, unsqueeze_66, unsqueeze_90, unsqueeze_102, unsqueeze_126, unsqueeze_138, unsqueeze_150, tangents_1 = args
        args.clear()
        assert_size_stride(primals_7, (64, ), (1, ))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_19, (128, ), (1, ))
        assert_size_stride(primals_25, (128, ), (1, ))
        assert_size_stride(primals_26, (128, ), (1, ))
        assert_size_stride(primals_31, (320, ), (1, ))
        assert_size_stride(primals_33, (1, ), (1, ))
        assert_size_stride(primals_38, (320, ), (1, ))
        assert_size_stride(primals_44, (320, ), (1, ))
        assert_size_stride(primals_45, (320, ), (1, ))
        assert_size_stride(primals_50, (512, ), (1, ))
        assert_size_stride(primals_56, (512, ), (1, ))
        assert_size_stride(primals_62, (512, ), (1, ))
        assert_size_stride(primals_63, (512, ), (1, ))
        assert_size_stride(convolution, (512, 54, 32, 32), (55296, 1, 1728, 54))
        assert_size_stride(convert_element_type_2, (64, 54, 3, 3), (486, 1, 162, 54))
        assert_size_stride(convolution_1, (512, 64, 32, 32), (65536, 1, 2048, 64))
        assert_size_stride(squeeze_1, (64, ), (1, ))
        assert_size_stride(relu, (512, 64, 32, 32), (65536, 1, 2048, 64))
        assert_size_stride(convert_element_type_5, (128, 64, 3, 3), (576, 1, 192, 64))
        assert_size_stride(convolution_2, (512, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(squeeze_4, (128, ), (1, ))
        assert_size_stride(relu_1, (512, 128, 32, 32), (131072, 1, 4096, 128))
        assert_size_stride(getitem_4, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(getitem_5, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(convert_element_type_8, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_3, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(squeeze_7, (128, ), (1, ))
        assert_size_stride(relu_2, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(convert_element_type_11, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_4, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(getitem_9, (1, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(rsqrt_3, (1, 128, 1, 1), (128, 1, 128, 128))
        assert_size_stride(add_20, (512, 128, 16, 16), (32768, 1, 2048, 128))
        assert_size_stride(convert_element_type_14, (320, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(convolution_5, (512, 320, 16, 16), (81920, 1, 5120, 320))
        assert_size_stride(squeeze_13, (320, ), (1, ))
        assert_size_stride(relu_4, (512, 320, 16, 16), (81920, 1, 5120, 320))
        assert_size_stride(getitem_12, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(getitem_13, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(convert_element_type_17, (320, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(convolution_6, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(squeeze_16, (320, ), (1, ))
        assert_size_stride(relu_5, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(convert_element_type_20, (320, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(convolution_7, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(getitem_17, (1, 320, 1, 1), (320, 1, 320, 320))
        assert_size_stride(rsqrt_6, (1, 320, 1, 1), (320, 1, 320, 320))
        assert_size_stride(convert_element_type_23, (512, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(convert_element_type_24, (512, 320, 8, 8), (20480, 1, 2560, 320))
        assert_size_stride(convolution_8, (512, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(squeeze_22, (512, ), (1, ))
        assert_size_stride(relu_7, (512, 512, 8, 8), (32768, 1, 4096, 512))
        assert_size_stride(getitem_20, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(getitem_21, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(convert_element_type_27, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(convolution_9, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(squeeze_25, (512, ), (1, ))
        assert_size_stride(relu_8, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(convert_element_type_30, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(convolution_10, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(getitem_25, (1, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(rsqrt_9, (1, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(add_52, (512, 512, 4, 4), (8192, 1, 2048, 512))
        assert_size_stride(getitem_27, (512, 512, 1, 1), (512, 1, 512, 512))
        assert_size_stride(view, (512, 512), (512, 1))
        assert_size_stride(permute_3, (10, 512), (512, 1))
        assert_size_stride(unsqueeze_54, (1, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(unsqueeze_66, (1, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(unsqueeze_90, (1, 320, 1, 1), (320, 1, 1, 1))
        assert_size_stride(unsqueeze_102, (1, 320, 1, 1), (320, 1, 1, 1))
        assert_size_stride(unsqueeze_126, (1, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(unsqueeze_138, (1, 128, 1, 1), (128, 1, 1, 1))
        assert_size_stride(unsqueeze_150, (1, 64, 1, 1), (64, 1, 1, 1))
        assert_size_stride(tangents_1, (512, 10), (10, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((512, 10), (10, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_0.run(tangents_1, buf0, 5120, stream=stream0)
            del tangents_1
            buf1 = empty_strided_cuda((10, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (10, 512), (1, 10), 0), view, out=buf1)
            del view
            buf2 = empty_strided_cuda((512, 16), (16, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_1.run(buf0, buf2, 8192, stream=stream0)
            del buf0
            buf3 = empty_strided_cuda((16, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_2.run(permute_3, buf3, 8192, stream=stream0)
            del permute_3
            buf4 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
            extern_kernels.mm(buf2, buf3, out=buf4)
            del buf2
            del buf3
            buf5 = empty_strided_cuda((10, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1, buf5, 5120, stream=stream0)
            del buf1
            buf6 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.bfloat16)
            buf7 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
            # Topologically Sorted Source Nodes: [max_pool2d_3, input_34, input_35], Original ATen: [aten.view, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_max_pool2d_with_indices_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_view_4.run(getitem_27, buf4, convolution_10, getitem_25, rsqrt_9, primals_62, primals_63, buf6, buf7, 4194304, stream=stream0)
            del buf4
            del getitem_27
            del primals_63
            buf8 = empty_strided_cuda((512, 64), (1, 512), torch.float32)
            buf10 = empty_strided_cuda((512, 64), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_5.run(buf7, convolution_10, getitem_25, buf8, buf10, 32768, 128, stream=stream0)
            buf9 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_batch_norm_backward_6.run(buf8, buf9, 512, 64, stream=stream0)
            del buf8
            buf11 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf12 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7.run(buf10, rsqrt_9, buf11, buf12, 512, 64, stream=stream0)
            del buf10
            buf13 = convolution_10; del convolution_10  # reuse
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_8.run(buf13, buf7, getitem_25, buf11, rsqrt_9, buf9, primals_62, 4194304, stream=stream0)
            del buf7
            del getitem_25
            del primals_62
            del rsqrt_9
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
            buf14 = torch.ops.aten.convolution_backward.default(buf13, relu_8, convert_element_type_30, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf13
            del convert_element_type_30
            buf15 = buf14[0]
            assert_size_stride(buf15, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf15, 16, 'torch.ops.aten.convolution_backward.default')
            buf16 = buf14[1]
            assert_size_stride(buf16, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf16, 16, 'torch.ops.aten.convolution_backward.default')
            del buf14
            buf17 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf16, buf17, 2359296, stream=stream0)
            del buf16
            buf18 = empty_strided_cuda((512, 64), (1, 512), torch.float32)
            buf20 = empty_strided_cuda((512, 64), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_10.run(relu_8, buf15, convolution_9, unsqueeze_54, buf18, buf20, 32768, 128, stream=stream0)
            buf19 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_per_fused_native_batch_norm_backward_6.run(buf18, buf19, 512, 64, stream=stream0)
            del buf18
            buf21 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf22 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_7.run(buf20, squeeze_25, buf21, buf22, 512, 64, stream=stream0)
            del buf20
            buf23 = relu_8; del relu_8  # reuse
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_11.run(buf23, buf15, convolution_9, unsqueeze_54, buf21, squeeze_25, buf19, primals_56, 4194304, stream=stream0)
            del buf15
            del convolution_9
            del primals_56
            del squeeze_25
            del unsqueeze_54
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf24 = torch.ops.aten.convolution_backward.default(buf23, getitem_20, convert_element_type_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf23
            del convert_element_type_27
            del getitem_20
            buf25 = buf24[0]
            assert_size_stride(buf25, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf25, 16, 'torch.ops.aten.convolution_backward.default')
            buf26 = buf24[1]
            assert_size_stride(buf26, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf26, 16, 'torch.ops.aten.convolution_backward.default')
            del buf24
            buf27 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf26, buf27, 2359296, stream=stream0)
            del buf26
            buf28 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_12.run(buf28, buf25, 4194304, stream=stream0)
            del buf25
            buf29 = empty_strided_cuda((512, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_13.run(getitem_21, buf28, buf29, 16777216, stream=stream0)
            del buf28
            del getitem_21
            buf30 = empty_strided_cuda((512, 128), (1, 512), torch.float32)
            buf32 = empty_strided_cuda((512, 128), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_14.run(relu_7, buf29, convolution_8, unsqueeze_66, buf30, buf32, 65536, 256, stream=stream0)
            buf31 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_threshold_backward_15.run(buf30, buf31, 512, 128, stream=stream0)
            buf33 = empty_strided_cuda((512, ), (1, ), torch.float32)
            buf34 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_16.run(buf32, squeeze_22, buf33, buf34, 512, 128, stream=stream0)
            buf35 = relu_7; del relu_7  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_17.run(buf35, buf29, convolution_8, unsqueeze_66, buf33, squeeze_22, buf31, primals_50, 16777216, stream=stream0)
            del buf29
            del buf33
            del convolution_8
            del primals_50
            del squeeze_22
            del unsqueeze_66
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf36 = torch.ops.aten.convolution_backward.default(buf35, convert_element_type_24, convert_element_type_23, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf35
            del convert_element_type_23
            del convert_element_type_24
            buf37 = buf36[0]
            assert_size_stride(buf37, (512, 320, 8, 8), (20480, 1, 2560, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf37, 16, 'torch.ops.aten.convolution_backward.default')
            buf38 = buf36[1]
            assert_size_stride(buf38, (512, 320, 3, 3), (2880, 1, 960, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf38, 16, 'torch.ops.aten.convolution_backward.default')
            del buf36
            buf39 = empty_strided_cuda((512, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_18.run(buf38, buf39, 1474560, stream=stream0)
            del buf38
            buf40 = empty_strided_cuda((512, 320, 8, 8), (20480, 1, 2560, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(convolution_7, getitem_17, rsqrt_6, primals_44, primals_45, buf40, 10485760, stream=stream0)
            del primals_45
            buf42 = empty_strided_cuda((1, 1, 1, 1, 320, 256), (81920, 81920, 81920, 81920, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_mul_sum_20.run(buf37, buf40, buf42, 81920, 128, stream=stream0)
            buf43 = empty_strided_cuda((1, 1, 1, 1, 320), (320, 320, 320, 320, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_mul_sum_21.run(buf42, buf43, 320, 256, stream=stream0)
            buf44 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_mul_sum_22.run(buf43, buf44, 1, 320, stream=stream0)
            buf45 = reinterpret_tensor(buf42, (320, 256), (1, 320), 0); del buf42  # reuse
            buf47 = empty_strided_cuda((320, 256), (1, 320), torch.float32)
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_23.run(buf40, buf37, primals_33, convolution_7, getitem_17, buf45, buf47, 81920, 128, stream=stream0)
            buf46 = reinterpret_tensor(buf43, (320, ), (1, ), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24.run(buf45, buf46, 320, 256, stream=stream0)
            buf48 = empty_strided_cuda((320, ), (1, ), torch.float32)
            buf50 = empty_strided_cuda((320, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25.run(buf47, rsqrt_6, buf48, buf50, 320, 256, stream=stream0)
            buf51 = empty_strided_cuda((512, 320, 8, 8), (20480, 1, 2560, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.threshold_backward, aten._to_copy, aten.mul, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_backward_mul_native_batch_norm_backward_threshold_backward_26.run(buf40, buf37, primals_33, convolution_7, getitem_17, buf48, rsqrt_6, buf46, primals_44, buf51, 10485760, stream=stream0)
            del buf40
            del convolution_7
            del getitem_17
            del primals_33
            del primals_44
            del rsqrt_6
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
            buf52 = torch.ops.aten.convolution_backward.default(buf51, relu_5, convert_element_type_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf51
            del convert_element_type_20
            buf53 = buf52[0]
            assert_size_stride(buf53, (512, 320, 8, 8), (20480, 1, 2560, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf53, 16, 'torch.ops.aten.convolution_backward.default')
            buf54 = buf52[1]
            assert_size_stride(buf54, (320, 320, 3, 3), (2880, 1, 960, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf54, 16, 'torch.ops.aten.convolution_backward.default')
            del buf52
            buf55 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_27.run(buf54, buf55, 921600, stream=stream0)
            del buf54
            buf56 = buf47; del buf47  # reuse
            buf58 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_28.run(relu_5, buf53, convolution_6, unsqueeze_90, buf56, buf58, 81920, 128, stream=stream0)
            buf57 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_native_batch_norm_backward_threshold_backward_24.run(buf56, buf57, 320, 256, stream=stream0)
            del buf56
            buf59 = empty_strided_cuda((320, ), (1, ), torch.float32)
            buf60 = empty_strided_cuda((320, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional__to_copy_mul_native_batch_norm_backward_threshold_backward_25.run(buf58, squeeze_16, buf59, buf60, 320, 256, stream=stream0)
            del buf58
            buf61 = relu_5; del relu_5  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_29.run(buf61, buf53, convolution_6, unsqueeze_90, buf59, squeeze_16, buf57, primals_38, 10485760, stream=stream0)
            del buf53
            del convolution_6
            del primals_38
            del squeeze_16
            del unsqueeze_90
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf62 = torch.ops.aten.convolution_backward.default(buf61, getitem_12, convert_element_type_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf61
            del convert_element_type_17
            del getitem_12
            buf63 = buf62[0]
            assert_size_stride(buf63, (512, 320, 8, 8), (20480, 1, 2560, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf63, 16, 'torch.ops.aten.convolution_backward.default')
            buf64 = buf62[1]
            assert_size_stride(buf64, (320, 320, 3, 3), (2880, 1, 960, 320), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf64, 16, 'torch.ops.aten.convolution_backward.default')
            del buf62
            buf65 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_27.run(buf64, buf65, 921600, stream=stream0)
            del buf64
            buf66 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_30.run(buf66, buf63, 10485760, stream=stream0)
            del buf63
            buf67 = empty_strided_cuda((512, 320, 16, 16), (81920, 1, 5120, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._to_copy, aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_31.run(getitem_13, buf66, buf67, 41943040, stream=stream0)
            del buf66
            del getitem_13
            buf68 = empty_strided_cuda((320, 208), (1, 320), torch.float32)
            buf70 = empty_strided_cuda((320, 208), (1, 320), torch.float32)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_32.run(relu_4, buf67, convolution_5, unsqueeze_102, buf68, buf70, 66560, 631, stream=stream0)
            buf69 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_threshold_backward_33.run(buf68, buf69, 320, 208, stream=stream0)
            del buf68
            buf71 = empty_strided_cuda((320, ), (1, ), torch.float32)
            buf72 = empty_strided_cuda((320, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_34.run(buf70, squeeze_13, buf71, buf72, 320, 208, stream=stream0)
            del buf70
            buf73 = relu_4; del relu_4  # reuse
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_35.run(buf73, buf67, convolution_5, unsqueeze_102, buf71, squeeze_13, buf69, primals_31, 41943040, stream=stream0)
            del buf67
            del buf71
            del convolution_5
            del primals_31
            del squeeze_13
            del unsqueeze_102
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf74 = torch.ops.aten.convolution_backward.default(buf73, add_20, convert_element_type_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del add_20
            del buf73
            del convert_element_type_14
            buf75 = buf74[0]
            assert_size_stride(buf75, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf75, 16, 'torch.ops.aten.convolution_backward.default')
            buf76 = buf74[1]
            assert_size_stride(buf76, (320, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf76, 16, 'torch.ops.aten.convolution_backward.default')
            del buf74
            buf77 = empty_strided_cuda((320, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_36.run(buf76, buf77, 368640, stream=stream0)
            del buf76
            buf78 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.relu, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_threshold_backward_37.run(convolution_4, getitem_9, rsqrt_3, primals_25, primals_26, buf75, buf78, 16777216, stream=stream0)
            del primals_26
            buf79 = reinterpret_tensor(buf32, (128, 512), (1, 128), 0); del buf32  # reuse
            buf81 = reinterpret_tensor(buf30, (128, 512), (1, 128), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_38.run(buf78, convolution_4, getitem_9, buf79, buf81, 65536, 256, stream=stream0)
            buf80 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_39.run(buf79, buf80, 128, 512, stream=stream0)
            buf82 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf83 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40.run(buf81, rsqrt_3, buf82, buf83, 128, 512, stream=stream0)
            buf84 = convolution_4; del convolution_4  # reuse
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41.run(buf84, buf78, getitem_9, buf82, rsqrt_3, buf80, primals_25, 16777216, stream=stream0)
            del buf78
            del getitem_9
            del primals_25
            del rsqrt_3
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
            buf85 = torch.ops.aten.convolution_backward.default(buf84, relu_2, convert_element_type_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf84
            del convert_element_type_11
            buf86 = buf85[0]
            assert_size_stride(buf86, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf86, 16, 'torch.ops.aten.convolution_backward.default')
            buf87 = buf85[1]
            assert_size_stride(buf87, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf87, 16, 'torch.ops.aten.convolution_backward.default')
            del buf85
            buf88 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(buf87, buf88, 147456, stream=stream0)
            del buf87
            buf89 = buf81; del buf81  # reuse
            buf91 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_43.run(relu_2, buf86, convolution_3, unsqueeze_126, buf89, buf91, 65536, 256, stream=stream0)
            buf90 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_39.run(buf89, buf90, 128, 512, stream=stream0)
            del buf89
            buf92 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf93 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40.run(buf91, squeeze_7, buf92, buf93, 128, 512, stream=stream0)
            del buf91
            buf94 = relu_2; del relu_2  # reuse
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_44.run(buf94, buf86, convolution_3, unsqueeze_126, buf92, squeeze_7, buf90, primals_19, 16777216, stream=stream0)
            del buf86
            del convolution_3
            del primals_19
            del squeeze_7
            del unsqueeze_126
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf95 = torch.ops.aten.convolution_backward.default(buf94, getitem_4, convert_element_type_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf94
            del convert_element_type_8
            del getitem_4
            buf96 = buf95[0]
            assert_size_stride(buf96, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf96, 16, 'torch.ops.aten.convolution_backward.default')
            buf97 = buf95[1]
            assert_size_stride(buf97, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf97, 16, 'torch.ops.aten.convolution_backward.default')
            del buf95
            buf98 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_42.run(buf97, buf98, 147456, stream=stream0)
            del buf97
            buf99 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_45.run(buf99, buf96, 16777216, stream=stream0)
            del buf96
            buf100 = empty_strided_cuda((512, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_46.run(getitem_5, buf99, buf100, 67108864, stream=stream0)
            del buf99
            del getitem_5
            buf101 = empty_strided_cuda((128, 622), (1, 128), torch.float32)
            buf103 = empty_strided_cuda((128, 622), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47.run(relu_1, buf100, convolution_2, unsqueeze_138, buf101, buf103, 79616, 843, stream=stream0)
            buf102 = buf92; del buf92  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_threshold_backward_48.run(buf101, buf102, 128, 622, stream=stream0)
            del buf101
            buf104 = empty_strided_cuda((128, ), (1, ), torch.float32)
            buf105 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_49.run(buf103, squeeze_4, buf104, buf105, 128, 622, stream=stream0)
            del buf103
            buf106 = relu_1; del relu_1  # reuse
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(buf106, buf100, convolution_2, unsqueeze_138, buf104, squeeze_4, buf102, primals_13, 67108864, stream=stream0)
            del buf100
            del buf104
            del convolution_2
            del primals_13
            del squeeze_4
            del unsqueeze_138
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf107 = torch.ops.aten.convolution_backward.default(buf106, relu, convert_element_type_5, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
            del buf106
            del convert_element_type_5
            buf108 = buf107[0]
            assert_size_stride(buf108, (512, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf108, 16, 'torch.ops.aten.convolution_backward.default')
            buf109 = buf107[1]
            assert_size_stride(buf109, (128, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf109, 16, 'torch.ops.aten.convolution_backward.default')
            del buf107
            buf110 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_51.run(buf109, buf110, 73728, stream=stream0)
            del buf109
            buf111 = empty_strided_cuda((64, 622), (1, 64), torch.float32)
            buf113 = empty_strided_cuda((64, 622), (1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_52.run(relu, buf108, convolution_1, unsqueeze_150, buf111, buf113, 39808, 843, stream=stream0)
            buf112 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_threshold_backward_53.run(buf111, buf112, 64, 622, stream=stream0)
            del buf111
            buf114 = empty_strided_cuda((64, ), (1, ), torch.float32)
            buf115 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_54.run(buf113, squeeze_1, buf114, buf115, 64, 622, stream=stream0)
            del buf113
            buf116 = relu; del relu  # reuse
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_55.run(buf116, buf108, convolution_1, unsqueeze_150, buf114, squeeze_1, buf112, primals_7, 33554432, stream=stream0)
            del buf108
            del buf114
            del convolution_1
            del primals_7
            del squeeze_1
            del unsqueeze_150
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
            buf117 = torch.ops.aten.convolution_backward.default(buf116, convolution, convert_element_type_2, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
            del buf116
            del convert_element_type_2
            del convolution
            buf118 = buf117[1]
            assert_size_stride(buf118, (64, 54, 3, 3), (486, 1, 162, 54), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf118, 16, 'torch.ops.aten.convolution_backward.default')
            del buf117
            buf119 = empty_strided_cuda((64, 54, 3, 3), (486, 1, 162, 54), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_56.run(buf118, buf119, 31104, stream=stream0)
            del buf118
        return (None, None, buf119, None, None, None, buf115, buf112, buf110, None, None, None, buf105, buf102, buf98, None, None, None, buf93, buf90, buf88, None, None, None, buf83, buf80, buf77, None, None, None, buf72, buf69, reinterpret_tensor(buf44, (1, ), (1, ), 0), buf65, None, None, None, buf60, buf57, buf55, None, None, None, buf50, buf46, buf39, None, None, None, buf34, buf31, buf27, None, None, None, buf22, buf19, buf17, None, None, None, buf12, buf9, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((512, 54, 32, 32), (55296, 1, 1728, 54), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_2 = rand_strided((64, 54, 3, 3), (486, 1, 162, 54), device='cuda:0', dtype=torch.bfloat16)
    convolution_1 = rand_strided((512, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((512, 64, 32, 32), (65536, 1, 2048, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_5 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_2 = rand_strided((512, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((512, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.int8)
    convert_element_type_8 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_3 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_11 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_4 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    add_20 = rand_strided((512, 128, 16, 16), (32768, 1, 2048, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_14 = rand_strided((320, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_5 = rand_strided((512, 320, 16, 16), (81920, 1, 5120, 320), device='cuda:0', dtype=torch.bfloat16)
    squeeze_13 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((512, 320, 16, 16), (81920, 1, 5120, 320), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.int8)
    convert_element_type_17 = rand_strided((320, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.bfloat16)
    convolution_6 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bfloat16)
    squeeze_16 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_20 = rand_strided((320, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.bfloat16)
    convolution_7 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((1, 320, 1, 1), (320, 1, 320, 320), device='cuda:0', dtype=torch.float32)
    convert_element_type_23 = rand_strided((512, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_24 = rand_strided((512, 320, 8, 8), (20480, 1, 2560, 320), device='cuda:0', dtype=torch.bfloat16)
    convolution_8 = rand_strided((512, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((512, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.int8)
    convert_element_type_27 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_9 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_30 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_10 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    add_52 = rand_strided((512, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_27 = rand_strided((512, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.int8)
    view = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_3 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    unsqueeze_54 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_66 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_90 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_102 = rand_strided((1, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_126 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_138 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_150 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((512, 10), (10, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_7, primals_13, primals_19, primals_25, primals_26, primals_31, primals_33, primals_38, primals_44, primals_45, primals_50, primals_56, primals_62, primals_63, convolution, convert_element_type_2, convolution_1, squeeze_1, relu, convert_element_type_5, convolution_2, squeeze_4, relu_1, getitem_4, getitem_5, convert_element_type_8, convolution_3, squeeze_7, relu_2, convert_element_type_11, convolution_4, getitem_9, rsqrt_3, add_20, convert_element_type_14, convolution_5, squeeze_13, relu_4, getitem_12, getitem_13, convert_element_type_17, convolution_6, squeeze_16, relu_5, convert_element_type_20, convolution_7, getitem_17, rsqrt_6, convert_element_type_23, convert_element_type_24, convolution_8, squeeze_22, relu_7, getitem_20, getitem_21, convert_element_type_27, convolution_9, squeeze_25, relu_8, convert_element_type_30, convolution_10, getitem_25, rsqrt_9, add_52, getitem_27, view, permute_3, unsqueeze_54, unsqueeze_66, unsqueeze_90, unsqueeze_102, unsqueeze_126, unsqueeze_138, unsqueeze_150, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
