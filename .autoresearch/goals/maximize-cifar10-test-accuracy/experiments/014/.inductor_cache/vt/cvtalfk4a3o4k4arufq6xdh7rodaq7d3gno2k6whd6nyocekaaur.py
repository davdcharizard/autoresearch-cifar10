# AOT ID: ['2_forward']
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


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/eb/cebcfwtipsj6coyxs25kmyfrgypescpqssnvpihky7bmbmkekcjm.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_1
# Graph fragment:
#   %primals_2 : Tensor "f32[512, 3, 32, 32][3072, 1, 96, 3]cuda:0" = PlaceHolder[target=primals_2]
#   %convert_element_type_1 : Tensor "bf16[512, 3, 32, 32][3072, 1, 96, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   return %convert_element_type_1
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/3d/c3d52advnosfcqzspx4pntwhiigvz2dmvfoqnjid6rbkqpjx6xjt.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type
# Graph fragment:
#   %primals_1 : Tensor "f32[54, 3, 3, 3][27, 1, 9, 3]cuda:0" = PlaceHolder[target=primals_1]
#   %convert_element_type : Tensor "bf16[54, 3, 3, 3][27, 1, 9, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1458
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fr/cfrw7fpjnuqbfkmyjh6s4lbmcl6d5omjpiaj5kjxaknafdlv4erv.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_2
# Graph fragment:
#   %primals_3 : Tensor "f32[64, 54, 3, 3][486, 1, 162, 54]cuda:0" = PlaceHolder[target=primals_3]
#   %convert_element_type_2 : Tensor "bf16[64, 54, 3, 3][486, 1, 162, 54]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 248832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/a3/ca37hpodkhe2t46sn56udbkulfsfiljsq6fpleptcadwmrywwy77.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_3, var_mean
# Graph fragment:
#   %convolution_1 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf5,%buf6,%buf7
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 68071680, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
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
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 0.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 1.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp14 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp13, tmp14, tmp15
        )
        tmp16_mean = tl.where(r0_mask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask & xmask, tmp16_weight_next, tmp16_weight)
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp17[:, None]
    tmp20 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/4c/c4cbdmzkpkoesjfhpwq5u6mzh43pt5di7pnkefksch3vfdphwsft.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_3, var_mean
# Graph fragment:
#   %buf5 : Tensor "f32[1, 64, 1, 1, 622][39808, 1, 39808, 39808, 64]cuda:0" = PlaceHolder[target=buf5]
#   %buf6 : Tensor "f32[1, 64, 1, 1, 622][39808, 1, 39808, 39808, 64]cuda:0" = PlaceHolder[target=buf6]
#   %buf7 : Tensor "f32[1, 64, 1, 1, 622][39808, 1, 39808, 39808, 64]cuda:0" = PlaceHolder[target=buf7]
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf8,%buf9,%buf10
triton_red_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 487680, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 320
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 125*x1
        tmp1 = tl.full([1, 1], 622, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 8000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_combine(
            tmp9_mean, tmp9_m2, tmp9_weight,
            tmp6, tmp7, tmp8
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/d2/cd2kxsvtz4oecsmtu6opqygvrocjzksq5deb5fdh2tyo6w4el4vo.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_2 => add_1, add_2, add_3, convert_element_type_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze, squeeze_2, var_mean
# Graph fragment:
#   %buf8 : Tensor "f32[1, 64, 1, 1, 5][320, 1, 320, 320, 64]cuda:0" = PlaceHolder[target=buf8]
#   %buf9 : Tensor "f32[1, 64, 1, 1, 5][320, 1, 320, 320, 64]cuda:0" = PlaceHolder[target=buf9]
#   %buf10 : Tensor "f32[1, 64, 1, 1, 5][320, 1, 320, 320, 64]cuda:0" = PlaceHolder[target=buf10]
#   %buf12 : Tensor "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0" = PlaceHolder[target=buf12]
#   %getitem_1 : Tensor "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0" = PlaceHolder[target=getitem_1]
#   %copy__1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=copy__1]
#   %add_2 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=add_2]
#   %copy__2 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=copy__2]
#   %add_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=add_3]
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %squeeze : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_1, [0, 2, 3]), kwargs = {})
#   %mul_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %squeeze_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem, [0, 2, 3]), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.0000019073522708), kwargs = {})
#   %mul_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.9), kwargs = {})
#   %add_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %copy__1 : Tensor "f32[64][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_2), kwargs = {})
#   %copy__2 : Tensor "f32[64][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %add_3), kwargs = {})
#   return %getitem_1,%buf12,%rsqrt,%add_2,%buf139,%add_3,%buf142
triton_per_fused__native_batch_norm_legit_functional_copy__5 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6912, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 524288.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000019073522708
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ww/cwwewccjfhey2xjoja7zltounwthtwfbq5cyrsvqyigtlevhxboy.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, add_4, convert_element_type_3, convert_element_type_4, mul, mul_6, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, var_mean
#   input_3 => relu
# Graph fragment:
#   %convolution_1 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %getitem_1 : Tensor "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf12 : Tensor "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0" = PlaceHolder[target=buf12]
#   %primals_7 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_8 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_8]
#   %convert_element_type_3 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_3, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_7, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %mul_6 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_8, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %add_4 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %relu : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_4,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_functional_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 524288.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/j7/cj7cgsxxas5ifdmmhrliqmqkl6mpvqcirhtm5d4lzo2k3bthtv6v.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_5
# Graph fragment:
#   %primals_9 : Tensor "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=primals_9]
#   %convert_element_type_5 : Tensor "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.bfloat16), kwargs = {})
#   return %convert_element_type_5
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/yk/cykanle6x4p3gbeda2vpb7mtf3tck2yzidrjdup4vtn2wxtkjqgz.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_6, var_mean_1
# Graph fragment:
#   %convolution_2 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_6, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf18,%buf19,%buf20
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 136143360, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
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
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 0.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 1.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp14 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp13, tmp14, tmp15
        )
        tmp16_mean = tl.where(r0_mask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask & xmask, tmp16_weight_next, tmp16_weight)
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp17[:, None]
    tmp20 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/zs/czsm3fxz4boxi7izezabczhzcdjg3kejlcgxu2rxinm3sofb4ibh.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_6, var_mean_1
# Graph fragment:
#   %buf18 : Tensor "f32[1, 128, 1, 1, 622][79616, 1, 79616, 79616, 128]cuda:0" = PlaceHolder[target=buf18]
#   %buf19 : Tensor "f32[1, 128, 1, 1, 622][79616, 1, 79616, 79616, 128]cuda:0" = PlaceHolder[target=buf19]
#   %buf20 : Tensor "f32[1, 128, 1, 1, 622][79616, 1, 79616, 79616, 128]cuda:0" = PlaceHolder[target=buf20]
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_6, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf21,%buf22,%buf23
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 975360, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 128
    x0 = (xindex % 128)
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 125*x1
        tmp1 = tl.full([1, 1], 622, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 16000*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_combine(
            tmp9_mean, tmp9_m2, tmp9_weight,
            tmp6, tmp7, tmp8
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/4x/c4xynsretysw64ayhrfyi5dhyzc3lhjf6xbwps7rt54pirny6kil.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_5 => add_6, add_7, add_8, convert_element_type_6, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_3, squeeze_5, var_mean_1
# Graph fragment:
#   %buf21 : Tensor "f32[1, 128, 1, 1, 5][640, 1, 640, 640, 128]cuda:0" = PlaceHolder[target=buf21]
#   %buf22 : Tensor "f32[1, 128, 1, 1, 5][640, 1, 640, 640, 128]cuda:0" = PlaceHolder[target=buf22]
#   %buf23 : Tensor "f32[1, 128, 1, 1, 5][640, 1, 640, 640, 128]cuda:0" = PlaceHolder[target=buf23]
#   %buf25 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=buf25]
#   %getitem_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_3]
#   %copy__4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__4]
#   %add_7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_7]
#   %copy__5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__5]
#   %add_8 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_8]
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_6, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %squeeze_3 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_3, [0, 2, 3]), kwargs = {})
#   %mul_8 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 0.1), kwargs = {})
#   %mul_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, 0.9), kwargs = {})
#   %add_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %squeeze_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_2, [0, 2, 3]), kwargs = {})
#   %mul_10 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0000019073522708), kwargs = {})
#   %mul_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.1), kwargs = {})
#   %mul_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.9), kwargs = {})
#   %add_8 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %copy__4 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_11, %add_7), kwargs = {})
#   %copy__5 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %add_8), kwargs = {})
#   return %getitem_3,%buf25,%rsqrt_1,%add_7,%buf147,%add_8,%buf150
triton_per_fused__native_batch_norm_legit_functional_copy__10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13824, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 524288.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000019073522708
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fc/cfc4kielsvnwwinljslqsjoxf5nmjoozaeog45hnr2pkaaibq6bv.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_6, add_9, convert_element_type_6, convert_element_type_7, mul_13, mul_7, rsqrt_1, sub_1, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, var_mean_1
#   input_6 => relu_1
# Graph fragment:
#   %convolution_2 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %getitem_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf25 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=buf25]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_14 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_14]
#   %convert_element_type_6 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_6, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_1 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %getitem_3), kwargs = {})
#   %mul_7 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_13, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_13 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_14, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_9 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_7), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_7,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402655232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 524288.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/p7/cp7oldq6lwnmw5kudg5am4eacf7roddjhh26meys5r5oncgqovqo.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_7 => _low_memory_max_pool_with_offsets, getitem_4, getitem_5
# Graph fragment:
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=relu_1]
#   %_low_memory_max_pool_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
#   %getitem_5 : Tensor "i8[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 1), kwargs = {})
#   return %getitem_4,%getitem_5
triton_poi_fused_max_pool2d_with_indices_12 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 234881024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_12(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x1 + 8192*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 8192*x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 256*x1 + 8192*x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4224 + x0 + 256*x1 + 8192*x2), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 > tmp1
    tmp8 = tmp0 == tmp1
    tmp9 = tmp0 != tmp0
    tmp10 = tmp1 != tmp1
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp0, tmp1)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp22 = tmp20 > tmp3
    tmp23 = tmp20 == tmp3
    tmp24 = tmp20 != tmp20
    tmp25 = tmp3 != tmp3
    tmp26 = tmp24 > tmp25
    tmp27 = tmp22 | tmp26
    tmp28 = tmp24 & tmp25
    tmp29 = tmp23 | tmp28
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp21 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tmp27 | tmp32
    tmp34 = tl.where(tmp33, tmp20, tmp3)
    tmp35 = tl.where(tmp33, tmp21, tmp30)
    tmp36 = tmp34 > tmp5
    tmp37 = tmp34 == tmp5
    tmp38 = tmp34 != tmp34
    tmp39 = tmp5 != tmp5
    tmp40 = tmp38 > tmp39
    tmp41 = tmp36 | tmp40
    tmp42 = tmp38 & tmp39
    tmp43 = tmp37 | tmp42
    tmp44 = tl.full([1], 3, tl.int64)
    tmp45 = tmp35 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp41 | tmp46
    tmp48 = tl.where(tmp47, tmp34, tmp5)
    tmp49 = tl.where(tmp47, tmp35, tmp44)
    tmp50 = tmp49.to(tl.int8)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/l2/cl23ynfyg7a2n7kuieffmgsvaltu2ka5sxjjdsevb25e6a6uy6hd.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_8 => convert_element_type_8
# Graph fragment:
#   %primals_15 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=primals_15]
#   %convert_element_type_8 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_15, torch.bfloat16), kwargs = {})
#   return %convert_element_type_8
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/by/cbyssthgfeawyr6xk7h3sp7k6tauo2dk253ookgqni5dz224kkst.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_9, var_mean_2
# Graph fragment:
#   %convolution_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_3]
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_9, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf33,%buf34,%buf35
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 35127296, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_14(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
    tl.store(out_ptr2 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/pm/cpmvnh4qcui2utxp2w5noqtinb64pb45xvsmflucb3wqa2txr54r.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_9 => convert_element_type_9, var_mean_2
# Graph fragment:
#   %buf33 : Tensor "f32[1, 128, 1, 1, 512][65536, 1, 65536, 65536, 128]cuda:0" = PlaceHolder[target=buf33]
#   %buf34 : Tensor "f32[1, 128, 1, 1, 512][65536, 1, 65536, 65536, 128]cuda:0" = PlaceHolder[target=buf34]
#   %buf35 : Tensor "f32[1, 128, 1, 1, 512][65536, 1, 65536, 65536, 128]cuda:0" = PlaceHolder[target=buf35]
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_9, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf36,%buf37,%buf38
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(out_ptr2 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/sl/cslz3f3ymakwsymualo473zp4rdtdm26amasp3ginh2ngfysfhce.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_9 => add_11, add_12, add_13, convert_element_type_9, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_6, squeeze_8, var_mean_2
# Graph fragment:
#   %buf36 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf36]
#   %buf37 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf37]
#   %buf38 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf38]
#   %buf40 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=buf40]
#   %getitem_7 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_7]
#   %copy__7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__7]
#   %add_12 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_12]
#   %copy__8 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__8]
#   %add_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_13]
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_9, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %squeeze_6 : Tensor "f32[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_7, [0, 2, 3]), kwargs = {})
#   %mul_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_6, 0.1), kwargs = {})
#   %mul_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_17, 0.9), kwargs = {})
#   %add_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %squeeze_8 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_6, [0, 2, 3]), kwargs = {})
#   %mul_17 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_8, 1.0000076294527394), kwargs = {})
#   %mul_18 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, 0.1), kwargs = {})
#   %mul_19 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, 0.9), kwargs = {})
#   %add_13 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %mul_19), kwargs = {})
#   %copy__7 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_17, %add_12), kwargs = {})
#   %copy__8 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_18, %add_13), kwargs = {})
#   return %getitem_7,%buf40,%rsqrt_2,%add_12,%buf155,%add_13,%buf158
triton_per_fused__native_batch_norm_legit_functional_copy__16 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
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
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ap/capz7n6jwcim7eaimgsqmbhvrtph4dbqy3fingjamormy343bz5p.py
# Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_2
#   input_9 => add_11, add_14, convert_element_type_10, convert_element_type_9, mul_14, mul_20, rsqrt_2, sub_2, unsqueeze_10, unsqueeze_11, unsqueeze_8, unsqueeze_9, var_mean_2
# Graph fragment:
#   %convolution_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_3]
#   %getitem_7 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf40 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=buf40]
#   %primals_19 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_19]
#   %primals_20 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_20]
#   %convert_element_type_9 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_9, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_2 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %getitem_7), kwargs = {})
#   %mul_14 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_19, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %mul_20 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_9), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_20, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %add_14 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_11), kwargs = {})
#   %convert_element_type_10 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.bfloat16), kwargs = {})
#   %relu_2 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_10,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100665344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 131072.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/o2/co2wbr3hxoefssc6ultffud6cbespxndzfous2h5ttn7zwhqkmcl.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_12 => add_16, add_17, add_18, convert_element_type_12, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_11, squeeze_9, var_mean_3
# Graph fragment:
#   %buf49 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf49]
#   %buf50 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf50]
#   %buf51 : Tensor "f32[1, 128, 1, 1, 4][512, 1, 512, 512, 128]cuda:0" = PlaceHolder[target=buf51]
#   %buf53 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=buf53]
#   %getitem_9 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %copy__10 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__10]
#   %add_17 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_17]
#   %copy__11 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=copy__11]
#   %add_18 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=add_18]
#   %convert_element_type_12 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_12, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %squeeze_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2, 3]), kwargs = {})
#   %mul_22 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_9, 0.1), kwargs = {})
#   %mul_23 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_23, 0.9), kwargs = {})
#   %add_17 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %mul_23), kwargs = {})
#   %squeeze_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_8, [0, 2, 3]), kwargs = {})
#   %mul_24 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_11, 1.0000076294527394), kwargs = {})
#   %mul_25 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, 0.1), kwargs = {})
#   %mul_26 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_24, 0.9), kwargs = {})
#   %add_18 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %mul_26), kwargs = {})
#   %copy__10 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_23, %add_17), kwargs = {})
#   %copy__11 : Tensor "f32[128][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_24, %add_18), kwargs = {})
#   return %getitem_9,%buf53,%rsqrt_3,%add_17,%buf163,%add_18,%buf166
triton_per_fused__native_batch_norm_legit_functional_copy__18 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__18', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11264, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
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
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/w5/cw5t34kfmhyncusbd22uoupxrtlmslgc2kmyeysyvcistogskwyn.py
# Topologically Sorted Source Nodes: [input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_12 => add_19, convert_element_type_13, mul_21, mul_27, sub_3, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15
#   input_13 => relu_3
#   input_14 => add_20
# Graph fragment:
#   %getitem_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem_4]
#   %convolution_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_4]
#   %getitem_9 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_3 : Tensor "f32[1, 128, 1, 1][128, 1, 128, 128]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_25 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_25]
#   %primals_26 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_26]
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
#   %add_20 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %relu_3), kwargs = {})
#   return %add_20
triton_poi_fused__native_batch_norm_legit_functional_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134219776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/kl/cklpcdkbfn33o2nfpgcvani2jybru342j43stlhn3xidisgahkx6.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_14
# Graph fragment:
#   %primals_27 : Tensor "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=primals_27]
#   %convert_element_type_14 : Tensor "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_27, torch.bfloat16), kwargs = {})
#   return %convert_element_type_14
triton_poi_fused__to_copy_20 = async_compile.triton('triton_poi_fused__to_copy_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ai/caisbyjusgaxk7t364jo5l5ec2lr7pto7e7padvzknezexaax5de.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_15, var_mean_4
# Graph fragment:
#   %convolution_5 : Tensor "bf16[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %convert_element_type_15 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf59,%buf60,%buf61
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 69043200, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_21(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 79360
    r0_numel = 423
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp16_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp16_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 423*x1
        tmp1 = tl.full([1, 1], 131072, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 256*(((r0_2 + 423*x1) % 131072))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 0.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 1.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp14 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp15 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp16_mean_next, tmp16_m2_next, tmp16_weight_next = triton_helpers.welford_combine(
            tmp16_mean, tmp16_m2, tmp16_weight,
            tmp13, tmp14, tmp15
        )
        tmp16_mean = tl.where(r0_mask & xmask, tmp16_mean_next, tmp16_mean)
        tmp16_m2 = tl.where(r0_mask & xmask, tmp16_m2_next, tmp16_m2)
        tmp16_weight = tl.where(r0_mask & xmask, tmp16_weight_next, tmp16_weight)
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp16_mean, tmp16_m2, tmp16_weight, 1)
    tmp16 = tmp17[:, None]
    tmp20 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/gv/cgvnooe5lsja2eorlstys5djo6cvkl4ua57vfym6glh5ov7ra3mr.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_16 => convert_element_type_15, var_mean_4
# Graph fragment:
#   %buf59 : Tensor "f32[1, 256, 1, 1, 310][79360, 1, 79360, 79360, 256]cuda:0" = PlaceHolder[target=buf59]
#   %buf60 : Tensor "f32[1, 256, 1, 1, 310][79360, 1, 79360, 79360, 256]cuda:0" = PlaceHolder[target=buf60]
#   %buf61 : Tensor "f32[1, 256, 1, 1, 310][79360, 1, 79360, 79360, 256]cuda:0" = PlaceHolder[target=buf61]
#   %convert_element_type_15 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf62,%buf63,%buf64
triton_red_fused__native_batch_norm_legit_functional_22 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 976896, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 104
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 256
    x0 = (xindex % 256)
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 104*x1
        tmp1 = tl.full([1, 1], 310, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 26624*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 26624*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 26624*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_combine(
            tmp9_mean, tmp9_m2, tmp9_weight,
            tmp6, tmp7, tmp8
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/cr/ccrtvg7r2xzd5xjn2tt2lhpxfaicfh5pn6ucu3gvlfunnwvw5wt2.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_16 => add_22, add_23, add_24, convert_element_type_15, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_12, squeeze_14, var_mean_4
# Graph fragment:
#   %buf62 : Tensor "f32[1, 256, 1, 1, 3][768, 1, 768, 768, 256]cuda:0" = PlaceHolder[target=buf62]
#   %buf63 : Tensor "f32[1, 256, 1, 1, 3][768, 1, 768, 768, 256]cuda:0" = PlaceHolder[target=buf63]
#   %buf64 : Tensor "f32[1, 256, 1, 1, 3][768, 1, 768, 768, 256]cuda:0" = PlaceHolder[target=buf64]
#   %buf66 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf66]
#   %getitem_11 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_11]
#   %copy__13 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__13]
#   %add_23 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_23]
#   %copy__14 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__14]
#   %add_24 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_24]
#   %convert_element_type_15 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_22 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %squeeze_12 : Tensor "f32[256][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_11, [0, 2, 3]), kwargs = {})
#   %mul_29 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_12, 0.1), kwargs = {})
#   %mul_30 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_29, 0.9), kwargs = {})
#   %add_23 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %mul_30), kwargs = {})
#   %squeeze_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_10, [0, 2, 3]), kwargs = {})
#   %mul_31 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_14, 1.0000076294527394), kwargs = {})
#   %mul_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, 0.1), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_30, 0.9), kwargs = {})
#   %add_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %mul_33), kwargs = {})
#   %copy__13 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_29, %add_23), kwargs = {})
#   %copy__14 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_30, %add_24), kwargs = {})
#   return %getitem_11,%buf66,%rsqrt_4,%add_23,%buf171,%add_24,%buf174
triton_per_fused__native_batch_norm_legit_functional_copy__23 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__23', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 21504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/d2/cd2q25cjl3as5vcmlocd3koesbidrg5p2rp7lb5mrbbsjyghzsrn.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_22, add_25, convert_element_type_15, convert_element_type_16, mul_28, mul_34, rsqrt_4, sub_4, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, var_mean_4
#   input_17 => relu_4
# Graph fragment:
#   %convolution_5 : Tensor "bf16[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=convolution_5]
#   %getitem_11 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_11]
#   %buf66 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf66]
#   %primals_31 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_31]
#   %primals_32 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_32]
#   %convert_element_type_15 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_5, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_22 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %sub_4 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %getitem_11), kwargs = {})
#   %mul_28 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_31, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %mul_34 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_17), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_32, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %add_25 : Tensor "f32[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_19), kwargs = {})
#   %convert_element_type_16 : Tensor "bf16[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_25, torch.bfloat16), kwargs = {})
#   %relu_4 : Tensor "bf16[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_16,), kwargs = {})
#   return %relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201330688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 131072.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/73/c73prp4cfmvmccsyy2on67zyussulqzfiyt4kzb3osfp3fgcqxip.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_18 => _low_memory_max_pool_with_offsets_1, getitem_12, getitem_13
# Graph fragment:
#   %relu_4 : Tensor "bf16[512, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = PlaceHolder[target=relu_4]
#   %_low_memory_max_pool_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_4, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_12 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_1, 0), kwargs = {})
#   %getitem_13 : Tensor "i8[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_1, 1), kwargs = {})
#   return %getitem_12,%getitem_13
triton_poi_fused_max_pool2d_with_indices_25 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 117440512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 8)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 8192*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 8192*x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 512*x1 + 8192*x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4352 + x0 + 512*x1 + 8192*x2), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 > tmp1
    tmp8 = tmp0 == tmp1
    tmp9 = tmp0 != tmp0
    tmp10 = tmp1 != tmp1
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp0, tmp1)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp22 = tmp20 > tmp3
    tmp23 = tmp20 == tmp3
    tmp24 = tmp20 != tmp20
    tmp25 = tmp3 != tmp3
    tmp26 = tmp24 > tmp25
    tmp27 = tmp22 | tmp26
    tmp28 = tmp24 & tmp25
    tmp29 = tmp23 | tmp28
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp21 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tmp27 | tmp32
    tmp34 = tl.where(tmp33, tmp20, tmp3)
    tmp35 = tl.where(tmp33, tmp21, tmp30)
    tmp36 = tmp34 > tmp5
    tmp37 = tmp34 == tmp5
    tmp38 = tmp34 != tmp34
    tmp39 = tmp5 != tmp5
    tmp40 = tmp38 > tmp39
    tmp41 = tmp36 | tmp40
    tmp42 = tmp38 & tmp39
    tmp43 = tmp37 | tmp42
    tmp44 = tl.full([1], 3, tl.int64)
    tmp45 = tmp35 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp41 | tmp46
    tmp48 = tl.where(tmp47, tmp34, tmp5)
    tmp49 = tl.where(tmp47, tmp35, tmp44)
    tmp50 = tmp49.to(tl.int8)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/se/cse25nl5vw6kewk3idqykcgmdwqnh6ssy5hodf6ysl5jrnff6nr3.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_19 => convert_element_type_17
# Graph fragment:
#   %primals_34 : Tensor "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=primals_34]
#   %convert_element_type_17 : Tensor "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_34, torch.bfloat16), kwargs = {})
#   return %convert_element_type_17
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/td/ctdutco7c5ub5ih2pwzmz5374tuhp5byauvlekyaiay6i3quyndd.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_20 => convert_element_type_18, var_mean_5
# Graph fragment:
#   %convolution_6 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0" = PlaceHolder[target=convolution_6]
#   %convert_element_type_18 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf74,%buf75,%buf76
triton_red_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18350080, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_27(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
    tl.store(out_ptr2 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/up/cup424qj3uivwgsudhvpc5a4mvljc4gxyt4g64pef73bpkquv3qg.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_20 => convert_element_type_18, var_mean_5
# Graph fragment:
#   %buf74 : Tensor "f32[1, 256, 1, 1, 256][65536, 1, 65536, 65536, 256]cuda:0" = PlaceHolder[target=buf74]
#   %buf75 : Tensor "f32[1, 256, 1, 1, 256][65536, 1, 65536, 65536, 256]cuda:0" = PlaceHolder[target=buf75]
#   %buf76 : Tensor "f32[1, 256, 1, 1, 256][65536, 1, 65536, 65536, 256]cuda:0" = PlaceHolder[target=buf76]
#   %convert_element_type_18 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf77,%buf78,%buf79
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(out_ptr2 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/jb/cjbxaoyhbumkxaqutwmege62nea4q6e55eexjmufeddltdhqs6nw.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_20 => add_27, add_28, add_29, convert_element_type_18, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_15, squeeze_17, var_mean_5
# Graph fragment:
#   %buf77 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf77]
#   %buf78 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf78]
#   %buf79 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf79]
#   %buf81 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf81]
#   %getitem_15 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_15]
#   %copy__16 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__16]
#   %add_28 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_28]
#   %copy__17 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__17]
#   %add_29 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_29]
#   %convert_element_type_18 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_27 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %squeeze_15 : Tensor "f32[256][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_15, [0, 2, 3]), kwargs = {})
#   %mul_36 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_15, 0.1), kwargs = {})
#   %mul_37 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_36, 0.9), kwargs = {})
#   %add_28 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %mul_37), kwargs = {})
#   %squeeze_17 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_14, [0, 2, 3]), kwargs = {})
#   %mul_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_17, 1.000030518509476), kwargs = {})
#   %mul_39 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, 0.1), kwargs = {})
#   %mul_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_37, 0.9), kwargs = {})
#   %add_29 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %mul_40), kwargs = {})
#   %copy__16 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_36, %add_28), kwargs = {})
#   %copy__17 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_37, %add_29), kwargs = {})
#   return %getitem_15,%buf81,%rsqrt_5,%add_28,%buf179,%add_29,%buf182
triton_per_fused__native_batch_norm_legit_functional_copy__29 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__29', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
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
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/kz/ckzstx5afdl7b5viihzmmmq4fdit7icapy7el7cmxotevlcgwksj.py
# Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_20 => add_27, add_30, convert_element_type_18, convert_element_type_19, mul_35, mul_41, rsqrt_5, sub_5, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, var_mean_5
#   input_21 => relu_5
# Graph fragment:
#   %convolution_6 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0" = PlaceHolder[target=convolution_6]
#   %getitem_15 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf81 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf81]
#   %primals_38 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_38]
#   %primals_39 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_39]
#   %convert_element_type_18 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_6, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_18, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_27 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %sub_5 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %getitem_15), kwargs = {})
#   %mul_35 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_38, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_41 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_39, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_30 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_23), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_30, torch.bfloat16), kwargs = {})
#   %relu_5 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_19,), kwargs = {})
#   return %relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50335744}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 32768.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/hf/chf6y4z3kha6vfu7hmekx3vozmxxgj7v3jorwckf2wire2yil2xh.py
# Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_23 => add_32, add_33, add_34, convert_element_type_21, mul_43, mul_44, mul_45, mul_46, mul_47, rsqrt_6, squeeze_18, squeeze_20, var_mean_6
# Graph fragment:
#   %buf90 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf90]
#   %buf91 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf91]
#   %buf92 : Tensor "f32[1, 256, 1, 1, 2][512, 1, 512, 512, 256]cuda:0" = PlaceHolder[target=buf92]
#   %buf94 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=buf94]
#   %getitem_17 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_17]
#   %copy__19 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__19]
#   %add_33 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_33]
#   %copy__20 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=copy__20]
#   %add_34 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=add_34]
#   %convert_element_type_21 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_21, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_32 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %squeeze_18 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_17, [0, 2, 3]), kwargs = {})
#   %mul_43 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_18, 0.1), kwargs = {})
#   %mul_44 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_42, 0.9), kwargs = {})
#   %add_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %mul_44), kwargs = {})
#   %squeeze_20 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_16, [0, 2, 3]), kwargs = {})
#   %mul_45 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_20, 1.000030518509476), kwargs = {})
#   %mul_46 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, 0.1), kwargs = {})
#   %mul_47 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_43, 0.9), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %mul_47), kwargs = {})
#   %copy__19 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_42, %add_33), kwargs = {})
#   %copy__20 : Tensor "f32[256][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_43, %add_34), kwargs = {})
#   return %getitem_17,%buf94,%rsqrt_6,%add_33,%buf187,%add_34,%buf190
triton_per_fused__native_batch_norm_legit_functional_copy__31 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__31', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
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
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 32768.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/d2/cd2rwamz4m7b7velfxwqjwdv5y3bdqeqmtvonvii4woojfqp52fx.py
# Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_26 => convert_element_type_23
# Graph fragment:
#   %primals_46 : Tensor "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = PlaceHolder[target=primals_46]
#   %convert_element_type_23 : Tensor "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_46, torch.bfloat16), kwargs = {})
#   return %convert_element_type_23
triton_poi_fused__to_copy_32 = async_compile.triton('triton_poi_fused__to_copy_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/gw/cgwgyexwqqesayf5ncbqxtjzsxfe3lcvfknqc5aws2fpqin63765.py
# Topologically Sorted Source Nodes: [input_23, input_24, mul, input_25, input_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   input_23 => add_35, convert_element_type_22, mul_42, mul_48, sub_6, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27
#   input_24 => relu_6
#   input_25 => add_36
#   input_26 => convert_element_type_24
#   mul => mul_49
# Graph fragment:
#   %getitem_12 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0" = PlaceHolder[target=getitem_12]
#   %primals_33 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_33]
#   %convolution_7 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0" = PlaceHolder[target=convolution_7]
#   %getitem_17 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=getitem_17]
#   %rsqrt_6 : Tensor "f32[1, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=rsqrt_6]
#   %primals_44 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_44]
#   %primals_45 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_45]
#   %sub_6 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %getitem_17), kwargs = {})
#   %mul_42 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_44, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %mul_48 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %unsqueeze_25), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_45, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %add_35 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %unsqueeze_27), kwargs = {})
#   %convert_element_type_22 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.bfloat16), kwargs = {})
#   %relu_6 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_22,), kwargs = {})
#   %mul_49 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_33, %relu_6), kwargs = {})
#   %add_36 : Tensor "f32[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, %mul_49), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[512, 256, 8, 8][16384, 1, 2048, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_36, torch.bfloat16), kwargs = {})
#   return %convert_element_type_24
triton_poi_fused__native_batch_norm_legit_functional__to_copy_add_mul_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional__to_copy_add_mul_relu_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__to_copy_add_mul_relu_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 67112960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_add_mul_relu_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp3 * tmp17
    tmp19 = tmp1 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/hk/chkh5slu2dl24bkyu6v4gcyyg6dwwin3wymduimos7iuzjjy4nzi.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_27 => convert_element_type_25, var_mean_7
# Graph fragment:
#   %convolution_8 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_8]
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_25, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf100,%buf101,%buf102
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 35127296, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
    tl.store(out_ptr2 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/wt/cwtfb46kaqwio5nztygz4do5xofqsc76y3uwhujoktecmmmdwdna.py
# Topologically Sorted Source Nodes: [input_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_27 => add_38, add_39, add_40, convert_element_type_25, mul_51, mul_52, mul_53, mul_54, mul_55, rsqrt_7, squeeze_21, squeeze_23, var_mean_7
# Graph fragment:
#   %buf100 : Tensor "f32[1, 512, 1, 1, 128][65536, 1, 65536, 65536, 512]cuda:0" = PlaceHolder[target=buf100]
#   %buf101 : Tensor "f32[1, 512, 1, 1, 128][65536, 1, 65536, 65536, 512]cuda:0" = PlaceHolder[target=buf101]
#   %buf102 : Tensor "f32[1, 512, 1, 1, 128][65536, 1, 65536, 65536, 512]cuda:0" = PlaceHolder[target=buf102]
#   %buf104 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=buf104]
#   %getitem_19 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_19]
#   %copy__22 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__22]
#   %add_39 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_39]
#   %copy__23 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__23]
#   %add_40 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_40]
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_25, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %squeeze_21 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_19, [0, 2, 3]), kwargs = {})
#   %mul_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_21, 0.1), kwargs = {})
#   %mul_52 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_48, 0.9), kwargs = {})
#   %add_39 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %mul_52), kwargs = {})
#   %squeeze_23 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_18, [0, 2, 3]), kwargs = {})
#   %mul_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_23, 1.000030518509476), kwargs = {})
#   %mul_54 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, 0.1), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_49, 0.9), kwargs = {})
#   %add_40 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %mul_55), kwargs = {})
#   %copy__22 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_48, %add_39), kwargs = {})
#   %copy__23 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_49, %add_40), kwargs = {})
#   return %getitem_19,%buf104,%rsqrt_7,%add_39,%buf195,%add_40,%buf198
triton_red_fused__native_batch_norm_legit_functional_copy__35 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_copy__35', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_copy__35', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 811008, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_copy__35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = 32768.0
    tmp13 = (tmp10 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = 0.1
    tmp18 = tmp6 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = 1.000030518509476
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24 * tmp17
    tmp27 = tmp26 * tmp20
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/hf/chfy5pptx3x4n7dxoc77xkha54bk35wn6j2q4gic5alzhvjwd5zj.py
# Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_27 => add_38, add_41, convert_element_type_25, convert_element_type_26, mul_50, mul_56, rsqrt_7, sub_7, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31, var_mean_7
#   input_28 => relu_7
# Graph fragment:
#   %convolution_8 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_8]
#   %getitem_19 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_19]
#   %buf104 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=buf104]
#   %primals_50 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_50]
#   %primals_51 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_51]
#   %convert_element_type_25 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_8, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_25, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_38 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_38,), kwargs = {})
#   %sub_7 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %getitem_19), kwargs = {})
#   %mul_50 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_50, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_56 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_51, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_41 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_31), kwargs = {})
#   %convert_element_type_26 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_41, torch.bfloat16), kwargs = {})
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_26,), kwargs = {})
#   return %relu_7
triton_poi_fused__native_batch_norm_legit_functional_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100671488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 32768.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/xw/cxwh3kj6mj72jsajwkknaudahqt24vy2va2w6tghh676qlppk3vk.py
# Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_29 => _low_memory_max_pool_with_offsets_2, getitem_20, getitem_21
# Graph fragment:
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=relu_7]
#   %_low_memory_max_pool_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_20 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_2, 0), kwargs = {})
#   %getitem_21 : Tensor "i8[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_2, 1), kwargs = {})
#   return %getitem_20,%getitem_21
triton_poi_fused_max_pool2d_with_indices_37 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 58720256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_37(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 4)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*x1 + 8192*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 1024*x1 + 8192*x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (4096 + x0 + 1024*x1 + 8192*x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4608 + x0 + 1024*x1 + 8192*x2), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 > tmp1
    tmp8 = tmp0 == tmp1
    tmp9 = tmp0 != tmp0
    tmp10 = tmp1 != tmp1
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp0, tmp1)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp22 = tmp20 > tmp3
    tmp23 = tmp20 == tmp3
    tmp24 = tmp20 != tmp20
    tmp25 = tmp3 != tmp3
    tmp26 = tmp24 > tmp25
    tmp27 = tmp22 | tmp26
    tmp28 = tmp24 & tmp25
    tmp29 = tmp23 | tmp28
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp21 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tmp27 | tmp32
    tmp34 = tl.where(tmp33, tmp20, tmp3)
    tmp35 = tl.where(tmp33, tmp21, tmp30)
    tmp36 = tmp34 > tmp5
    tmp37 = tmp34 == tmp5
    tmp38 = tmp34 != tmp34
    tmp39 = tmp5 != tmp5
    tmp40 = tmp38 > tmp39
    tmp41 = tmp36 | tmp40
    tmp42 = tmp38 & tmp39
    tmp43 = tmp37 | tmp42
    tmp44 = tl.full([1], 3, tl.int64)
    tmp45 = tmp35 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp41 | tmp46
    tmp48 = tl.where(tmp47, tmp34, tmp5)
    tmp49 = tl.where(tmp47, tmp35, tmp44)
    tmp50 = tmp49.to(tl.int8)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp50, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/yb/cybxtb6ocyt7wzh7u2skrumdshecyqhuzqfq3fxmtuyjxg2fxyow.py
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_30 => convert_element_type_27
# Graph fragment:
#   %primals_52 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=primals_52]
#   %convert_element_type_27 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_52, torch.bfloat16), kwargs = {})
#   return %convert_element_type_27
triton_poi_fused__to_copy_38 = async_compile.triton('triton_poi_fused__to_copy_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ih/cihl43ai6vuutsk4ep2nh3isxmwwhqgc47c6d6o6rdpmfqk6feor.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_31 => convert_element_type_28, var_mean_8
# Graph fragment:
#   %convolution_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_9]
#   %convert_element_type_28 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_28, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf112,%buf113,%buf114
triton_red_fused__native_batch_norm_legit_functional_39 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9175040, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_39(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp4, tmp5, tmp6 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
    tl.store(out_ptr2 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fc/cfcgy2zxbm45u4qtohpk6j3zdxziroxyxrq6kslm5xda7iqv6kgy.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_31 => add_43, add_44, add_45, convert_element_type_28, mul_58, mul_59, mul_60, mul_61, mul_62, rsqrt_8, squeeze_24, squeeze_26, var_mean_8
# Graph fragment:
#   %buf112 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf112]
#   %buf113 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf113]
#   %buf114 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf114]
#   %buf116 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=buf116]
#   %getitem_23 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_23]
#   %copy__25 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__25]
#   %add_44 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_44]
#   %copy__26 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__26]
#   %add_45 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_45]
#   %convert_element_type_28 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_28, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_43 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_43,), kwargs = {})
#   %squeeze_24 : Tensor "f32[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_23, [0, 2, 3]), kwargs = {})
#   %mul_58 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_24, 0.1), kwargs = {})
#   %mul_59 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_54, 0.9), kwargs = {})
#   %add_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %mul_59), kwargs = {})
#   %squeeze_26 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_22, [0, 2, 3]), kwargs = {})
#   %mul_60 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_26, 1.0001220852154804), kwargs = {})
#   %mul_61 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, 0.1), kwargs = {})
#   %mul_62 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_55, 0.9), kwargs = {})
#   %add_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %mul_62), kwargs = {})
#   %copy__25 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_54, %add_44), kwargs = {})
#   %copy__26 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_55, %add_45), kwargs = {})
#   return %getitem_23,%buf116,%rsqrt_8,%add_44,%buf203,%add_45,%buf206
triton_per_fused__native_batch_norm_legit_functional_copy__40 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__40', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__40', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 417792, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/f3/cf3jp7lftggip7cwkj7fcyv5uctcqak6jlagdtlgzej3z5a4foio.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   input_31 => add_43, add_46, convert_element_type_28, convert_element_type_29, mul_57, mul_63, rsqrt_8, sub_8, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, var_mean_8
#   input_32 => relu_8
# Graph fragment:
#   %convolution_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_9]
#   %getitem_23 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf116 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=buf116]
#   %primals_56 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_56]
#   %primals_57 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_57]
#   %convert_element_type_28 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_28, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_43 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-05), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_43,), kwargs = {})
#   %sub_8 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %getitem_23), kwargs = {})
#   %mul_57 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_8), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_56, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %mul_63 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_33), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_57, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %add_46 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %unsqueeze_35), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_46, torch.bfloat16), kwargs = {})
#   %relu_8 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_29,), kwargs = {})
#   return %relu_8
triton_poi_fused__native_batch_norm_legit_functional_relu_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25174016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 8192.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/f7/cf7wdwxwm6iti7hs5lyp2g7sk3hmrojefhsmijulchxxfi7bzvw5.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_34 => add_48, add_49, add_50, convert_element_type_31, mul_65, mul_66, mul_67, mul_68, mul_69, rsqrt_9, squeeze_27, squeeze_29, var_mean_9
# Graph fragment:
#   %buf122 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf122]
#   %buf123 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf123]
#   %buf124 : Tensor "f32[1, 512, 1, 1, 64][32768, 1, 32768, 32768, 512]cuda:0" = PlaceHolder[target=buf124]
#   %buf126 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=buf126]
#   %getitem_25 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_25]
#   %copy__28 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__28]
#   %add_49 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_49]
#   %copy__29 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=copy__29]
#   %add_50 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=add_50]
#   %convert_element_type_31 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_31, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_48 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_9 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %squeeze_27 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_25, [0, 2, 3]), kwargs = {})
#   %mul_65 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_27, 0.1), kwargs = {})
#   %mul_66 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_60, 0.9), kwargs = {})
#   %add_49 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %mul_66), kwargs = {})
#   %squeeze_29 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_24, [0, 2, 3]), kwargs = {})
#   %mul_67 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_29, 1.0001220852154804), kwargs = {})
#   %mul_68 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, 0.1), kwargs = {})
#   %mul_69 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_61, 0.9), kwargs = {})
#   %add_50 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %mul_69), kwargs = {})
#   %copy__28 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_60, %add_49), kwargs = {})
#   %copy__29 : Tensor "f32[512][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_61, %add_50), kwargs = {})
#   return %getitem_25,%buf126,%rsqrt_9,%add_49,%buf211,%add_50,%buf214
triton_per_fused__native_batch_norm_legit_functional_copy__42 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__42', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 413696, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0001220852154804
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/en/cena7d2v2shjiuv2izkxaexlzgqfzabdrspq2t6ytp66y2qs3ao3.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_34 => add_51, convert_element_type_32, mul_64, mul_70, sub_9, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_35 => relu_9
#   input_36 => add_52
# Graph fragment:
#   %getitem_20 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_20]
#   %convolution_10 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %getitem_25 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=getitem_25]
#   %rsqrt_9 : Tensor "f32[1, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=rsqrt_9]
#   %primals_62 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_62]
#   %primals_63 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_63]
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
#   %add_52 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, %relu_9), kwargs = {})
#   return %add_52
triton_poi_fused__native_batch_norm_legit_functional_add_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33562624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tmp0 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/vv/cvvdzjivwlzajalkvba2e6fys6ashibl2v6ctmdd7wtoqsrz4xfh.py
# Topologically Sorted Source Nodes: [max_pool2d_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d_3 => _low_memory_max_pool_with_offsets_3, getitem_27
# Graph fragment:
#   %add_52 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=add_52]
#   %_low_memory_max_pool_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%add_52, [4, 4], [4, 4], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_27 : Tensor "i8[512, 512, 1, 1][512, 1, 512, 512]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets_3, 1), kwargs = {})
#   return %getitem_26,%getitem_27
triton_poi_fused_max_pool2d_with_indices_44 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9961472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_44(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + 8192*x1), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + 8192*x1), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + 8192*x1), None).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (2048 + x0 + 8192*x1), None).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (2560 + x0 + 8192*x1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (3072 + x0 + 8192*x1), None).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (3584 + x0 + 8192*x1), None).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (4096 + x0 + 8192*x1), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (4608 + x0 + 8192*x1), None).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (5120 + x0 + 8192*x1), None).to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (5632 + x0 + 8192*x1), None).to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (6144 + x0 + 8192*x1), None).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (6656 + x0 + 8192*x1), None).to(tl.float32)
    tmp27 = tl.load(in_ptr0 + (7168 + x0 + 8192*x1), None).to(tl.float32)
    tmp29 = tl.load(in_ptr0 + (7680 + x0 + 8192*x1), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = tmp0 > tmp1
    tmp32 = tmp0 == tmp1
    tmp33 = tmp0 != tmp0
    tmp34 = tmp1 != tmp1
    tmp35 = tmp33 > tmp34
    tmp36 = tmp31 | tmp35
    tmp37 = tmp33 & tmp34
    tmp38 = tmp32 | tmp37
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tl.full([1], 1, tl.int64)
    tmp41 = tmp39 < tmp40
    tmp42 = tmp38 & tmp41
    tmp43 = tmp36 | tmp42
    tmp44 = tl.where(tmp43, tmp0, tmp1)
    tmp45 = tl.where(tmp43, tmp39, tmp40)
    tmp46 = tmp44 > tmp3
    tmp47 = tmp44 == tmp3
    tmp48 = tmp44 != tmp44
    tmp49 = tmp3 != tmp3
    tmp50 = tmp48 > tmp49
    tmp51 = tmp46 | tmp50
    tmp52 = tmp48 & tmp49
    tmp53 = tmp47 | tmp52
    tmp54 = tl.full([1], 2, tl.int64)
    tmp55 = tmp45 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tmp51 | tmp56
    tmp58 = tl.where(tmp57, tmp44, tmp3)
    tmp59 = tl.where(tmp57, tmp45, tmp54)
    tmp60 = tmp58 > tmp5
    tmp61 = tmp58 == tmp5
    tmp62 = tmp58 != tmp58
    tmp63 = tmp5 != tmp5
    tmp64 = tmp62 > tmp63
    tmp65 = tmp60 | tmp64
    tmp66 = tmp62 & tmp63
    tmp67 = tmp61 | tmp66
    tmp68 = tl.full([1], 3, tl.int64)
    tmp69 = tmp59 < tmp68
    tmp70 = tmp67 & tmp69
    tmp71 = tmp65 | tmp70
    tmp72 = tl.where(tmp71, tmp58, tmp5)
    tmp73 = tl.where(tmp71, tmp59, tmp68)
    tmp74 = tmp72 > tmp7
    tmp75 = tmp72 == tmp7
    tmp76 = tmp72 != tmp72
    tmp77 = tmp7 != tmp7
    tmp78 = tmp76 > tmp77
    tmp79 = tmp74 | tmp78
    tmp80 = tmp76 & tmp77
    tmp81 = tmp75 | tmp80
    tmp82 = tl.full([1], 4, tl.int64)
    tmp83 = tmp73 < tmp82
    tmp84 = tmp81 & tmp83
    tmp85 = tmp79 | tmp84
    tmp86 = tl.where(tmp85, tmp72, tmp7)
    tmp87 = tl.where(tmp85, tmp73, tmp82)
    tmp88 = tmp86 > tmp9
    tmp89 = tmp86 == tmp9
    tmp90 = tmp86 != tmp86
    tmp91 = tmp9 != tmp9
    tmp92 = tmp90 > tmp91
    tmp93 = tmp88 | tmp92
    tmp94 = tmp90 & tmp91
    tmp95 = tmp89 | tmp94
    tmp96 = tl.full([1], 5, tl.int64)
    tmp97 = tmp87 < tmp96
    tmp98 = tmp95 & tmp97
    tmp99 = tmp93 | tmp98
    tmp100 = tl.where(tmp99, tmp86, tmp9)
    tmp101 = tl.where(tmp99, tmp87, tmp96)
    tmp102 = tmp100 > tmp11
    tmp103 = tmp100 == tmp11
    tmp104 = tmp100 != tmp100
    tmp105 = tmp11 != tmp11
    tmp106 = tmp104 > tmp105
    tmp107 = tmp102 | tmp106
    tmp108 = tmp104 & tmp105
    tmp109 = tmp103 | tmp108
    tmp110 = tl.full([1], 6, tl.int64)
    tmp111 = tmp101 < tmp110
    tmp112 = tmp109 & tmp111
    tmp113 = tmp107 | tmp112
    tmp114 = tl.where(tmp113, tmp100, tmp11)
    tmp115 = tl.where(tmp113, tmp101, tmp110)
    tmp116 = tmp114 > tmp13
    tmp117 = tmp114 == tmp13
    tmp118 = tmp114 != tmp114
    tmp119 = tmp13 != tmp13
    tmp120 = tmp118 > tmp119
    tmp121 = tmp116 | tmp120
    tmp122 = tmp118 & tmp119
    tmp123 = tmp117 | tmp122
    tmp124 = tl.full([1], 7, tl.int64)
    tmp125 = tmp115 < tmp124
    tmp126 = tmp123 & tmp125
    tmp127 = tmp121 | tmp126
    tmp128 = tl.where(tmp127, tmp114, tmp13)
    tmp129 = tl.where(tmp127, tmp115, tmp124)
    tmp130 = tmp128 > tmp15
    tmp131 = tmp128 == tmp15
    tmp132 = tmp128 != tmp128
    tmp133 = tmp15 != tmp15
    tmp134 = tmp132 > tmp133
    tmp135 = tmp130 | tmp134
    tmp136 = tmp132 & tmp133
    tmp137 = tmp131 | tmp136
    tmp138 = tl.full([1], 8, tl.int64)
    tmp139 = tmp129 < tmp138
    tmp140 = tmp137 & tmp139
    tmp141 = tmp135 | tmp140
    tmp142 = tl.where(tmp141, tmp128, tmp15)
    tmp143 = tl.where(tmp141, tmp129, tmp138)
    tmp144 = tmp142 > tmp17
    tmp145 = tmp142 == tmp17
    tmp146 = tmp142 != tmp142
    tmp147 = tmp17 != tmp17
    tmp148 = tmp146 > tmp147
    tmp149 = tmp144 | tmp148
    tmp150 = tmp146 & tmp147
    tmp151 = tmp145 | tmp150
    tmp152 = tl.full([1], 9, tl.int64)
    tmp153 = tmp143 < tmp152
    tmp154 = tmp151 & tmp153
    tmp155 = tmp149 | tmp154
    tmp156 = tl.where(tmp155, tmp142, tmp17)
    tmp157 = tl.where(tmp155, tmp143, tmp152)
    tmp158 = tmp156 > tmp19
    tmp159 = tmp156 == tmp19
    tmp160 = tmp156 != tmp156
    tmp161 = tmp19 != tmp19
    tmp162 = tmp160 > tmp161
    tmp163 = tmp158 | tmp162
    tmp164 = tmp160 & tmp161
    tmp165 = tmp159 | tmp164
    tmp166 = tl.full([1], 10, tl.int64)
    tmp167 = tmp157 < tmp166
    tmp168 = tmp165 & tmp167
    tmp169 = tmp163 | tmp168
    tmp170 = tl.where(tmp169, tmp156, tmp19)
    tmp171 = tl.where(tmp169, tmp157, tmp166)
    tmp172 = tmp170 > tmp21
    tmp173 = tmp170 == tmp21
    tmp174 = tmp170 != tmp170
    tmp175 = tmp21 != tmp21
    tmp176 = tmp174 > tmp175
    tmp177 = tmp172 | tmp176
    tmp178 = tmp174 & tmp175
    tmp179 = tmp173 | tmp178
    tmp180 = tl.full([1], 11, tl.int64)
    tmp181 = tmp171 < tmp180
    tmp182 = tmp179 & tmp181
    tmp183 = tmp177 | tmp182
    tmp184 = tl.where(tmp183, tmp170, tmp21)
    tmp185 = tl.where(tmp183, tmp171, tmp180)
    tmp186 = tmp184 > tmp23
    tmp187 = tmp184 == tmp23
    tmp188 = tmp184 != tmp184
    tmp189 = tmp23 != tmp23
    tmp190 = tmp188 > tmp189
    tmp191 = tmp186 | tmp190
    tmp192 = tmp188 & tmp189
    tmp193 = tmp187 | tmp192
    tmp194 = tl.full([1], 12, tl.int64)
    tmp195 = tmp185 < tmp194
    tmp196 = tmp193 & tmp195
    tmp197 = tmp191 | tmp196
    tmp198 = tl.where(tmp197, tmp184, tmp23)
    tmp199 = tl.where(tmp197, tmp185, tmp194)
    tmp200 = tmp198 > tmp25
    tmp201 = tmp198 == tmp25
    tmp202 = tmp198 != tmp198
    tmp203 = tmp25 != tmp25
    tmp204 = tmp202 > tmp203
    tmp205 = tmp200 | tmp204
    tmp206 = tmp202 & tmp203
    tmp207 = tmp201 | tmp206
    tmp208 = tl.full([1], 13, tl.int64)
    tmp209 = tmp199 < tmp208
    tmp210 = tmp207 & tmp209
    tmp211 = tmp205 | tmp210
    tmp212 = tl.where(tmp211, tmp198, tmp25)
    tmp213 = tl.where(tmp211, tmp199, tmp208)
    tmp214 = tmp212 > tmp27
    tmp215 = tmp212 == tmp27
    tmp216 = tmp212 != tmp212
    tmp217 = tmp27 != tmp27
    tmp218 = tmp216 > tmp217
    tmp219 = tmp214 | tmp218
    tmp220 = tmp216 & tmp217
    tmp221 = tmp215 | tmp220
    tmp222 = tl.full([1], 14, tl.int64)
    tmp223 = tmp213 < tmp222
    tmp224 = tmp221 & tmp223
    tmp225 = tmp219 | tmp224
    tmp226 = tl.where(tmp225, tmp212, tmp27)
    tmp227 = tl.where(tmp225, tmp213, tmp222)
    tmp228 = tmp226 > tmp29
    tmp229 = tmp226 == tmp29
    tmp230 = tmp226 != tmp226
    tmp231 = tmp29 != tmp29
    tmp232 = tmp230 > tmp231
    tmp233 = tmp228 | tmp232
    tmp234 = tmp230 & tmp231
    tmp235 = tmp229 | tmp234
    tmp236 = tl.full([1], 15, tl.int64)
    tmp237 = tmp227 < tmp236
    tmp238 = tmp235 & tmp237
    tmp239 = tmp233 | tmp238
    tmp240 = tl.where(tmp239, tmp226, tmp29)
    tmp241 = tl.where(tmp239, tmp227, tmp236)
    tmp242 = tmp241.to(tl.int8)
    tl.store(out_ptr0 + (x2), tmp30, None)
    tl.store(out_ptr1 + (x2), tmp242, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/wj/cwjscp7f5qmr3n5wkqu5iagfuapp3om46tq6ugb7wpyvx3yf6kkm.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_33, permute
# Graph fragment:
#   %primals_64 : Tensor "f32[10, 512][512, 1]cuda:0" = PlaceHolder[target=primals_64]
#   %convert_element_type_33 : Tensor "bf16[10, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_64, torch.bfloat16), kwargs = {})
#   %permute : Tensor "bf16[512, 10][1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_33, [1, 0]), kwargs = {})
#   return %permute
triton_poi_fused__to_copy_t_45 = async_compile.triton('triton_poi_fused__to_copy_t_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/tl/ctlekna7rlqsejz23ahhl67vj6g67egdoodnj4zoont7rwfxtsew.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_71
# Graph fragment:
#   %mm : Tensor "bf16[512, 10][10, 1]cuda:0" = PlaceHolder[target=mm]
#   %mul_71 : Tensor "bf16[512, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 0.125), kwargs = {})
#   return %mul_71
triton_poi_fused_mul_46 = async_compile.triton('triton_poi_fused_mul_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_46(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/4z/c4zqjsr4d62rs4qfwnsygtdp4el75ygtdnph3eqw25utbdya7cq2.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %copy_ : Tensor "i64[][]cuda:0" = PlaceHolder[target=copy_]
#   %add : Tensor "i64[][]cuda:0" = PlaceHolder[target=add]
#   %add : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1), kwargs = {})
#   %copy_ : Tensor "i64[][]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add), kwargs = {})
#   return %add,%buf136
triton_poi_fused_add_copy__47 = async_compile.triton('triton_poi_fused_add_copy__47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__47', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__47(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64 = args
        args.clear()
        assert_size_stride(primals_1, (54, 3, 3, 3), (27, 1, 9, 3))
        assert_size_stride(primals_2, (512, 3, 32, 32), (3072, 1, 96, 3))
        assert_size_stride(primals_3, (64, 54, 3, 3), (486, 1, 162, 54))
        assert_size_stride(primals_4, (), ())
        assert_size_stride(primals_5, (64, ), (1, ))
        assert_size_stride(primals_6, (64, ), (1, ))
        assert_size_stride(primals_7, (64, ), (1, ))
        assert_size_stride(primals_8, (64, ), (1, ))
        assert_size_stride(primals_9, (128, 64, 3, 3), (576, 1, 192, 64))
        assert_size_stride(primals_10, (), ())
        assert_size_stride(primals_11, (128, ), (1, ))
        assert_size_stride(primals_12, (128, ), (1, ))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_14, (128, ), (1, ))
        assert_size_stride(primals_15, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_16, (), ())
        assert_size_stride(primals_17, (128, ), (1, ))
        assert_size_stride(primals_18, (128, ), (1, ))
        assert_size_stride(primals_19, (128, ), (1, ))
        assert_size_stride(primals_20, (128, ), (1, ))
        assert_size_stride(primals_21, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_22, (), ())
        assert_size_stride(primals_23, (128, ), (1, ))
        assert_size_stride(primals_24, (128, ), (1, ))
        assert_size_stride(primals_25, (128, ), (1, ))
        assert_size_stride(primals_26, (128, ), (1, ))
        assert_size_stride(primals_27, (256, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(primals_28, (), ())
        assert_size_stride(primals_29, (256, ), (1, ))
        assert_size_stride(primals_30, (256, ), (1, ))
        assert_size_stride(primals_31, (256, ), (1, ))
        assert_size_stride(primals_32, (256, ), (1, ))
        assert_size_stride(primals_33, (1, ), (1, ))
        assert_size_stride(primals_34, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_35, (), ())
        assert_size_stride(primals_36, (256, ), (1, ))
        assert_size_stride(primals_37, (256, ), (1, ))
        assert_size_stride(primals_38, (256, ), (1, ))
        assert_size_stride(primals_39, (256, ), (1, ))
        assert_size_stride(primals_40, (256, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_41, (), ())
        assert_size_stride(primals_42, (256, ), (1, ))
        assert_size_stride(primals_43, (256, ), (1, ))
        assert_size_stride(primals_44, (256, ), (1, ))
        assert_size_stride(primals_45, (256, ), (1, ))
        assert_size_stride(primals_46, (512, 256, 3, 3), (2304, 1, 768, 256))
        assert_size_stride(primals_47, (), ())
        assert_size_stride(primals_48, (512, ), (1, ))
        assert_size_stride(primals_49, (512, ), (1, ))
        assert_size_stride(primals_50, (512, ), (1, ))
        assert_size_stride(primals_51, (512, ), (1, ))
        assert_size_stride(primals_52, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(primals_53, (), ())
        assert_size_stride(primals_54, (512, ), (1, ))
        assert_size_stride(primals_55, (512, ), (1, ))
        assert_size_stride(primals_56, (512, ), (1, ))
        assert_size_stride(primals_57, (512, ), (1, ))
        assert_size_stride(primals_58, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(primals_59, (), ())
        assert_size_stride(primals_60, (512, ), (1, ))
        assert_size_stride(primals_61, (512, ), (1, ))
        assert_size_stride(primals_62, (512, ), (1, ))
        assert_size_stride(primals_63, (512, ), (1, ))
        assert_size_stride(primals_64, (10, 512), (512, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((512, 3, 32, 32), (3072, 1, 96, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(primals_2, buf0, 1572864, stream=stream0)
            del primals_2
            buf1 = empty_strided_cuda((54, 3, 3, 3), (27, 1, 9, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(primals_1, buf1, 1458, stream=stream0)
            del primals_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (512, 54, 32, 32), (55296, 1, 1728, 54), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = empty_strided_cuda((64, 54, 3, 3), (486, 1, 162, 54), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_3, buf3, 31104, stream=stream0)
            del primals_3
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf4, (512, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            buf5 = empty_strided_cuda((1, 64, 1, 1, 622), (39808, 1, 39808, 39808, 64), torch.float32)
            buf6 = empty_strided_cuda((1, 64, 1, 1, 622), (39808, 1, 39808, 39808, 64), torch.float32)
            buf7 = empty_strided_cuda((1, 64, 1, 1, 622), (39808, 1, 39808, 39808, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_3.run(buf4, buf5, buf6, buf7, 39808, 843, stream=stream0)
            buf8 = empty_strided_cuda((1, 64, 1, 1, 5), (320, 1, 320, 320, 64), torch.float32)
            buf9 = empty_strided_cuda((1, 64, 1, 1, 5), (320, 1, 320, 320, 64), torch.float32)
            buf10 = empty_strided_cuda((1, 64, 1, 1, 5), (320, 1, 320, 320, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_4.run(buf5, buf6, buf7, buf8, buf9, buf10, 320, 125, stream=stream0)
            del buf5
            del buf6
            del buf7
            buf11 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
            buf12 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
            buf14 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__5.run(buf8, buf9, buf10, primals_5, primals_6, buf11, buf12, buf14, primals_5, primals_6, 64, 5, stream=stream0)
            del buf10
            del buf8
            del buf9
            del primals_5
            del primals_6
            buf15 = empty_strided_cuda((512, 64, 32, 32), (65536, 1, 2048, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf4, buf11, buf12, primals_7, primals_8, buf15, 33554432, stream=stream0)
            del buf12
            del primals_8
            buf16 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(primals_9, buf16, 73728, stream=stream0)
            del primals_9
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
            buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf17, (512, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            buf18 = empty_strided_cuda((1, 128, 1, 1, 622), (79616, 1, 79616, 79616, 128), torch.float32)
            buf19 = empty_strided_cuda((1, 128, 1, 1, 622), (79616, 1, 79616, 79616, 128), torch.float32)
            buf20 = empty_strided_cuda((1, 128, 1, 1, 622), (79616, 1, 79616, 79616, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_8.run(buf17, buf18, buf19, buf20, 79616, 843, stream=stream0)
            buf21 = empty_strided_cuda((1, 128, 1, 1, 5), (640, 1, 640, 640, 128), torch.float32)
            buf22 = empty_strided_cuda((1, 128, 1, 1, 5), (640, 1, 640, 640, 128), torch.float32)
            buf23 = empty_strided_cuda((1, 128, 1, 1, 5), (640, 1, 640, 640, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_9.run(buf18, buf19, buf20, buf21, buf22, buf23, 640, 125, stream=stream0)
            del buf18
            del buf19
            del buf20
            buf24 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            buf25 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            buf27 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf21, buf22, buf23, primals_11, primals_12, buf24, buf25, buf27, primals_11, primals_12, 128, 5, stream=stream0)
            del buf21
            del buf22
            del buf23
            del primals_11
            del primals_12
            buf28 = empty_strided_cuda((512, 128, 32, 32), (131072, 1, 4096, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf17, buf24, buf25, primals_13, primals_14, buf28, 67108864, stream=stream0)
            del primals_14
            buf29 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.bfloat16)
            buf30 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.int8)
            # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_12.run(buf28, buf29, buf30, 16777216, stream=stream0)
            buf31 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(primals_15, buf31, 147456, stream=stream0)
            del primals_15
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
            buf32 = extern_kernels.convolution(buf29, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf32, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            buf33 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            buf34 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            buf35 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_14.run(buf32, buf33, buf34, buf35, 65536, 256, stream=stream0)
            buf36 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), torch.float32)
            buf37 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), torch.float32)
            buf38 = empty_strided_cuda((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_15.run(buf33, buf34, buf35, buf36, buf37, buf38, 512, 128, stream=stream0)
            del buf33
            del buf34
            del buf35
            buf39 = buf25; del buf25  # reuse
            buf40 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            buf42 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__16.run(buf36, buf37, buf38, primals_17, primals_18, buf39, buf40, buf42, primals_17, primals_18, 128, 4, stream=stream0)
            del primals_17
            del primals_18
            buf43 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_17.run(buf32, buf39, buf40, primals_19, primals_20, buf43, 16777216, stream=stream0)
            del primals_20
            buf44 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(primals_21, buf44, 147456, stream=stream0)
            del primals_21
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf45, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            buf46 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            buf47 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            buf48 = empty_strided_cuda((1, 128, 1, 1, 512), (65536, 1, 65536, 65536, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_14.run(buf45, buf46, buf47, buf48, 65536, 256, stream=stream0)
            buf49 = buf38; del buf38  # reuse
            buf50 = buf37; del buf37  # reuse
            buf51 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_15.run(buf46, buf47, buf48, buf49, buf50, buf51, 512, 128, stream=stream0)
            del buf46
            del buf47
            del buf48
            buf52 = buf40; del buf40  # reuse
            buf55 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__18.run(buf49, buf50, buf51, primals_23, primals_24, buf52, buf55, primals_23, primals_24, 128, 4, stream=stream0)
            del primals_23
            del primals_24
            buf56 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_add_relu_19.run(buf29, buf45, buf52, buf55, primals_25, primals_26, buf56, 16777216, stream=stream0)
            buf57 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_20.run(primals_27, buf57, 294912, stream=stream0)
            del primals_27
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
            buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf58, (512, 256, 16, 16), (65536, 1, 4096, 256), 'torch.ops.aten.convolution.default')
            buf59 = empty_strided_cuda((1, 256, 1, 1, 310), (79360, 1, 79360, 79360, 256), torch.float32)
            buf60 = empty_strided_cuda((1, 256, 1, 1, 310), (79360, 1, 79360, 79360, 256), torch.float32)
            buf61 = empty_strided_cuda((1, 256, 1, 1, 310), (79360, 1, 79360, 79360, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_21.run(buf58, buf59, buf60, buf61, 79360, 423, stream=stream0)
            buf62 = empty_strided_cuda((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), torch.float32)
            buf63 = empty_strided_cuda((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), torch.float32)
            buf64 = empty_strided_cuda((1, 256, 1, 1, 3), (768, 1, 768, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_22.run(buf59, buf60, buf61, buf62, buf63, buf64, 768, 104, stream=stream0)
            del buf59
            del buf60
            del buf61
            buf65 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            buf66 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            buf68 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__23.run(buf62, buf63, buf64, primals_29, primals_30, buf65, buf66, buf68, primals_29, primals_30, 256, 3, stream=stream0)
            del buf62
            del buf63
            del buf64
            del primals_29
            del primals_30
            buf69 = empty_strided_cuda((512, 256, 16, 16), (65536, 1, 4096, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_24.run(buf58, buf65, buf66, primals_31, primals_32, buf69, 33554432, stream=stream0)
            del primals_32
            buf70 = empty_strided_cuda((512, 256, 8, 8), (16384, 1, 2048, 256), torch.bfloat16)
            buf71 = empty_strided_cuda((512, 256, 8, 8), (16384, 1, 2048, 256), torch.int8)
            # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_25.run(buf69, buf70, buf71, 8388608, stream=stream0)
            buf72 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_26.run(primals_34, buf72, 589824, stream=stream0)
            del primals_34
            # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
            buf73 = extern_kernels.convolution(buf70, buf72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf73, (512, 256, 8, 8), (16384, 1, 2048, 256), 'torch.ops.aten.convolution.default')
            buf74 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            buf75 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            buf76 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_27.run(buf73, buf74, buf75, buf76, 65536, 128, stream=stream0)
            buf77 = reinterpret_tensor(buf51, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf51  # reuse
            buf78 = reinterpret_tensor(buf50, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf50  # reuse
            buf79 = reinterpret_tensor(buf49, (1, 256, 1, 1, 2), (512, 1, 512, 512, 256), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_28.run(buf74, buf75, buf76, buf77, buf78, buf79, 512, 128, stream=stream0)
            del buf74
            del buf75
            del buf76
            buf80 = buf66; del buf66  # reuse
            buf81 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            buf83 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__29.run(buf77, buf78, buf79, primals_36, primals_37, buf80, buf81, buf83, primals_36, primals_37, 256, 2, stream=stream0)
            del primals_36
            del primals_37
            buf84 = empty_strided_cuda((512, 256, 8, 8), (16384, 1, 2048, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_30.run(buf73, buf80, buf81, primals_38, primals_39, buf84, 8388608, stream=stream0)
            del primals_39
            buf85 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_26.run(primals_40, buf85, 589824, stream=stream0)
            del primals_40
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
            buf86 = extern_kernels.convolution(buf84, buf85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf86, (512, 256, 8, 8), (16384, 1, 2048, 256), 'torch.ops.aten.convolution.default')
            buf87 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            buf88 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            buf89 = empty_strided_cuda((1, 256, 1, 1, 256), (65536, 1, 65536, 65536, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_27.run(buf86, buf87, buf88, buf89, 65536, 128, stream=stream0)
            buf90 = buf79; del buf79  # reuse
            buf91 = buf78; del buf78  # reuse
            buf92 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_28.run(buf87, buf88, buf89, buf90, buf91, buf92, 512, 128, stream=stream0)
            del buf87
            del buf88
            del buf89
            buf93 = buf81; del buf81  # reuse
            buf96 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
            # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__31.run(buf90, buf91, buf92, primals_42, primals_43, buf93, buf96, primals_42, primals_43, 256, 2, stream=stream0)
            del primals_42
            del primals_43
            buf97 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_32.run(primals_46, buf97, 1179648, stream=stream0)
            del primals_46
            buf98 = empty_strided_cuda((512, 256, 8, 8), (16384, 1, 2048, 256), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_23, input_24, mul, input_25, input_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional__to_copy_add_mul_relu_33.run(buf70, primals_33, buf86, buf93, buf96, primals_44, primals_45, buf98, 8388608, stream=stream0)
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
            buf99 = extern_kernels.convolution(buf98, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf99, (512, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            buf100 = empty_strided_cuda((1, 512, 1, 1, 128), (65536, 1, 65536, 65536, 512), torch.float32)
            buf101 = empty_strided_cuda((1, 512, 1, 1, 128), (65536, 1, 65536, 65536, 512), torch.float32)
            buf102 = empty_strided_cuda((1, 512, 1, 1, 128), (65536, 1, 65536, 65536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_34.run(buf99, buf100, buf101, buf102, 65536, 256, stream=stream0)
            buf103 = reinterpret_tensor(buf92, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf92  # reuse
            buf104 = reinterpret_tensor(buf91, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf91  # reuse
            buf106 = reinterpret_tensor(buf90, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_copy__35.run(buf100, buf101, buf102, primals_48, primals_49, buf103, buf104, buf106, primals_48, primals_49, 512, 128, stream=stream0)
            del buf100
            del buf101
            del buf102
            del primals_48
            del primals_49
            buf107 = empty_strided_cuda((512, 512, 8, 8), (32768, 1, 4096, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_36.run(buf99, buf103, buf104, primals_50, primals_51, buf107, 16777216, stream=stream0)
            del primals_51
            buf108 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.bfloat16)
            buf109 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.int8)
            # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_37.run(buf107, buf108, buf109, 4194304, stream=stream0)
            buf110 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(primals_52, buf110, 2359296, stream=stream0)
            del primals_52
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
            buf111 = extern_kernels.convolution(buf108, buf110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf111, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution.default')
            buf112 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            buf113 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            buf114 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_39.run(buf111, buf112, buf113, buf114, 32768, 128, stream=stream0)
            buf115 = buf104; del buf104  # reuse
            buf116 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            buf118 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__40.run(buf112, buf113, buf114, primals_54, primals_55, buf115, buf116, buf118, primals_54, primals_55, 512, 64, stream=stream0)
            del buf112
            del buf113
            del buf114
            del primals_54
            del primals_55
            buf119 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_relu_41.run(buf111, buf115, buf116, primals_56, primals_57, buf119, 4194304, stream=stream0)
            del primals_57
            buf120 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_38.run(primals_58, buf120, 2359296, stream=stream0)
            del primals_58
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
            buf121 = extern_kernels.convolution(buf119, buf120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf121, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution.default')
            buf122 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            buf123 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            buf124 = empty_strided_cuda((1, 512, 1, 1, 64), (32768, 1, 32768, 32768, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional]
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_39.run(buf121, buf122, buf123, buf124, 32768, 128, stream=stream0)
            buf125 = buf116; del buf116  # reuse
            buf128 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf122, buf123, buf124, primals_60, primals_61, buf125, buf128, primals_60, primals_61, 512, 64, stream=stream0)
            del buf122
            del buf123
            del buf124
            del primals_60
            del primals_61
            buf129 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_add_relu_43.run(buf108, buf121, buf125, buf128, primals_62, primals_63, buf129, 4194304, stream=stream0)
            buf130 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 1, 1), torch.bfloat16)
            buf131 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 512, 512), torch.int8)
            # Topologically Sorted Source Nodes: [max_pool2d_3], Original ATen: [aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused_max_pool2d_with_indices_44.run(buf129, buf130, buf131, 262144, stream=stream0)
            buf132 = empty_strided_cuda((512, 10), (1, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_45.run(primals_64, buf132, 5120, stream=stream0)
            del primals_64
            buf133 = empty_strided_cuda((512, 10), (10, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, linear], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf130, (512, 512), (512, 1), 0), buf132, out=buf133)
            buf134 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_46.run(buf134, 5120, stream=stream0)
            # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_4, primals_4, 1, stream=stream0)
            del primals_4
            # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_10, primals_10, 1, stream=stream0)
            del primals_10
            # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_16, primals_16, 1, stream=stream0)
            del primals_16
            # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_22, primals_22, 1, stream=stream0)
            del primals_22
            # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_28, primals_28, 1, stream=stream0)
            del primals_28
            # Topologically Sorted Source Nodes: [add__5], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_35, primals_35, 1, stream=stream0)
            del primals_35
            # Topologically Sorted Source Nodes: [add__6], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_41, primals_41, 1, stream=stream0)
            del primals_41
            # Topologically Sorted Source Nodes: [add__7], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_47, primals_47, 1, stream=stream0)
            del primals_47
            # Topologically Sorted Source Nodes: [add__8], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_53, primals_53, 1, stream=stream0)
            del primals_53
            # Topologically Sorted Source Nodes: [add__9], Original ATen: [aten.add, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__47.run(primals_59, primals_59, 1, stream=stream0)
            del primals_59
        return (buf134, primals_7, primals_13, primals_19, primals_25, primals_26, primals_31, primals_33, primals_38, primals_44, primals_45, primals_50, primals_56, primals_62, primals_63, buf2, buf3, buf4, reinterpret_tensor(buf14, (64, ), (1, ), 0), buf15, buf16, buf17, reinterpret_tensor(buf27, (128, ), (1, ), 0), buf28, buf29, buf30, buf31, buf32, reinterpret_tensor(buf42, (128, ), (1, ), 0), buf43, buf44, buf45, buf52, buf55, buf56, buf57, buf58, reinterpret_tensor(buf68, (256, ), (1, ), 0), buf69, buf70, buf71, buf72, buf73, reinterpret_tensor(buf83, (256, ), (1, ), 0), buf84, buf85, buf86, buf93, buf96, buf97, buf98, buf99, reinterpret_tensor(buf106, (512, ), (1, ), 0), buf107, buf108, buf109, buf110, buf111, reinterpret_tensor(buf118, (512, ), (1, ), 0), buf119, buf120, buf121, buf125, buf128, buf129, buf131, reinterpret_tensor(buf130, (512, 512), (512, 1), 0), reinterpret_tensor(buf132, (10, 512), (512, 1), 0), reinterpret_tensor(buf115, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf103, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf80, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf65, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf39, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf11, (1, 64, 1, 1), (64, 1, 1, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((54, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 3, 32, 32), (3072, 1, 96, 3), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, 54, 3, 3), (486, 1, 162, 54), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
