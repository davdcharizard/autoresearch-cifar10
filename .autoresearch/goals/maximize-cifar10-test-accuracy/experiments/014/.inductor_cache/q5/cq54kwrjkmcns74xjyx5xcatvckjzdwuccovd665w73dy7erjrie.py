# AOT ID: ['1_inference']
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


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/of/cofchrm3oecgwfioas2dkgpmryi3toucb7ysoocwxkf7s6dzadiw.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_1
# Graph fragment:
#   %arg1_1 : Tensor "f32[512, 3, 32, 32][3072, 1, 96, 3]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type_1 : Tensor "bf16[512, 3, 32, 32][3072, 1, 96, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12582912}},
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


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/lo/clofqe7sqg22g2vw5zfrdm2euptt6e2etopr7y5n62cvn3udouow.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type
# Graph fragment:
#   %arg0_1 : Tensor "f32[54, 3, 3, 3][27, 1, 9, 3]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "bf16[54, 3, 3, 3][27, 1, 9, 3]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11664}},
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


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/wa/cwa5q3c3hdkabxk53ko3zwoyublbznpea5gilvp67z4lyyb2bglx.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_2
# Graph fragment:
#   %arg2_1 : Tensor "f32[64, 54, 3, 3][486, 1, 162, 54]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type_2 : Tensor "bf16[64, 54, 3, 3][486, 1, 162, 54]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 248832}},
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


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/fh/cfhddpepy6j3ut5quqobudzukgkovc2fi4ezd63okwq73yvvfvn4.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add, add_1, convert_element_type_5, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   input_3 => relu
# Graph fragment:
#   %convolution_1 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg6_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg4_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %relu : Tensor "bf16[512, 64, 32, 32][65536, 1, 2048, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_5,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ix/cix5zrvylxer7kjlwcypzdbfobaqjvyjhidrcvjvafhemuzrvjww.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_6
# Graph fragment:
#   %arg7_1 : Tensor "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_6 : Tensor "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_6
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/5i/c5iz44inds4l5w4ibk6cwbd4af3vebgs7whit2i3perfaodhtq4b.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_2, add_3, convert_element_type_9, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_6 => relu_1
# Graph fragment:
#   %convolution_2 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg8_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %unsqueeze_8 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg9_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg11_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_9,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402655232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/tc/ctcrkwm3nxco5nssf3tqc6dcbkkk4wtm65efd5py6krvraxe6etm.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_5 => add_2, add_3, convert_element_type_9, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   input_6 => relu_1
#   input_7 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0" = PlaceHolder[target=relu_1]
#   %unsqueeze_8 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg9_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg11_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %relu_1 : Tensor "bf16[512, 128, 32, 32][131072, 1, 4096, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_9,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201326592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/gj/cgjezhg34jrq3ayk77oepsh6hh36gfz4hdnmwv6jem4pj4appf3y.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_8 => convert_element_type_10
# Graph fragment:
#   %arg12_1 : Tensor "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg12_1]
#   %convert_element_type_10 : Tensor "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_10
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/qp/cqpt3hre75c4xjt6hz2kmrrj2b5fmv45npyxqurh2vcnox3okj2r.py
# Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_2
#   input_9 => add_4, add_5, convert_element_type_13, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
# Graph fragment:
#   %convolution_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_3]
#   %arg13_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %unsqueeze_16 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg13_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg14_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg16_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %relu_2 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_13,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100665344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/nq/cnqizapjr7jd42q67drzd5yys7y2gbhj47b7gl7wdf3iuyqut7nm.py
# Topologically Sorted Source Nodes: [input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_12 => add_6, add_7, convert_element_type_17, mul_10, mul_11, mul_9, reciprocal_3, sqrt_3, sub_3, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31
#   input_13 => relu_3
#   input_14 => add_8
# Graph fragment:
#   %getitem : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=getitem]
#   %convolution_4 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = PlaceHolder[target=convolution_4]
#   %arg18_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %arg19_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %arg20_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %unsqueeze_24 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg18_1, -1), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, -1), kwargs = {})
#   %sub_3 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_25), kwargs = {})
#   %add_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg19_1, 1e-05), kwargs = {})
#   %sqrt_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %reciprocal_3 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_3,), kwargs = {})
#   %mul_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1), kwargs = {})
#   %unsqueeze_26 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_9, -1), kwargs = {})
#   %unsqueeze_27 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_26, -1), kwargs = {})
#   %mul_10 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg20_1, -1), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, -1), kwargs = {})
#   %mul_11 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg21_1, -1), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_30, -1), kwargs = {})
#   %add_7 : Tensor "f32[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.bfloat16), kwargs = {})
#   %relu_3 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_17,), kwargs = {})
#   %add_8 : Tensor "bf16[512, 128, 16, 16][32768, 1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %relu_3), kwargs = {})
#   return %add_8
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134219776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1], 0, tl.int32)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/4c/c4c4kw4xeekqtfhd27jbnyoflt4lbhefq5bous724pjk4tb63xst.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_15 => convert_element_type_18
# Graph fragment:
#   %arg22_1 : Tensor "f32[320, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_18 : Tensor "bf16[320, 128, 3, 3][1152, 1, 384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_18
triton_poi_fused__to_copy_10 = async_compile.triton('triton_poi_fused__to_copy_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2949120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/g5/cg5q44azln2x2jvkgobcd7mqdzqpxcf3vckknj6aup33kfjrn66q.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_10, add_9, convert_element_type_21, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_17 => relu_4
# Graph fragment:
#   %convolution_5 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=convolution_5]
#   %arg23_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %arg24_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %arg26_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %unsqueeze_32 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg23_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_33), kwargs = {})
#   %add_9 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg26_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_10 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   %relu_4 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_21,), kwargs = {})
#   return %relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 251663360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41943040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/7y/c7ytdokpiwi3hw7ie23j6kako7btfhmrjjfc434d52tt2e3l7ebk.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_16 => add_10, add_9, convert_element_type_21, mul_12, mul_13, mul_14, reciprocal_4, sqrt_4, sub_4, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39
#   input_17 => relu_4
#   input_18 => _low_memory_max_pool_with_offsets_1
# Graph fragment:
#   %relu_4 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0" = PlaceHolder[target=relu_4]
#   %unsqueeze_32 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg23_1, -1), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, -1), kwargs = {})
#   %sub_4 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_33), kwargs = {})
#   %add_9 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg24_1, 1e-05), kwargs = {})
#   %sqrt_4 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %reciprocal_4 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_4,), kwargs = {})
#   %mul_12 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_4, 1), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, -1), kwargs = {})
#   %mul_13 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, -1), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_36, -1), kwargs = {})
#   %mul_14 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg26_1, -1), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, -1), kwargs = {})
#   %add_10 : Tensor "f32[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
#   %relu_4 : Tensor "bf16[512, 320, 16, 16][81920, 1, 5120, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_21,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_4, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_2
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 125829120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 320)
    x1 = ((xindex // 320) % 8)
    x2 = xindex // 2560
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 640*x1 + 10240*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (320 + x0 + 640*x1 + 10240*x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (5120 + x0 + 640*x1 + 10240*x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (5440 + x0 + 640*x1 + 10240*x2), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/hu/chuzjqmcw2wxvmkeefbevqupurw74z7ngdq3hvautepjzdfn3tsg.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_19 => convert_element_type_22
# Graph fragment:
#   %arg28_1 : Tensor "f32[320, 320, 3, 3][2880, 1, 960, 320]cuda:0" = PlaceHolder[target=arg28_1]
#   %convert_element_type_22 : Tensor "bf16[320, 320, 3, 3][2880, 1, 960, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg28_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_22
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7372800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 921600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/kd/ckdpzdmlx7ccq5nbyakghgavgu4dxzzg2bkaug6c2lwmynsiesb5.py
# Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_20 => add_11, add_12, convert_element_type_25, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   input_21 => relu_5
# Graph fragment:
#   %convolution_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_6]
#   %arg29_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %arg31_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %arg32_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %unsqueeze_40 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_41), kwargs = {})
#   %add_11 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg30_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_11,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg31_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_12 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %convert_element_type_25 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
#   %relu_5 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_25,), kwargs = {})
#   return %relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 62919680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ay/cayw6fngcp2m4563t2hch6dpj44fdiv64mhk6kepttrw7q6ceubd.py
# Topologically Sorted Source Nodes: [input_23, input_24, mul, input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   input_23 => add_13, add_14, convert_element_type_29, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   input_24 => relu_6
#   input_25 => add_15
#   input_26 => convert_element_type_31
#   mul => mul_21
# Graph fragment:
#   %getitem_2 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=getitem_2]
#   %arg27_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %convolution_7 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg34_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %arg37_1 : Tensor "f32[320][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %unsqueeze_48 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_49), kwargs = {})
#   %add_13 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg35_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[320][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg36_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[320, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[320, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_14 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.bfloat16), kwargs = {})
#   %relu_6 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_29,), kwargs = {})
#   %mul_21 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg27_1, %relu_6), kwargs = {})
#   %add_15 : Tensor "f32[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %mul_21), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[512, 320, 8, 8][20480, 1, 2560, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   return %convert_element_type_31
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_mul_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_mul_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_mul_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 83891200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_mul_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = (tmp12 / tmp11)
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp3 * tmp24
    tmp26 = tmp1 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp27, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/pj/cpjiygd3jdl6quncvkigyxgu4ib6bwgmu3hrh5fzv62gganhiey6.py
# Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_26 => convert_element_type_30
# Graph fragment:
#   %arg38_1 : Tensor "f32[512, 320, 3, 3][2880, 1, 960, 320]cuda:0" = PlaceHolder[target=arg38_1]
#   %convert_element_type_30 : Tensor "bf16[512, 320, 3, 3][2880, 1, 960, 320]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg38_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_30
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 11796480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1474560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/uq/cuqhhiszsvmhnxletkdmhn5yy632cze6t2nbrzr5xgesunxz2xt3.py
# Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_27 => add_16, add_17, convert_element_type_34, mul_22, mul_23, mul_24, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   input_28 => relu_7
# Graph fragment:
#   %convolution_8 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=convolution_8]
#   %arg39_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %arg41_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg41_1]
#   %arg42_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg42_1]
#   %unsqueeze_56 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_57), kwargs = {})
#   %add_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg40_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_22 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_22, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_23 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg41_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_24 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_17 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_63), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_34,), kwargs = {})
#   return %relu_7
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100671488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ei/ceiet73uknogtkmpfuywhejgyr5gmeeno5qs3aetvgkqyepqkg3s.py
# Topologically Sorted Source Nodes: [input_27, input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_27 => add_16, add_17, convert_element_type_34, mul_22, mul_23, mul_24, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   input_28 => relu_7
#   input_29 => _low_memory_max_pool_with_offsets_2
# Graph fragment:
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = PlaceHolder[target=relu_7]
#   %unsqueeze_56 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_57), kwargs = {})
#   %add_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg40_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_22 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_22, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_23 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg41_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_24 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg42_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_17 : Tensor "f32[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_63), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.bfloat16), kwargs = {})
#   %relu_7 : Tensor "bf16[512, 512, 8, 8][32768, 1, 4096, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_34,), kwargs = {})
#   %_low_memory_max_pool_with_offsets_2 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu_7, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_4
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 50331648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/ar/carf5hnilusqteyb763zja3pohcjb6xltbhmaiwtmxymyuhhouna.py
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_30 => convert_element_type_35
# Graph fragment:
#   %arg43_1 : Tensor "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = PlaceHolder[target=arg43_1]
#   %convert_element_type_35 : Tensor "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_35
triton_poi_fused__to_copy_19 = async_compile.triton('triton_poi_fused__to_copy_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/is/cisdkig6q4z4mokfjeoyzdixf52rziobrsf5ellxk4qug4o6xmis.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_31 => add_18, add_19, convert_element_type_38, mul_25, mul_26, mul_27, reciprocal_8, sqrt_8, sub_8, unsqueeze_64, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, unsqueeze_71
#   input_32 => relu_8
# Graph fragment:
#   %convolution_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_9]
#   %arg44_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %arg45_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg45_1]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %arg47_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %unsqueeze_64 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg44_1, -1), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_64, -1), kwargs = {})
#   %sub_8 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_65), kwargs = {})
#   %add_18 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg45_1, 1e-05), kwargs = {})
#   %sqrt_8 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_18,), kwargs = {})
#   %reciprocal_8 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_8,), kwargs = {})
#   %mul_25 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_8, 1), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_25, -1), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, -1), kwargs = {})
#   %mul_26 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg46_1, -1), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, -1), kwargs = {})
#   %mul_27 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_69), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg47_1, -1), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_70, -1), kwargs = {})
#   %add_19 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_71), kwargs = {})
#   %convert_element_type_38 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
#   %relu_8 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_38,), kwargs = {})
#   return %relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25174016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/gh/cghv7id6fno6mrmqvasfqdopmjv2lbmcmb4qkye4qpfhpzz5ahmc.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_34 => add_20, add_21, convert_element_type_42, mul_28, mul_29, mul_30, reciprocal_9, sqrt_9, sub_9, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, unsqueeze_77, unsqueeze_78, unsqueeze_79
#   input_35 => relu_9
#   input_36 => add_22
# Graph fragment:
#   %getitem_4 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=getitem_4]
#   %convolution_10 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=convolution_10]
#   %arg49_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg49_1]
#   %arg50_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg50_1]
#   %arg51_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg51_1]
#   %arg52_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %unsqueeze_72 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg49_1, -1), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, -1), kwargs = {})
#   %sub_9 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_73), kwargs = {})
#   %add_20 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg50_1, 1e-05), kwargs = {})
#   %sqrt_9 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_20,), kwargs = {})
#   %reciprocal_9 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_9,), kwargs = {})
#   %mul_28 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 1), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_28, -1), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, -1), kwargs = {})
#   %mul_29 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg51_1, -1), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, -1), kwargs = {})
#   %mul_30 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_77), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, -1), kwargs = {})
#   %add_21 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %unsqueeze_79), kwargs = {})
#   %convert_element_type_42 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.bfloat16), kwargs = {})
#   %relu_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_42,), kwargs = {})
#   %add_22 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %relu_9), kwargs = {})
#   return %add_22
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33562624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1], 0, tl.int32)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tmp0 + tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/dr/cdrvt2ftd5jzibwx7ql6g5gakmgo23n3hmpkukcdxc4zeq56klub.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36, max_pool2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_34 => add_20, add_21, convert_element_type_42, mul_28, mul_29, mul_30, reciprocal_9, sqrt_9, sub_9, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, unsqueeze_77, unsqueeze_78, unsqueeze_79
#   input_35 => relu_9
#   input_36 => add_22
#   max_pool2d_3 => _low_memory_max_pool_with_offsets_3
# Graph fragment:
#   %add_22 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0" = PlaceHolder[target=add_22]
#   %unsqueeze_72 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg49_1, -1), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, -1), kwargs = {})
#   %sub_9 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_73), kwargs = {})
#   %add_20 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg50_1, 1e-05), kwargs = {})
#   %sqrt_9 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_20,), kwargs = {})
#   %reciprocal_9 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_9,), kwargs = {})
#   %mul_28 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 1), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_28, -1), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, -1), kwargs = {})
#   %mul_29 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg51_1, -1), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, -1), kwargs = {})
#   %mul_30 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_77), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, -1), kwargs = {})
#   %add_21 : Tensor "f32[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %unsqueeze_79), kwargs = {})
#   %convert_element_type_42 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.bfloat16), kwargs = {})
#   %relu_9 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_42,), kwargs = {})
#   %add_22 : Tensor "bf16[512, 512, 4, 4][8192, 1, 2048, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %relu_9), kwargs = {})
#   %_low_memory_max_pool_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%add_22, [4, 4], [4, 4], [0, 0], [1, 1], False), kwargs = {})
#   return %getitem_6
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=78, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/3n/c3nolgsjw54znz24lvnzp3ivgnnuqufgv2vc2vfaqzvqlcfuec3f.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_43
# Graph fragment:
#   %arg53_1 : Tensor "f32[10, 512][512, 1]cuda:0" = PlaceHolder[target=arg53_1]
#   %convert_element_type_43 : Tensor "bf16[10, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg53_1, torch.bfloat16), kwargs = {})
#   return %convert_element_type_43
triton_poi_fused__to_copy_23 = async_compile.triton('triton_poi_fused__to_copy_23', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/014/.inductor_cache/6y/c6y2zaituqfcbg5bb55spdhodph5zuemm4ezwzdyalhczgmt2oyl.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_31
# Graph fragment:
#   %mm : Tensor "bf16[512, 10][10, 1]cuda:0" = PlaceHolder[target=mm]
#   %mul_31 : Tensor "bf16[512, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 0.125), kwargs = {})
#   return %mul_31
triton_poi_fused_mul_24 = async_compile.triton('triton_poi_fused_mul_24', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 30720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_24(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1 = args
        args.clear()
        assert_size_stride(arg0_1, (54, 3, 3, 3), (27, 1, 9, 3))
        assert_size_stride(arg1_1, (512, 3, 32, 32), (3072, 1, 96, 3))
        assert_size_stride(arg2_1, (64, 54, 3, 3), (486, 1, 162, 54))
        assert_size_stride(arg3_1, (64, ), (1, ))
        assert_size_stride(arg4_1, (64, ), (1, ))
        assert_size_stride(arg5_1, (64, ), (1, ))
        assert_size_stride(arg6_1, (64, ), (1, ))
        assert_size_stride(arg7_1, (128, 64, 3, 3), (576, 1, 192, 64))
        assert_size_stride(arg8_1, (128, ), (1, ))
        assert_size_stride(arg9_1, (128, ), (1, ))
        assert_size_stride(arg10_1, (128, ), (1, ))
        assert_size_stride(arg11_1, (128, ), (1, ))
        assert_size_stride(arg12_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg13_1, (128, ), (1, ))
        assert_size_stride(arg14_1, (128, ), (1, ))
        assert_size_stride(arg15_1, (128, ), (1, ))
        assert_size_stride(arg16_1, (128, ), (1, ))
        assert_size_stride(arg17_1, (128, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg18_1, (128, ), (1, ))
        assert_size_stride(arg19_1, (128, ), (1, ))
        assert_size_stride(arg20_1, (128, ), (1, ))
        assert_size_stride(arg21_1, (128, ), (1, ))
        assert_size_stride(arg22_1, (320, 128, 3, 3), (1152, 1, 384, 128))
        assert_size_stride(arg23_1, (320, ), (1, ))
        assert_size_stride(arg24_1, (320, ), (1, ))
        assert_size_stride(arg25_1, (320, ), (1, ))
        assert_size_stride(arg26_1, (320, ), (1, ))
        assert_size_stride(arg27_1, (1, ), (1, ))
        assert_size_stride(arg28_1, (320, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(arg29_1, (320, ), (1, ))
        assert_size_stride(arg30_1, (320, ), (1, ))
        assert_size_stride(arg31_1, (320, ), (1, ))
        assert_size_stride(arg32_1, (320, ), (1, ))
        assert_size_stride(arg33_1, (320, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(arg34_1, (320, ), (1, ))
        assert_size_stride(arg35_1, (320, ), (1, ))
        assert_size_stride(arg36_1, (320, ), (1, ))
        assert_size_stride(arg37_1, (320, ), (1, ))
        assert_size_stride(arg38_1, (512, 320, 3, 3), (2880, 1, 960, 320))
        assert_size_stride(arg39_1, (512, ), (1, ))
        assert_size_stride(arg40_1, (512, ), (1, ))
        assert_size_stride(arg41_1, (512, ), (1, ))
        assert_size_stride(arg42_1, (512, ), (1, ))
        assert_size_stride(arg43_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg44_1, (512, ), (1, ))
        assert_size_stride(arg45_1, (512, ), (1, ))
        assert_size_stride(arg46_1, (512, ), (1, ))
        assert_size_stride(arg47_1, (512, ), (1, ))
        assert_size_stride(arg48_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
        assert_size_stride(arg49_1, (512, ), (1, ))
        assert_size_stride(arg50_1, (512, ), (1, ))
        assert_size_stride(arg51_1, (512, ), (1, ))
        assert_size_stride(arg52_1, (512, ), (1, ))
        assert_size_stride(arg53_1, (10, 512), (512, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((512, 3, 32, 32), (3072, 1, 96, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_0.run(arg1_1, buf0, 1572864, stream=stream0)
            del arg1_1
            buf1 = empty_strided_cuda((54, 3, 3, 3), (27, 1, 9, 3), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_1.run(arg0_1, buf1, 1458, stream=stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy, aten.convolution]
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (512, 54, 32, 32), (55296, 1, 1728, 54), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            buf3 = empty_strided_cuda((64, 54, 3, 3), (486, 1, 162, 54), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(arg2_1, buf3, 31104, stream=stream0)
            del arg2_1
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.convolution]
            buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf4, (512, 64, 32, 32), (65536, 1, 2048, 64), 'torch.ops.aten.convolution.default')
            del buf2
            del buf3
            buf5 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, arg3_1, arg4_1, arg5_1, arg6_1, 33554432, stream=stream0)
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            buf6 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_4.run(arg7_1, buf6, 73728, stream=stream0)
            del arg7_1
            # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
            buf7 = extern_kernels.convolution(buf5, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf7, (512, 128, 32, 32), (131072, 1, 4096, 128), 'torch.ops.aten.convolution.default')
            del buf5
            del buf6
            buf8 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, arg8_1, arg9_1, arg10_1, arg11_1, 67108864, stream=stream0)
            del arg10_1
            del arg11_1
            del arg8_1
            del arg9_1
            buf9 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_6.run(buf8, buf9, 16777216, stream=stream0)
            del buf8
            buf10 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg12_1, buf10, 147456, stream=stream0)
            del arg12_1
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._to_copy, aten.convolution]
            buf11 = extern_kernels.convolution(buf9, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf11, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf12, arg13_1, arg14_1, arg15_1, arg16_1, 16777216, stream=stream0)
            del arg13_1
            del arg14_1
            del arg15_1
            del arg16_1
            buf13 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(arg17_1, buf13, 147456, stream=stream0)
            del arg17_1
            # Topologically Sorted Source Nodes: [input_9, input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
            buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf14, (512, 128, 16, 16), (32768, 1, 2048, 128), 'torch.ops.aten.convolution.default')
            del buf12
            del buf13
            buf15 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [input_12, input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf15, buf14, arg18_1, arg19_1, arg20_1, arg21_1, 16777216, stream=stream0)
            del arg18_1
            del arg19_1
            del arg20_1
            del arg21_1
            del buf14
            buf16 = empty_strided_cuda((320, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_10.run(arg22_1, buf16, 368640, stream=stream0)
            del arg22_1
            # Topologically Sorted Source Nodes: [input_12, input_13, input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten._to_copy, aten.convolution]
            buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf17, (512, 320, 16, 16), (81920, 1, 5120, 320), 'torch.ops.aten.convolution.default')
            del buf15
            del buf16
            buf18 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf18, arg23_1, arg24_1, arg25_1, arg26_1, 41943040, stream=stream0)
            del arg23_1
            del arg24_1
            del arg25_1
            del arg26_1
            buf19 = empty_strided_cuda((512, 320, 8, 8), (20480, 1, 2560, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_16, input_17, input_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_12.run(buf18, buf19, 10485760, stream=stream0)
            del buf18
            buf20 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg28_1, buf20, 921600, stream=stream0)
            del arg28_1
            # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten._to_copy, aten.convolution]
            buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf21, (512, 320, 8, 8), (20480, 1, 2560, 320), 'torch.ops.aten.convolution.default')
            buf22 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf22, arg29_1, arg30_1, arg31_1, arg32_1, 10485760, stream=stream0)
            del arg29_1
            del arg30_1
            del arg31_1
            del arg32_1
            buf23 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_13.run(arg33_1, buf23, 921600, stream=stream0)
            del arg33_1
            # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
            buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf24, (512, 320, 8, 8), (20480, 1, 2560, 320), 'torch.ops.aten.convolution.default')
            del buf22
            del buf23
            buf25 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [input_23, input_24, mul, input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.add, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_mul_relu_15.run(buf25, arg27_1, buf24, arg34_1, arg35_1, arg36_1, arg37_1, 10485760, stream=stream0)
            del arg27_1
            del arg34_1
            del arg35_1
            del arg36_1
            del arg37_1
            del buf24
            buf26 = empty_strided_cuda((512, 320, 3, 3), (2880, 1, 960, 320), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_16.run(arg38_1, buf26, 1474560, stream=stream0)
            del arg38_1
            # Topologically Sorted Source Nodes: [input_23, input_24, mul, input_25, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul, aten.add, aten._to_copy, aten.convolution]
            buf27 = extern_kernels.convolution(buf25, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf27, (512, 512, 8, 8), (32768, 1, 4096, 512), 'torch.ops.aten.convolution.default')
            del buf25
            del buf26
            buf28 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf28, arg39_1, arg40_1, arg41_1, arg42_1, 16777216, stream=stream0)
            del arg39_1
            del arg40_1
            del arg41_1
            del arg42_1
            buf29 = empty_strided_cuda((512, 512, 4, 4), (8192, 1, 2048, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_27, input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_18.run(buf28, buf29, 4194304, stream=stream0)
            del buf28
            buf30 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_19.run(arg43_1, buf30, 2359296, stream=stream0)
            del arg43_1
            # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._to_copy, aten.convolution]
            buf31 = extern_kernels.convolution(buf29, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf31, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution.default')
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf32, arg44_1, arg45_1, arg46_1, arg47_1, 4194304, stream=stream0)
            del arg44_1
            del arg45_1
            del arg46_1
            del arg47_1
            buf33 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_19.run(arg48_1, buf33, 2359296, stream=stream0)
            del arg48_1
            # Topologically Sorted Source Nodes: [input_31, input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
            buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf34, (512, 512, 4, 4), (8192, 1, 2048, 512), 'torch.ops.aten.convolution.default')
            del buf32
            del buf33
            buf35 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_21.run(buf35, buf34, arg49_1, arg50_1, arg51_1, arg52_1, 4194304, stream=stream0)
            del arg49_1
            del arg50_1
            del arg51_1
            del arg52_1
            del buf34
            buf36 = empty_strided_cuda((512, 512, 1, 1), (512, 1, 1, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [input_34, input_35, input_36, max_pool2d_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.max_pool2d_with_indices]
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_22.run(buf35, buf36, 262144, stream=stream0)
            del buf35
            buf37 = empty_strided_cuda((10, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_23.run(arg53_1, buf37, 5120, stream=stream0)
            del arg53_1
            buf38 = empty_strided_cuda((512, 10), (10, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x_1, linear], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf36, (512, 512), (512, 1), 0), reinterpret_tensor(buf37, (512, 10), (1, 512), 0), out=buf38)
            del buf36
            del buf37
            buf39 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_24.run(buf39, 5120, stream=stream0)
        return (buf39, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((54, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 3, 32, 32), (3072, 1, 96, 3), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 54, 3, 3), (486, 1, 162, 54), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((320, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((320, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((320, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, 320, 3, 3), (2880, 1, 960, 320), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((10, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
