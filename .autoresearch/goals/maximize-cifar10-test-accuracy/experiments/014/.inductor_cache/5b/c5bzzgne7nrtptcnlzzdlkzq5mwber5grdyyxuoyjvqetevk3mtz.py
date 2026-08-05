
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '168E8D72A2911C3B3764FBC9919500356717EE403883CBFEFAE47391A8A6994A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 146800640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
