
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
