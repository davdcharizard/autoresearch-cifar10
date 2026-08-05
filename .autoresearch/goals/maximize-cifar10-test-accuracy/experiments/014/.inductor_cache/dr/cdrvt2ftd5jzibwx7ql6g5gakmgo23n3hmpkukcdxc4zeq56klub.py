
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
