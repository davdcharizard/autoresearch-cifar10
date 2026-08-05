
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
