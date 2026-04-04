Candidate Card
shape: m16 (exact m16, k=7168)
deleted_cost_center: hand-rolled mxfp4_pack_a_fixed quantization work on exact m16 path (A-pack pack cost)
expected_upside_source: replace per-element bit/branch quantization with CDNA4 FP4 scale-pack builtin, reducing A-pack time (dominant in m16)
why_larger_than_noise: A-pack dominates m16 path; 2x pack speed is >0.2us geomean leverage

touched_symbols_or_regions: fp8-mm/hip_phase2_working.py (mxfp4_pack_a_fixed_kernel, _get_a_contract_mfma_fp4_compiled)
forbidden_edits: no contract changes, no A-pack duplication-law changes, no non-m16 kernel changes, no wrapper/dispatch changes outside pack
success_gate: PACK_DEBUG_COMPARE mismatch == 0; benchmark geomean improves >=0.2us with no shape regression >0.1us
