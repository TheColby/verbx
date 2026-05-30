# Illustrated Guide

This chapter collects the visual reference material used throughout the guide: signal-flow diagrams, graphs of key DSP tradeoffs, analysis dashboards, and topology sketches. The figures are generated with `python3 scripts/generate_userguide_figures.py` so the PDF can be rebuilt reproducibly.

## System Flow

![End-to-end render signal flow.](assets/userguide_figures/01_signal_flow.png)

![Realtime latency budget by block size.](assets/userguide_figures/02_realtime_latency.png)

![CLI command map.](assets/userguide_figures/18_cli_command_map.png)

![Analysis JSON structure.](assets/userguide_figures/22_json_tree.png)

## Reverb Physics and Analysis

![RT60 decay families.](assets/userguide_figures/03_rt60_decay_families.png)

![Energy decay curve fitting windows.](assets/userguide_figures/04_edc_fit_windows.png)

![Frequency-dependent decay bands.](assets/userguide_figures/09_multiband_decay.png)

![Room size inference curves.](assets/userguide_figures/16_room_size_inference.png)

![Analysis metrics dashboard.](assets/userguide_figures/17_analysis_dashboard.png)

## Algorithms and Processing

![Feedback matrix texture heatmap.](assets/userguide_figures/05_fdn_matrix_heatmap.png)

![Partitioned convolution layout.](assets/userguide_figures/11_partitioned_convolution.png)

![IR morphing blend space.](assets/userguide_figures/12_ir_morph_space.png)

![Shimmer feedback path.](assets/userguide_figures/15_shimmer_feedback.png)

![Infinite-style reverb tail behavior.](assets/userguide_figures/24_infinite_reverb.png)

## Controls and Tradeoffs

![Analysis window function shapes.](assets/userguide_figures/06_window_functions.png)

![Limiter transfer curves.](assets/userguide_figures/07_limiter_transfer.png)

![Reverb ducking envelope.](assets/userguide_figures/08_ducking_envelope.png)

![Dereverb strength versus artifact tradeoff.](assets/userguide_figures/10_dereverb_tradeoff.png)

![Block size CPU and latency tradeoff.](assets/userguide_figures/21_cpu_block_tradeoff.png)

![Preset design radar.](assets/userguide_figures/23_preset_radar.png)

## Spatial and Library Views

![Spatial layout families.](assets/userguide_figures/13_spatial_layouts.png)

![Ambisonics order channel growth.](assets/userguide_figures/14_ambisonics_order.png)

![IR library coverage grid.](assets/userguide_figures/20_ir_library_grid.png)

![Reference corpus shape.](assets/userguide_figures/19_reference_corpus.png)

## Additional Diagnostics and Design Graphs

![Early reflection timing.](assets/userguide_figures/25_early_reflections.png)

![Pre-delay perception curve.](assets/userguide_figures/26_predelay_perception.png)

![Diffusion build-up curve.](assets/userguide_figures/27_diffusion_buildup.png)

![Damping EQ target curves.](assets/userguide_figures/28_damping_eq_targets.png)

![Modulation depth safety heatmap.](assets/userguide_figures/29_modulation_depth_safety.png)

![Stereo width correlation curve.](assets/userguide_figures/30_stereo_width_correlation.png)

![Haas zone timing bands.](assets/userguide_figures/31_haas_zone.png)

![Gate tail shape timeline.](assets/userguide_figures/32_gate_tail_shapes.png)

![Reverse reverb envelope.](assets/userguide_figures/33_reverse_reverb_envelope.png)

![Spectral tilt analyzer.](assets/userguide_figures/34_spectral_tilt_analyzer.png)

![Loudness normalization path.](assets/userguide_figures/35_loudness_normalization_path.png)

![Sample-rate cost chart.](assets/userguide_figures/36_sample_rate_cost.png)

![Oversampling alias guard.](assets/userguide_figures/37_oversampling_alias_guard.png)

![Lookahead limiter timing.](assets/userguide_figures/38_lookahead_limiter_timing.png)

![Dry wet crossfade laws.](assets/userguide_figures/39_dry_wet_crossfade_laws.png)

![IR trim finder timeline.](assets/userguide_figures/40_ir_trim_finder.png)

![Silence detector thresholds.](assets/userguide_figures/41_silence_detector_thresholds.png)

![Batch render throughput.](assets/userguide_figures/42_batch_render_throughput.png)

![Cache hit savings.](assets/userguide_figures/43_cache_hit_savings.png)

![Native parity slice.](assets/userguide_figures/44_native_parity_slice.png)

![Test matrix coverage.](assets/userguide_figures/45_test_matrix_coverage.png)

![Preset morph trajectory.](assets/userguide_figures/46_preset_morph_trajectory.png)

![Realtime dropout risk.](assets/userguide_figures/47_realtime_dropout_risk.png)

![Release readiness dashboard.](assets/userguide_figures/48_release_readiness_dashboard.png)

## Extended Figure Atlas

![Comb filter notches.](assets/userguide_figures/49_comb_filter_notches.png)

![Allpass diffuser response.](assets/userguide_figures/50_allpass_diffuser_response.png)

![FDN delay distribution.](assets/userguide_figures/51_fdn_delay_distribution.png)

![Modal density growth.](assets/userguide_figures/52_modal_density_growth.png)

![Schroeder frequency estimate.](assets/userguide_figures/53_schroeder_frequency_estimate.png)

![Air absorption roll-off.](assets/userguide_figures/54_air_absorption_rolloff.png)

![Material absorption map.](assets/userguide_figures/55_material_absorption_map.png)

![Early late balance.](assets/userguide_figures/56_early_late_balance.png)

![Source distance cue.](assets/userguide_figures/57_source_distance_cue.png)

![Mic pattern pickup.](assets/userguide_figures/58_mic_pattern_pickup.png)

![Sidechain detector modes.](assets/userguide_figures/59_sidechain_detector_modes.png)

![Ducking release families.](assets/userguide_figures/60_ducking_release_families.png)

![Limiter knee families.](assets/userguide_figures/61_limiter_knee_families.png)

![True peak margin.](assets/userguide_figures/62_true_peak_margin.png)

![LUFS integration windows.](assets/userguide_figures/63_lufs_integration_windows.png)

![Crest factor map.](assets/userguide_figures/64_crest_factor_map.png)

![Transient preservation.](assets/userguide_figures/65_transient_preservation.png)

![Dereverb mask strength.](assets/userguide_figures/66_dereverb_mask_strength.png)

![Spectral gate residuals.](assets/userguide_figures/67_spectral_gate_residuals.png)

![Noise floor tracking.](assets/userguide_figures/68_noise_floor_tracking.png)

![Multichannel routing matrix.](assets/userguide_figures/69_multichannel_routing_matrix.png)

![Ambisonic decode spread.](assets/userguide_figures/70_ambisonic_decode_spread.png)

![Binaural HRTF blend.](assets/userguide_figures/71_binaural_hrtf_blend.png)

![Speaker layout coverage.](assets/userguide_figures/72_speaker_layout_coverage.png)

![IR capture checklist.](assets/userguide_figures/73_ir_capture_checklist.png)

![Sweep deconvolution path.](assets/userguide_figures/74_sweep_deconvolution_path.png)

![IR tail trim decision.](assets/userguide_figures/75_ir_tail_trim_decision.png)

![IR normalization modes.](assets/userguide_figures/76_ir_normalization_modes.png)

![Convolution partition plan.](assets/userguide_figures/77_convolution_partition_plan.png)

![FFT size efficiency.](assets/userguide_figures/78_fft_size_efficiency.png)

![SIMD batch shape.](assets/userguide_figures/79_simd_batch_shape.png)

![Memory bandwidth pressure.](assets/userguide_figures/80_memory_bandwidth_pressure.png)

![Thread pool scaling.](assets/userguide_figures/81_thread_pool_scaling.png)

![Realtime callback budget.](assets/userguide_figures/82_realtime_callback_budget.png)

![XRuns by block size.](assets/userguide_figures/83_xruns_by_block_size.png)

![Device buffer stack.](assets/userguide_figures/84_device_buffer_stack.png)

![CLI option families.](assets/userguide_figures/85_cli_option_families.png)

![Preset taxonomy.](assets/userguide_figures/86_preset_taxonomy.png)

![JSON schema coverage.](assets/userguide_figures/87_json_schema_coverage.png)

![Analysis regression bands.](assets/userguide_figures/88_analysis_regression_bands.png)

![Golden audio drift.](assets/userguide_figures/89_golden_audio_drift.png)

![Documentation build pipeline.](assets/userguide_figures/90_documentation_build_pipeline.png)

![Table wrap stress test.](assets/userguide_figures/91_table_wrap_stress_test.png)

![Reference annotation flow.](assets/userguide_figures/92_reference_annotation_flow.png)

![Citation corpus growth.](assets/userguide_figures/93_citation_corpus_growth.png)

![Release branch flow.](assets/userguide_figures/94_release_branch_flow.png)

![Homebrew formula refresh.](assets/userguide_figures/95_homebrew_formula_refresh.png)

![Platform support grid.](assets/userguide_figures/96_platform_support_grid.png)

![Error message quality.](assets/userguide_figures/97_error_message_quality.png)

![User workflow map.](assets/userguide_figures/98_user_workflow_map.png)

![Feature maturity radar.](assets/userguide_figures/99_feature_maturity_radar.png)

![End-to-end confidence map.](assets/userguide_figures/100_end_to_end_confidence_map.png)
