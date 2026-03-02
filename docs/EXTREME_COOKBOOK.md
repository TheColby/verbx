# verbx Extreme Workflow Cookbook (100 Recipes)

This cookbook is a high-intensity command library for experimental and
large-space reverb design in `verbx`.

Assumptions:

- Input file is `in.wav`
- Output files are written into `out/`
- Replace IR names with your own files as needed

Create an output folder first:

```bash
mkdir -p out
```

## 1) Algorithmic Extremes (1-10)

1. `verbx render in.wav out/001_algo_long.wav --engine algo --rt60 120 --wet 0.95 --dry 0.1 --beast-mode 6`
2. `verbx render in.wav out/002_algo_dark.wav --engine algo --rt60 90 --damping 0.9 --wet 0.9 --dry 0.15`
3. `verbx render in.wav out/003_algo_wide.wav --engine algo --rt60 75 --width 2.0 --wet 0.85 --dry 0.2`
4. `verbx render in.wav out/004_algo_mod_slow.wav --engine algo --rt60 80 --mod-depth-ms 12 --mod-rate-hz 0.05`
5. `verbx render in.wav out/005_algo_mod_fast.wav --engine algo --rt60 65 --mod-depth-ms 8 --mod-rate-hz 1.2`
6. `verbx render in.wav out/006_algo_predelay_cloud.wav --engine algo --rt60 70 --pre-delay-ms 220 --wet 0.9`
7. `verbx render in.wav out/007_algo_splash.wav --engine algo --rt60 45 --beast-mode 10 --repeat 2`
8. `verbx render in.wav out/008_algo_massive.wav --engine algo --rt60 180 --wet 1.0 --dry 0.0 --beast-mode 12`
9. `verbx render in.wav out/009_algo_air.wav --engine algo --rt60 50 --damping 0.2 --highcut 18000 --tilt 3`
10. `verbx render in.wav out/010_algo_mud.wav --engine algo --rt60 110 --lowcut 30 --highcut 1800 --tilt -5`

## 2) Freeze + Repeat Chains (11-20)

11. `verbx render in.wav out/011_freeze_short.wav --freeze --start 2 --end 3 --repeat 3 --engine algo`
12. `verbx render in.wav out/012_freeze_wide.wav --freeze --start 4 --end 6 --repeat 4 --width 1.8`
13. `verbx render in.wav out/013_freeze_beast.wav --freeze --start 1 --end 2.2 --repeat 5 --beast-mode 14`
14. `verbx render in.wav out/014_freeze_dark.wav --freeze --start 3 --end 5 --repeat 3 --damping 0.85 --tilt -4`
15. `verbx render in.wav out/015_freeze_shimmer.wav --freeze --start 2.5 --end 4 --repeat 2 --shimmer`
16. `verbx render in.wav out/016_repeat_algo.wav --engine algo --rt60 55 --repeat 6 --normalize-stage per-pass`
17. `verbx render in.wav out/017_repeat_conv.wav --engine conv --ir hall.wav --repeat 4 --normalize-stage per-pass`
18. `verbx render in.wav out/018_repeat_duck.wav --engine algo --repeat 5 --duck --duck-attack 5 --duck-release 800`
19. `verbx render in.wav out/019_repeat_bloom.wav --engine algo --repeat 4 --bloom 4.5 --wet 0.95`
20. `verbx render in.wav out/020_repeat_floor.wav --engine algo --rt60 140 --repeat 3 --target-lufs -26`

## 3) Convolution Heavy Modes (21-30)

21. `verbx render in.wav out/021_conv_hall.wav --engine conv --ir hall.wav --partition-size 32768`
22. `verbx render in.wav out/022_conv_plate.wav --engine conv --ir plate.wav --ir-normalize peak`
23. `verbx render in.wav out/023_conv_church.wav --engine conv --ir church.wav --tail-limit 150`
24. `verbx render in.wav out/024_conv_mono_to_all.wav --engine conv --ir mono_ir.wav --wet 0.9 --dry 0.2`
25. `verbx render in.wav out/025_conv_surround.wav --engine conv --ir matrix_5p1.wav --ir-matrix-layout output-major`
26. `verbx render in.wav out/026_conv_input_major.wav --engine conv --ir matrix_5p1.wav --ir-matrix-layout input-major`
27. `verbx render in.wav out/027_conv_fast.wav --engine conv --ir hall.wav --partition-size 65536 --normalize-stage none`
28. `verbx render in.wav out/028_conv_dense.wav --engine conv --ir hall.wav --repeat 3 --beast-mode 5`
29. `verbx render in.wav out/029_conv_trimmed.wav --engine conv --ir hall.wav --tail-limit 20`
30. `verbx render in.wav out/030_conv_fulltail.wav --engine conv --ir hall.wav --output-peak-norm input`

## 4) Self-Convolution and Feedback Smear (31-40)

31. `verbx render in.wav out/031_self_base.wav --self-convolve --normalize-stage none`
32. `verbx render in.wav out/032_self_beast.wav --self-convolve --beast-mode 20 --normalize-stage none`
33. `verbx render in.wav out/033_self_longtail.wav --self-convolve --tail-limit 200 --partition-size 32768`
34. `verbx render in.wav out/034_self_bright.wav --self-convolve --tilt 5 --highcut 18000`
35. `verbx render in.wav out/035_self_dark.wav --self-convolve --tilt -6 --highcut 2500`
36. `verbx render in.wav out/036_self_duck.wav --self-convolve --duck --duck-attack 3 --duck-release 600`
37. `verbx render in.wav out/037_self_shimmer.wav --self-convolve --shimmer --shimmer-mix 0.55`
38. `verbx render in.wav out/038_self_repeat.wav --self-convolve --repeat 3 --normalize-stage per-pass`
39. `verbx render in.wav out/039_self_loud.wav --self-convolve --target-lufs -16 --target-peak-dbfs -1`
40. `verbx render in.wav out/040_self_huge.wav --self-convolve --beast-mode 40 --repeat 2`

## 5) Shimmer, Ducking, Bloom, Tilt (41-50)

41. `verbx render in.wav out/041_shimmer_oct.wav --engine algo --shimmer --shimmer-semitones 12 --shimmer-mix 0.35`
42. `verbx render in.wav out/042_shimmer_double.wav --engine algo --shimmer --shimmer-semitones 24 --shimmer-feedback 0.8`
43. `verbx render in.wav out/043_shimmer_fifth.wav --engine algo --shimmer --shimmer-semitones 7 --shimmer-mix 0.5`
44. `verbx render in.wav out/044_duck_hard.wav --engine algo --duck --duck-attack 2 --duck-release 900`
45. `verbx render in.wav out/045_duck_soft.wav --engine algo --duck --duck-attack 60 --duck-release 180`
46. `verbx render in.wav out/046_bloom_long.wav --engine algo --bloom 5 --rt60 90`
47. `verbx render in.wav out/047_bloom_shimmer.wav --engine algo --bloom 3 --shimmer --shimmer-mix 0.4`
48. `verbx render in.wav out/048_tilt_up.wav --engine algo --tilt 6 --lowcut 120`
49. `verbx render in.wav out/049_tilt_down.wav --engine algo --tilt -6 --highcut 4500`
50. `verbx render in.wav out/050_combo_extreme.wav --engine algo --duck --bloom 4 --tilt 4 --shimmer --beast-mode 8`

## 6) Loudness and Output Format Stress (51-60)

51. `verbx render in.wav out/051_lufs_24.wav --target-lufs -24 --target-peak-dbfs -2`
52. `verbx render in.wav out/052_lufs_18.wav --target-lufs -18 --target-peak-dbfs -1 --true-peak`
53. `verbx render in.wav out/053_sample_peak.wav --target-peak-dbfs -0.5 --sample-peak`
54. `verbx render in.wav out/054_per_pass.wav --repeat 4 --normalize-stage per-pass --repeat-target-lufs -22`
55. `verbx render in.wav out/055_no_limiter.wav --engine algo --rt60 70 --no-limiter`
56. `verbx render in.wav out/056_float32.wav --engine conv --ir hall.wav --out-subtype float32`
57. `verbx render in.wav out/057_float64.wav --engine conv --ir hall.wav --out-subtype float64`
58. `verbx render in.wav out/058_pcm24.wav --engine conv --ir hall.wav --out-subtype pcm24`
59. `verbx render in.wav out/059_peak_input.wav --engine algo --output-peak-norm input`
60. `verbx render in.wav out/060_peak_target.wav --engine algo --output-peak-norm target --output-peak-target-dbfs -9`

## 7) Synthetic IR Factory Workflows (61-70)

61. `verbx ir gen out/061_ir_hybrid.wav --mode hybrid --length 120 --seed 61`
62. `verbx ir gen out/062_ir_fdn.wav --mode fdn --length 180 --fdn-lines 12 --seed 62`
63. `verbx ir gen out/063_ir_stochastic.wav --mode stochastic --length 240 --density 1.8 --seed 63`
64. `verbx ir gen out/064_ir_modal.wav --mode modal --length 90 --modal-count 96 --seed 64`
65. `verbx ir gen out/065_ir_tuned.wav --mode modal --length 120 --f0 64Hz --seed 65`
66. `verbx ir gen out/066_ir_from_input.wav --mode hybrid --analyze-input in.wav --seed 66`
67. `verbx ir gen out/067_ir_resonator.wav --mode hybrid --resonator --resonator-mix 0.6 --seed 67`
68. `verbx ir process out/067_ir_resonator.wav out/068_ir_processed.wav --tilt -4 --normalize peak`
69. `verbx ir analyze out/068_ir_processed.wav --json-out out/069_ir_analysis.json`
70. `verbx render in.wav out/070_ir_render.wav --engine conv --ir out/068_ir_processed.wav --repeat 2`

## 8) Multichannel and Spatial Chaos (71-80)

71. `verbx render in_5p1.wav out/071_5p1_conv.wav --engine conv --ir ir_5p1_matrix.wav --ir-matrix-layout output-major`
72. `verbx render in_7p1.wav out/072_7p1_conv.wav --engine conv --ir ir_7p1_matrix.wav --ir-matrix-layout output-major`
73. `verbx render in_5p1.wav out/073_5p1_algo.wav --engine algo --rt60 85 --width 1.6`
74. `verbx render in_7p1.wav out/074_7p1_algo_beast.wav --engine algo --beast-mode 12 --repeat 3`
75. `verbx render in_5p1.wav out/075_5p1_freeze.wav --engine algo --freeze --start 3 --end 4.5 --repeat 2`
76. `verbx render in_7p1.wav out/076_7p1_shimmer.wav --engine algo --shimmer --shimmer-mix 0.45`
77. `verbx render in_5p1.wav out/077_5p1_target.wav --target-lufs -23 --target-peak-dbfs -2`
78. `verbx render in_7p1.wav out/078_7p1_fullscale.wav --output-peak-norm full-scale --out-subtype float32`
79. `verbx render in_5p1.wav out/079_5p1_tailcap.wav --engine conv --ir ir_5p1_matrix.wav --tail-limit 12`
80. `verbx render in_7p1.wav out/080_7p1_longtail.wav --engine conv --ir ir_7p1_matrix.wav --tail-limit 240`

## 9) Tempo, Suggest, Analyze, and Batch Pressure (81-90)

81. `verbx render in.wav out/081_predelay_8d.wav --pre-delay 1/8D --bpm 96 --engine algo`
82. `verbx render in.wav out/082_predelay_triplet.wav --pre-delay 1/16T --bpm 132 --engine algo`
83. `verbx analyze in.wav --lufs --edr --json-out out/083_analysis.json`
84. `verbx analyze in.wav --frames-out out/084_frames.csv --edr`
85. `verbx suggest in.wav`
86. `verbx batch template > out/086_manifest.json`
87. `verbx batch render manifest.json --jobs 8 --schedule longest-first --retries 1`
88. `verbx batch render manifest.json --jobs 4 --schedule shortest-first --dry-run`
89. `verbx cache info`
90. `verbx cache clear`

## 10) Lucky-Mode Wildcards (91-100)

91. `verbx render in.wav out/lucky.wav --lucky 5 --lucky-out-dir out/lucky_01`
92. `verbx render in.wav out/lucky.wav --lucky 10 --lucky-out-dir out/lucky_02 --lucky-seed 2026`
93. `verbx render in.wav out/lucky.wav --lucky 25 --lucky-out-dir out/lucky_03 --device auto`
94. `verbx render in.wav out/lucky.wav --lucky 50 --lucky-out-dir out/lucky_04 --no-progress`
95. `verbx render in.wav out/lucky.wav --lucky 8 --lucky-out-dir out/lucky_05 --engine algo`
96. `verbx render in.wav out/lucky.wav --lucky 8 --lucky-out-dir out/lucky_06 --engine conv --ir hall.wav`
97. `verbx render in.wav out/lucky.wav --lucky 12 --lucky-out-dir out/lucky_07 --self-convolve`
98. `verbx render in.wav out/lucky.wav --lucky 15 --lucky-out-dir out/lucky_08 --target-lufs -20`
99. `verbx render in.wav out/lucky.wav --lucky 30 --lucky-out-dir out/lucky_09 --out-subtype float32`
100. `verbx render in.wav out/lucky.wav --lucky 100 --lucky-out-dir out/lucky_10 --lucky-seed 404`

## Notes for Large Runs

- Start with `--lucky 3` to confirm runtime and output expectations.
- Use `--no-progress` for log-friendly batch terminals.
- Keep `--lucky-seed` fixed for reproducible extreme sets.
- Long tails and high repeats can create large output files; monitor storage.

