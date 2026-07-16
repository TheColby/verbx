# CLI Reference

This file is generated from live CLI help output via `scripts/generate_cli_reference.py`.
Do not edit manually.

## `verbx --help`

```text

 Usage: root [OPTIONS] COMMAND [ARGS]...

 Extreme reverb CLI with scalable DSP architecture.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ realtime    Run realtime duplex monitoring with selectable input/output      │
│             devices.                                                         │
│ room-model  Inspect a room geometry or infer one from RT60 and absorption.   │
│ version     Print CLI/package version.                                       │
│ quickstart  Print minimal copy/paste commands for first successful renders.  │
│ doctor      Print runtime diagnostics for launch-day troubleshooting.        │
│ analyze     Analyze an audio file and print a summary table.                 │
│ compare     Side-by-side comparison of two audio files.                      │
│ presets     Print available presets or one preset payload.                   │
│ suggest     Suggest practical render defaults from input analysis.           │
│ render                                                                       │
│ dereverb                                                                     │
│ ir          Impulse response workflows.                                      │
│ cache       IR cache inspection and cleanup.                                 │
│ batch       Batch manifest generation and rendering.                         │
│ immersive   Immersive production interoperability workflows.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx render --help`

```text

 Usage: root render [OPTIONS] {infile} {outfile}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    infile       <path>  [required]                                         │
│ *    outfile      <path>  [required]                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --preset                                 <str>             Named preset      │
│                                                            baseline (see     │
│                                                            `verbx presets`)  │
│                                                            or dynamic room   │
│                                                            shorthand         │
│                                                            `room:<width>x<d… │
│                                                            Explicitly        │
│                                                            supplied CLI      │
│                                                            options override  │
│                                                            preset values.    │
│ --auto-fit                               <none|speech|mus  Apply             │
│                                          ic|drums|ambient  target-oriented   │
│                                          >                 heuristic         │
│                                                            profile: none,    │
│                                                            speech, music,    │
│                                                            drums, ambient.   │
│                                                            [default: none]   │
│ --engine                                 <conv|algo|auto>  Engine: conv,     │
│                                                            algo, or auto.    │
│                                                            [default: auto]   │
│ --rt60                                   <float range>     [default: 60.0]   │
│                                          [0.1<=x<=3600.0]                    │
│ --wet                                    <float range>     [default: 0.8]    │
│                                          [0.0<=x<=1.0]                       │
│ --dry                                    <float range>     [default: 0.2]    │
│                                          [0.0<=x<=1.0]                       │
│ --repeat                                 <int range>       [default: 1]      │
│                                          [x>=1]                              │
│ --freeze                                                   Enable freeze     │
│                                                            segment mode.     │
│ --start                                  <float range>                       │
│                                          [x>=0.0]                            │
│ --end                                    <float range>                       │
│                                          [x>=0.0]                            │
│ --pre-delay-ms                           <float range>     [default: 20.0]   │
│                                          [x>=0.0]                            │
│ --pre-delay                              <str>                               │
│ --bpm                                    <float range>                       │
│                                          [x>=1.0]                            │
│ --damping                                <float range>     [default: 0.45]   │
│                                          [0.0<=x<=1.0]                       │
│ --width                                  <float range>     [default: 1.0]    │
│                                          [0.0<=x<=2.0]                       │
│ --mod-depth-ms                           <float range>     [default: 2.0]    │
│                                          [x>=0.0]                            │
│ --mod-rate-hz                            <float range>     [default: 0.1]    │
│                                          [x>=0.0]                            │
│ --mod-target                             <none|mix|wet|ga  Dynamic parameter │
│                                          in-db>            target: none,     │
│                                                            mix/wet, or       │
│                                                            gain-db.          │
│                                                            [default: none]   │
│ --mod-source                             <str>             Repeatable        │
│                                                            modulation source │
│                                                            spec. Examples:   │
│                                                            lfo:sine:0.08:1.… │
│                                                            env:20:350,       │
│                                                            audio-env:sidech… │
│                                                            const:0.5.        │
│ --mod-route                              <str>             Repeatable        │
│                                                            advanced route:   │
│                                                            <target>:<min>:<… │
│                                                            (target:          │
│                                                            mix|wet|gain-db). │
│ --mod-min                                <float>           Minimum mapped    │
│                                                            value for the     │
│                                                            modulation        │
│                                                            target.           │
│                                                            [default: 0.0]    │
│ --mod-max                                <float>           Maximum mapped    │
│                                                            value for the     │
│                                                            modulation        │
│                                                            target.           │
│                                                            [default: 1.0]    │
│ --mod-combine                            <sum|avg|max>     How multiple      │
│                                                            sources are       │
│                                                            combined: sum,    │
│                                                            avg, or max.      │
│                                                            [default: sum]    │
│ --mod-smooth-ms                          <float range>     One-pole          │
│                                          [x>=0.0]          smoothing time    │
│                                                            for modulation    │
│                                                            control signals.  │
│                                                            [default: 20.0]   │
│ --allpass-stages                         <int range>       Number of         │
│                                          [0<=x<=64]        Schroeder allpass │
│                                                            diffusion stages  │
│                                                            (0 disables       │
│                                                            diffusion).       │
│                                                            [default: 6]      │
│ --allpass-gain                           <str>             Allpass gain. Use │
│                                                            one value (e.g.   │
│                                                            0.7) for all      │
│                                                            stages, or a      │
│                                                            comma-separated   │
│                                                            list (e.g.        │
│                                                            0.72,0.70,0.68,0… │
│                                                            for per-stage     │
│                                                            gains.            │
│                                                            [default: 0.7]    │
│ --allpass-delays…                        <str>             Optional          │
│                                                            comma-separated   │
│                                                            allpass delay     │
│                                                            list in           │
│                                                            milliseconds.     │
│                                                            Example:          │
│                                                            5,7,11,17,23,29   │
│ --comb-delays-ms                         <str>             Optional          │
│                                                            comma-separated   │
│                                                            FDN comb-like     │
│                                                            delay list in     │
│                                                            milliseconds.     │
│                                                            Example:          │
│                                                            31,37,41,43,47,5… │
│ --comb-cloud         --no-comb-cloud                       Enable an         │
│                                                            optional pre-FDN  │
│                                                            cloud of          │
│                                                            decorrelated      │
│                                                            feedback comb     │
│                                                            filters.          │
│                                                            [default:         │
│                                                            no-comb-cloud]    │
│ --comb-cloud-cou…                        <int range>       Number of comb    │
│                                          [1<=x<=128]       filters generated │
│                                                            for the optional  │
│                                                            comb cloud.       │
│                                                            [default: 24]     │
│ --comb-cloud-fee…                        <float range>     Feedback amount   │
│                                          [0.0<=x<=0.95]    used by the       │
│                                                            optional comb     │
│                                                            cloud (0..0.95).  │
│                                                            [default: 0.35]   │
│ --comb-cloud-mix                         <float range>     Blend from        │
│                                          [0.0<=x<=1.0]     diffusion output  │
│                                                            into comb-cloud   │
│                                                            color output      │
│                                                            (0..1).           │
│                                                            [default: 0.25]   │
│ --comb-cloud-del…                        <str>             Optional          │
│                                                            comma-separated   │
│                                                            delay list in     │
│                                                            milliseconds for  │
│                                                            the comb cloud.   │
│                                                            Providing this    │
│                                                            auto-enables the  │
│                                                            mode.             │
│ --comb-cloud-seed                        <int>             Deterministic     │
│                                                            seed used when    │
│                                                            generating the    │
│                                                            optional comb     │
│                                                            cloud.            │
│                                                            [default: 2026]   │
│ --fdn-lines                              <int range>       FDN line count    │
│                                          [1<=x<=64]        used when         │
│                                                            --comb-delays-ms  │
│                                                            is not provided.  │
│                                                            [default: 8]      │
│ --fdn-matrix                             <str>             FDN matrix        │
│                                                            topology:         │
│                                                            hadamard,         │
│                                                            householder,      │
│                                                            random_orthogona… │
│                                                            circulant,        │
│                                                            elliptic,         │
│                                                            tv_unitary,       │
│                                                            graph, or         │
│                                                            sdn_hybrid.       │
│                                                            Default resolves  │
│                                                            to hadamard.      │
│                                                            [default: auto]   │
│ --fdn-tv-rate-hz                         <float range>     Block-rate update │
│                                          [x>=0.0]          speed for         │
│                                                            --fdn-matrix      │
│                                                            tv_unitary (Hz).  │
│                                                            [default: 0.0]    │
│ --fdn-tv-depth                           <float range>     Blend depth for   │
│                                          [0.0<=x<=1.0]     --fdn-matrix      │
│                                                            tv_unitary        │
│                                                            (0..1).           │
│                                                            [default: 0.0]    │
│ --fdn-dfm-delays…                        <str>             Optional          │
│                                                            delay-feedback-m… │
│                                                            delays in         │
│                                                            milliseconds.     │
│                                                            Provide one value │
│                                                            for broadcast or  │
│                                                            one per FDN line. │
│ --fdn-sparse         --no-fdn-sparse                       Enable sparse     │
│                                                            high-order FDN    │
│                                                            pair-mixing mode. │
│                                                            [default:         │
│                                                            no-fdn-sparse]    │
│ --fdn-sparse-deg…                        <int range>       Number of sparse  │
│                                          [1<=x<=16]        pair-mixing       │
│                                                            stages used when  │
│                                                            --fdn-sparse is   │
│                                                            enabled.          │
│                                                            [default: 2]      │
│ --fdn-cascade        --no-fdn-cascade                      Enable            │
│                                                            nested/cascaded   │
│                                                            FDN mode (small   │
│                                                            fast network into │
│                                                            late network).    │
│                                                            [default:         │
│                                                            no-fdn-cascade]   │
│ --fdn-cascade-mix                        <float range>     Injection amount  │
│                                          [0.0<=x<=1.0]     from nested FDN   │
│                                                            into the main     │
│                                                            late-field        │
│                                                            network (0..1).   │
│                                                            [default: 0.35]   │
│ --fdn-cascade-de…                        <float range>     Delay scaling for │
│                                          [0.2<=x<=1.0]     nested FDN        │
│                                                            relative to       │
│                                                            primary FDN       │
│                                                            delays            │
│                                                            (0.2..1.0).       │
│                                                            [default: 0.5]    │
│ --fdn-cascade-rt…                        <float range>     RT60 ratio for    │
│                                          [0.1<=x<=1.0]     nested FDN        │
│                                                            relative to       │
│                                                            --rt60            │
│                                                            (0.1..1.0).       │
│                                                            [default: 0.55]   │
│ --fdn-rt60-low                           <float range>     Low-band RT60     │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-mid                           <float range>     Mid-band RT60     │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-high                          <float range>     High-band RT60    │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-tilt                          <float range>     Jot-style         │
│                                          [-1.0<=x<=1.0]    low/high RT skew  │
│                                                            around mid band   │
│                                                            (-1..1). Positive │
│                                                            extends low-band  │
│                                                            decay and         │
│                                                            shortens highs.   │
│                                                            [default: 0.0]    │
│ --fdn-tonal-corr…                        <float range>     Track C tonal     │
│                                          [0.0<=x<=1.0]     correction        │
│                                                            strength for      │
│                                                            multiband/tilted  │
│                                                            FDN response      │
│                                                            (0..1). Higher    │
│                                                            values apply      │
│                                                            stronger          │
│                                                            decay-color       │
│                                                            equalization.     │
│                                                            [default: 0.0]    │
│ --fdn-xover-low-…                        <float range>     Low/mid crossover │
│                                          [x>=20.0]         frequency used by │
│                                                            multiband FDN     │
│                                                            decay shaping.    │
│                                                            [default: 250.0]  │
│ --fdn-xover-high…                        <float range>     Mid/high          │
│                                          [x>=100.0]        crossover         │
│                                                            frequency used by │
│                                                            multiband FDN     │
│                                                            decay shaping.    │
│                                                            [default: 4000.0] │
│ --fdn-link-filter                        <str>             Feedback-link     │
│                                                            filter mode       │
│                                                            inside the FDN    │
│                                                            matrix path:      │
│                                                            none, lowpass, or │
│                                                            highpass.         │
│                                                            [default: none]   │
│ --fdn-link-filte…                        <float range>     Cutoff frequency  │
│                                          [x>=20.0]         used by           │
│                                                            --fdn-link-filter │
│                                                            (Hz).             │
│                                                            [default: 2500.0] │
│ --fdn-link-filte…                        <float range>     Wet mix of        │
│                                          [0.0<=x<=1.0]     feedback-link     │
│                                                            filter processing │
│                                                            (0..1).           │
│                                                            [default: 1.0]    │
│ --fdn-graph-topo…                        <str>             Graph topology    │
│                                                            for --fdn-matrix  │
│                                                            graph: ring,      │
│                                                            path, star, or    │
│                                                            random.           │
│                                                            [default: ring]   │
│ --fdn-graph-degr…                        <int range>       Graph             │
│                                          [1<=x<=32]        neighborhood/con… │
│                                                            degree for        │
│                                                            --fdn-matrix      │
│                                                            graph.            │
│                                                            [default: 2]      │
│ --fdn-graph-seed                         <int>             Deterministic     │
│                                                            seed used to      │
│                                                            build             │
│                                                            graph-structured  │
│                                                            FDN pairings.     │
│                                                            [default: 2026]   │
│ --fdn-matrix-mor…                        <str>             Optional target   │
│                                                            matrix family for │
│                                                            gradual           │
│                                                            feedback-matrix   │
│                                                            morphing.         │
│ --fdn-matrix-mor…                        <float range>     Duration          │
│                                          [x>=0.0]          (seconds) for     │
│                                                            matrix morph from │
│                                                            --fdn-matrix to   │
│                                                            --fdn-matrix-mor… │
│                                                            [default: 0.0]    │
│ --fdn-spatial-co…                        <none|adjacent|f  Directional       │
│                                          ront_rear|bed_to  wet-bus coupling  │
│                                          p|all_to_all>     mode: none,       │
│                                                            adjacent,         │
│                                                            front_rear,       │
│                                                            bed_top,          │
│                                                            all_to_all.       │
│                                                            [default: none]   │
│ --fdn-spatial-co…                        <float range>     Wet-bus           │
│                                          [0.0<=x<=1.0]     directional       │
│                                                            coupling amount   │
│                                                            (0..1).           │
│                                                            [default: 0.0]    │
│ --fdn-nonlineari…                        <none|tanh|softc  Optional in-loop  │
│                                          lip>              nonlinearity:     │
│                                                            none, tanh, or    │
│                                                            softclip.         │
│                                                            [default: none]   │
│ --fdn-nonlineari…                        <float range>     Blend amount for  │
│                                          [0.0<=x<=1.0]     in-loop           │
│                                                            nonlinearity      │
│                                                            shaping (0..1).   │
│                                                            [default: 0.0]    │
│ --fdn-nonlineari…                        <float range>     Drive multiplier  │
│                                          [0.1<=x<=8.0]     for in-loop       │
│                                                            nonlinearity      │
│                                                            shaping.          │
│                                                            [default: 1.0]    │
│ --room-size-macro                        <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    room-size macro   │
│                                                            (-1..1) mapped to │
│                                                            decay-time and    │
│                                                            spacing behavior. │
│                                                            [default: 0.0]    │
│ --clarity-macro                          <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    clarity macro     │
│                                                            (-1..1) mapped to │
│                                                            decay, damping,   │
│                                                            and wet balance.  │
│                                                            [default: 0.0]    │
│ --warmth-macro                           <float range>     Perceptual warmth │
│                                          [-1.0<=x<=1.0]    macro (-1..1)     │
│                                                            mapped to damping │
│                                                            and spectral      │
│                                                            decay tilt.       │
│                                                            [default: 0.0]    │
│ --envelopment-ma…                        <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    envelopment macro │
│                                                            (-1..1) mapped to │
│                                                            width/decorrelat… │
│                                                            emphasis.         │
│                                                            [default: 0.0]    │
│ --beast-mode                             <int range>       Scales core       │
│                                          [1<=x<=100]       reverb parameters │
│                                                            by an intensity   │
│                                                            multiplier        │
│                                                            (1-100) to push   │
│                                                            denser, longer,   │
│                                                            freeze-like       │
│                                                            tails.            │
│                                                            [default: 1]      │
│ --ir                                     <path>                              │
│ --ir-blend                               <path>            Repeatable        │
│                                                            additional IR     │
│                                                            path for          │
│                                                            render-time       │
│                                                            convolution       │
│                                                            blending.         │
│                                                            Requires          │
│                                                            convolution       │
│                                                            render path.      │
│ --ir-blend-mix                           <float range>     Repeatable blend  │
│                                          [0.0<=x<=1.0]     coefficient for   │
│                                                            each --ir-blend   │
│                                                            IR (0..1).        │
│                                                            Provide one value │
│                                                            to broadcast to   │
│                                                            all blend IRs.    │
│ --ir-blend-mode                          <str>             IR blend morph    │
│                                                            mode: linear,     │
│                                                            equal-power,      │
│                                                            spectral, or      │
│                                                            envelope-aware.   │
│                                                            [default:         │
│                                                            equal-power]      │
│ --ir-blend-early…                        <float range>     Early/late split  │
│                                          [x>=0.0]          time (ms) used by │
│                                                            envelope-aware    │
│                                                            and split         │
│                                                            blending modes.   │
│                                                            [default: 80.0]   │
│ --ir-blend-early…                        <float range>     Optional override │
│                                          [0.0<=x<=1.0]     alpha for         │
│                                                            early-reflection  │
│                                                            blend region.     │
│ --ir-blend-late-…                        <float range>     Optional override │
│                                          [0.0<=x<=1.0]     alpha for         │
│                                                            late-tail blend   │
│                                                            region.           │
│ --ir-blend-align…    --no-ir-blend-a…                      Enable RT60       │
│                                                            alignment before  │
│                                                            morphing to       │
│                                                            stabilize blend   │
│                                                            trajectories.     │
│                                                            [default:         │
│                                                            ir-blend-align-d… │
│ --ir-blend-phase…                        <float range>     Phase-coherence   │
│                                          [0.0<=x<=1.0]     safeguard         │
│                                                            strength for      │
│                                                            spectral/envelop… │
│                                                            blending.         │
│                                                            [default: 0.75]   │
│ --ir-blend-spect…                        <int range>       Frequency         │
│                                          [0<=x<=128]       smoothing radius  │
│                                                            (FFT bins) used   │
│                                                            by spectral blend │
│                                                            modes.            │
│                                                            [default: 3]      │
│ --ir-blend-misma…                        <coerce|strict>   Mismatch behavior │
│                                                            for blend-source  │
│                                                            sample-rate/chan… │
│                                                            differences:      │
│                                                            coerce            │
│                                                            (resample/align)  │
│                                                            or strict (fail). │
│                                                            [default: coerce] │
│ --ir-blend-cache…                        <str>             Cache directory   │
│                                                            for               │
│                                                            blended/morphed   │
│                                                            IR artifacts used │
│                                                            by render         │
│                                                            workflow.         │
│                                                            [default:         │
│                                                            .verbx_cache/ir_… │
│ --self-convolve                                            Use INFILE as its │
│                                                            own IR and force  │
│                                                            fast partitioned  │
│                                                            convolution       │
│                                                            (equivalent to    │
│                                                            --engine conv     │
│                                                            --ir INFILE).     │
│ --ir-route-map                           <str>             Convolution       │
│                                                            route-map mode:   │
│                                                            auto, diagonal,   │
│                                                            broadcast, or     │
│                                                            full.             │
│                                                            [default: auto]   │
│ --input-layout                           <auto|mono|stere  Input signal      │
│                                          o|LCR|5.1|7.1|7.  channel layout:   │
│                                          1.2|7.1.4|7.2.4|  auto, mono,       │
│                                          8.0|16.0|64.4>    stereo, LCR, 5.1, │
│                                                            7.1, 7.1.2,       │
│                                                            7.1.4, 7.2.4,     │
│                                                            8.0, 16.0, 64.4   │
│                                                            [default: auto]   │
│ --output-layout                          <auto|mono|stere  Output signal     │
│                                          o|LCR|5.1|7.1|7.  channel layout:   │
│                                          1.2|7.1.4|7.2.4|  auto, mono,       │
│                                          8.0|16.0|64.4>    stereo, LCR, 5.1, │
│                                                            7.1, 7.1.2,       │
│                                                            7.1.4, 7.2.4,     │
│                                                            8.0, 16.0, 64.4   │
│                                                            [default: auto]   │
│ --ir-normalize                           <peak|rms|none>   [default: peak]   │
│ --ir-matrix-layo…                        <output-major|in  [default:         │
│                                          put-major>        output-major]     │
│ --conv-route-sta…                        <str>             Convolution       │
│                                                            trajectory start  │
│                                                            position (index   │
│                                                            or alias, e.g.    │
│                                                            left, rear-left). │
│ --conv-route-end                         <str>             Convolution       │
│                                                            trajectory end    │
│                                                            position (index   │
│                                                            or alias).        │
│ --conv-route-cur…                        <str>             Convolution       │
│                                                            trajectory curve: │
│                                                            linear or         │
│                                                            equal-power.      │
│                                                            [default:         │
│                                                            equal-power]      │
│ --ambi-order                             <int range>       Ambisonics order  │
│                                          [0<=x<=7]         (0 disables       │
│                                                            Ambisonics-speci… │
│                                                            processing).      │
│                                                            [default: 0]      │
│ --ambi-normaliza…                        <auto|sn3d|n3d|f  Ambisonics        │
│                                          uma>              normalization     │
│                                                            convention: auto, │
│                                                            sn3d, n3d, or     │
│                                                            fuma.             │
│                                                            [default: auto]   │
│ --channel-order                          <auto|acn|fuma>   Ambisonics        │
│                                                            channel order     │
│                                                            convention: auto, │
│                                                            acn, or fuma.     │
│                                                            [default: auto]   │
│ --ambi-encode-fr…                        <none|mono|stere  Encode input bus  │
│                                          o>                into FOA before   │
│                                                            render: none,     │
│                                                            mono, or stereo.  │
│                                                            [default: none]   │
│ --ambi-decode-to                         <none|stereo>     Decode Ambisonics │
│                                                            output after      │
│                                                            render: none or   │
│                                                            stereo.           │
│                                                            [default: none]   │
│ --ambi-rotate-ya…                        <float>           Listener yaw      │
│                                                            rotation in       │
│                                                            degrees applied   │
│                                                            in Ambisonic      │
│                                                            domain.           │
│                                                            [default: 0.0]    │
│ --algo-front-var…                        <float range>     Algorithmic       │
│                                          [0.0<=x<=1.0]     surround          │
│                                                            decorrelation     │
│                                                            variance for      │
│                                                            front channels.   │
│                                                            [default: 0.0]    │
│ --algo-rear-vari…                        <float range>     Algorithmic       │
│                                          [0.0<=x<=1.0]     surround          │
│                                                            decorrelation     │
│                                                            variance for rear │
│                                                            channels.         │
│                                                            [default: 0.0]    │
│ --algo-top-varia…                        <float range>     Algorithmic       │
│                                          [0.0<=x<=1.0]     surround          │
│                                                            decorrelation     │
│                                                            variance for top  │
│                                                            channels.         │
│                                                            [default: 0.0]    │
│ --tail-limit                             <float range>                       │
│                                          [x>=0.0]                            │
│ --tail-stop-thre…                        <float range>     Tail completion   │
│                                          [-240.0<=x<=0.0]  threshold in dBFS │
│                                                            used for final    │
│                                                            zero-tail         │
│                                                            writeout.         │
│                                                            [default: -120.0] │
│ --tail-stop-hold…                        <float range>     Explicit          │
│                                          [x>=0.0]          zero-hold         │
│                                                            duration appended │
│                                                            after tail        │
│                                                            completion.       │
│                                                            [default: 10.0]   │
│ --tail-stop-metr…                        <peak|rms>        Tail stop         │
│                                                            detector metric:  │
│                                                            peak or rms.      │
│                                                            [default: peak]   │
│ --threads                                <int range>                         │
│                                          [x>=1]                              │
│ --device                                 <auto|cpu|cuda|m  Compute device    │
│                                          ps>               preference: auto, │
│                                                            cpu, cuda, or mps │
│                                                            (Apple Silicon).  │
│                                                            [default: auto]   │
│ --algo-stream        --no-algo-stream                      Use               │
│                                                            algorithmic-to-c… │
│                                                            proxy streaming   │
│                                                            path for long     │
│                                                            algorithmic       │
│                                                            renders.          │
│                                                            [default:         │
│                                                            no-algo-stream]   │
│ --algo-proxy-ir-…                        <float range>     Maximum proxy-IR  │
│                                          [x>=1.0]          duration used by  │
│                                                            --algo-stream.    │
│                                                            [default: 120.0]  │
│ --algo-gpu-proxy     --no-algo-gpu-p…                      Route algorithmic │
│                                                            render through    │
│                                                            proxy convolution │
│                                                            path to leverage  │
│                                                            CUDA convolution. │
│                                                            [default:         │
│                                                            no-algo-gpu-prox… │
│ --partition-size                         <int range>       [default: 16384]  │
│                                          [x>=256]                            │
│ --quality-preset                         <str>             Output-definition │
│                                                            preset: sd=44.1   │
│                                                            kHz PCM16, md=48  │
│                                                            kHz PCM24, hd=192 │
│                                                            kHz float32       │
│                                                            (default).        │
│                                                            Explicit          │
│                                                            --target-sr/--ou… │
│                                                            override.         │
│                                                            [default: hd]     │
│ --target-sr                              <int range>       Optional          │
│                                          [x>=1]            output/render     │
│                                                            sample rate (Hz). │
│                                                            Input is          │
│                                                            resampled         │
│                                                            internally if     │
│                                                            needed.           │
│ --ir-gen                                                                     │
│ --ir-gen-mode                            <fdn|stochastic|  [default: hybrid] │
│                                          modal|hybrid>                       │
│ --ir-gen-length                          <float range>     [default: 60.0]   │
│                                          [x>=0.1]                            │
│ --ir-gen-seed                            <int>             [default: 0]      │
│ --ir-gen-cache-d…                        <str>             [default:         │
│                                                            .verbx_cache/irs] │
│ --block-size                             <int range>       [default: 4096]   │
│                                          [x>=256]                            │
│ --target-lufs                            <float>                             │
│ --target-peak-db…                        <float>                             │
│ --true-peak          --sample-peak                         [default:         │
│                                                            true-peak]        │
│ --limiter            --no-limiter                          [default:         │
│                                                            limiter]          │
│ --limiter-mode                           <tanh|arctan|sof  Limiter transfer  │
│                                          tsign|hard>       curve: tanh,      │
│                                                            arctan, softsign, │
│                                                            or hard.          │
│                                                            [default: tanh]   │
│ --limiter-detect                         <peak|rms>        Limiter detector  │
│                                                            mode: peak or     │
│                                                            rms.              │
│                                                            [default: peak]   │
│ --limiter-thresh…                        <float>           Limiter onset     │
│                                                            threshold in      │
│                                                            dBFS. Defaults to │
│                                                            the active peak   │
│                                                            target/ceiling.   │
│ --limiter-ceilin…                        <float>           Limiter output    │
│                                                            ceiling in dBFS.  │
│                                                            Defaults to the   │
│                                                            active peak       │
│                                                            target or -1      │
│                                                            dBFS.             │
│ --limiter-knee-db                        <float range>     [default: 6.0]    │
│                                          [x>=0.0]                            │
│ --limiter-drive                          <float range>     [default: 1.0]    │
│                                          [x>=1e-06]                          │
│ --limiter-mix                            <float range>     [default: 1.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --limiter-attack…                        <float range>     [default: 0.5]    │
│                                          [x>=0.0]                            │
│ --limiter-releas…                        <float range>     [default: 80.0]   │
│                                          [x>=0.0]                            │
│ --limiter-lookah…                        <float range>     [default: 1.5]    │
│                                          [x>=0.0]                            │
│ --limiter-stereo…    --no-limiter-st…                      Link channels in  │
│                                                            the limiter       │
│                                                            detector to       │
│                                                            preserve stereo   │
│                                                            image.            │
│                                                            [default:         │
│                                                            limiter-stereo-l… │
│ --limiter-oversa…                        <int range>       [default: 2]      │
│                                          [1<=x<=16]                          │
│ --limiter-pre-ga…                        <float range>     [default: 0.0]    │
│                                          [-48.0<=x<=48.0]                    │
│ --limiter-post-g…                        <float range>     [default: 0.0]    │
│                                          [-48.0<=x<=48.0]                    │
│ --limiter-dc-blo…    --no-limiter-dc…                      Apply a gentle DC │
│                                                            blocker before    │
│                                                            limiter           │
│                                                            detection.        │
│                                                            [default:         │
│                                                            no-limiter-dc-bl… │
│ --normalize-stage                        <none|post|per-p  [default: post]   │
│                                          ass>                                │
│ --repeat-target-…                        <float>                             │
│ --repeat-target-…                        <float>                             │
│ --out-subtype                            <auto|float32|fl  Output file       │
│                                          oat64|pcm16|pcm2  subtype. Internal │
│                                          4|pcm32>          DSP runs in       │
│                                                            float64           │
│                                                            regardless of     │
│                                                            container         │
│                                                            subtype; use      │
│                                                            float64/float32/… │
│                                                            per delivery      │
│                                                            needs.            │
│                                                            [default: auto]   │
│ --output-contain…                        <auto|wav|w64|rf  Output container  │
│                                          64>               mode: auto, wav,  │
│                                                            w64, or rf64.     │
│                                                            [default: auto]   │
│ --output-peak-no…                        <none|input|targ  Final peak        │
│                                          et|full-scale>    normalization     │
│                                                            mode: none, input │
│                                                            peak match,       │
│                                                            explicit target,  │
│                                                            or full-scale.    │
│                                                            [default: none]   │
│ --output-peak-ta…                        <float>           Target dBFS used  │
│                                                            when              │
│                                                            --output-peak-no… │
│                                                            target is         │
│                                                            selected.         │
│ --shimmer                                                                    │
│ --shimmer-semito…                        <float>           [default: 12.0]   │
│ --shimmer-mix                            <float range>     [default: 0.25]   │
│                                          [0.0<=x<=1.0]                       │
│ --shimmer-feedba…                        <float range>     [default: 0.35]   │
│                                          [0.0<=x<=1.25]                      │
│ --shimmer-highcut                        <float range>     [default:         │
│                                          [x>=10.0]         10000.0]          │
│ --shimmer-lowcut                         <float range>     [default: 300.0]  │
│                                          [x>=10.0]                           │
│ --shimmer-spatial    --no-shimmer-sp…                      Enable            │
│                                                            multichannel      │
│                                                            shimmer spatial   │
│                                                            decorrelation.    │
│                                                            [default:         │
│                                                            no-shimmer-spati… │
│ --shimmer-spread…                        <float range>     Per-channel       │
│                                          [x>=0.0]          shimmer detune    │
│                                                            spread in cents   │
│                                                            (multichannel).   │
│                                                            [default: 8.0]    │
│ --shimmer-decorr…                        <float range>     Per-channel       │
│                                          [x>=0.0]          shimmer delay     │
│                                                            spread in         │
│                                                            milliseconds.     │
│                                                            [default: 1.5]    │
│ --er-geometry        --no-er-geometry                      Enable            │
│                                                            first-order       │
│                                                            image-source      │
│                                                            early-reflection  │
│                                                            pre-stage.        │
│                                                            [default:         │
│                                                            no-er-geometry]   │
│ --er-room-dims-m                         <str>             Room dimensions   │
│                                                            in meters: L,W,H  │
│                                                            [default: 10,7,3] │
│ --er-source-pos-m                        <str>             Source position   │
│                                                            in meters: x,y,z  │
│                                                            [default:         │
│                                                            2,2,1.5]          │
│ --er-listener-po…                        <str>             Listener position │
│                                                            in meters: x,y,z  │
│                                                            [default:         │
│                                                            5,3.5,1.5]        │
│ --er-absorption                          <float range>     Wall absorption   │
│                                          [0.0<=x<=0.99]    coefficient for   │
│                                                            early-reflection  │
│                                                            stage.            │
│                                                            [default: 0.35]   │
│ --er-material                            <str>             Early-reflection  │
│                                                            material preset:  │
│                                                            anechoic, dead,   │
│                                                            studio, hall,     │
│                                                            stone, or custom. │
│                                                            [default: studio] │
│ --unsafe-self-os…    --safe-no-self-…                      UNSAFE: permit    │
│                                                            feedback-path     │
│                                                            gains above unity │
│                                                            in algorithmic    │
│                                                            mode for          │
│                                                            self-oscillating  │
│                                                            tails.            │
│                                                            [default:         │
│                                                            safe-no-self-osc… │
│ --unsafe-loop-ga…                        <float range>     UNSAFE loop-gain  │
│                                          [0.01<=x<=1.25]   scale used with   │
│                                                            --unsafe-self-os… │
│                                                            Values >1.0       │
│                                                            encourage         │
│                                                            self-oscillation. │
│                                                            [default: 1.02]   │
│ --duck                                                                       │
│ --duck-attack                            <float range>     [default: 20.0]   │
│                                          [x>=0.1]                            │
│ --duck-release                           <float range>     [default: 350.0]  │
│                                          [x>=0.1]                            │
│ --duck-strength                          <float range>     How strongly the  │
│                                          [0.0<=x<=1.0]     wet field is      │
│                                                            attenuated when   │
│                                                            the sidechain     │
│                                                            rises.            │
│                                                            [default: 0.75]   │
│ --duck-floor                             <float range>     Minimum wet gain  │
│                                          [0.0<=x<=1.0]     held during       │
│                                                            ducking; useful   │
│                                                            for softer        │
│                                                            pumping.          │
│                                                            [default: 0.0]    │
│ --bloom                                  <float range>     [default: 0.0]    │
│                                          [x>=0.0]                            │
│ --bloom-mix                              <float range>     Override bloom    │
│                                          [0.0<=x<=1.0]     blend amount.     │
│                                                            Default           │
│                                                            auto-scales from  │
│                                                            --bloom.          │
│ --lowcut                                 <float range>                       │
│                                          [x>=10.0]                           │
│ --lowcut-order                           <int range>       Butterworth order │
│                                          [1<=x<=8]         used by the       │
│                                                            post-wet          │
│                                                            high-pass filter. │
│                                                            [default: 2]      │
│ --highcut                                <float range>                       │
│                                          [x>=10.0]                           │
│ --highcut-order                          <int range>       Butterworth order │
│                                          [1<=x<=8]         used by the       │
│                                                            post-wet low-pass │
│                                                            filter.           │
│                                                            [default: 2]      │
│ --tilt                                   <float>           [default: 0.0]    │
│ --tilt-pivot-hz                          <float range>     Pivot frequency   │
│                                          [x>=20.0]         used by the       │
│                                                            post-wet tilt EQ. │
│                                                            [default: 1000.0] │
│ --automation-file                        <path>            JSON/CSV          │
│                                                            automation lanes  │
│                                                            used for          │
│                                                            time-varying      │
│                                                            render control.   │
│ --automation-mode                        <auto|sample|blo  Automation        │
│                                          ck>               evaluation mode:  │
│                                                            auto, sample, or  │
│                                                            block.            │
│                                                            [default: auto]   │
│ --automation-blo…                        <float range>     Control block     │
│                                          [x>=0.1]          size in           │
│                                                            milliseconds when │
│                                                            automation mode   │
│                                                            is block.         │
│                                                            [default: 20.0]   │
│ --automation-smo…                        <float range>     Default smoothing │
│                                          [x>=0.0]          time (ms) applied │
│                                                            to automation     │
│                                                            lanes.            │
│                                                            [default: 20.0]   │
│ --automation-sle…                        <float range>     Optional max      │
│                                          [x>=0.0]          control slew as   │
│                                                            target-range      │
│                                                            fraction per      │
│                                                            second; 0/None    │
│                                                            disables slew     │
│                                                            guard.            │
│ --automation-dea…                        <float range>     Optional control  │
│                                          [0.0<=x<=1.0]     deadband as       │
│                                                            target-range      │
│                                                            fraction; small   │
│                                                            changes below     │
│                                                            threshold are     │
│                                                            suppressed.       │
│                                                            [default: 0.0]    │
│ --automation-cla…                        <str>             Clamp override in │
│                                                            target:min:max    │
│                                                            format            │
│                                                            (repeatable).     │
│ --automation-poi…                        <str>             Inline automation │
│                                                            control point in  │
│                                                            target:time_s:va… │
│                                                            format            │
│                                                            (repeatable).     │
│ --automation-tra…                        <str>             Optional CSV path │
│                                                            for resolved      │
│                                                            sample-level      │
│                                                            automation        │
│                                                            curves.           │
│ --feature-vector…                        <str>             Feature-vector    │
│                                                            mapping lane      │
│                                                            (repeatable).     │
│                                                            Format:           │
│                                                            target=<target>,… │
│ --feature-vector…                        <float range>     Frame size used   │
│                                          [x>=1.0]          for               │
│                                                            feature-vector    │
│                                                            extraction (ms).  │
│                                                            [default: 40.0]   │
│ --feature-vector…                        <float range>     Hop size used for │
│                                          [x>=1.0]          feature-vector    │
│                                                            extraction (ms).  │
│                                                            [default: 20.0]   │
│ --feature-guide                          <path>            Optional external │
│                                                            guide audio used  │
│                                                            for               │
│                                                            feature-vector    │
│                                                            extraction        │
│                                                            instead of INFILE │
│                                                            (Track B external │
│                                                            feature-guide     │
│                                                            ingest).          │
│ --feature-guide-…                        <align|strict>    Mismatch policy   │
│                                                            for               │
│                                                            --feature-guide   │
│                                                            relative to       │
│                                                            render context:   │
│                                                            align             │
│                                                            (deterministic    │
│                                                            resample +        │
│                                                            hold/trim +       │
│                                                            mixdown) or       │
│                                                            strict.           │
│                                                            [default: align]  │
│ --feature-vector…                        <str>             Optional CSV path │
│                                                            for               │
│                                                            feature+parameter │
│                                                            trace exports.    │
│ --frames-out                             <str>                               │
│ --analysis-out                           <str>                               │
│ --lucky                                  <int range>       Generate N wild   │
│                                          [1<=x<=500]       random renders    │
│                                                            from one input    │
│                                                            using randomized  │
│                                                            parameters.       │
│                                                            Outputs are       │
│                                                            written to        │
│                                                            --lucky-out-dir   │
│                                                            (or OUTFILE       │
│                                                            parent by         │
│                                                            default).         │
│ --lucky-out-dir                          <path>            Output directory  │
│                                                            used when --lucky │
│                                                            is enabled.       │
│ --lucky-seed                             <int>             Optional          │
│                                                            deterministic     │
│                                                            seed for --lucky  │
│                                                            render            │
│                                                            generation.       │
│ --quiet                                                    Suppress console  │
│                                                            summary tables    │
│                                                            while still       │
│                                                            writing output    │
│                                                            and analysis      │
│                                                            artifacts.        │
│ --verbosity                              <int range>       Console detail    │
│                                          [0<=x<=2]         level: 0=minimal  │
│                                                            summary,          │
│                                                            1=summary +       │
│                                                            output features   │
│                                                            (default), 2=also │
│                                                            include input     │
│                                                            feature table.    │
│                                                            [default: 1]      │
│ --silent                                                   Disable analysis  │
│                                                            JSON generation   │
│                                                            and console       │
│                                                            output.           │
│ --dry-run                                                  Validate inputs   │
│                                                            and print         │
│                                                            resolved render   │
│                                                            plan without      │
│                                                            writing audio.    │
│ --repro-bundle                                             Write a           │
│                                                            reproducibility/… │
│                                                            JSON bundle next  │
│                                                            to OUTFILE.       │
│ --repro-bundle-o…                        <path>            Optional explicit │
│                                                            path for          │
│                                                            reproducibility/… │
│                                                            JSON bundle.      │
│ --failure-report…                        <path>            Optional JSON     │
│                                                            report path       │
│                                                            populated when    │
│                                                            render execution  │
│                                                            fails.            │
│ --progress           --no-progress                         [default:         │
│                                                            progress]         │
│ --json-out                               <path>            Optional path to  │
│                                                            write the full    │
│                                                            render report as  │
│                                                            JSON.             │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx realtime --help`

```text

 Usage: root realtime [OPTIONS]

 Run realtime duplex monitoring with selectable input/output devices.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --live-mode                              <reverb|dereverb  Realtime          │
│                                          |dereverb-reverb  processing mode:  │
│                                          >                 reverb only,      │
│                                                            dereverb only, or │
│                                                            dereverb feeding  │
│                                                            the live reverb   │
│                                                            path.             │
│                                                            [default: reverb] │
│ --engine                                 <auto|conv|algo>  Realtime engine:  │
│                                                            convolution IR,   │
│                                                            or algorithmic    │
│                                                            proxy rendered    │
│                                                            into a live       │
│                                                            convolver.        │
│                                                            [default: auto]   │
│ --ir                                     <path>            Impulse response  │
│                                                            path for realtime │
│                                                            convolution mode. │
│ --input-device                           <str>             Input device      │
│                                                            index or          │
│                                                            case-insensitive  │
│                                                            name substring.   │
│ --output-device                          <str>             Output device     │
│                                                            index or          │
│                                                            case-insensitive  │
│                                                            name substring.   │
│ --list-devices                                             List available    │
│                                                            realtime audio    │
│                                                            devices and exit. │
│ --sample-rate                            <int range>       [default: 48000]  │
│                                          [x>=8000]                           │
│ --block-size                             <int range>       [default: 512]    │
│                                          [x>=64]                             │
│ --partition-size                         <int range>       [default: 2048]   │
│                                          [x>=256]                            │
│ --input-channels                         <int range>       Requested live    │
│                                          [x>=1]            input channel     │
│                                                            count. Defaults   │
│                                                            to mono or stereo │
│                                                            depending on      │
│                                                            device.           │
│ --input-channel-…                        <str>             Comma-separated   │
│                                                            1-based hardware  │
│                                                            input channels to │
│                                                            feed the          │
│                                                            processor, for    │
│                                                            example 1,3 or    │
│                                                            1,3,5,7.          │
│ --output-channels                        <int range>       Requested live    │
│                                          [x>=1]            output channel    │
│                                                            count. Defaults   │
│                                                            to the            │
│                                                            processor's       │
│                                                            natural output    │
│                                                            width.            │
│ --output-channel…                        <str>             Comma-separated   │
│                                                            1-based hardware  │
│                                                            output channels   │
│                                                            that receive      │
│                                                            processor         │
│                                                            outputs, in       │
│                                                            order.            │
│ --duration                               <float range>     Optional duration │
│                                          [x>=0.0]          in seconds. Omit  │
│                                                            to run until      │
│                                                            Ctrl-C.           │
│ --dereverb-mode                          <wiener|spectral  Low-latency       │
│                                          _sub>             dereverb kernel   │
│                                                            used by           │
│                                                            --live-mode       │
│                                                            dereverb*.        │
│                                                            [default: wiener] │
│ --dereverb-stren…                        <float range>     [default: 0.7]    │
│                                          [0.0<=x<=2.0]                       │
│ --dereverb-floor                         <float range>     [default: 0.08]   │
│                                          [1e-06<=x<=1.0]                     │
│ --dereverb-windo…                        <float range>     [default: 16.0]   │
│                                          [x>=2.0]                            │
│ --dereverb-hop-ms                        <float range>     [default: 8.0]    │
│                                          [x>=1.0]                            │
│ --dereverb-tail-…                        <float range>     [default: 120.0]  │
│                                          [x>=10.0]                           │
│ --dereverb-pre-e…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=0.98]                      │
│ --dereverb-mix                           <float range>     [default: 1.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --dereverb-max-a…                        <float range>     [default: 18.0]   │
│                                          [0.0<=x<=48.0]                      │
│ --dereverb-stere…    --no-dereverb-s…                      Link stereo gain  │
│                                                            decisions to      │
│                                                            reduce image      │
│                                                            wobble.           │
│                                                            [default:         │
│                                                            dereverb-stereo-… │
│ --dereverb-input…                        <float range>     [default: 0.0]    │
│                                          [-24.0<=x<=24.0]                    │
│ --dereverb-outpu…                        <float range>     [default: 0.0]    │
│                                          [-24.0<=x<=24.0]                    │
│ --dereverb-windo…                        <str>             Live dereverb     │
│                                                            analysis window   │
│                                                            family (hann,     │
│                                                            hamming,          │
│                                                            blackman, kaiser, │
│                                                            dpss, tukey,      │
│                                                            chebwin, and many │
│                                                            more).            │
│                                                            [default: hann]   │
│ --dereverb-synth…                        <str>             Optional live     │
│                                                            dereverb          │
│                                                            synthesis window  │
│                                                            family. Defaults  │
│                                                            to the analysis   │
│                                                            window.           │
│ --dereverb-windo…    --dereverb-wind…                      Use symmetric     │
│                                                            instead of        │
│                                                            periodic live     │
│                                                            dereverb windows. │
│                                                            [default:         │
│                                                            dereverb-window-… │
│ --dereverb-windo…                        <float range>     [default: 0.5]    │
│                                          [x>=0.0]                            │
│ --dereverb-windo…                        <float range>     [default: 14.0]   │
│                                          [x>=0.0]                            │
│ --dereverb-windo…                        <float range>     [default: 2.5]    │
│                                          [x>=1e-06]                          │
│ --dereverb-windo…                        <float range>     [default: 1.5]    │
│                                          [x>=1e-06]                          │
│ --dereverb-windo…                        <float range>     [default: 100.0]  │
│                                          [x>=0.001]                          │
│ --dereverb-windo…                        <int range>       [default: 4]      │
│                                          [x>=2]                              │
│ --dereverb-windo…                        <float range>     [default: 2.5]    │
│                                          [x>=0.001]                          │
│ --dereverb-windo…                        <float range>     [default: 3.0]    │
│                                          [x>=1e-06]                          │
│ --dereverb-windo…                        <str>             Optional          │
│                                                            comma-separated   │
│                                                            weights for       │
│                                                            general_cosine    │
│                                                            live dereverb     │
│                                                            windows.          │
│ --wet                                    <float range>     [default: 0.8]    │
│                                          [0.0<=x<=1.0]                       │
│ --dry                                    <float range>     [default: 0.2]    │
│                                          [0.0<=x<=1.0]                       │
│ --rt60                                   <float range>     [default: 6.0]    │
│                                          [x>=0.1]                            │
│ --pre-delay-ms                           <float range>     [default: 20.0]   │
│                                          [x>=0.0]                            │
│ --damping                                <float range>     [default: 0.45]   │
│                                          [0.0<=x<=1.0]                       │
│ --width                                  <float range>     [default: 1.0]    │
│                                          [0.0<=x<=2.0]                       │
│ --mod-depth-ms                           <float range>     [default: 2.0]    │
│                                          [x>=0.0]                            │
│ --mod-rate-hz                            <float range>     [default: 0.1]    │
│                                          [x>=0.0]                            │
│ --fdn-lines                              <int range>       [default: 8]      │
│                                          [1<=x<=64]                          │
│ --fdn-matrix                             <str>             Algorithmic proxy │
│                                                            matrix for        │
│                                                            realtime --engine │
│                                                            algo.             │
│                                                            [default:         │
│                                                            hadamard]         │
│ --fdn-tv-rate-hz                         <float range>     [default: 0.0]    │
│                                          [x>=0.0]                            │
│ --fdn-tv-depth                           <float range>     [default: 0.0]    │
│                                          [x>=0.0]                            │
│ --fdn-dfm-delays…                        <str>             Comma-separated   │
│                                                            delay-feedback    │
│                                                            modulation taps   │
│                                                            in milliseconds.  │
│ --fdn-sparse                                                                 │
│ --fdn-sparse-deg…                        <int range>       [default: 2]      │
│                                          [1<=x<=16]                          │
│ --fdn-cascade                                                                │
│ --fdn-cascade-mix                        <float range>     [default: 0.35]   │
│                                          [0.0<=x<=1.0]                       │
│ --fdn-cascade-de…                        <float range>     [default: 0.5]    │
│                                          [0.2<=x<=1.0]                       │
│ --fdn-cascade-rt…                        <float range>     [default: 0.55]   │
│                                          [0.1<=x<=1.0]                       │
│ --fdn-rt60-low                           <float range>                       │
│                                          [x>=0.1]                            │
│ --fdn-rt60-mid                           <float range>                       │
│                                          [x>=0.1]                            │
│ --fdn-rt60-high                          <float range>                       │
│                                          [x>=0.1]                            │
│ --fdn-rt60-tilt                          <float range>     [default: 0.0]    │
│                                          [-1.0<=x<=1.0]                      │
│ --fdn-tonal-corr…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --fdn-xover-low-…                        <float range>     [default: 250.0]  │
│                                          [x>=10.0]                           │
│ --fdn-xover-high…                        <float range>     [default: 4000.0] │
│                                          [x>=10.0]                           │
│ --fdn-link-filter                        <none|lowpass|hi  [default: none]   │
│                                          ghpass>                             │
│ --fdn-link-filte…                        <float range>     [default: 2500.0] │
│                                          [x>=10.0]                           │
│ --fdn-link-filte…                        <float range>     [default: 1.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --fdn-graph-topo…                        <ring|path|star|  [default: ring]   │
│                                          random>                             │
│ --fdn-graph-degr…                        <int range>       [default: 2]      │
│                                          [1<=x<=16]                          │
│ --fdn-graph-seed                         <int>             [default: 2026]   │
│ --fdn-matrix-mor…                        <str>             Optional second   │
│                                                            FDN matrix family │
│                                                            for gradual       │
│                                                            morphing.         │
│ --fdn-matrix-mor…                        <float range>     [default: 0.0]    │
│                                          [x>=0.0]                            │
│ --fdn-spatial-co…                        <none|adjacent|f  [default: none]   │
│                                          ront_rear|bed_to                    │
│                                          p|all_to_all>                       │
│ --fdn-spatial-co…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --fdn-nonlineari…                        <none|tanh|softc  [default: none]   │
│                                          lip>                                │
│ --fdn-nonlineari…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --fdn-nonlineari…                        <float range>     [default: 1.0]    │
│                                          [0.1<=x<=8.0]                       │
│ --room-size-macro                        <float range>     [default: 0.0]    │
│                                          [-1.0<=x<=1.0]                      │
│ --clarity-macro                          <float range>     [default: 0.0]    │
│                                          [-1.0<=x<=1.0]                      │
│ --warmth-macro                           <float range>     [default: 0.0]    │
│                                          [-1.0<=x<=1.0]                      │
│ --envelopment-ma…                        <float range>     [default: 0.0]    │
│                                          [-1.0<=x<=1.0]                      │
│ --algo-decorrela…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --algo-decorrela…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --algo-decorrela…                        <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --allpass-stages                         <int range>       [default: 6]      │
│                                          [0<=x<=64]                          │
│ --allpass-gain                           <str>             Single allpass    │
│                                                            gain or           │
│                                                            comma-separated   │
│                                                            per-stage gains.  │
│                                                            [default: 0.7]    │
│ --allpass-delays…                        <str>             Comma-separated   │
│                                                            diffusion delays  │
│                                                            in milliseconds.  │
│ --comb-delays-ms                         <str>             Comma-separated   │
│                                                            FDN/comb delay    │
│                                                            taps in           │
│                                                            milliseconds.     │
│ --freeze                                                   Realtime algo     │
│                                                            only: approximate │
│                                                            a frozen-space    │
│                                                            sustain by        │
│                                                            forcing a long    │
│                                                            near-infinite     │
│                                                            proxy tail.       │
│ --shimmer                                                                    │
│ --shimmer-semito…                        <float>           [default: 12.0]   │
│ --shimmer-mix                            <float range>     [default: 0.25]   │
│                                          [0.0<=x<=1.0]                       │
│ --shimmer-feedba…                        <float range>     [default: 0.35]   │
│                                          [0.0<=x<=1.25]                      │
│ --shimmer-highcut                        <float range>                       │
│                                          [x>=10.0]                           │
│ --shimmer-lowcut                         <float range>                       │
│                                          [x>=10.0]                           │
│ --shimmer-spatial                                                            │
│ --shimmer-spread…                        <float range>     [default: 8.0]    │
│                                          [x>=0.0]                            │
│ --shimmer-decorr…                        <float range>     [default: 1.5]    │
│                                          [x>=0.0]                            │
│ --unsafe-self-os…                                                            │
│ --unsafe-loop-ga…                        <float range>     [default: 1.02]   │
│                                          [x>=0.001]                          │
│ --algo-proxy-ir-…                        <float range>     Maximum rendered  │
│                                          [x>=0.1]          proxy IR duration │
│                                                            used by realtime  │
│                                                            --engine algo.    │
│                                                            [default: 120.0]  │
│ --lowcut                                 <float range>                       │
│                                          [x>=10.0]                           │
│ --highcut                                <float range>                       │
│                                          [x>=10.0]                           │
│ --tilt                                   <float range>     [default: 0.0]    │
│                                          [-18.0<=x<=18.0]                    │
│ --json-out                               <path>            Optional path for │
│                                                            a                 │
│                                                            machine-readable  │
│                                                            realtime session  │
│                                                            report JSON.      │
│ --quiet                                                    Reduce console    │
│                                                            output.           │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx room-model --help`

```text

 Usage: root room-model [OPTIONS]

 Inspect a room geometry or infer one from RT60 and absorption.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --dims-m                <str>                      Explicit room dimensions  │
│                                                    as width,depth,height in  │
│                                                    meters.                   │
│ --rt60                  <float range> [x>=0.05]    Infer a rectangular room  │
│                                                    from RT60 plus            │
│                                                    absorption/material       │
│                                                    assumptions.              │
│ --absorption            <float range>              Mean absorption           │
│                         [0.01<=x<=0.99]            coefficient used with     │
│                                                    --rt60 inference.         │
│ --material              <str>                      Material preset for wall  │
│                                                    absorption when           │
│                                                    --absorption is not       │
│                                                    given.                    │
│                                                    [default: studio]         │
│ --source-pos-m          <str>                      Source position as x,y,z  │
│                                                    in meters.                │
│                                                    [default: 2.0,2.0,1.5]    │
│ --listener-pos-m        <str>                      Listener position as      │
│                                                    x,y,z in meters.          │
│                                                    [default: 5.0,3.5,1.5]    │
│ --json-out              <path>                     Optional path to write    │
│                                                    the full room-model       │
│                                                    payload as JSON.          │
│ --help                                             Show this message and     │
│                                                    exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx analyze --help`

```text

 Usage: root analyze [OPTIONS] {infile}

 Analyze an audio file and print a summary table.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    infile      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --json-out                  <path>                                           │
│ --lufs                                             Include                   │
│                                                    LUFS/true-peak/LRA        │
│                                                    metrics.                  │
│ --edr                                              Include EDR (Energy Decay │
│                                                    Relief) summary metrics.  │
│ --frames-out                <path>                                           │
│ --ambi-order                <int range> [0<=x<=7]  Enable Ambisonics spatial │
│                                                    metrics for the given     │
│                                                    order.                    │
│                                                    [default: 0]              │
│ --ambi-normalization        <auto|sn3d|n3d|fuma>   Ambisonics normalization  │
│                                                    convention for analysis   │
│                                                    mode.                     │
│                                                    [default: auto]           │
│ --channel-order             <auto|acn|fuma>        Ambisonics channel order  │
│                                                    convention for analysis   │
│                                                    mode.                     │
│                                                    [default: auto]           │
│ --room                                             Estimate room size,       │
│                                                    dimensions, absorption,   │
│                                                    critical distance, and    │
│                                                    class from the signal's   │
│                                                    reverberant decay         │
│                                                    characteristics. Works    │
│                                                    best on reverberant       │
│                                                    recordings or rendered    │
│                                                    impulse responses.        │
│ --help                                             Show this message and     │
│                                                    exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx suggest --help`

```text

 Usage: root suggest [OPTIONS] {infile}

 Suggest practical render defaults from input analysis.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    infile      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --pin         <path>  Write suggested parameters as a JSON preset file.      │
│ --help                Show this message and exit.                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx quickstart --help`

```text

 Usage: root quickstart [OPTIONS]

 Print minimal copy/paste commands for first successful renders.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --verify                       Run startup readiness checks for first-run    │
│                                confidence.                                   │
│ --strict                       Exit non-zero when --verify finds one or more │
│                                failed checks.                                │
│ --json-out             <path>  Optional path to write quickstart             │
│                                verification/smoke JSON.                      │
│ --smoke-test                   Run a tiny end-to-end render smoke test with  │
│                                synthetic input audio.                        │
│ --smoke-out-dir        <path>  Optional output directory for smoke-test      │
│                                artifacts.                                    │
│ --help                         Show this message and exit.                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx doctor --help`

```text

 Usage: root doctor [OPTIONS]

 Print runtime diagnostics for launch-day troubleshooting.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --json-out                 <path>  Optional path to write machine-readable   │
│                                    diagnostics JSON.                         │
│ --strict                           Exit non-zero when startup checks fail.   │
│ --render-smoke-test                Run a tiny end-to-end render smoke test   │
│                                    after diagnostics.                        │
│ --smoke-out-dir            <path>  Optional output directory for doctor      │
│                                    smoke-test artifacts.                     │
│ --help                             Show this message and exit.               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx presets --help`

```text

 Usage: root presets [OPTIONS]

 Print available presets or one preset payload.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --show            <str>  Show resolved values for one preset.                │
│ --validate        <str>  Validate a preset's fields against RenderConfig and │
│                          report any errors.                                  │
│ --help                   Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx version --help`

```text

 Usage: root version [OPTIONS]

 Print CLI/package version.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir --help`

```text

 Usage: root ir [OPTIONS] COMMAND [ARGS]...

 Impulse response workflows.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ gen                                                                          │
│ analyze                                                                      │
│ sofa-info                                                                    │
│ sofa-extract                                                                 │
│ trace                                                                        │
│ process                                                                      │
│ morph                                                                        │
│ morph-sweep                                                                  │
│ fit                                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir gen --help`

```text

 Usage: root ir gen [OPTIONS] {out_ir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    out_ir      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --format                                 <auto|wav|flac|a  [default: auto]   │
│                                          iff|aif|ogg|caf>                    │
│ --mode                                   <fdn|stochastic|  [default: hybrid] │
│                                          modal|hybrid>                       │
│ --length                                 <float range>     [default: 60.0]   │
│                                          [x>=0.1]                            │
│ --sr                                     <int range>       [default: 48000]  │
│                                          [x>=8000]                           │
│ --channels                               <int range>       [default: 2]      │
│                                          [x>=1]                              │
│ --seed                                   <int>             [default: 0]      │
│ --rt60                                   <float range>                       │
│                                          [0.1<=x<=3600.0]                    │
│ --rt60-low                               <float range>                       │
│                                          [0.1<=x<=3600.0]                    │
│ --rt60-high                              <float range>                       │
│                                          [0.1<=x<=3600.0]                    │
│ --damping                                <float range>     [default: 0.4]    │
│                                          [0.0<=x<=1.0]                       │
│ --lowcut                                 <float range>                       │
│                                          [x>=10.0]                           │
│ --highcut                                <float range>                       │
│                                          [x>=10.0]                           │
│ --tilt                                   <float>           [default: 0.0]    │
│ --normalize                              <none|peak|rms>   [default: peak]   │
│ --peak-dbfs                              <float>           [default: -1.0]   │
│ --target-lufs                            <float>                             │
│ --true-peak          --sample-peak                         [default:         │
│                                                            true-peak]        │
│ --er-count                               <int range>       [default: 24]     │
│                                          [x>=0]                              │
│ --er-max-delay-ms                        <float range>     [default: 90.0]   │
│                                          [x>=1.0]                            │
│ --er-decay-shape                         <str>             [default: exp]    │
│ --er-stereo-width                        <float range>     [default: 1.0]    │
│                                          [0.0<=x<=2.0]                       │
│ --er-room                                <float range>     [default: 1.0]    │
│                                          [x>=0.1]                            │
│ --diffusion                              <float range>     [default: 0.5]    │
│                                          [0.0<=x<=1.0]                       │
│ --mod-depth-ms                           <float range>     [default: 1.5]    │
│                                          [x>=0.0]                            │
│ --mod-rate-hz                            <float range>     [default: 0.12]   │
│                                          [x>=0.0]                            │
│ --density                                <float range>     [default: 1.0]    │
│                                          [x>=0.01]                           │
│ --tuning                                 <str>             [default: A4=440] │
│ --modal-count                            <int range>       [default: 48]     │
│                                          [x>=1]                              │
│ --modal-q-min                            <float range>     [default: 5.0]    │
│                                          [x>=0.5]                            │
│ --modal-q-max                            <float range>     [default: 60.0]   │
│                                          [x>=0.5]                            │
│ --modal-spread-c…                        <float range>     [default: 5.0]    │
│                                          [x>=0.0]                            │
│ --modal-low-hz                           <float range>     [default: 80.0]   │
│                                          [x>=20.0]                           │
│ --modal-high-hz                          <float range>     [default:         │
│                                          [x>=50.0]         12000.0]          │
│ --fdn-lines                              <int range>       [default: 8]      │
│                                          [x>=1]                              │
│ --fdn-matrix                             <str>             FDN matrix        │
│                                                            topology:         │
│                                                            hadamard,         │
│                                                            householder,      │
│                                                            random_orthogona… │
│                                                            circulant,        │
│                                                            elliptic,         │
│                                                            tv_unitary,       │
│                                                            graph, or         │
│                                                            sdn_hybrid.       │
│                                                            [default:         │
│                                                            hadamard]         │
│ --fdn-tv-rate-hz                         <float range>     Block-rate update │
│                                          [x>=0.0]          speed for         │
│                                                            --fdn-matrix      │
│                                                            tv_unitary (Hz).  │
│                                                            [default: 0.0]    │
│ --fdn-tv-depth                           <float range>     Blend depth for   │
│                                          [0.0<=x<=1.0]     --fdn-matrix      │
│                                                            tv_unitary        │
│                                                            (0..1).           │
│                                                            [default: 0.0]    │
│ --fdn-dfm-delays…                        <str>             Optional          │
│                                                            delay-feedback-m… │
│                                                            delays in         │
│                                                            milliseconds.     │
│                                                            Provide one value │
│                                                            for broadcast or  │
│                                                            one per FDN line. │
│ --fdn-sparse         --no-fdn-sparse                       Enable sparse     │
│                                                            high-order FDN    │
│                                                            pair-mixing mode. │
│                                                            [default:         │
│                                                            no-fdn-sparse]    │
│ --fdn-sparse-deg…                        <int range>       Number of sparse  │
│                                          [1<=x<=16]        pair-mixing       │
│                                                            stages used when  │
│                                                            --fdn-sparse is   │
│                                                            enabled.          │
│                                                            [default: 2]      │
│ --fdn-cascade        --no-fdn-cascade                      Enable            │
│                                                            nested/cascaded   │
│                                                            FDN mode (small   │
│                                                            fast network into │
│                                                            late network).    │
│                                                            [default:         │
│                                                            no-fdn-cascade]   │
│ --fdn-cascade-mix                        <float range>     Injection amount  │
│                                          [0.0<=x<=1.0]     from nested FDN   │
│                                                            into the main     │
│                                                            late-field        │
│                                                            network (0..1).   │
│                                                            [default: 0.35]   │
│ --fdn-cascade-de…                        <float range>     Delay scaling for │
│                                          [0.2<=x<=1.0]     nested FDN        │
│                                                            relative to       │
│                                                            primary FDN       │
│                                                            delays            │
│                                                            (0.2..1.0).       │
│                                                            [default: 0.5]    │
│ --fdn-cascade-rt…                        <float range>     RT60 ratio for    │
│                                          [0.1<=x<=1.0]     nested FDN        │
│                                                            relative to       │
│                                                            --rt60            │
│                                                            (0.1..1.0).       │
│                                                            [default: 0.55]   │
│ --fdn-rt60-low                           <float range>     Low-band RT60     │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-mid                           <float range>     Mid-band RT60     │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-high                          <float range>     High-band RT60    │
│                                          [0.1<=x<=3600.0]  target for        │
│                                                            multiband FDN     │
│                                                            decay shaping     │
│                                                            (seconds).        │
│ --fdn-rt60-tilt                          <float range>     Jot-style         │
│                                          [-1.0<=x<=1.0]    low/high RT skew  │
│                                                            around mid band   │
│                                                            (-1..1). Positive │
│                                                            extends low-band  │
│                                                            decay and         │
│                                                            shortens highs.   │
│                                                            [default: 0.0]    │
│ --fdn-tonal-corr…                        <float range>     Track C tonal     │
│                                          [0.0<=x<=1.0]     correction        │
│                                                            strength for      │
│                                                            multiband/tilted  │
│                                                            FDN response      │
│                                                            (0..1). Higher    │
│                                                            values apply      │
│                                                            stronger          │
│                                                            decay-color       │
│                                                            equalization.     │
│                                                            [default: 0.0]    │
│ --fdn-xover-low-…                        <float range>     Low/mid crossover │
│                                          [x>=20.0]         frequency used by │
│                                                            multiband FDN     │
│                                                            decay shaping.    │
│                                                            [default: 250.0]  │
│ --fdn-xover-high…                        <float range>     Mid/high          │
│                                          [x>=100.0]        crossover         │
│                                                            frequency used by │
│                                                            multiband FDN     │
│                                                            decay shaping.    │
│                                                            [default: 4000.0] │
│ --fdn-link-filter                        <str>             Feedback-link     │
│                                                            filter mode       │
│                                                            inside the FDN    │
│                                                            matrix path:      │
│                                                            none, lowpass, or │
│                                                            highpass.         │
│                                                            [default: none]   │
│ --fdn-link-filte…                        <float range>     Cutoff frequency  │
│                                          [x>=20.0]         used by           │
│                                                            --fdn-link-filter │
│                                                            (Hz).             │
│                                                            [default: 2500.0] │
│ --fdn-link-filte…                        <float range>     Wet mix of        │
│                                          [0.0<=x<=1.0]     feedback-link     │
│                                                            filter processing │
│                                                            (0..1).           │
│                                                            [default: 1.0]    │
│ --fdn-graph-topo…                        <str>             Graph topology    │
│                                                            for --fdn-matrix  │
│                                                            graph: ring,      │
│                                                            path, star, or    │
│                                                            random.           │
│                                                            [default: ring]   │
│ --fdn-graph-degr…                        <int range>       Graph             │
│                                          [1<=x<=32]        neighborhood/con… │
│                                                            degree for        │
│                                                            --fdn-matrix      │
│                                                            graph.            │
│                                                            [default: 2]      │
│ --fdn-graph-seed                         <int>             Deterministic     │
│                                                            seed used to      │
│                                                            build             │
│                                                            graph-structured  │
│                                                            FDN pairings.     │
│                                                            [default: 2026]   │
│ --fdn-spatial-co…                        <none|adjacent|f  Directional       │
│                                          ront_rear|bed_to  wet-bus coupling  │
│                                          p|all_to_all>     mode: none,       │
│                                                            adjacent,         │
│                                                            front_rear,       │
│                                                            bed_top,          │
│                                                            all_to_all.       │
│                                                            [default: none]   │
│ --fdn-spatial-co…                        <float range>     Wet-bus           │
│                                          [0.0<=x<=1.0]     directional       │
│                                                            coupling amount   │
│                                                            (0..1).           │
│                                                            [default: 0.0]    │
│ --fdn-nonlineari…                        <none|tanh|softc  Optional in-loop  │
│                                          lip>              nonlinearity:     │
│                                                            none, tanh, or    │
│                                                            softclip.         │
│                                                            [default: none]   │
│ --fdn-nonlineari…                        <float range>     Blend amount for  │
│                                          [0.0<=x<=1.0]     in-loop           │
│                                                            nonlinearity      │
│                                                            shaping (0..1).   │
│                                                            [default: 0.0]    │
│ --fdn-nonlineari…                        <float range>     Drive multiplier  │
│                                          [0.1<=x<=8.0]     for in-loop       │
│                                                            nonlinearity      │
│                                                            shaping.          │
│                                                            [default: 1.0]    │
│ --room-size-macro                        <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    room-size macro   │
│                                                            (-1..1) mapped to │
│                                                            decay-time and    │
│                                                            spacing behavior. │
│                                                            [default: 0.0]    │
│ --clarity-macro                          <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    clarity macro     │
│                                                            (-1..1) mapped to │
│                                                            decay, damping,   │
│                                                            and wet balance.  │
│                                                            [default: 0.0]    │
│ --warmth-macro                           <float range>     Perceptual warmth │
│                                          [-1.0<=x<=1.0]    macro (-1..1)     │
│                                                            mapped to damping │
│                                                            and spectral      │
│                                                            decay tilt.       │
│                                                            [default: 0.0]    │
│ --envelopment-ma…                        <float range>     Perceptual        │
│                                          [-1.0<=x<=1.0]    envelopment macro │
│                                                            (-1..1) mapped to │
│                                                            width/decorrelat… │
│                                                            emphasis.         │
│                                                            [default: 0.0]    │
│ --fdn-stereo-inj…                        <float range>     [default: 1.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --f0                                     <str>             e.g. 64, 64Hz, or │
│                                                            64 Hz             │
│ --analyze-input                          <path>            Input audio to    │
│                                                            estimate          │
│                                                            fundamentals/har… │
│                                                            for IR tuning     │
│ --harmonic-align…                        <float range>     [default: 0.75]   │
│                                          [0.0<=x<=1.0]                       │
│ --resonator          --no-resonator                        Enable            │
│                                                            Modalys-inspired  │
│                                                            physical          │
│                                                            modal-bank        │
│                                                            late-tail         │
│                                                            coloration.       │
│                                                            [default:         │
│                                                            no-resonator]     │
│ --resonator-mix                          <float range>     [default: 0.35]   │
│                                          [0.0<=x<=1.0]                       │
│ --resonator-modes                        <int range>       [default: 32]     │
│                                          [x>=1]                              │
│ --resonator-q-min                        <float range>     [default: 8.0]    │
│                                          [x>=0.5]                            │
│ --resonator-q-max                        <float range>     [default: 90.0]   │
│                                          [x>=0.5]                            │
│ --resonator-low-…                        <float range>     [default: 50.0]   │
│                                          [x>=20.0]                           │
│ --resonator-high…                        <float range>     [default: 9000.0] │
│                                          [x>=30.0]                           │
│ --resonator-late…                        <float range>     [default: 80.0]   │
│                                          [x>=0.0]                            │
│ --cache-dir                              <str>             [default:         │
│                                                            .verbx_cache/irs] │
│ --lucky                                  <int range>       Generate N        │
│                                          [1<=x<=500]       randomized IR     │
│                                                            files from one    │
│                                                            base setup.       │
│                                                            Outputs are       │
│                                                            written to        │
│                                                            --lucky-out-dir   │
│                                                            (or OUT_IR parent │
│                                                            by default).      │
│ --lucky-out-dir                          <path>            Output directory  │
│                                                            used when --lucky │
│                                                            is enabled.       │
│ --lucky-seed                             <int>             Optional          │
│                                                            deterministic     │
│                                                            seed for --lucky  │
│                                                            IR generation.    │
│ --silent                                                                     │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir analyze --help`

```text

 Usage: root ir analyze [OPTIONS] {ir_file}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    ir_file      <path>  [required]                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --json-out        <path>                                                     │
│ --help                    Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir process --help`

```text

 Usage: root ir process [OPTIONS] {in_ir} {out_ir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    in_ir       <path>  [required]                                          │
│ *    out_ir      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --damping                           <float range>        [default: 0.4]      │
│                                     [0.0<=x<=1.0]                            │
│ --lowcut                            <float range>                            │
│                                     [x>=10.0]                                │
│ --highcut                           <float range>                            │
│                                     [x>=10.0]                                │
│ --tilt                              <float>              [default: 0.0]      │
│ --normalize                         <none|peak|rms>      [default: peak]     │
│ --peak-dbfs                         <float>              [default: -1.0]     │
│ --target-lufs                       <float>                                  │
│ --true-peak        --sample-peak                         [default:           │
│                                                          true-peak]          │
│ --lucky                             <int range>          Generate N          │
│                                     [1<=x<=500]          randomized          │
│                                                          processed IR files  │
│                                                          from one input IR.  │
│                                                          Outputs are written │
│                                                          to --lucky-out-dir  │
│                                                          (or OUT_IR parent   │
│                                                          by default).        │
│ --lucky-out-dir                     <path>               Output directory    │
│                                                          used when --lucky   │
│                                                          is enabled.         │
│ --lucky-seed                        <int>                Optional            │
│                                                          deterministic seed  │
│                                                          for --lucky IR      │
│                                                          processing.         │
│ --silent                                                                     │
│ --help                                                   Show this message   │
│                                                          and exit.           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir morph --help`

```text

 Usage: root ir morph [OPTIONS] {ir_a} {ir_b} {out_ir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    ir_a        <path>  [required]                                          │
│ *    ir_b        <path>  [required]                                          │
│ *    out_ir      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --mode                                   <str>             Morph mode:       │
│                                                            linear,           │
│                                                            equal-power,      │
│                                                            spectral, or      │
│                                                            envelope-aware.   │
│                                                            [default:         │
│                                                            equal-power]      │
│ --alpha                                  <float range>     [default: 0.5]    │
│                                          [0.0<=x<=1.0]                       │
│ --early-ms                               <float range>     Early/late split  │
│                                          [x>=0.0]          used by           │
│                                                            split/envelope-a… │
│                                                            morphing (ms).    │
│                                                            [default: 80.0]   │
│ --early-alpha                            <float range>     Optional alpha    │
│                                          [0.0<=x<=1.0]     override for      │
│                                                            early-reflection  │
│                                                            region.           │
│ --late-alpha                             <float range>     Optional alpha    │
│                                          [0.0<=x<=1.0]     override for      │
│                                                            late-tail region. │
│ --align-decay        --no-align-decay                      Align decay       │
│                                                            profiles before   │
│                                                            morphing for      │
│                                                            stable RT         │
│                                                            trajectories.     │
│                                                            [default:         │
│                                                            align-decay]      │
│ --phase-coherence                        <float range>     Phase-coherence   │
│                                          [0.0<=x<=1.0]     safeguard         │
│                                                            strength for      │
│                                                            spectral          │
│                                                            morphing.         │
│                                                            [default: 0.75]   │
│ --spectral-smoot…                        <int range>       Frequency         │
│                                          [0<=x<=128]       smoothing radius  │
│                                                            (FFT bins) used   │
│                                                            by spectral       │
│                                                            modes.            │
│                                                            [default: 3]      │
│ --mismatch-policy                        <coerce|strict>   Mismatch behavior │
│                                                            for               │
│                                                            sample-rate/chan… │
│                                                            differences:      │
│                                                            coerce (align) or │
│                                                            strict (fail).    │
│                                                            [default: coerce] │
│ --target-sr                              <int range>       Optional target   │
│                                          [x>=1]            sample rate for   │
│                                                            morph processing  │
│                                                            and output.       │
│ --cache-dir                              <str>             [default:         │
│                                                            .verbx_cache/ir_… │
│ --silent                                                                     │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir morph-sweep --help`

```text

 Usage: root ir morph-sweep [OPTIONS] {ir_a} {ir_b} {out_dir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    ir_a         <path>  [required]                                         │
│ *    ir_b         <path>  [required]                                         │
│ *    out_dir      <path>  [required]                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --mode                                   <str>             Morph mode:       │
│                                                            linear,           │
│                                                            equal-power,      │
│                                                            spectral, or      │
│                                                            envelope-aware.   │
│                                                            [default:         │
│                                                            equal-power]      │
│ --alpha                                  <float range>     Explicit alpha    │
│                                          [0.0<=x<=1.0]     point. Repeat to  │
│                                                            define custom     │
│                                                            sweep timeline.   │
│ --alpha-start                            <float range>     [default: 0.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --alpha-end                              <float range>     [default: 1.0]    │
│                                          [0.0<=x<=1.0]                       │
│ --alpha-steps                            <int range>       [default: 9]      │
│                                          [2<=x<=257]                         │
│ --out-prefix                             <str>             Output filename   │
│                                                            prefix for        │
│                                                            generated sweep   │
│                                                            IR files.         │
│                                                            [default: morph]  │
│ --early-ms                               <float range>     [default: 80.0]   │
│                                          [x>=0.0]                            │
│ --early-alpha                            <float range>                       │
│                                          [0.0<=x<=1.0]                       │
│ --late-alpha                             <float range>                       │
│                                          [0.0<=x<=1.0]                       │
│ --align-decay        --no-align-decay                      Align decay       │
│                                                            profiles before   │
│                                                            morphing for      │
│                                                            stable RT         │
│                                                            trajectories.     │
│                                                            [default:         │
│                                                            align-decay]      │
│ --phase-coherence                        <float range>     [default: 0.75]   │
│                                          [0.0<=x<=1.0]                       │
│ --spectral-smoot…                        <int range>       [default: 3]      │
│                                          [0<=x<=128]                         │
│ --mismatch-policy                        <coerce|strict>   Mismatch behavior │
│                                                            for               │
│                                                            sample-rate/chan… │
│                                                            differences:      │
│                                                            coerce (align) or │
│                                                            strict (fail).    │
│                                                            [default: coerce] │
│ --target-sr                              <int range>                         │
│                                          [x>=1]                              │
│ --cache-dir                              <str>             [default:         │
│                                                            .verbx_cache/ir_… │
│ --workers                                <int range>       0 = auto          │
│                                          [x>=0]            [default: 0]      │
│ --schedule                               <fifo|shortest-f  [default:         │
│                                          irst|longest-fir  longest-first]    │
│                                          st>                                 │
│ --retries                                <int range>       [default: 0]      │
│                                          [x>=0]                              │
│ --continue-on-er…    --fail-fast                           [default:         │
│                                                            fail-fast]        │
│ --fail-if-any-fa…    --allow-failed                        Exit non-zero     │
│                                                            when any sweep    │
│                                                            step fails.       │
│                                                            [default:         │
│                                                            fail-if-any-fail… │
│ --checkpoint-file                        <path>            Optional          │
│                                                            checkpoint JSON   │
│                                                            path for          │
│                                                            resume-safe sweep │
│                                                            execution.        │
│ --resume                                                   Resume from       │
│                                                            --checkpoint-fil… │
│ --qa-json-out                            <path>            Summary JSON      │
│                                                            output path       │
│                                                            (default:         │
│                                                            <out_dir>/morph_… │
│ --qa-csv-out                             <path>            Per-step QA       │
│                                                            metrics CSV path  │
│                                                            (default:         │
│                                                            <out_dir>/morph_… │
│ --silent                                                                     │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx ir fit --help`

```text

 Usage: root ir fit [OPTIONS] {infile} {out_ir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    infile      <path>  [required]                                          │
│ *    out_ir      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --top-k                                 <int range>        [default: 3]      │
│                                         [x>=1]                               │
│ --base-mode                             <fdn|stochastic|m  [default: hybrid] │
│                                         odal|hybrid>                         │
│ --length                                <float range>      [default: 60.0]   │
│                                         [x>=0.1]                             │
│ --seed                                  <int>              [default: 0]      │
│ --candidate-pool                        <int range>        [default: 12]     │
│                                         [x>=1]                               │
│ --fit-workers                           <int range>        0 = auto          │
│                                         [x>=0]             [default: 0]      │
│ --analyze-tuning    --no-analyze-tu…                       [default:         │
│                                                            analyze-tuning]   │
│ --cache-dir                             <str>              [default:         │
│                                                            .verbx_cache/irs] │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch --help`

```text

 Usage: root batch [OPTIONS] COMMAND [ARGS]...

 Batch manifest generation and rendering.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ template                                                                     │
│ augment-template                                                             │
│ augment-profiles                                                             │
│ augment                                                                      │
│ render                                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch template --help`

```text

 Usage: root batch template [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch augment-template --help`

```text

 Usage: root batch augment-template [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch augment-profiles --help`

```text

 Usage: root batch augment-profiles [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --json          Emit profile definitions as JSON instead of a table.         │
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch augment --help`

```text

 Usage: root batch augment [OPTIONS] {manifest}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    manifest      <path>  [required]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --output-root                            <path>            Override output   │
│                                                            root directory    │
│                                                            from manifest.    │
│ --profile                                <str>             Optional profile  │
│                                                            override (for     │
│                                                            quick profile A/B │
│                                                            against one       │
│                                                            manifest).        │
│ --seed                                   <int>             Optional          │
│                                                            deterministic     │
│                                                            seed override.    │
│ --variants-per-i…                        <int range>       Optional          │
│                                          [1<=x<=500]       variants-per-sou… │
│                                                            override.         │
│ --write-analysis     --no-write-anal…                      Override manifest │
│                                                            write_analysis    │
│                                                            behavior.         │
│ --copy-dry           --no-copy-dry                         Copy clean source │
│                                                            files into output │
│                                                            tree (paired      │
│                                                            dry/wet dataset   │
│                                                            layout).          │
│                                                            [default:         │
│                                                            no-copy-dry]      │
│ --verify-split-i…    --allow-split-o…                      Require one       │
│                                                            source id/input   │
│                                                            file to belong to │
│                                                            exactly one split │
│                                                            (prevents         │
│                                                            train/val/test    │
│                                                            leakage).         │
│                                                            [default:         │
│                                                            verify-split-iso… │
│ --jobs                                   <int range>       0 = auto          │
│                                          [x>=0]            [default: 0]      │
│ --schedule                               <fifo|shortest-f  [default:         │
│                                          irst|longest-fir  longest-first]    │
│                                          st>                                 │
│ --retries                                <int range>       [default: 0]      │
│                                          [x>=0]                              │
│ --continue-on-er…    --fail-fast                           [default:         │
│                                                            fail-fast]        │
│ --fail-if-any-fa…    --allow-failed                        Exit non-zero     │
│                                                            when any          │
│                                                            augmentation      │
│                                                            render fails.     │
│                                                            [default:         │
│                                                            fail-if-any-fail… │
│ --dry-run                                                                    │
│ --jsonl-out                              <path>            Path for dataset  │
│                                                            metadata JSONL    │
│                                                            (default:         │
│                                                            <output_root>/au… │
│ --summary-out                            <path>            Path for run      │
│                                                            summary JSON      │
│                                                            (default:         │
│                                                            <output_root>/au… │
│ --dataset-card-o…                        <path>            Optional Markdown │
│                                                            dataset-card path │
│                                                            for ML dataset    │
│                                                            documentation.    │
│ --metrics-csv-out                        <path>            Optional CSV path │
│                                                            for per-output    │
│                                                            acoustic          │
│                                                            features.         │
│ --metrics-includ…    --metrics-fast                        Include           │
│                                                            LUFS/true-peak/L… │
│                                                            in metrics CSV    │
│                                                            export (slower).  │
│                                                            [default:         │
│                                                            metrics-fast]     │
│ --qa-bundle-out                          <path>            Optional QA       │
│                                                            bundle JSON path  │
│                                                            (default:         │
│                                                            <output_root>/au… │
│ --baseline-summa…                        <path>            Optional prior    │
│                                                            augmentation      │
│                                                            summary/QA bundle │
│                                                            JSON used for     │
│                                                            regeneration      │
│                                                            deltas.           │
│ --provenance-hash    --no-provenance…                      Emit              │
│                                                            deterministic     │
│                                                            provenance hash   │
│                                                            over manifest     │
│                                                            payload + source  │
│                                                            signatures.       │
│                                                            [default:         │
│                                                            no-provenance-ha… │
│ --help                                                     Show this message │
│                                                            and exit.         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx batch render --help`

```text

 Usage: root batch render [OPTIONS] {manifest}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    manifest      <path>  [required]                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --jobs                                <int range> [x>=0]  0 = auto           │
│                                                           [default: 0]       │
│ --schedule                            <fifo|shortest-fir  [default:          │
│                                       st|longest-first>   longest-first]     │
│ --retries                             <int range> [x>=0]  [default: 0]       │
│ --continue-on-error    --fail-fast                        [default:          │
│                                                           fail-fast]         │
│ --checkpoint-file                     <path>              Optional           │
│                                                           checkpoint file    │
│                                                           used to persist    │
│                                                           per-job completion │
│                                                           state.             │
│ --resume                                                  Resume from        │
│                                                           --checkpoint-file  │
│                                                           and skip already   │
│                                                           completed jobs.    │
│ --dry-run                                                                    │
│ --lucky                               <int range>         For each manifest  │
│                                       [1<=x<=500]         job, generate N    │
│                                                           wild random render │
│                                                           variants. Outputs  │
│                                                           are written to     │
│                                                           --lucky-out-dir    │
│                                                           (or each job       │
│                                                           OUTFILE parent by  │
│                                                           default).          │
│ --lucky-out-dir                       <path>              Output directory   │
│                                                           used when --lucky  │
│                                                           is enabled.        │
│ --lucky-seed                          <int>               Optional           │
│                                                           deterministic seed │
│                                                           for --lucky batch  │
│                                                           generation.        │
│ --progress-json                       <path>              Append one JSONL   │
│                                                           line per completed │
│                                                           job to this file.  │
│                                                           Each line contains │
│                                                           index, outfile,    │
│                                                           success,           │
│                                                           duration_seconds,  │
│                                                           and error.         │
│ --help                                                    Show this message  │
│                                                           and exit.          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive --help`

```text

 Usage: root immersive [OPTIONS] COMMAND [ARGS]...

 Immersive production interoperability workflows.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ template                                                                     │
│ handoff                                                                      │
│ qc                                                                           │
│ queue     Distributed immersive queue workflows.                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive template --help`

```text

 Usage: root immersive template [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive handoff --help`

```text

 Usage: root immersive handoff [OPTIONS] {scene_file} {out_dir}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    scene_file      <path>  [required]                                      │
│ *    out_dir         <path>  [required]                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --strict    --warn-only      Fail if policy/QC errors are detected.          │
│                              [default: strict]                               │
│ --help                       Show this message and exit.                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive qc --help`

```text

 Usage: root immersive qc [OPTIONS] {infile}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    infile      <path>  [required]                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --layout                      <str>                  Channel layout hint:    │
│                                                      auto, mono, stereo,     │
│                                                      lcr, 5.1, 7.1, 7.1.2,   │
│                                                      7.1.4, 7.2.4, 8.0,      │
│                                                      16.0, 64.4              │
│                                                      [default: auto]         │
│ --target-lufs                 <float>                [default: -18.0]        │
│ --lufs-tolerance              <float range>          [default: 3.0]          │
│                               [x>=0.0]                                       │
│ --max-true-peak-dbfs          <float>                [default: -1.0]         │
│ --max-fold-down-delta…        <float range>          [default: 4.0]          │
│                               [x>=0.0]                                       │
│ --min-channel-occupan…        <float range>          [default: 0.34]         │
│                               [0.0<=x<=1.0]                                  │
│ --occupancy-threshold…        <float>                [default: -45.0]        │
│ --json-out                    <path>                 Optional output path    │
│                                                      for QC JSON payload.    │
│ --fail-on-violation                                  Exit with code 2 when   │
│                                                      any QC gate fails.      │
│ --help                                               Show this message and   │
│                                                      exit.                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive queue --help`

```text

 Usage: root immersive queue [OPTIONS] COMMAND [ARGS]...

 Distributed immersive queue workflows.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ template                                                                     │
│ status                                                                       │
│ worker                                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive queue template --help`

```text

 Usage: root immersive queue template [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive queue status --help`

```text

 Usage: root immersive queue status [OPTIONS] {queue_file}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    queue_file      <path>  [required]                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## `verbx immersive queue worker --help`

```text

 Usage: root immersive queue worker [OPTIONS] {queue_file}

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    queue_file      <path>  [required]                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --worker-id                           <str>               Worker identifier. │
│                                                           Defaults to host   │
│                                                           PID-based value.   │
│ --heartbeat-dir                       <path>              Directory for      │
│                                                           per-worker         │
│                                                           heartbeat JSON     │
│                                                           files.             │
│                                                           [default:          │
│                                                           .verbx_queue_hear… │
│ --poll-ms                             <int range>         [default: 800]     │
│                                       [x>=50]                                │
│ --max-jobs                            <int range> [x>=0]  0 = run until      │
│                                                           queue drain        │
│                                                           [default: 0]       │
│ --stale-claim-seco…                   <float range>       [default: 120.0]   │
│                                       [x>=1.0]                               │
│ --continue-on-error    --fail-fast                        [default:          │
│                                                           continue-on-error] │
│ --fail-if-any-fail…                                       Exit with code 2   │
│                                                           if any queue jobs  │
│                                                           end in failed      │
│                                                           state.             │
│ --help                                                    Show this message  │
│                                                           and exit.          │
╰──────────────────────────────────────────────────────────────────────────────╯
```
