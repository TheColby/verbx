# System Orientation Through Block Diagrams

This chapter is a visual map of the signal paths, state boundaries, and evidence loops used throughout verbx. Read each diagram from left to right, then use the paragraph beneath it to identify what may happen in realtime, what must be prepared off-thread, and what should be preserved in a report.

## The complete verbx workflow

The figure below introduces **The complete verbx workflow**. Every workflow begins with an identified source and ends with both an audible artifact and machine-readable evidence. Read it from left to right before using the detailed boundary notes beneath the visual.

![The complete verbx workflow.](assets/intro_block_diagrams/01_the_complete_verbx_workflow.png)

**Figure: The complete verbx workflow.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Algorithmic render path

The figure below introduces **Algorithmic render path**. The algorithmic engine separates arrival cues, density growth, sustained decay, and delivery processing. Read it from left to right before using the detailed boundary notes beneath the visual.

![Algorithmic render path.](assets/intro_block_diagrams/02_algorithmic_render_path.png)

**Figure: Algorithmic render path.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Convolution render path

The figure below introduces **Convolution render path**. Partitioned convolution trades a small fixed scheduling delay for efficient processing of long and multichannel impulse responses. Read it from left to right before using the detailed boundary notes beneath the visual.

![Convolution render path.](assets/intro_block_diagrams/03_convolution_render_path.png)

**Figure: Convolution render path.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Realtime callback boundary

The figure below introduces **Realtime callback boundary**. The callback path must remain bounded and allocation-free; preparation, file access, and reporting belong outside it. Read it from left to right before using the detailed boundary notes beneath the visual.

![Realtime callback boundary.](assets/intro_block_diagrams/04_realtime_callback_boundary.png)

**Figure: Realtime callback boundary.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## End-to-end latency stack

The figure below introduces **End-to-end latency stack**. Perceived realtime latency is the sum of device buffers, host scheduling, algorithmic lookahead, and physical propagation. Read it from left to right before using the detailed boundary notes beneath the visual.

![End-to-end latency stack.](assets/intro_block_diagrams/05_end_to_end_latency_stack.png)

**Figure: End-to-end latency stack.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Reverb event anatomy

The figure below introduces **Reverb event anatomy**. A useful listening model treats reverberation as an ordered event rather than a single wet signal. Read it from left to right before using the detailed boundary notes beneath the visual.

![Reverb event anatomy.](assets/intro_block_diagrams/06_reverb_event_anatomy.png)

**Figure: Reverb event anatomy.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## RT60 control hierarchy

The figure below introduces **RT60 control hierarchy**. Coarse and fine controls establish time scale while frequency-dependent decay and freeze determine how energy persists. Read it from left to right before using the detailed boundary notes beneath the visual.

![RT60 control hierarchy.](assets/intro_block_diagrams/07_rt60_control_hierarchy.png)

**Figure: RT60 control hierarchy.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## FDN feedback loop

The figure below introduces **FDN feedback loop**. Delay lengths create modes; filters set color; the matrix disperses energy; calibrated gain establishes decay. Read it from left to right before using the detailed boundary notes beneath the visual.

![FDN feedback loop.](assets/intro_block_diagrams/08_fdn_feedback_loop.png)

**Figure: FDN feedback loop.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Impulse-response lifecycle

The figure below introduces **Impulse-response lifecycle**. An impulse response is an auditable asset whose provenance and preparation affect every convolution result. Read it from left to right before using the detailed boundary notes beneath the visual.

![Impulse-response lifecycle.](assets/intro_block_diagrams/09_impulse_response_lifecycle.png)

**Figure: Impulse-response lifecycle.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Dereverberation path

The figure below introduces **Dereverberation path**. Dereverberation balances late-energy reduction against speech, transient, and ambience preservation. Read it from left to right before using the detailed boundary notes beneath the visual.

![Dereverberation path.](assets/intro_block_diagrams/10_dereverberation_path.png)

**Figure: Dereverberation path.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Dynamics around reverb

The figure below introduces **Dynamics around reverb**. Ducking creates foreground space before a limiter protects the combined output and meters verify the result. Read it from left to right before using the detailed boundary notes beneath the visual.

![Dynamics around reverb.](assets/intro_block_diagrams/11_dynamics_around_reverb.png)

**Figure: Dynamics around reverb.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Spatial routing

The figure below introduces **Spatial routing**. Channel identity must remain explicit from source through room processing and final reproduction. Read it from left to right before using the detailed boundary notes beneath the visual.

![Spatial routing.](assets/intro_block_diagrams/12_spatial_routing.png)

**Figure: Spatial routing.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Automation contract

The figure below introduces **Automation contract**. Automation is safest when parsed and validated off-thread, then applied deterministically at block boundaries. Read it from left to right before using the detailed boundary notes beneath the visual.

![Automation contract.](assets/intro_block_diagrams/13_automation_contract.png)

**Figure: Automation contract.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Analysis evidence chain

The figure below introduces **Analysis evidence chain**. Analysis becomes operational evidence when measurements, policy thresholds, warnings, and provenance share one report. Read it from left to right before using the detailed boundary notes beneath the visual.

![Analysis evidence chain.](assets/intro_block_diagrams/14_analysis_evidence_chain.png)

**Figure: Analysis evidence chain.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Preset lifecycle

The figure below introduces **Preset lifecycle**. A preset is more than values: it needs intent, compatibility metadata, calibrated auditioning, and deterministic recall. Read it from left to right before using the detailed boundary notes beneath the visual.

![Preset lifecycle.](assets/intro_block_diagrams/15_preset_lifecycle.png)

**Figure: Preset lifecycle.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Plug-in host contract

The figure below introduces **Plug-in host contract**. AUv3 and VST3 integration wrap the same DSP in host-specific state, layout, and lifecycle contracts. Read it from left to right before using the detailed boundary notes beneath the visual.

![Plug-in host contract.](assets/intro_block_diagrams/16_plug_in_host_contract.png)

**Figure: Plug-in host contract.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Regression workflow

The figure below introduces **Regression workflow**. Deterministic fixtures make audible and numerical drift discoverable before release. Read it from left to right before using the detailed boundary notes beneath the visual.

![Regression workflow.](assets/intro_block_diagrams/17_regression_workflow.png)

**Figure: Regression workflow.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.

## Learning loop

The figure below introduces **Learning loop**. The textbook exercises use repeated prediction, experiment, measurement, and critical listening to connect controls with perception. Read it from left to right before using the detailed boundary notes beneath the visual.

![Learning loop.](assets/intro_block_diagrams/18_learning_loop.png)

**Figure: Learning loop.** The arrows indicate processing or evidence flow, not elapsed-time scale. Every box names a boundary at which parameters, latency, channel identity, or provenance should be checked.
