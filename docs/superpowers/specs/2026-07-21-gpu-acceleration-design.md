# GPU Acceleration Design for the Coilgun Numerical Solver

**Date:** 2026-07-21  
**Status:** Approved design, pending implementation plan  
**Scope:** CUDA execution of single high-resolution simulations and batched parameter sweeps

## 1. Context

The project implements a synchronous induction coilgun simulation using the current
filament method. The canonical numerical model is documented in
`docs/NumericalModel.md`. The dominant physical calculation is the position-dependent
mutual inductance and mutual-inductance gradient between every active driving-coil
filament distribution and every armature filament. Each pair is evaluated by a 4D
Gauss-Legendre quadrature whose integrand contains complete elliptic integrals.

The repository already has a CUDA backend. The current backend provides CUDA paths for:

- 4D mutual-inductance and gradient evaluation in `gpu_mutual_pipeline.cu`;
- force reduction, acceleration and Euler state update in `gpu_state_kernels.cu`;
- temperature, resistivity, resistance and Joule-energy update in `gpu_thermal.cu`;
- batched dense solving through cuBLAS batched LU in `gpu_solver.cu`;
- partial CUDA Graph support for the mutual-inductance segment;
- CPU fallback and execution reporting.

The remaining execution path is not fully device-resident. In particular,
`GpuEngine::assemble_physical_system()` assembles the system matrix and RHS on the
host, and `GpuEngine::execute_physical_pipeline()` uploads and downloads several
buffers at every step. The current thermal path also copies input and output arrays
for every update. These transfers and synchronizations limit end-to-end speedup,
especially when the physical workload is small enough that mutual-inductance kernels
are no longer dominant.

The design therefore targets two workloads with one shared physical-kernel layer:

1. **Single high-resolution simulation:** minimize end-to-end latency for one
   simulation with a large armature mesh or many active stages.
2. **Batched parameter sweep:** maximize throughput for independent simulations that
   share geometry and differ in voltage, trigger policy, initial state or other
   per-simulation parameters.

The implementation must preserve the existing three optimization levels:

- `Standard`: FP64 mutual-inductance integrand, no distance cutoff, 9-point
  Gauss-Legendre quadrature per dimension. This is the validation/reference GPU path.
- `Full`: FP64 mutual-inductance integrand, distance cutoff, 9-point quadrature. This
  is the default production path.
- `Aggressive`: FP32 mutual-inductance integrand with FP64 accumulation/reduction and
  distance cutoff. This is the high-throughput sweep path.

The optimization level changes numerical evaluation policy, not the physical model or
state layout. Any approximation introduced by `Aggressive` must remain isolated to the
existing precision policy and must not leak into `Standard` or `Full`.

## 2. Goals

- Keep geometry, fixed mutual inductance data and dynamic simulation state on the GPU
  for the duration of a run whenever the CUDA path is selected.
- Move dynamic system-matrix and RHS assembly to the GPU.
- Connect the GPU mutual pipeline directly to the GPU assembly and solver buffers
  without downloading `M` and `dM/dx` after every step.
- Keep current derivatives, currents, motion state and thermal state device-resident
  between steps.
- Capture a complete fixed-shape time-step pipeline in CUDA Graph mode, not only the
  mutual-inductance segment.
- Preserve a direct mode for debugging, dynamic boundaries and unsupported graph
  variants.
- Improve batch throughput by sharing immutable geometry and fixed inductance matrices
  across batch members and by moving per-member control reductions toward the GPU.
- Preserve CPU fallback and make fallback decisions explicit in `ExecutionReport`.
- Validate every optimization level against the CPU reference and existing GPU tests.
- Measure actual wall-clock performance for both single-run latency and sweep
  throughput; do not infer speedup from kernel time alone.

## 3. Non-Goals

- Replacing the current filament physical model with a finite-element model.
- Changing the forward Euler physical semantics as part of the first GPU residency
  work.
- Making GPU RK4 available. The current GPU wrappers explicitly reject `RK4Stepper`;
  this design keeps that contract unchanged.
- Implementing a persistent resident control-stream protocol in the first phase.
  `Persistent` remains a separate follow-up and may continue to resolve to safe CPU
  fallback until its stream ownership and shutdown protocol are complete.
- Replacing the dense solver with a sparse solver without numerical evidence that the
  matrix structure and conditioning justify it.
- Removing the CPU reference implementation or its tests.

## 4. Existing Alternatives and Decision

### 4.1 Alternative A: Extend Individual CUDA Kernels

This approach adds kernels for more isolated operations, such as matrix assembly, but
retains the current host orchestration and per-step downloads.

**Advantages:**

- Smallest initial code change.
- Easy to validate one kernel at a time.
- Low risk to the existing public API.

**Disadvantages:**

- Does not remove the main synchronization and transfer overhead.
- GPU work remains split by host boundaries.
- End-to-end latency may improve little even if individual kernels are faster.

**Decision:** Use only as an intermediate implementation technique inside the
recommended plan, not as the final architecture.

### 4.2 Alternative B: Device-Resident Single and Batched Engines

This approach keeps the simulation state and dynamic physical buffers on the GPU,
performs matrix/RHS assembly on the GPU, and uses separate execution policies for
single-run latency and batch throughput.

**Advantages:**

- Removes the largest recurring Host/Device transfers.
- Allows the existing mutual, force, state and thermal kernels to form one pipeline.
- Supports both `B=1` latency optimization and large-`B` sweep throughput.
- Fits the existing `GpuEngine`, `GpuStateLayout`, execution policy and graph variant
  concepts.
- Allows incremental rollout with direct mode and CPU fallback.

**Disadvantages:**

- Requires device-side state ownership and explicit boundary synchronization rules.
- Requires careful treatment of stage masks, trigger masks, inactive batch members and
  thermal state.
- Dense solver integration must be changed so that matrix and RHS buffers can remain
  on the device.

**Decision:** Recommended architecture and first implementation target.

### 4.3 Alternative C: Structured Block Solver

This approach uses the block structure of the circuit matrix:

\[
\begin{bmatrix}
L_{cc} & M_{ca} \\
M_{ac} & L_{aa}
\end{bmatrix}
\]

to apply block elimination or a Schur complement instead of factoring the complete
`(S+F) x (S+F)` dense matrix.

**Advantages:**

- Potentially lower arithmetic and memory cost for large filament counts.
- Better opportunity to reuse fixed `L_aa` structure and stage blocks.
- May reduce the performance sensitivity of large single simulations to dense LU.

**Disadvantages:**

- Higher implementation and numerical-stability risk.
- The armature filament mutual matrix is generally dense, so the problem is not
  automatically sparse.
- Requires conditioning analysis, residual tests and a robust fallback to dense solve.

**Decision:** Recommended second-stage optimization after device residency and full
   pipeline measurements are established.

## 5. Recommended Architecture

### 5.1 Shared Device-Resident State

The existing logical layout remains row-major and indexed as `[batch][stage][filament]`
or `[batch][dimension]`. The implementation should add device-owned counterparts for:

- currents and current derivatives;
- position, velocity, acceleration and force;
- stage voltages and stage masks;
- mutual-inductance values and gradients;
- dynamic system matrices and RHS/solutions;
- temperatures, resistances, resistivities and Joule energy;
- active and trigger masks;
- fixed geometry and fixed stage/filament mutual-inductance matrices.

The host `GpuEngineState` remains the public synchronous representation and is updated
only at explicit observation boundaries. A run may request one of these boundaries:

- final result only;
- periodic sampled history;
- every-step debug/validation mode;
- explicit host synchronization requested by the caller.

The existing CPU fallback continues to use the host representation directly.

### 5.2 Fixed and Dynamic Data Separation

The following data is immutable for a simulation geometry and should be uploaded once:

- driving-coil geometry and turns;
- armature filament geometry and relative axial positions;
- stage self-inductances and resistances;
- filament self-inductances;
- inter-stage mutual inductance matrix;
- inter-filament mutual inductance matrix;
- filament masses, reference resistances and material identifiers;
- quadrature nodes and weights;
- material lookup tables when the selected thermal mode needs them.

The following data changes per step and should remain in device buffers:

- armature position and velocity;
- active, trigger, stage and mutual masks;
- stage voltages;
- currents and current derivatives;
- position-dependent `M_ca` and `dM_ca`;
- temperature-dependent resistance and thermal outputs;
- force and acceleration.

Only boundary changes should require host-to-device control updates. Geometry arrays
must not be uploaded at every step.

### 5.3 GPU Time-Step Pipeline

The physical pipeline follows the order already represented by `PipelineStage`, with
the host boundary removed from the interior:

1. Update dynamic position-dependent separations from device position state.
2. Evaluate `M_ca` and `dM_ca` using the selected `Standard`, `Full` or `Aggressive`
   mutual kernel.
3. Assemble the circuit matrix and RHS on the GPU.
4. Solve for current derivatives on the GPU.
5. Reduce electromagnetic force and compute acceleration.
6. Apply Euler current, velocity and position updates on the GPU.
7. Update thermal state and temperature-dependent resistance on the GPU when enabled.
8. Evaluate device-side active/finished flags and expose a compact boundary result.

The first implementation may use separate kernels for these stages. Kernel fusion is
not required initially; the priority is eliminating host synchronization. Fusion can
be evaluated after profiling identifies launch overhead as a material fraction of the
step time.

### 5.4 Matrix and RHS Assembly

The device assembly must preserve the current physical matrix semantics:

- stage self and inter-stage mutual inductances form the stage block;
- filament self and fixed inter-filament mutual inductances form the armature block;
- dynamic coil-filament mutual values populate the off-diagonal coupling blocks;
- dynamic `dM/dx` and velocity/current products contribute motional EMF terms to the
  RHS;
- stage and filament masks determine whether a physical slot participates;
- inactive slots use the same identity-row behavior as the current CPU engine.

The fixed matrices should be stored in a solver-friendly layout during initialization.
The assembly kernel should write only the dynamic portions when possible, rather than
clearing and rebuilding all matrix entries every step. A full rebuild remains an
acceptable first implementation if profiling confirms that it is not dominant.

The implementation must add device-side matrix/RHS tests against the host assembly for
all three precision levels. These tests should compare the assembled matrix and RHS
before solving, because a final-state-only comparison cannot distinguish assembly and
solver errors.

### 5.5 Solver Policy

Solver selection remains dimension- and batch-dependent:

- Small systems may continue to use CPU Eigen in the initial direct/reference path if
  measurements show that launch and transfer overhead exceed GPU solve time.
- Medium and large batched systems use device-resident batched LU where it is stable
  and beneficial.
- Large single systems use the GPU dense solver when the problem size justifies it;
  CUDA Graph mode must avoid repeating host-side matrix conversion and synchronization.
- The solver must expose a device-buffer interface so matrix, RHS and solution do not
  require Host/Device copies for every step.
- Residual checks must be available in validation mode. Production mode may use a
  device-side residual/reduction and copy back only a compact status value.
- Any factorization failure, non-finite result or residual failure must lock the
  engine to the documented CPU fallback path without committing the failed step.

The dense solver redesign is deliberately separate from the structured block solver.
The first solver milestone is device residency and stable batched execution, not a
new mathematical factorization.

### 5.6 Single-Simulation Path

For `B=1`, the engine should optimize latency rather than maximize occupancy:

- retain geometry and fixed matrices for the entire run;
- select `Eigen` for small systems when it is faster in measured end-to-end time;
- select device dense solve for high-resolution systems;
- use direct mode when masks or controls change frequently;
- use Graph mode when the system shape and kernel topology remain stable;
- synchronize to the host only at requested history or result boundaries.

The public compatibility wrappers may continue to record every step by default for API
compatibility. Internally, a configurable observation policy should make it possible to
run without copying the full state at every step. The default policy must not silently
change existing history semantics; any reduced-observation mode must be explicit.

### 5.7 Batched Sweep Path

For `B>1`, each batch member has independent dynamic state but shares immutable geometry
and fixed matrices. The batch path should:

- store all per-member state in contiguous SoA-like flat arrays;
- process mutual pairs as a three-dimensional `[batch][stage][filament]` workload;
- use masks rather than compacting members during a run;
- advance excitation, trigger and finished state on the device where practical;
- use a compact device-to-host status buffer for members that stop or require a graph
  variant change;
- keep completed members inactive while other members continue;
- avoid host loops over every active member on every step once device control is enabled.

The host may still apply complex user-defined trigger policies at a declared boundary.
The device control path initially supports the existing position and time-delay trigger
modes. Unsupported policies must use direct CPU control or an explicit fallback, not an
implicit approximation.

### 5.8 CUDA Graph Strategy

Graph capture is extended in stages:

1. Keep the current mutual-only graph as a fallback implementation.
2. Capture matrix assembly, solver, force and state update for a fixed layout.
3. Add thermal kernels to the graph when thermal mode is enabled.
4. Add device-side status reduction and boundary flags.

Graph variants are keyed by all values that change graph topology or kernel selection:

- batch capacity and active batch shape;
- stage and filament dimensions;
- stage/mutual mask topology;
- precision mode;
- thermal mode;
- solver mode;
- any fixed launch configuration.

Stage trigger changes that alter masks require selection or capture of another variant at
a step boundary. Graph capture failures lock the graph path to direct or CPU fallback
according to the existing reporting contract.

## 6. Precision-Level Behavior

### 6.1 Standard

- FP64 input geometry and arithmetic in mutual kernels.
- No distance cutoff.
- FP64 reduction.
- Full validation against CPU mutual values, gradients, matrix/RHS assembly and final
  trajectories.
- Preferred for diagnostics, reference comparisons and numerical regression tests.

### 6.2 Full

- Same FP64 arithmetic as Standard for the mutual integrand.
- Distance cutoff enabled according to the current GPU policy.
- FP64 reduction and solver inputs.
- Default production mode.
- Must be compared against Standard on representative near-field and far-field cases
  to verify that cutoff behavior does not alter physically relevant interactions.

### 6.3 Aggressive

- FP32 mutual integrand and elliptic evaluation where currently defined.
- FP64 accumulation/reduction and FP64 state/solver storage unless a later design
  explicitly approves mixed-precision solving.
- Distance cutoff enabled.
- Intended for large parameter sweeps, not the reference path.
- Existing tolerance policy remains the authority. The current benchmark records
  approximately `1e-5` to `1e-4` relative trajectory differences for representative
  quantities; future changes must update measured tolerances rather than assume a
  universal bound.

No precision mode may change the circuit equations, force expression, motion update
order or thermal update order.

## 7. Control, Trigger and Thermal Semantics

The GPU residency work must preserve the current explicit-Euler step boundary. The
implementation must document and test whether a value is pre-step or post-step for:

- current derivative evaluation;
- capacitor/excitation advancement;
- temperature and Joule heating update;
- force recording;
- trigger checks;
- quiet-stage completion.

The existing numerical model specifies pre-step values for the Euler updates in
`docs/NumericalModel.md`, while the current CPU and wrapper implementations contain
post-step interactions in some paths. This is a model-contract issue, not a GPU
optimization detail. Before declaring GPU residency complete, the implementation plan
must include focused regression tests for the chosen canonical ordering. If the
ordering is corrected, CPU and GPU paths must be changed together and the API and
numerical documentation must be synchronized.

Thermal state must remain device-resident in GPU thermal mode. The kernel must consume
the previous-step current/resistance values required by Eq. (6.10), then update
temperature, resistivity, resistance and Joule energy in one consistent step. CPU
thermal mode remains available for reference and fallback.

## 8. Error Handling and Fallback

The following conditions must not commit a partially updated state:

- invalid device buffer or dimension;
- CUDA launch or synchronization failure;
- non-finite mutual, matrix, RHS or solution value;
- factorization failure;
- residual beyond the configured validation threshold;
- graph capture or replay failure;
- unsupported trigger/control policy;
- device memory allocation failure.

On failure:

1. preserve the pre-step host and device state until the physical step succeeds;
2. record the failure category and message in `ExecutionReport`;
3. lock the failing CUDA mode for the current engine instance;
4. execute the equivalent CPU step from the pre-step snapshot;
5. keep all subsequent steps on the resolved fallback path unless an explicit reset or
   new engine construction selects another policy.

The fallback must be observable through `backend`, `solver`, `thermal`, fallback reason
and `gpu_executed`. A fallback timing row must never be reported as GPU performance.

## 9. Recommended Implementation Order

### Phase 0: Contract and Baseline

- Freeze representative CPU reference cases for single-stage, multi-stage, thermal and
  batch workloads.
- Add matrix/RHS assembly observability and comparison tests.
- Record `Standard`, `Full` and `Aggressive` correctness tolerances separately.
- Separate kernel time, transfer time, host assembly time, solver time and control time
  in the benchmark report.
- Confirm the canonical pre-step/post-step ordering for excitation, thermal state and
  force history before moving those operations to the device.

**Exit criteria:** existing CPU/GPU tests pass, baseline benchmark is reproducible and
the physical state contract is explicit.

### Phase 1: GPU-Resident State and Dynamic Buffers

- Add a device state/workspace owner to `GpuEngine`.
- Upload fixed geometry, quadrature, fixed mutual matrices and material data once.
- Keep dynamic currents, position, velocity, voltages, masks, `M_ca` and `dM_ca` in
  device buffers.
- Keep the current direct mutual kernel and existing state kernels as the first
  resident pipeline.
- Add explicit observation/synchronization boundaries.

**Exit criteria:** a multi-step GPU run produces the same `Standard` and `Full` results
as the current GPU path and no per-step geometry or mutual-result download is needed.

### Phase 2: GPU Matrix/RHS Assembly

- Implement device assembly for the coupled circuit matrix and RHS.
- Reuse fixed stage and filament blocks where possible.
- Add assembly-only tests for masks, mutual coupling, motional EMF, thermal resistance
  and inactive batch members.
- Preserve host assembly as a debug/reference path.

**Exit criteria:** device and host assembly agree within the precision-level tolerance;
the solver receives device-resident matrix and RHS data.

### Phase 3: Device-Resident Solver Integration

- Add a solver API that accepts device buffers and returns device-resident solutions.
- Remove row-major/column-major host staging from the steady-state path.
- Move residual checking to a selectable device validation kernel plus compact status
  copy where practical.
- Keep CPU Eigen and existing host batched solving as fallback paths.

**Exit criteria:** high-resolution single runs and batch runs show measured end-to-end
benefit without loss of stability; solver fallback tests pass.

### Phase 4: Complete CUDA Graph Pipeline

- Extend graph capture from mutual-only to assembly, solve, force, state and thermal
  stages.
- Cache variants by layout, masks, precision, thermal and solver policy.
- Rebuild only when a boundary control changes graph topology.
- Keep Direct mode for dynamic or unsupported cases.

**Exit criteria:** graph replay reduces host orchestration time in fixed-shape runs and
graph failure cleanly falls back without corrupting state.

### Phase 5: Batch Control Migration

- Move position/time-delay trigger checks, excitation progression, quiet-stage checks
  and active-member status toward device kernels.
- Reduce per-step host loops over `SimInstance` objects.
- Keep a compact host control boundary for user-visible history and unsupported policies.

**Exit criteria:** batch throughput scales with batch size until device compute or memory
bandwidth becomes the limiting factor, and results remain per-member correct.

### Phase 6: Structured Solver Investigation

- Profile dense solver cost after Phases 1-5.
- Evaluate block elimination and Schur-complement formulations using the fixed
  armature block and low-dimensional stage block.
- Measure conditioning, residuals, memory use and precision-level behavior.
- Use dense GPU/CPU solve as a per-case fallback.

**Exit criteria:** adopt the structured solver only if it improves measured latency or
throughput on representative high-resolution workloads without violating the existing
precision tolerances.

## 10. Verification Strategy

### Numerical Verification

- Compare filament-level `M` and `dM/dx` against the CPU implementation.
- Compare full coil-level quadrature results for near-field, crossover and far-field
  separations.
- Compare host/device matrix and RHS assembly before solving.
- Compare current derivatives, currents, force, position and velocity step by step.
- Compare thermal temperature, resistivity, resistance and Joule energy step by step.
- Test all stage-mask, trigger-mask, inactive-batch and finished-stage combinations.
- Test graph/direct/fallback equivalence.
- Test `Standard`, `Full` and `Aggressive` independently; do not use an Aggressive
  result to loosen Standard or Full tolerances.

### Performance Verification

Benchmark at minimum:

- small single-stage workload;
- medium multi-stage workload;
- large high-resolution single-stage workload;
- thermal single-stage workload;
- batch sizes 1, 8, 32, 128 and a device-appropriate larger case;
- direct versus graph for fixed-shape runs;
- all three precision levels where numerically valid.

Report separately:

- total wall time;
- device kernel time;
- host assembly time;
- solver time;
- thermal time;
- transfer time;
- control and synchronization time;
- graph capture/rebuild count;
- achieved steps per second and simulations per second for batch runs.

The benchmark must identify CPU fallback rows and exclude them from GPU speedup claims.

### Build and Runtime Verification

- CPU-only configure, build and tests.
- CUDA configure, build and tests.
- Runtime tests on the available NVIDIA device.
- `compute-sanitizer` for new device buffer and kernel paths.
- Device allocation reuse checks across multiple steps and reset cycles.
- Failure injection for allocation, launch, graph, solver and residual failures.

## 11. Risks and Mitigations

**Numerical drift from changed evaluation order:** Use Standard as a stepwise reference,
retain host assembly, and compare matrix/RHS before final trajectories.

**Dense solver instability:** Check residuals, condition indicators and factorization
status; fall back to Eigen or the existing host path.

**Graph invalidation from dynamic masks:** Treat mask topology as a graph key and
rebuild only at explicit boundaries.

**Small workloads becoming slower:** Keep a planner threshold and permit CPU Eigen for
small single systems; select based on measured end-to-end cost, not kernel throughput.

**Batch divergence:** Use masks and inactive-member preservation; never compact state
arrays implicitly or change logical batch indices.

**Host history compatibility:** Keep per-step recording as the compatibility default;
make reduced observation explicit and document its data semantics.

**Thermal ordering mismatch:** Add pre-step/post-step tests before moving thermal state
to device and use the same contract in CPU and GPU implementations.

**Persistent protocol hazards:** Do not enable resident polling until it has dedicated
control-stream ownership, bounded shutdown behavior and failure recovery tests.

## 12. Success Criteria

The design is considered successfully implemented when:

- Single high-resolution runs have lower measured end-to-end latency than the current
  GPU path on representative workloads.
- Batched sweeps achieve higher simulations-per-second than independent CPU runs and
  scale across the intended batch range.
- The majority of steady-state simulation data remains device-resident between explicit
  observation boundaries.
- `Standard` and `Full` remain within their existing validation tolerances.
- `Aggressive` remains within its documented engineering tolerance and is clearly
  reported as a lower-precision mode.
- CPU fallback remains correct, deterministic where previously required and visible in
  execution reports.
- CUDA Graph mode improves fixed-shape runs without changing physical results.
- New device paths pass CPU-only and CUDA test suites, sanitizer checks and failure
  injection tests.

## 13. Open Implementation Decisions

The following choices are intentionally deferred to the implementation plan and
benchmark data:

- exact device matrix layout and whether the assembly kernel writes full matrices or
  updates only dynamic blocks;
- the threshold for selecting Eigen versus device dense solve for `B=1`;
- the maximum batch size before splitting work into multiple launches or tiles;
- the observation API for reduced-history runs;
- whether device-side excitation control should be a separate kernel or fused with the
  state update;
- the block solver formulation, if Phase 6 proves worthwhile.

These decisions must not change the three existing optimization-level meanings or the
CPU fallback contract.
