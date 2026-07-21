# Final Integrated Design: Coupled Simulation Contracts and GPU Execution

**Date:** 2026-07-21  
**Status:** Final design for implementation planning  
**Scope:** CPU coupled integration correctness, excitation and stage lifecycle semantics, public API and resource contracts, and device-resident CUDA execution for single simulations and batched parameter sweeps.

## 1. Executive Decision

The project will proceed in two ordered tracks that share one physical-state contract:

1. **Correctness and contract track:** define one complete simulation state, make CPU Euler and RK4 obey explicit time-layer semantics, separate continuous integration from discrete events, validate public inputs, and make stage lifecycle state explicit.
2. **Execution track:** preserve that contract while moving the CUDA steady-state pipeline from host orchestration to device-resident state, device matrix/RHS assembly, device solver buffers, complete fixed-shape CUDA Graph capture, and batched device control.

Correctness work is a prerequisite for GPU residency. No GPU optimization may change circuit equations, force calculation, thermal ordering, stage lifecycle semantics, history timestamps, or fallback behavior.

The first implementation does **not** include GPU RK4, a persistent resident control-stream protocol, or a replacement structured solver. Those capabilities require separate feasibility gates after the shared state and execution contracts are stable.

## 2. Context and Current Constraints

The project simulates a synchronous induction coilgun using the current filament method. The dominant physical operation is the position-dependent mutual inductance and mutual-inductance gradient between each active driving-coil filament distribution and each armature filament. Each pair is evaluated by four-dimensional Gauss-Legendre quadrature whose kernel contains complete elliptic integrals.

The current repository contains:

- A CPU reference and production simulation path with Euler and an existing, but incomplete, RK4 policy.
- CUDA mutual-inductance, force, state, thermal, and batched dense-solver components.
- CPU fallback and execution reporting.
- Partial CUDA Graph support for the mutual-inductance segment.
- Public wrappers for single-stage, multi-stage, and batched execution.

The current review and design documents identify the following blockers to calling the implementation fully correct or fully resident:

- CPU RK4 does not integrate excitation state, thermal state, or stage lifecycle state as one coupled system.
- CPU Euler advances excitation and thermal state with post-step current although the documented model requires pre-step current.
- A finished stage can leave a non-zero current frozen in public state after its circuit row becomes inactive.
- Public constructors and configuration boundaries do not consistently reject invalid values before derived calculations or numerical libraries are used.
- CUDA constant tables, thermal workspaces, and adaptor buffers do not have sufficiently explicit ownership and capacity contracts.
- GPU matrix/RHS assembly and much of the step orchestration remain on the host, causing repeated transfers and synchronization.
- CUDA Graph captures only the mutual segment; it does not currently represent a complete physical step.
- CPU and CUDA optimization-level names describe different policies and must not be conflated.

The design below resolves these issues without replacing the current filament physical model.

## 3. Goals

### 3.1 Correctness and API goals

- Define complete ownership for continuous state, excitation runtime state, discrete modes, derived values, and immutable configuration.
- Make CPU Euler match the documented pre-step explicit Euler equations.
- Make CPU RK4 a genuine coupled RK4 for continuous state, with explicit event handling for discrete transitions.
- Keep excitation, thermal, resistance, force, trigger, stage completion, and history semantics deterministic and auditable.
- Reject invalid public inputs before division, allocation, Eigen operations, or CUDA operations.
- Preserve explicit unsupported behavior for GPU RK4. A GPU RK4 request must not silently execute Euler.
- Make result timestamps, sampling, summary statistics, CPU/CUDA presets, and bilingual documentation contracts explicit.

### 3.2 CUDA execution goals

- Keep immutable geometry, fixed inductance data, and dynamic simulation state on the GPU for the duration of a CUDA run.
- Move dynamic circuit matrix and RHS assembly to the GPU.
- Connect mutual-inductance results directly to device assembly and solver buffers.
- Keep currents, current derivatives, motion, masks, thermal state, and control state device-resident between observation boundaries.
- Capture the complete fixed-shape step in CUDA Graph mode.
- Preserve a direct CUDA mode for debugging and dynamic boundaries.
- Improve batch throughput by sharing immutable data and moving per-member control reductions toward the GPU.
- Keep CPU fallback explicit, correct, and visible in `ExecutionReport`.
- Measure end-to-end latency and throughput rather than inferring performance from kernel time.

## 4. Non-Goals and Explicitly Deferred Work

- Replacing the current filament method with a finite-element model.
- Changing the physical model or circuit equations as part of residency work.
- Implementing GPU RK4 in this implementation cycle.
- Implementing a persistent resident control-stream protocol. `Persistent` may continue to resolve to safe CPU fallback.
- Replacing the dense solver with a block, Schur-complement, sparse, or structured solver before conditioning and residual evidence exists.
- Removing the CPU reference implementation or its tests.
- Redesigning the global CPU mutual-inductance LRU in the first correctness cycle. Its non-thread-safe behavior remains an explicit accepted risk; hot parallel paths must bypass it or use caller-owned/thread-local caches.
- Silently changing public history behavior when introducing reduced-observation GPU execution.

## 5. Contract Vocabulary

The implementation must use the following terms consistently.

### 5.1 Time layers

- `pre_state`: the complete state at the beginning of a macro-step or integration segment.
- `derivative_state`: the state supplied to one derivative evaluation.
- `post_state`: the state committed after the macro-step or event segment.
- `post_step history`: a record of `post_state` whose timestamp is `(step_count + 1) * dt`.

No operation may implicitly mix these layers. In particular, capacitor discharge and Joule heating in an Euler step consume pre-step current and pre-step resistance.

### 5.2 Continuous and discrete state

Continuous state is integrated numerically. Discrete state changes only at an event or declared step boundary. Derived values are recalculated from the current continuous state and immutable configuration; they are not independent integration variables.

### 5.3 Backend and precision policy

Backend selection, solver selection, thermal selection, precision selection, fallback reason, and observation policy are separate decisions. A fallback in one subsystem must be reported without falsely claiming that another subsystem executed on the GPU.

The CPU and CUDA optimization names are intentionally separate:

| Layer | Policy names | Meaning |
|---|---|---|
| CPU multi-stage | `Reference`, `LookupTable`, `Full` | Component/runtime CPU policy. `LookupTable` remains a compatibility path until its runtime semantics are expanded. |
| CUDA | `Standard`, `Full`, `Aggressive` | Mutual-kernel precision and cutoff policy. `Standard` is the GPU reference path. |

## 6. Complete Simulation State

### 6.1 Immutable configuration

Immutable configuration includes:

- Driving-coil and armature geometry, turns, masses, material identifiers, and reference resistances.
- Fixed self-inductance and inter-coil/inter-filament mutual-inductance data.
- Capacitance, initial excitation configuration, waveform configuration, trigger policy, termination policy, and integration step size.
- Quadrature nodes and weights selected by the execution policy.
- Thermal material tables and their identity/version.

### 6.2 Continuous state

The CPU and logical GPU state contains:

- Driving-coil and armature-filament currents.
- Armature position and velocity.
- Filament temperatures when thermal mode is enabled.
- Excitation-specific continuous variables owned by their excitation snapshots, such as capacitor voltage or waveform time.

Temperature is the thermal dynamic state. Temperature-dependent resistance is derived as `R(T)` during each derivative evaluation and must not be used as hidden RK4 scratch state.

### 6.3 Discrete state

Discrete state includes, as applicable:

- Stage triggered/not triggered.
- Excitation finished/not finished.
- Circuit active/inactive.
- Stage completed/not completed.
- Crowbar diode off/on.
- Batch member active/finished.

These flags are distinct. In particular, excitation completion does not imply circuit completion.

### 6.4 Excitation ownership

`Excitation` remains polymorphic. Its polymorphic `ExcitationSnapshot` owns every mutable runtime field of a concrete excitation and provides `snapshot()` and `restore()` operations plus read-only evaluation needed by RK4.

Required snapshot content includes:

- Capacitor voltage and finished state.
- Crowbar capacitor voltage, diode mode, and finished state.
- Waveform time and finished state.

Future excitation types must include all mutable runtime data in their snapshot. Capacitor voltage and waveform time must not be duplicated between the simulation state and an excitation snapshot.

## 7. Pure Derivative Evaluation

The derivative evaluator is the common boundary for CPU reference execution and future device-side assembly semantics. It receives:

- A complete derivative state.
- Immutable geometry and configuration.
- An explicit `DerivativeWorkspace`.
- The relevant discrete mode.

It returns continuous derivatives and event observations. It must:

- Not mutate the simulation object or input state.
- Not advance or mutate real excitation objects.
- Not write shared `M`, `dM/dx`, matrix, RHS, resistance, or active-stage scratch members.
- Use workspace owned by the current integration call or simulation instance.
- Be reusable for sequential RK4 evaluations and event localization, but not concurrently shared by multiple integration calls.

The workspace contains reusable mutual-inductance arrays, system matrix, RHS, active-stage indices, derived resistance arrays, and temporary reductions.

## 8. Numerical Integration Contract

### 8.1 Euler

At the beginning of every Euler macro-step, capture the complete pre-step state and excitation snapshots. Evaluate all updates from the same time layer:

```text
I_n = I_(n-1) + dt * dI(pre_state)
v_n = v_(n-1) + dt * dv(pre_state)
z_n = z_(n-1) + dt * dz(pre_state)
U_n = U_(n-1) - dt * I_(n-1) / C
T_n = T_(n-1) + dt * I_(n-1)^2 * R(T_(n-1))
      / (mass * cp(T_(n-1)))
```

The post-step state is committed once. Events are then applied according to the event contract, and history records the resulting post-step state. The newly committed current must not be used to advance excitation or thermal state for the same macro-step.

### 8.2 Event-aware RK4

RK4 integrates the complete continuous state using four derivative evaluations. Each trial state uses an excitation snapshot and thermal-derived resistance corresponding to that trial state. Discrete transitions are not represented as ordinary continuous derivatives.

Events include:

- Capacitor voltage crossing zero.
- Waveform reaching end time.
- Position or time-delay trigger crossing.
- Current crossing the decay threshold.
- Excitation completion and stage completion.

For each integration segment:

1. Compute a trial RK4 step under the current discrete mode.
2. Detect a boundary crossing and bracket the earliest event.
3. Locate the event with protected bisection, recomputing every trial from the segment start.
4. Apply the transition at the event point.
5. Continue over the remaining time.

Events within the configured event tolerance are processed as one batch with this priority:

1. Continuous boundary clamp, such as `U_C = 0`.
2. Crowbar conduction transition.
3. Waveform end-time transition.
4. Excitation-finished transition.
5. Stage trigger.
6. Current decay and stage completion.

Stage triggers in one batch are processed by increasing stage index. The integrator has a finite event iteration/batch limit and detects zero-duration repeated events. Failed localization throws an exception containing event type, stage, and time interval.

### 8.3 Physical step order

The canonical forward-Euler order is:

1. Capture pre-step state, modes, and derived values.
2. Evaluate position-dependent mutual inductance and gradient at the pre-step position required by the selected contract.
3. Assemble and solve the circuit derivative.
4. Compute force and acceleration from the contractually selected time layer.
5. Update current, velocity, and position.
6. Update excitation and thermal state from pre-step values.
7. Apply diode, trigger, excitation, and stage lifecycle events.
8. Record post-step history at `(step_count + 1) * dt`.
9. Evaluate termination.

CPU and GPU paths must use the same ordering. If an existing implementation differs, the contract is corrected in both paths together and the numerical/API documentation is updated.

## 9. Stage Lifecycle

Every stage has explicit lifecycle state:

- `triggered`: the stage is activated.
- `excitation_finished`: the source no longer supplies external voltage.
- `circuit_active`: the stage still participates in circuit and force evaluation.
- `stage_completed`: current has decayed below threshold and the stage may be removed.

Waveform end, capacitor depletion, and crowbar transition stop or change external excitation. They do not immediately remove a stage from the circuit. A non-zero coil current continues through the RL/mutual decay path while `circuit_active` is true.

Only at `stage_completed`:

- The stage current is explicitly set to zero.
- The stage row and column are replaced by the inactive identity block.
- The stage is removed from force and active-stage summaries.
- The lifecycle state is recorded in history and summary data.

No inactive identity row may coexist with a non-zero public stage current.

## 10. Public Input and Resource Validation

Shared validation helpers are used at all public construction and configuration boundaries. Validation occurs before derived division, allocation, Eigen operation, CUDA operation, or waveform execution.

Required validation includes:

- All scalar physical and integration values are finite unless positive infinity is explicitly allowed for a terminal trigger or no-end-time policy.
- `0 <= inner_radius < outer_radius`.
- Positive length, turns, wire area, density, mass, capacitance, and `dt`.
- `0 < fill_factor <= 1`.
- Positive axial and radial division counts.
- Non-null excitation pointers.
- Non-empty waveform functions.
- Valid non-negative waveform end time, with positive infinity reserved for supported no-end-time behavior.
- Finite, positive derived resistance, inductance, and filament mass.
- `coils.size() == excitations.size()` and `trigger_configs.size() == coils.size() - 1`.
- `sampled(every_n)` requires `every_n > 0`.

All invalid arguments throw `std::invalid_argument` with the parameter name and violated constraint.

`DrivingCoil` uses `std::optional<double>` for position: `nullopt` means the default center position, while every finite explicit coordinate, including a negative coordinate, is preserved.

### 10.1 CUDA adaptor contract

`GpuAdaptor` tracks configured state, device identity, single and batch capacities, and pair counts. Upload and download sizes must match configured capacities:

- Single separation count: `n_stages * n_filaments`.
- Batch separation count: `batch_size * n_stages * n_filaments`.
- Single result download count: configured single pair count.

Unconfigured use, negative counts, mismatched sizes, integer overflow, null buffers, and invalid dimensions throw `std::invalid_argument`. Every CUDA allocation, copy, symbol operation, launch, and synchronization call is checked with an operation-specific message.

### 10.2 CUDA ownership

`GpuExecutionContext` owns device initialization and immutable device resources. The mutual pipeline uses a fixed nine-node Gauss-Legendre table for the first implementation. Normal launches do not rewrite device-global constant symbols, and `n_nodes == 9` is required for the CUDA mutual path.

`ThermalWorkspace` records:

- Owning device.
- Material-table identity/version.
- Temperature range.
- Allocation dimensions.

Workspace reuse requires all ownership and table keys to match. Resource lifetime is attached to the owning CUDA context rather than an implicit thread-local object.

The CPU global mutual-inductance LRU remains a documented non-thread-safe facility in this phase. Default hot parallel paths bypass it. Explicit cache use must be serial, caller-owned, thread-local, or otherwise externally synchronized.

## 11. Circuit and Thermal Semantics

The coupled circuit preserves the existing filament model:

- Stage self and inter-stage mutual inductances form the stage block.
- Filament self and fixed inter-filament mutual inductances form the armature block.
- Position-dependent `M_ca` and `dM_ca` populate coupling blocks and motional EMF terms.
- Temperature-dependent filament resistance is derived from the current trial temperature.
- Stage, mutual, active, and trigger masks determine participating slots.
- Inactive slots use the existing identity-row behavior only after the lifecycle contract marks them completed.

Thermal state is adiabatic. For each filament, one step consumes previous-step current and resistance, then updates temperature, specific heat, resistivity, resistance, and Joule energy consistently. CPU thermal mode remains the reference and fallback path. GPU thermal mode keeps the same state and ordering on the device.

## 12. CUDA Execution Architecture

### 12.1 Fixed versus dynamic device data

Upload once per geometry/configuration:

- Driving-coil geometry and turns.
- Armature filament geometry and relative axial positions.
- Stage self-inductance/resistance and inter-stage mutual data.
- Filament self-inductance and inter-filament mutual matrix.
- Filament masses, reference resistances, material identifiers.
- Quadrature nodes and weights.
- Material lookup tables when required.

Keep resident and update per step:

- Armature position and velocity.
- Stage voltages.
- Currents and current derivatives.
- Active, trigger, stage, and mutual masks.
- Position-dependent `M_ca` and `dM_ca`.
- Temperature, resistivity, resistance, Joule energy.
- Force, acceleration, active/finished flags, and compact status.

The host representation is updated only at explicit observation boundaries:

- Final result only.
- Periodic sampled history.
- Every-step debug/validation.
- Explicit caller-requested synchronization.

Compatibility wrappers retain per-step recording by default. Reduced observation is opt-in.

### 12.2 Device time-step pipeline

The complete device pipeline follows this order:

1. Read dynamic position and update separations.
2. Evaluate `M_ca` and `dM_ca` using `Standard`, `Full`, or `Aggressive` mutual policy.
3. Assemble the coupled matrix and RHS on the GPU.
4. Solve current derivatives on the GPU.
5. Reduce force and compute acceleration.
6. Apply Euler current, velocity, and position updates.
7. Update excitation and thermal state according to their selected device contract.
8. Evaluate device-side trigger, active, finished, and boundary flags.
9. Copy only compact status or requested observation data to the host.

Separate kernels are acceptable for the first implementation. Fusion is deferred until profiling shows launch overhead is material.

### 12.3 Matrix and RHS assembly

Device assembly must preserve the current physical matrix semantics, including:

- Fixed stage and filament blocks.
- Dynamic coupling blocks.
- Motional EMF terms from `dM/dx`, velocity, and currents.
- Temperature-dependent resistance.
- Stage and filament masks.
- Identity-row behavior for completed/inactive slots.

The fixed blocks should be stored in a solver-friendly layout during initialization. A first implementation may rebuild the full matrix; later versions may update only dynamic portions if profiling justifies the complexity.

Assembly-only tests compare host and device matrix/RHS before solving for all CUDA precision levels, masks, thermal modes, and inactive batch members.

### 12.4 Solver policy

Solver selection remains dimension- and batch-dependent:

- Small single systems may use CPU Eigen if launch and transfer overhead are lower.
- Medium and large batched systems use device-resident batched LU when stable and beneficial.
- Large single systems use a device dense solver when measured end-to-end time justifies it.
- Matrix, RHS, and solution remain device-resident in the steady-state path.
- Validation mode supports residual checks. Production mode may reduce residual status on-device and copy back only compact status.
- Factorization failure, non-finite output, or residual failure aborts the step and enters documented fallback.

The first solver milestone is device residency and stable execution, not a new mathematical factorization.

### 12.5 Single-simulation path

For batch size one:

- Retain geometry and fixed matrices for the run.
- Choose Eigen for small systems based on measured wall time.
- Choose device dense solve for high-resolution systems.
- Use Direct mode for frequently changing masks or controls.
- Use Graph mode for fixed topology and stable shape.
- Synchronize only at requested observation boundaries.

### 12.6 Batched sweep path

For batch size greater than one:

- Store per-member state in contiguous `[batch][stage][filament]` or `[batch][dimension]` flat arrays.
- Process mutual pairs as a three-dimensional workload.
- Use masks rather than compacting members.
- Advance supported position/time-delay triggers and finished state on the device.
- Keep completed members inactive while other members continue.
- Use compact device-to-host status for stops, graph changes, and user-visible boundaries.
- Keep unsupported user-defined policies on explicit direct CPU control or fallback.

### 12.7 CUDA Graph strategy

Graph rollout is incremental:

1. Retain the existing mutual-only graph as a fallback.
2. Capture assembly, solver, force, and state update for fixed layout.
3. Add thermal and excitation kernels when enabled and contractually supported.
4. Add device-side status reduction and boundary flags.

Graph variants are keyed by:

- Batch capacity and active shape.
- Stage and filament dimensions.
- Mask topology.
- CUDA precision mode.
- Thermal mode.
- Solver mode.
- Fixed launch configuration.

Mask topology changes select or capture a new variant at a declared step boundary. Capture or replay failure falls back without committing a failed step.

## 13. CUDA Precision Policies

### 13.1 Standard

- FP64 geometry and mutual integrand.
- No distance cutoff.
- FP64 reduction.
- Reference comparison against CPU mutual values, gradients, matrix/RHS, and trajectories.

### 13.2 Full

- Same FP64 mutual arithmetic as Standard.
- Distance cutoff according to the current GPU policy.
- FP64 reduction and solver inputs.
- Default production mode.

### 13.3 Aggressive

- FP32 mutual integrand and elliptic evaluation where defined.
- FP64 accumulation/reduction and FP64 state/solver storage unless separately approved.
- Distance cutoff enabled.
- Intended for large parameter sweeps, not reference validation.

No CUDA precision policy may change the circuit equations, force expression, update order, or lifecycle semantics. Tolerances are recorded independently; Aggressive results must not loosen Standard or Full tolerances.

## 14. Backend Planning and Fallback

The host planner resolves requested backend, solver, thermal, precision, determinism, dimensions, and capability snapshot without inspecting timings or creating CUDA resources.

Required rules:

- Explicit `Fallback` is CPU-only and must not create a CUDA context.
- Explicit `Graph` without graph capability resolves to fallback with `CapabilityUnavailable`.
- `Persistent` resolves to safe fallback until dedicated control-stream ownership, bounded shutdown, and recovery are implemented.
- `Direct` is the direct CUDA pipeline only after device selection and initialization succeed.
- `SolverMode::Batched` requires batched-solver capability; otherwise use Eigen with a recorded reason.
- `ThermalMode::Gpu` requires GPU thermal capability; otherwise use CPU thermal with a recorded reason.
- Runtime capability snapshots do not prove device availability; allocation, launch, synchronization, and solver failures remain runtime fallback cases.

On any failure that could produce partial state:

1. Preserve pre-step host and device state.
2. Record category and message in `ExecutionReport`.
3. Lock the failing CUDA mode for the engine instance.
4. Execute the equivalent CPU step from the pre-step snapshot.
5. Keep subsequent steps on the resolved fallback path unless explicitly reset or reconstructed.

Reports must expose backend, solver, thermal mode, fallback reason, and `gpu_executed`. Fallback timing rows are excluded from GPU speedup claims.

## 15. Result and Documentation Contracts

- The first post-step history timestamp is `dt`.
- `summary.total_time` equals the final history timestamp.
- `sampled(every_n)` rejects zero and negative values.
- Active counts are calculated per history point, not from a sticky flag.
- Peak current uses absolute-amplitude semantics where documented.
- Stage and global force summaries use the active circuit state at each history point.
- Stage history distinguishes excitation completion from circuit completion.
- `docs/API.md` and `docs/API_cn.md` remain equivalent.
- `README.md` and `README_cn.md` link only to maintained documents.
- Documentation states the C++20 requirement, actual CPU/CUDA preset behavior, fallback limitations, partial Graph status until complete capture is implemented, and unsupported GPU RK4.

## 16. Verification Strategy

### 16.1 Numerical verification

- Compare filament-level `M` and `dM/dx` with the CPU reference.
- Compare coil-level quadrature at near-field, crossover, and far-field separations.
- Compare host/device matrix and RHS before solving.
- Compare current derivatives, currents, force, position, and velocity step by step.
- Compare temperature, resistivity, resistance, and Joule energy step by step.
- Verify Euler first-step capacitor voltage and thermal state use zero pre-step current.
- Verify Euler and RK4 history timestamps begin at `dt`.
- Verify RK4 excitation and thermal convergence on controlled cases.
- Verify capacitor zero crossing, crowbar, waveform end, trigger, and stage completion event localization.
- Verify excitation-finished stages continue decay and completed stages have zero current.
- Verify all mask, inactive-member, trigger-order, and finished-state combinations.
- Verify Direct, Graph, and fallback equivalence.
- Verify Standard, Full, and Aggressive independently.

### 16.2 API and resource verification

- Invalid component, excitation, simulation, trigger, and sampling inputs throw documented exceptions.
- Negative explicit coil positions are preserved; omitted positions use the default.
- CUDA unconfigured use, transfer mismatch, overflow, null buffers, invalid dimensions, allocation, copy, launch, and synchronization failures are covered.
- Thermal workspace rejects wrong device or material-table identity.
- Unsupported GPU RK4 throws explicitly and never executes Euler.

### 16.3 Build and test verification

- CPU-only configure, build, and tests with CUDA explicitly disabled.
- CUDA configure, build, and tests with actual GPU execution evidence.
- C++20 strict compilation and documented compiler requirements.
- Separate CPU and CUDA presets with cache isolation.
- CTest labels: `fast`, `slow`, `validation`, and `gpu`.
- Explicit test timeouts and GPU resource lock.
- No-GPU tests report skip rather than ordinary success; required GPU CI checks `gpu_executed`.
- `compute-sanitizer` for new device buffers and kernels.
- Failure injection for allocation, launch, graph, solver, and residual errors.
- Allocation reuse checks across steps, reset cycles, and multiple engine instances.

### 16.4 Performance verification

Benchmark at minimum:

- Small single-stage workload.
- Medium multi-stage workload.
- Large high-resolution single-stage workload.
- Thermal single-stage workload.
- Batch sizes 1, 8, 32, 128, and a device-appropriate larger case.
- Direct versus Graph for fixed-shape runs.
- All valid precision policies.

Report separately:

- Total wall time.
- Device kernel time.
- Host assembly time.
- Solver time.
- Thermal time.
- Transfer time.
- Control and synchronization time.
- Graph capture/rebuild count.
- Steps per second and simulations per second.

## 17. Implementation Order and Exit Criteria

### Phase 0: Freeze contracts and baseline

- Freeze representative CPU reference cases for single-stage, multi-stage, thermal, and batch workloads.
- Add matrix/RHS observability.
- Add pre-step/post-step and history timestamp tests.
- Record CPU and CUDA precision tolerances separately.
- Split benchmark categories into kernel, transfer, assembly, solver, thermal, control, and wall time.
- Decide and document canonical force, excitation, thermal, and trigger time layers.

**Exit:** CPU/GPU baseline tests pass, benchmark is reproducible, and time-layer semantics are explicit.

### Phase 1: Correct CPU state and integration

- Introduce complete continuous/discrete state ownership.
- Add excitation snapshots and pure derivative evaluation.
- Correct Euler pre-step excitation and thermal updates.
- Implement coupled CPU RK4 with event-aware transitions.
- Separate excitation completion from circuit/stage completion.
- Explicitly zero current when a stage is removed.

**Exit:** controlled Euler/RK4 convergence and lifecycle regression tests pass; CPU history and summaries match the contract.

### Phase 2: Boundary, resource, and build contracts

- Add shared public input validation.
- Correct coil position sentinel behavior and sampling validation.
- Add CUDA adaptor capacity and error checks.
- Bind quadrature and thermal resources to explicit device context/workspace ownership.
- Unify the short-term CUDA quadrature contract to fixed 9-node execution.
- Raise and document the project language standard as C++20. CMake targets, presets, strict compilation checks, and public documentation must use C++20 consistently.
- Separate CPU/CUDA presets and make tests auditable.
- Synchronize English and Chinese API/README documentation.

**Exit:** invalid-input, resource, preset, documentation, and strict-build tests pass.

### Phase 3: GPU-resident state and dynamic buffers

- Add a device state/workspace owner to `GpuEngine`.
- Upload immutable geometry, fixed matrices, quadrature, and material data once.
- Keep dynamic state, masks, `M_ca`, and `dM_ca` on device.
- Retain direct mutual and state kernels as the first resident pipeline.
- Add explicit observation/synchronization boundaries.

**Exit:** multi-step Standard and Full runs agree with the current reference GPU path without per-step geometry or mutual-result downloads.

### Phase 4: GPU matrix/RHS assembly and solver

- Implement device assembly for the coupled matrix and RHS.
- Add assembly-only host/device comparisons.
- Add device-buffer solver interfaces.
- Remove steady-state host layout conversion and matrix/RHS staging.
- Add device residual status and CPU/Eigen fallback.

**Exit:** solver consumes device-resident inputs, high-resolution single and batch workloads show measured end-to-end benefit, and failure rollback tests pass.

### Phase 5: Complete Graph and batch control

- Extend Graph capture through assembly, solver, force, motion, excitation, and thermal stages where supported.
- Cache variants by shape, masks, precision, thermal, and solver policy.
- Move supported position/time-delay triggers and finished-member reductions toward device control.
- Preserve explicit host boundaries for unsupported policies and user-visible history.

**Exit:** fixed-shape Graph replay reduces host orchestration time and batch throughput scales across the intended batch range without changing physical results.

### Phase 6: Separate feasibility gates

Evaluate two independent follow-up designs:

1. **Structured solver gate:** conditioning, residuals, memory, and measured latency/throughput versus dense CPU/GPU fallback.
2. **GPU RK4 gate:** corrected CPU RK4 stability, complete device representation, excitation snapshots, event transitions, `k1..k4` buffers, solver placement, Graph boundary, and a single-stage prototype plan.

Neither feature starts until its gate passes. GPU RK4 implementation order, if approved, is single-stage, multi-stage, then batch.

## 18. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Changed evaluation order causes numerical drift | Standard/reference mode, matrix/RHS comparisons, and stepwise trajectory tests. |
| RK4 event localization is expensive or unstable | Protected bisection, finite iteration limits, deterministic priority, and controlled convergence tests. |
| Dense solver instability | Residual checks, factorization status, non-finite detection, and Eigen/host fallback. |
| Graph invalidation from dynamic masks | Include topology in graph key and rebuild only at declared boundaries. |
| Small workloads become slower | End-to-end planner threshold and measured Eigen crossover. |
| Batch divergence | Masks, stable logical indices, inactive-member preservation, no implicit compaction. |
| Host history compatibility breaks | Per-step recording remains default; reduced observation is explicit. |
| Thermal ordering mismatch | Same pre-step contract and stepwise CPU/GPU thermal comparisons. |
| Global CPU cache races | Keep cache use out of parallel hot paths; document non-thread-safe explicit cache semantics. |
| Resource ownership crosses devices or engines | Context-owned quadrature, thermal workspace, allocations, and device identity checks. |
| Persistent protocol hazards | Keep safe fallback until stream ownership, shutdown, and recovery are independently verified. |

## 19. Success Criteria

The integrated design is successfully implemented when:

- CPU Euler obeys the documented pre-step contract.
- CPU RK4 integrates the complete continuous state and handles discrete events deterministically.
- Completed stages have zero public current and correct history/lifecycle summaries.
- Public invalid inputs fail at the boundary with useful diagnostics.
- CPU and CUDA resource ownership, transfer sizes, presets, and test execution are auditable.
- Most steady-state CUDA data remains device-resident between observation boundaries.
- Device matrix/RHS assembly and solver execution agree with the CPU/reference path within policy-specific tolerances.
- Standard and Full remain within their existing validation tolerances.
- Aggressive remains within its documented engineering tolerance and is clearly reported as lower precision.
- Complete Graph mode improves fixed-shape runs without changing physical results.
- Batched execution improves simulations per second on representative workloads.
- CPU fallback remains correct, deterministic where previously required, and visible in execution reports.
- CPU-only and CUDA test suites, sanitizer checks, failure injection, and benchmark evidence pass.
- GPU RK4 and structured solver decisions are based on explicit post-implementation feasibility reviews rather than assumptions.

## 20. Open Decisions Reserved for Implementation Planning

The following details remain implementation-plan decisions and must be benchmarked or tested without changing the contracts above:

- Exact device matrix layout and full rebuild versus dynamic-block update.
- CPU Eigen versus device dense-solver crossover threshold for batch size one.
- Maximum batch tile size.
- Reduced-history observation API.
- Separate versus fused device excitation/control kernel.
- Event localization tolerance and iteration limits.
- Whether the global CPU cache is later replaced by per-simulation or thread-local ownership.
- Structured solver formulation and GPU RK4 design, subject to their feasibility gates.
