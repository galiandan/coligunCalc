# Coupled Integrator and Simulation Contracts Design

**Date:** 2026-07-21
**Status:** Approved for implementation planning
**Scope:** CPU coupled integration correctness, excitation and stage lifecycle semantics, public input validation, selected CUDA resource contracts, test/documentation contracts, and a gated GPU RK4 feasibility review.

## Context

The final code review report (`docs/code_review_final_2026-07-21.md`) identified three highest-priority correctness problems:

1. CPU `RK4Stepper` integrates only currents and motion while excitation, thermal resistance, and stage lifecycle state remain outside the integrated state.
2. CPU Euler advances excitation and thermal state with post-step current instead of the documented pre-step current.
3. Stage completion can replace a circuit row with an identity row while leaving non-zero current frozen in public state and history.

The review also identified boundary and resource-contract problems in the public CPU/CUDA APIs. This design addresses those issues in a staged way. It does not implement GPU RK4 in this scope.

## Goals

- Make CPU Euler match the documented pre-step explicit Euler equations.
- Make CPU RK4 a genuine coupled RK4 for continuous physical state.
- Make discrete events deterministic, localized, and separate from continuous derivatives.
- Make excitation, thermal, resistance, and stage lifecycle state ownership explicit.
- Reject invalid public inputs before derived values or numerical libraries are used.
- Add selected CUDA resource and transfer boundary checks without introducing hidden ownership.
- Make test, preset, and bilingual documentation contracts auditable.
- Establish an evidence-based gate for deciding whether GPU RK4 implementation should begin afterward.

## Non-goals

- Do not implement GPU RK4 in this change set.
- Do not redesign the CPU global filament mutual-inductance cache in this scope. Its lack of thread safety remains an accepted, explicitly tracked risk.
- Do not claim CUDA Graph captures a complete physical step. Existing partial Graph and honest CPU fallback behavior remain until a later GPU pipeline project.
- Do not silently downgrade a requested GPU RK4 wrapper to Euler.

## Design

### 1. Complete simulation state

The CPU runtime state is divided into continuous simulation values, polymorphic excitation runtime snapshots, discrete modes, and immutable configuration. The implementation must choose one owner for each mutable field; capacitor voltage and waveform time must not be duplicated in both the simulation state and an excitation snapshot.

Continuous simulation state includes:

- Driving-coil and armature-filament currents.
- Armature position and velocity.
- Filament temperatures when thermal mode is enabled.

Derived values include temperature-dependent filament resistance. Temperature is the only thermal dynamic state; resistance is calculated as `R(T)` from reference resistance and material properties during each derivative evaluation. A mutable simulation member such as `R_fil_` is not an independent state and must not be used as hidden RK4 scratch.

Discrete state includes, as applicable:

- Whether a stage has been triggered.
- Whether an excitation has finished providing external voltage.
- Whether the stage circuit is still active.
- Whether the stage is completed and can be removed from the circuit.
- Whether a crowbar diode is conducting.

`Excitation` remains a polymorphic abstraction. Its polymorphic `ExcitationSnapshot` is the owner of excitation-specific continuous and discrete runtime state. It gains `snapshot()` and `restore()` operations, plus a read-only derivative/state-evaluation capability needed by RK4. Snapshot data must cover every mutable runtime field of each concrete excitation:

- Capacitor voltage and finished state.
- Crowbar capacitor voltage, diode mode, and finished state.
- Waveform time and finished state.

Immutable excitation configuration remains owned by the concrete excitation object. A future excitation type must include all of its mutable state in its snapshot implementation.

### 2. Pure derivative evaluation and workspace ownership

`compute_derivatives()` is replaced or refactored into a pure evaluation boundary. It receives a complete state, immutable geometry/configuration, and an explicit `DerivativeWorkspace`.

The derivative evaluator:

- Does not mutate the simulation object.
- Does not mutate the input state.
- Does not advance or otherwise mutate the real excitation objects.
- Does not write shared `M1`, `dM1`, matrix, RHS, resistance, or active-stage scratch members.
- Uses a workspace owned by the current integration call or simulation instance.
- Returns continuous derivatives together with event observations needed by the integrator.

The workspace contains reusable mutual-inductance arrays, system matrix, RHS, active-stage indices, and other temporary numerical storage. One workspace may be reused across sequential RK4 evaluations, but it is not concurrently reusable by multiple integration calls.

### 3. Euler contract

At the beginning of every Euler macro-step, the integrator captures the complete pre-step state and excitation snapshots. All updates use the same pre-step time layer:

```text
I_n = I_(n-1) + dt * dI(pre_state)
v_n = v_(n-1) + dt * dv(pre_state)
z_n = z_(n-1) + dt * dz(pre_state)
U_n = U_(n-1) - dt * I_(n-1) / C
T_n = T_(n-1) + dt * I_(n-1)^2 * R(T_(n-1))
      / (mass * cp(T_(n-1)))
```

The post-step state is committed once, then events are applied according to the event contract, and the post-step history record is generated. No excitation or thermal update may consume the newly committed current for the current macro-step.

### 4. Event-aware RK4 contract

RK4 integrates the complete continuous state using four derivative evaluations. Each trial state uses excitation snapshots and its corresponding thermal-derived resistance. Discrete transitions are not treated as ordinary continuous derivatives.

Events include:

- Capacitor voltage crossing zero.
- Waveform reaching its end time.
- Position or time-delay stage trigger crossing.
- Current crossing the decay threshold.
- Excitation completion and stage completion.

For each integration segment, the integrator first performs a trial RK4 step under the current discrete mode. If an event function crosses a boundary, it brackets the earliest event and locates it with protected bisection. Each bisection trial is recomputed from the segment start, avoiding dependence on mutated intermediate state.

At the event point, the integrator applies the transition and continues over the remaining time. Events whose times differ by no more than the configured event tolerance are processed as one batch. The fixed priority is:

1. Continuous boundary clamp, such as `U_C = 0`.
2. Crowbar conduction transition.
3. Waveform end-time transition.
4. Excitation finished transition.
5. Stage trigger.
6. Current decay and stage completion.

Stage triggers in one event batch are processed by increasing stage index. Event processing has a finite batch/iteration limit and detects zero-duration repeat events. Failed localization throws an exception containing event type, stage, and time interval rather than silently continuing.

### 5. Stage lifecycle

Stage lifecycle state is explicit and does not overload one `finished` flag:

- `triggered`: the stage is activated.
- `excitation_finished`: the source no longer supplies external excitation.
- `circuit_active`: the stage still participates in circuit and force evaluation.
- `stage_completed`: the stage current has decayed below threshold and the stage can be removed.

An excitation finishing does not immediately remove the circuit. Waveform end, capacitor depletion, and crowbar transition stop or change external excitation, but a non-zero coil current continues through the RL/互感 decay path while `circuit_active` is true. Only at `stage_completed` is the stage current explicitly set to zero and the stage row/column replaced by the inactive identity block.

Stage force remains active while the stage circuit is active. Result history records the lifecycle state needed to distinguish source completion from circuit completion.

### 6. Public input validation

Shared validation helpers are used at public construction and configuration boundaries. Validation occurs before any derived division, allocation, Eigen operation, or CUDA operation.

Required constraints include:

- Finite values for all scalar physical and integration inputs.
- `0 <= inner_radius < outer_radius`.
- Positive length, turns, wire area, density, mass, capacitance, and `dt`.
- `0 < fill_factor <= 1`.
- Positive axial/radial division counts.
- Non-null excitation pointers.
- Non-empty waveform function objects.
- Valid, non-negative waveform end time, with positive infinity reserved for no end time where supported.
- Finite, positive derived resistance, inductance, and filament mass.

All invalid arguments throw `std::invalid_argument` with the parameter name and violated constraint.

`DrivingCoil` changes its optional position contract to `std::optional<double>`: `nullopt` means default center position, while any finite explicit value, including a negative coordinate, is preserved.

### 7. CUDA resource and transfer contracts

The CUDA mutual pipeline uses a fixed 9-node Gauss-Legendre constant table. `GpuExecutionContext` ensures the target device is initialized once. Normal pipeline launches do not rewrite the device-global constant symbols, and `n_nodes == 9` is required for this path. Graph replay therefore does not depend on mutable capture-external quadrature data.

`GpuAdaptor` tracks configured state, device identity, single and batch capacities, and pair counts. Upload and download sizes must exactly match configured capacities:

- Single separation count is `n_stages * n_filaments`.
- Batch separation count is `batch_size * n_stages * n_filaments`.
- Single result download count equals the configured single pair count.

Unconfigured use, negative counts, mismatched sizes, integer overflow, null buffers, and invalid dimensions throw `std::invalid_argument`. Every CUDA allocation, copy, symbol operation, launch, and synchronization call is checked with an operation-specific error message.

`ThermalWorkspace` records owning device, table identity/version, temperature range, and allocation dimensions. Workspace reuse requires all ownership and material-table keys to match. Resource lifetime is attached to the owning CUDA context rather than an implicit thread-local object.

The CPU default filament mutual-inductance API continues to use the existing global LRU cache for this project phase. This is not thread-safe and remains a documented unresolved risk.

### 8. Language, quadrature, and build contracts

The project minimum language standard is raised from C++17 to C++20. CMake targets, presets, documentation, and strict compilation checks use C++20 consistently.

The short-term quadrature contract is unified to the supported 4-node and 9-node set. The CUDA mutual pipeline specifically accepts only 9 nodes. Higher-order support requires a separate expansion of tables, workspaces, kernel indexing, and tests.

CPU and CUDA configure/test presets are separated so that cached options cannot silently select CUDA for a CPU run. CMake policies introduced after the declared minimum are version-guarded or the minimum version is raised consistently.

The lookup-table generator avoids unsigned arithmetic underflow when deriving worker counts from `hardware_concurrency()` and applies a reasonable upper bound.

### 9. Result, test, and documentation contracts

Post-step history timestamps use `(step_count + 1) * dt`; the first recorded post-step state is at `dt`, and summary total time equals the final history timestamp.

`sampled(every_n)` requires `every_n > 0` and rejects zero or negative values. Active counts and peak current summaries are calculated per history point. Peak current uses absolute amplitude semantics. Stage and global force summaries use the active circuit state at each history point rather than a sticky active flag.

CTest tests are labeled `fast`, `slow`, `validation`, and `gpu`, have explicit timeouts, and use a GPU resource lock where required. CPU and CUDA test presets are explicit.

CUDA testing uses two channels. Without a GPU, CUDA tests report an explicit skip state rather than ordinary success. GPU validation CI requires actual `gpu_executed` evidence. CPU-only CI runs the CPU test set independently.

`docs/API.md` and `docs/API_cn.md` remain equivalent. `README.md` and `README_cn.md` links point to existing maintained documents. Documentation states the C++20 requirement, actual preset/test behavior, CPU fallback limitations, partial Graph scope, and the fact that GPU RK4 is not yet implemented.

### 10. GPU RK4 feasibility gate

GPU RK4 is deliberately not implemented in this scope. Existing GPU wrappers retain an explicit unsupported RK4 contract and must not silently execute Euler.

After all preceding work is implemented and verified, a separate feasibility review is mandatory before GPU RK4 work begins. The review must evaluate:

- Stability and numerical agreement of the corrected CPU RK4 reference.
- A complete GPU state representation for continuous and discrete state.
- A feasible device or controlled host strategy for excitation snapshots and event transitions.
- Required `k1..k4` buffer sizes and ownership.
- Solver and matrix-assembly placement for staged evaluations.
- Whether the event contract can be mapped without reintroducing host/device time-layer drift.
- The capture boundary for a complete physical step.
- A single-stage GPU RK4 prototype plan and CPU-reference comparison tests.

Only if this review concludes that the state, event, buffer, solver, and Graph contracts are implementable should GPU RK4 implementation start. The implementation order after approval is single-stage, then multi-stage, then batch.

## Verification strategy

The implementation plan must add regression tests before or alongside each change:

- Euler first-step capacitor voltage and thermal state use zero pre-step current.
- Euler and RK4 history timestamps begin at `dt`.
- RK4 excitation and thermal state converge with expected order on controlled cases.
- Capacitor zero-crossing and crowbar events are localized and deterministic.
- Waveform end-time and stage trigger events occur at the same localized time regardless of event discovery order.
- Excitation-finished stages continue current decay and are removed only after the threshold.
- Completed stages have explicitly zero current.
- Invalid component, excitation, simulation, sampling, and CUDA adaptor inputs throw the documented exception type.
- Negative explicit coil positions are preserved while omitted positions use the default.
- CUDA transfer mismatch, unconfigured use, and CUDA error paths are covered.
- C++20 configure/build and separate CPU/CUDA presets are exercised.
- GPU tests distinguish skipped execution from actual `gpu_executed` execution.

The existing clean CPU and CUDA test results are baseline evidence only; they do not prove the new contracts until the regression tests above pass.

## Risks and accepted decisions

- The global CPU mutual-inductance LRU remains a concurrency risk by explicit decision. It must not be described as thread-safe.
- Event-aware RK4 may cost more near transitions because bisection repeats derivative evaluations.
- C++20 adoption changes the downstream compiler requirement and must be reflected in release documentation.
- Fixed 9-node CUDA constant tables limit quadrature flexibility until a separate resource-ownership redesign.
- GPU RK4 remains gated and is not part of the current implementation plan until the post-fix feasibility review passes.
