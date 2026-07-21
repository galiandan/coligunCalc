# Final Integrated Design Implementation Plan

## 执行约束

- 禁止调用 Subagent，单 agent 独立执行。
- 所有任务按本文档顺序在当前工作区执行，不使用并行代理分担实现或审查。
- 每个任务先补回归测试，再修改实现；测试失败必须先确认失败原因，再进入实现步骤。
- 不修改用户或其他 agent 已有的无关工作树变更；提交时只暂存本任务列出的文件。
- 不在 GPU fallback、Graph partial capture 或 GPU RK4 unsupported contract 尚未通过验证前宣称 GPU 已完成全流水线。

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task in the current session. Subagent dispatch is prohibited by the execution constraint above. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将最终设计落实为具有统一状态/时间步契约的 CPU 耦合仿真，并在保持物理结果和 fallback 语义不变的前提下逐步建立 device-resident CUDA 单实例与批量执行流水线。

**Architecture:** 先建立共享的完整状态、激励快照、纯导数评估和离散事件边界；CPU Euler/RK4 使用该契约作为参考实现。随后将同一逻辑状态映射到 GPU，先修正资源所有权和传输边界，再迁移矩阵/RHS 组装、solver、热状态和控制流水线，最后扩展完整 CUDA Graph 和 batch device control。结构化 solver 与 GPU RK4 只在独立可验证的 feasibility gate 通过后立项。

**Tech Stack:** C++20, CUDA, Eigen LDLT, Boost.Math, OpenMP, doctest, CMake presets, CTest, `compute-sanitizer`。

---

## 0. 文件地图与变更边界

### 0.1 新增文件

- Create: `include/coilgun/simulation/excitation_snapshot.hpp`，定义多态激励运行时快照、激励连续推进接口和事件观测结果。
- Create: `include/coilgun/simulation/integration_state.hpp`，定义独立于 single/multi simulator 的连续状态别名、阶段离散状态和可复制的积分工作状态，避免与 `single_stage_sim.hpp` 形成头文件循环。
- Create: `include/coilgun/simulation/derivative_workspace.hpp`，定义单次积分调用拥有的互感、矩阵、RHS、阻值、active-stage scratch、事件观测和导数结果。
- Create: `src/simulation/integration_state.cpp`，实现快照克隆、状态复制、离散事件状态初始化和边界状态转换。
- Create: `tests/test_simulation_contracts.cpp`，覆盖状态所有权、时间层、快照和生命周期契约。
- Create: `tests/test_input_validation.cpp`，覆盖组件、激励、仿真、触发和采样边界校验。
- Create: `tests/test_coupled_integrator.cpp`，覆盖 Euler、RK4、事件定位和 thermal coupling。
- Create: `tests/test_gpu_resource_contracts.cpp`，覆盖 CUDA adaptor/context/thermal workspace 的容量和所有权错误。
- Create: `tests/test_gpu_assembly.cpp`，覆盖 host/device matrix 与 RHS 组装一致性。
- Create: `tests/test_gpu_resident_pipeline.cpp`，覆盖 device allocation reuse、observation boundary 和完整流水线状态。
- Create: `tests/test_gpu_graph_pipeline.cpp`，覆盖完整 Graph variant 的捕获、重建和失败回退。
- Create: `tests/test_fixtures.hpp`，以 `inline` 函数定义 CPU 受控组件、激励和积分场景 factory，供单级/多级回归测试复用。
- Create: `tests/test_t_table_generator.cpp`，覆盖 worker count 在 0、1、2、3 hardware concurrency 下不下溢。

### 0.2 现有 CPU 文件

- Modify: `include/coilgun/simulation/excitation.hpp`，增加快照、纯电压查询、连续推进和事件观测接口。
- Modify: `src/simulation/excitation.cpp`，实现 capacitor、crowbar、waveform 快照及校验。
- Modify: `include/coilgun/simulation/time_stepper.hpp`，保留通用 Euler/RK4 数值算法，增加不含离散事件的 coupled trial API。
- Modify: `include/coilgun/simulation/single_stage_sim.hpp`，移除本地 `SimState` 定义和共享导数 scratch 的隐式所有权，改为包含 `integration_state.hpp` 并接入 `DerivativeWorkspace`。
- Modify: `src/simulation/single_stage_sim.cpp`，实现 pre-step Euler、纯导数、thermal 状态推进、历史时间戳和 reset。
- Modify: `include/coilgun/simulation/multi_stage_sim.hpp`，移除本地 `MultiStageState` 定义，将 `triggered_`/`finished_` 替换为显式 stage lifecycle，包含 `integration_state.hpp`。
- Modify: `src/simulation/multi_stage_sim.cpp`，实现多级 pre-step、事件优先级、stage decay 和完成时清零。
- Modify: `include/coilgun/components/driving_coil.hpp`，将位置参数改为 `std::optional<double>`，保留显式负坐标。
- Modify: `src/components/driving_coil.cpp`，增加构造边界验证并实现 `nullopt` 默认位置。
- Modify: `src/components/armature.cpp`，增加几何、质量、材料和 division 校验。
- Modify: `src/simulation/sim_result.cpp`，增加 `every_n > 0` 校验。
- Modify: `src/simulation/multi_stage_result.cpp`，增加 `every_n > 0` 校验。
- Modify: `docs/NumericalModel.md`，固定 pre-step/post-step、stage lifecycle 和事件顺序。
- Modify: `docs/API.md`、`docs/API_cn.md`，同步新接口、C++20、unsupported GPU RK4 和历史语义。
- Modify: `README.md`、`README_cn.md`，同步构建、测试、Graph partial 状态和文档链接。

### 0.3 现有 CUDA 文件

- Modify: `include/coilgun/simulation/cuda/gpu_adaptor.hpp`、`src/cuda/gpu_adaptor.cu`，增加 configured/capacity/device/pair-count 所有权和严格传输检查。
- Modify: `include/coilgun/simulation/cuda/gpu_execution_context.hpp`、`src/cuda/gpu_execution_context.cu`，统一 device 初始化、stream、constant table 和资源生命周期。
- Modify: `include/coilgun/simulation/cuda/gpu_thermal.hpp`、`src/cuda/gpu_thermal.cu`，增加 thermal workspace key 和显式 context ownership。
- Modify: `include/coilgun/simulation/cuda/gpu_engine.hpp`、`src/cuda/gpu_engine.cu`，逐阶段增加 device state、matrix/RHS、solver、observation 和 fallback rollback。
- Modify: `include/coilgun/simulation/cuda/gpu_state_layout.hpp`、`src/cuda/gpu_state_kernels.cu`，固定 `[B][D]`、`[B][S][F]` 布局并增加 device control buffers。
- Modify: `include/coilgun/simulation/cuda/gpu_solver.hpp`、`src/cuda/gpu_solver.cu`，增加 device-buffer matrix/RHS/solution 接口和 residual status。
- Modify: `include/coilgun/simulation/cuda/gpu_graph.hpp`、`src/cuda/gpu_graph.cu`，从 mutual-only variant 扩展完整固定形状 pipeline。
- Modify: `src/cuda/gpu_mutual_pipeline.cu`，固定九节点表初始化规则，禁止每次 launch 改写 device-global constant symbol。
- Modify: `src/cuda/gpu_single_stage_sim.cu`、`src/cuda/gpu_multi_stage_sim.cu`、`src/cuda/sim_batch.cu`，接入 observation boundary、lifecycle mask 和 compact status。

### 0.4 构建与测试文件

- Modify: `CMakeLists.txt`，统一 C++20、CUDA C++20、CMake policy guard 和测试选项。
- Modify: `src/CMakeLists.txt`，将新增的 CPU integration implementation 加入 `coilgun` target。
- Modify: `CMakePresets.json`，分离 `cpu-debug`、`cpu-release`、`cuda-debug`、`cuda-release` configure/build/test preset，并显式设置 `COILGUN_ENABLE_CUDA`。
- Modify: `tests/CMakeLists.txt`，添加测试 target、`fast/slow/validation/gpu` 标签、timeout 和 GPU `RESOURCE_LOCK`。
- Modify: `tools/generate_t_table.cpp`，修复 `hardware_concurrency()` unsigned 下溢并限制 worker 数。
- Modify: `src/CMakeLists.txt`，将 `simulation/integration_state.cpp` 和测试所需的公共状态实现加入 target。

---

## Task 1: 建立可重复基线与构建契约

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `CMakePresets.json`
- Modify: `src/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Create: `tests/test_simulation_contracts.cpp`
- Create: `tests/test_input_validation.cpp`
- Create: `tests/test_coupled_integrator.cpp`
- Test: `tests/test_simulation_contracts.cpp`
- Test: `tests/test_input_validation.cpp`

- [ ] **Step 1: 保存当前 baseline 结果**

Run:

```sh
cmake --preset ninja-debug
cmake --build --preset ninja-debug
ctest --preset debug --output-on-failure
cmake --preset ninja-cuda-debug
cmake --build --preset ninja-cuda-debug
ctest --preset debug --output-on-failure
```

Expected: 当前工作树的 CPU/CUDA 构建结果被记录；若 `debug` test preset 指向错误的 configure preset，记录该现状，不修改历史输出。

- [ ] **Step 2: 为构建契约添加失败测试入口**

在 `tests/CMakeLists.txt` 中先注册新测试 target，使用明确的 CPU/CUDA preset 构建。测试源文件先只包含以下契约测试：

```cpp
#include <doctest/doctest.h>
#include "coilgun/simulation/sim_result.hpp"
#include <stdexcept>

TEST_CASE("sampling rejects non-positive stride") {
    coilgun::simulation::SimResult result;
    CHECK_THROWS_AS(result.sampled(0), std::invalid_argument);
    CHECK_THROWS_AS(result.sampled(-1), std::invalid_argument);
}
```

Expected: 在尚未加入校验前，测试至少有一个失败；失败必须来自 `sampled()` 没有拒绝非法步长，而不是编译或 test discovery 错误。

CPU 测试 target 注册为：

```cmake
add_coilgun_test(test_simulation_contracts)
add_coilgun_test(test_input_validation)
add_coilgun_test(test_coupled_integrator)
```

CUDA API 测试 target 注册为：

```cmake
add_gpu_test(test_gpu_resource_contracts)
add_gpu_test(test_gpu_assembly)
add_gpu_test(test_gpu_resident_pipeline)
add_gpu_test(test_gpu_graph_pipeline)
```

`test_fixtures.hpp` 在 Task 2 创建；Task 1 只负责注册 target，后续任务再补齐各 `inline` factory 实现。每个测试源直接 `#include "test_fixtures.hpp"`，不需要额外链接 target。

- [ ] **Step 3: 统一语言标准和 preset 名称**

将 `CMAKE_CXX_STANDARD` 与 `CMAKE_CUDA_STANDARD` 设置为 `20`，并在 presets 中显式设置：

```json
"cacheVariables": {
  "COILGUN_ENABLE_CUDA": "OFF",
  "COILGUN_BUILD_TESTS": "ON"
}
```

CUDA preset 设置 `COILGUN_ENABLE_CUDA=ON`。新增 `cpu-debug`、`cpu-release`、`cuda-debug`、`cuda-release` 的 configure/build/test 对应关系；test preset 不再复用含糊的 `debug` 名称。

Expected: CPU configure 不会继承旧 cache 的 CUDA 开关；CUDA configure 明确启用 CUDA；项目和 CUDA target 均以 C++20 编译。

- [ ] **Step 4: 为测试设置标签、超时和 GPU 资源锁**

在 `tests/CMakeLists.txt` 的 helper 中加入：

```cmake
set_tests_properties(${name} PROPERTIES
  LABELS "fast"
  TIMEOUT 120)
```

GPU helper 使用 `LABELS "gpu"`、`TIMEOUT 300` 和 `RESOURCE_LOCK gpu`；数值回归使用 `LABELS "validation"`；端到端长测使用 `LABELS "slow"`。没有 GPU 时测试必须返回 CTest skip code，并由实际 GPU CI 检查 `ExecutionReport::gpu_executed`。

- [ ] **Step 5: 修复采样失败测试并运行 CPU 构建**

在两个 result 实现的入口最先加入：

```cpp
if (every_n <= 0) {
    throw std::invalid_argument("every_n must be positive");
}
```

同时包含 `<stdexcept>`。运行：

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

Expected: 新采样测试 PASS；既有 CPU 测试无回归。

- [ ] **Step 6: 提交构建契约里程碑**

```sh
git add CMakeLists.txt CMakePresets.json tests/CMakeLists.txt \
  src/CMakeLists.txt \
  include/coilgun/simulation/sim_result.hpp \
  include/coilgun/simulation/multi_stage_result.hpp \
  src/simulation/sim_result.cpp src/simulation/multi_stage_result.cpp \
  tests/test_simulation_contracts.cpp tests/test_input_validation.cpp
git commit -m "build: establish explicit CPU and CUDA test contracts"
```

---

## Task 2: 完成公共组件与激励输入验证

**Files:**
- Modify: `include/coilgun/components/driving_coil.hpp`
- Modify: `src/components/driving_coil.cpp`
- Modify: `src/components/armature.cpp`
- Modify: `include/coilgun/simulation/excitation.hpp`
- Modify: `src/simulation/excitation.cpp`
- Modify: `include/coilgun/simulation/trigger_config.hpp`
- Create: `tests/test_fixtures.hpp`
- Test: `tests/test_driving_coil.cpp`
- Test: `tests/test_armature.cpp`
- Test: `tests/test_input_validation.cpp`

- [ ] **Step 1: 写组件非法参数测试**

在 `tests/test_input_validation.cpp` 加入：

```cpp
using coilgun::components::Armature;
using coilgun::components::DrivingCoil;
using coilgun::physics::ALUMINUM;
using coilgun::physics::COPPER;

TEST_CASE("DrivingCoil rejects invalid geometry and electrical inputs") {
    CHECK_THROWS_AS(DrivingCoil(0.03, 0.01, 0.05, 100,
                                COPPER.resistivity_ref, 1e-6, 0.7), std::invalid_argument);
    CHECK_THROWS_AS(DrivingCoil(0.01, 0.03, 0.0, 100,
                                COPPER.resistivity_ref, 1e-6, 0.7), std::invalid_argument);
    CHECK_THROWS_AS(DrivingCoil(0.01, 0.03, 0.05, 0,
                                COPPER.resistivity_ref, 1e-6, 0.7), std::invalid_argument);
    CHECK_THROWS_AS(DrivingCoil(0.01, 0.03, 0.05, 100,
                                COPPER.resistivity_ref, 1e-6, 1.1), std::invalid_argument);
}

TEST_CASE("Armature rejects invalid dimensions and mass") {
    CHECK_THROWS_AS(Armature(0.03, 0.01, 0.08, ALUMINUM.resistivity_ref,
                             ALUMINUM.density, 0.0, 0.1, 5, 2, 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(Armature(0.005, 0.025, 0.08, ALUMINUM.resistivity_ref,
                             ALUMINUM.density, 0.0, 0.0, 5, 2, 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(Armature(0.005, 0.025, 0.08, ALUMINUM.resistivity_ref,
                             ALUMINUM.density, 0.0, 0.1, 0, 2, 0.0),
                    std::invalid_argument);
}
```

Expected: 构造器尚未统一校验时失败。

- [ ] **Step 2: 改造 `DrivingCoil` 的位置参数**

将声明改为：

```cpp
#include <optional>

DrivingCoil(double inner_radius, double outer_radius, double length,
            int turns, double resistivity, double wire_area,
            double fill_factor, std::optional<double> position = std::nullopt,
            bool force_exact_self_inductance = false);
```

实现规则：`nullopt` 使用 `length / 2.0`；显式有限值直接保存，包括负坐标；NaN、正负无穷抛出 `std::invalid_argument`。更新所有构造调用和 API/README 示例，使省略位置和显式位置语义清晰。

- [ ] **Step 3: 在构造器计算前加入有限性和正值校验**

`DrivingCoil` 在初始化列表计算 `nc_`、`R_`、`L_` 前，使用 delegating/private validation factory 或构造函数体先验证：

```cpp
if (!std::isfinite(inner_radius) || inner_radius < 0.0 ||
    !std::isfinite(outer_radius) || outer_radius <= inner_radius ||
    !std::isfinite(length) || length <= 0.0 || turns <= 0 ||
    !std::isfinite(resistivity) || resistivity <= 0.0 ||
    !std::isfinite(wire_area) || wire_area <= 0.0 ||
    !std::isfinite(fill_factor) || fill_factor <= 0.0 || fill_factor > 1.0) {
    throw std::invalid_argument("invalid DrivingCoil parameter");
}
```

不得在校验前执行除法或 `self_inductance()`。

- [ ] **Step 4: 在 `Armature` 构造器前加入校验**

校验 `inner_radius >= 0`、`outer_radius > inner_radius`、有限正 `length/resistivity/material_density/mass`、有限初始 position/velocity、`m_axial > 0`、`n_radial > 0`。校验失败消息必须包含具体参数名，例如 `"Armature mass must be positive"`。

- [ ] **Step 5: 校验激励和触发配置**

为 `CapacitorExcitation`、`CrowbarExcitation` 验证有限正初始电压和电容；`WaveformExcitation` 验证 `func_` 非空；`set_end_time()` 接受有限非负值或正无穷，拒绝 NaN/负无穷。保留 `TriggerConfig` 对 position 的有限值规则和对 time-delay 的非负规则，并为正无穷终止策略添加测试。

- [ ] **Step 6: 运行输入验证测试**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_(driving_coil|armature|input_validation|simulation_contracts)$" --output-on-failure
```

Expected: 所有新增非法输入测试 PASS，负位置测试保留 `-0.25` 而不是替换为默认中心。

`tests/test_fixtures.hpp` 在本任务中提供以下 `inline` factory，所有 factory 返回值按值返回，避免测试引用临时对象：

```cpp
struct SingleStageFixture {
    components::DrivingCoil coil;
    components::Armature armature;
};

SingleStageFixture make_test_coil_and_armature();
components::DrivingCoil make_test_coil();
components::Armature make_thermal_armature();
MultiStageSim<EulerStepper> make_waveform_stage_with_nonzero_current();
MultiStageSim<EulerStepper> make_stage_that_decays_below_threshold();
MultiStageSim<EulerStepper> make_two_stage_position_trigger(double position);
```

Factory 实现必须使用当前 API 可构造的最小几何：single-stage 使用 `m=1,n=1`，multi-stage 使用两个位置不同的 coil；所有 waveform/capacitor 使用有限正参数，termination 测试使用 `max_steps` 防止坏 fixture 无限循环。

- [ ] **Step 7: 提交输入边界里程碑**

```sh
git add include/coilgun/components/driving_coil.hpp \
  include/coilgun/simulation/excitation.hpp \
  include/coilgun/simulation/trigger_config.hpp \
  src/components/driving_coil.cpp src/components/armature.cpp \
  src/simulation/excitation.cpp tests/test_driving_coil.cpp \
  tests/test_armature.cpp tests/test_input_validation.cpp tests/test_fixtures.hpp
git commit -m "fix: validate public physical and excitation inputs"
```

---

## Task 2A: 消除默认 CPU 互感 API 的隐式全局缓存竞争

**Files:**
- Modify: `include/coilgun/physics/mutual_inductance.hpp`
- Modify: `src/physics/mutual_inductance.cpp`
- Test: `tests/test_mutual_inductance.cpp`
- Test: `tests/test_simulation_contracts.cpp`

- [ ] **Step 1: 写默认 overload 的并发语义测试**

先锁定默认 overload 与显式 `use_cache=false` 的数值等价，并在 OpenMP 多线程循环中重复默认调用：

```cpp
TEST_CASE("default filament mutual API bypasses shared cache") {
    const double expected = mutual_inductance_filament(0.02, 0.025, 0.01, false);
    for (int i = 0; i < 100; ++i)
        CHECK(mutual_inductance_filament(0.02, 0.025, 0.01) ==
              doctest::Approx(expected));
}

TEST_CASE("explicit cache remains opt-in") {
    const double uncached = mutual_inductance_gradient_filament(
        0.02, 0.025, -0.01, false);
    const double cached = mutual_inductance_gradient_filament(
        0.02, 0.025, -0.01, true);
    CHECK(cached == doctest::Approx(uncached));
}
```

Expected: 当前无参数 overload 访问 static `m_cache`/`grad_cache`，测试先固定其行为差异和需要修正的共享状态。

- [ ] **Step 2: 让默认 overload 委托到无锁路径**

将以下四个无 `use_cache` overload 改为直接调用对应的 `use_cache=false` overload：

```cpp
return mutual_inductance_filament(radius_a, radius_b, separation, false);
return mutual_inductance_gradient_filament(radius_a, radius_b, separation, false);
return mutual_inductance_coil(/* existing args */, false);
return mutual_inductance_gradient_coil(/* existing args */, false);
```

显式 `use_cache=true` 只允许串行 cold-path 或 caller-owned/thread-local cache。不要在本任务里用一把全局 mutex 掩盖 simulation hot path 的 ownership 问题。

- [ ] **Step 3: 更新线程安全文档**

在 header Doxygen 和 `docs/API.md`/`API_cn.md` 中明确：默认 API 不读写全局 LRU，适合 OpenMP；显式 cache 参数是非线程安全 opt-in，调用方必须提供外部同步或独立 cache。

- [ ] **Step 4: 运行互感和并发测试**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_mutual_inductance|test_simulation_contracts" --output-on-failure
```

Expected: 默认/显式 uncached 数值一致；多线程测试无 sanitizer data race。原有 cold-path cache 测试仍可通过 `use_cache=true` 保留。

- [ ] **Step 5: 提交缓存语义里程碑**

```sh
git add include/coilgun/physics/mutual_inductance.hpp \
  src/physics/mutual_inductance.cpp tests/test_mutual_inductance.cpp \
  tests/test_simulation_contracts.cpp
git commit -m "fix: make mutual-inductance caching explicitly opt-in"
```

---

## Task 3: 建立激励快照、完整积分状态和工作区

**Files:**
- Create: `include/coilgun/simulation/excitation_snapshot.hpp`
- Create: `include/coilgun/simulation/integration_state.hpp`
- Create: `include/coilgun/simulation/derivative_workspace.hpp`
- Create: `src/simulation/integration_state.cpp`
- Modify: `include/coilgun/simulation/excitation.hpp`
- Modify: `src/simulation/excitation.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `tests/test_simulation_contracts.cpp`

- [ ] **Step 1: 写快照所有权测试**

测试必须验证每个 concrete excitation 的全部 mutable runtime 字段可独立保存、修改、恢复：

```cpp
TEST_CASE("excitation snapshots restore capacitor runtime state") {
    coilgun::simulation::CapacitorExcitation source(100.0, 1.0);
    auto snapshot = source.snapshot();
    source.advance(0.25, 20.0);
    REQUIRE(source.capacitor_voltage() == doctest::Approx(95.0));
    source.restore(*snapshot);
    CHECK(source.capacitor_voltage() == doctest::Approx(100.0));
    CHECK_FALSE(source.finished());
}

TEST_CASE("crowbar snapshot includes diode state") {
    coilgun::simulation::CrowbarExcitation source(1.0, 1.0);
    source.advance(2.0, 1.0);
    REQUIRE(source.diode_on());
    auto snapshot = source.snapshot();
    source.reset();
    source.restore(*snapshot);
    CHECK(source.diode_on());
    CHECK(source.voltage() == doctest::Approx(0.0));
}

TEST_CASE("waveform snapshot includes time and completion") {
    coilgun::simulation::WaveformExcitation source([](double t) { return t; });
    source.set_end_time(1.0);
    source.advance(0.25, 0.0);
    auto snapshot = source.snapshot();
    source.advance(1.0, 0.0);
    source.restore(*snapshot);
    CHECK(source.voltage() == doctest::Approx(0.25));
    CHECK_FALSE(source.finished());
}
```

Expected: 先失败于接口不存在。

- [ ] **Step 2: 定义快照接口和派生类型**

在 `excitation_snapshot.hpp` 定义：

```cpp
class ExcitationSnapshot {
public:
    virtual ~ExcitationSnapshot() = default;
    virtual std::unique_ptr<ExcitationSnapshot> clone() const = 0;
};

enum class ExcitationEvent {
    CapacitorZero,
    CrowbarOn,
    WaveformEnd,
    Finished
};

struct ExcitationDerivative {
    double capacitor_voltage_rate = 0.0;
    double waveform_time_rate = 0.0;
};
```

在 `Excitation` 中增加：

```cpp
virtual std::unique_ptr<ExcitationSnapshot> snapshot() const = 0;
virtual void restore(const ExcitationSnapshot& snapshot) = 0;
virtual double voltage(const ExcitationSnapshot& snapshot) const = 0;
virtual ExcitationDerivative continuous_derivative(
    const ExcitationSnapshot& snapshot, double coil_current) const = 0;
virtual void advance_snapshot(ExcitationSnapshot& snapshot,
                              double dt, double coil_current) const = 0;
virtual void advance_snapshot_derivative(ExcitationSnapshot& snapshot,
                                          double dt,
                                          const ExcitationDerivative& derivative) const = 0;
virtual void apply_event(ExcitationSnapshot& snapshot,
                         ExcitationEvent event) const = 0;
```

`advance_snapshot()` 只能改传入的本地快照；不得修改真实 excitation。`apply_event()` 只处理 discrete transition，例如 capacitor clamp、crowbar on 和 waveform finish。

- [ ] **Step 3: 定义 `IntegrationState` 和 stage lifecycle**

在 `integration_state.hpp` 定义：

```cpp
struct StageRuntimeState {
    bool triggered = false;
    bool excitation_finished = false;
    bool circuit_active = false;
    bool stage_completed = false;
    bool crowbar_on = false;
    double trigger_time = 0.0;
    double trigger_position = 0.0;
};

struct ContinuousState {
    Eigen::VectorXd currents;
    double arm_position = 0.0;
    double arm_velocity = 0.0;
    Eigen::VectorXd filament_temperatures;

    ContinuousState& operator+=(const ContinuousState& rhs);
    ContinuousState& operator*=(double scalar);
};

using SimState = ContinuousState;
using MultiStageState = ContinuousState;

struct IntegrationState {
    ContinuousState physical;
    std::vector<std::unique_ptr<ExcitationSnapshot>> excitations;
    std::vector<StageRuntimeState> stages;
};
```

`integration_state.hpp` 只依赖 Eigen、`excitation_snapshot.hpp`、`memory` 和 `vector`，不包含 `single_stage_sim.hpp` 或 `multi_stage_sim.hpp`。提供 `clone_integration_state()`、`restore_integration_state()`、`stage_state()` 查询和 `mark_stage_completed()`。`excitation_finished` 与 `circuit_active` 必须允许同时为 true；只有 `stage_completed` 为 true 时才允许矩阵使用 identity row/column。`mark_stage_completed()` 必须先把对应 coil current 置零。

- [ ] **Step 4: 定义 `DerivativeWorkspace`**

在 `derivative_workspace.hpp` 定义与仿真维度匹配的复用缓冲：

```cpp
struct DerivativeWorkspace {
    Eigen::VectorXd mutual;
    Eigen::VectorXd mutual_gradient;
    Eigen::MatrixXd system_matrix;
    Eigen::VectorXd rhs;
    Eigen::VectorXd resistance;
    std::vector<int> active_stages;

    void resize(std::size_t stages, std::size_t filaments);
};

enum class EventType {
    CapacitorZero,
    CrowbarTransition,
    WaveformEnd,
    ExcitationFinished,
    StageTrigger,
    CurrentDecay,
    StageCompleted
};

struct EventObservation {
    EventType type;
    double value = 0.0;
    double normalized_time = 0.0;
    int stage = -1;
};

struct DerivativeResult {
    ContinuousState physical_derivative;
    ExcitationDerivative excitation_derivative;
    double force = 0.0;
    std::vector<EventObservation> events;
};
```

workspace 必须由 `SingleStageSim`/`MultiStageSim` 实例或一次 integration call 持有；`compute_derivatives()` 不得继续写 `M1_`、`dM1_`、`L_total_`、`RHS_` 等共享 scratch。

- [ ] **Step 5: 实现三种 excitation snapshot**

实现字段：

- `CapacitorSnapshot { double capacitor_voltage; bool finished; }`
- `CrowbarSnapshot { double capacitor_voltage; bool diode_on; bool finished; }`
- `WaveformSnapshot { double time; bool finished; }`

`continuous_derivative()` 对 capacitor 返回 `-I/C`，对 waveform 返回 `time_rate=1`，对 crowbar 在 diode-on 时返回 capacitor rate `0`。所有 voltage query 只读 snapshot。

- [ ] **Step 6: 运行契约测试并检查所有 mutable 字段**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_simulation_contracts$" --output-on-failure
```

Expected: 快照恢复、clone 独立性、stage completion 清零和 workspace resize 测试 PASS。

- [ ] **Step 7: 提交状态契约里程碑**

```sh
git add include/coilgun/simulation/excitation_snapshot.hpp \
  include/coilgun/simulation/integration_state.hpp \
  include/coilgun/simulation/derivative_workspace.hpp \
  include/coilgun/simulation/excitation.hpp \
  src/simulation/integration_state.cpp src/simulation/excitation.cpp \
  src/CMakeLists.txt \
  tests/test_simulation_contracts.cpp
git commit -m "refactor: define coupled integration state ownership"
```

---

## Task 4: 修正 CPU Euler 和单级仿真时间层

**Files:**
- Modify: `include/coilgun/simulation/single_stage_sim.hpp`
- Modify: `src/simulation/single_stage_sim.cpp`
- Modify: `include/coilgun/simulation/time_stepper.hpp`
- Modify: `include/coilgun/simulation/integration_state.hpp`
- Modify: `src/simulation/integration_state.cpp`
- Modify: `tests/test_fixtures.hpp`
- Test: `tests/test_single_stage_sim.cpp`
- Test: `tests/test_coupled_integrator.cpp`

- [ ] **Step 1: 写 Euler pre-step 回归测试**

新增受控测试，初始 coil current 为零、初始 filament current 为零：

```cpp
TEST_CASE("single-stage Euler uses pre-step current for capacitor and heat") {
    auto excitation = std::make_unique<CapacitorExcitation>(100.0, 1.0);
    SingleStageSim<EulerStepper> sim(make_test_coil(), make_thermal_armature(),
                                     std::move(excitation), 1e-4, true);
    const auto& first = sim.step();
    CHECK(first.time == doctest::Approx(1e-4));
    CHECK(first.cap_voltage == doctest::Approx(100.0));
    for (double temperature : first.filament_temperatures)
        CHECK(temperature == doctest::Approx(physics::T_REFERENCE));
}
```

Expected: 当前实现使用 post-step coil current，测试在电流非零案例中失败，且 history timestamp 当前为零时失败。

- [ ] **Step 2: 将导数评估改为纯函数边界**

将私有接口改为：

```cpp
DerivativeResult evaluate_derivatives(
    const SimState& state,
    const ExcitationSnapshot& excitation,
    DerivativeWorkspace& workspace) const;
```

`DerivativeResult` 包含 `SimState physical_derivative`、`ExcitationDerivative excitation_derivative`、`double force` 和事件观测。所有互感、矩阵、RHS、阻值计算写入传入 workspace；不得修改 `state_`、`excitation_` 或成员 scratch。

- [ ] **Step 3: 实现 Euler 的 pre-state 流程**

`SingleStageSim::step()` 按以下顺序实现：

```cpp
const IntegrationState pre = clone_integration_state(integration_state_);
DerivativeWorkspace& workspace = workspace_;
const auto derivative = evaluate_derivatives(pre.physical,
                                             *pre.excitations.front(),
                                             workspace);
IntegrationState post = clone_integration_state(pre);
post.physical += dt_ * derivative.physical_derivative;
post.excitations.front()->advance_snapshot_derivative(
    *post.excitations.front(), dt_, derivative.excitation_derivative);
if (enable_thermal_) {
    update_temperatures_from_pre_state(pre.physical, post.physical,
                                       dt_, workspace);
}
apply_events(post, derivative.events);
integration_state_ = std::move(post);
record_step(integration_state_, (step_count_ + 1) * dt_);
++step_count_;
```

实际实现可将字段拆开，但必须保留同一时间层：电容和 thermal 消耗 `pre.physical.currents`、`pre.physical` 温度和 `R(T_pre)`。

- [ ] **Step 4: 修正 thermal derived resistance**

`R_fil_` 不再作为 RK4 scratch 或隐藏 state。每次 derivative 根据 `pre`/trial temperature 计算 `R(T)`；Euler thermal update 使用 pre-step resistance，提交 post temperature 后再计算下一步 resistance。

- [ ] **Step 5: 修正历史记录和 summary**

`record_step()` 接受明确 `post_time`，首条记录时间为 `(0 + 1) * dt_`。`prepare_summary()` 的 peak current 使用 `std::abs()`；force 使用 post-state active circuit。reset 恢复 excitation snapshot、temperature、resistance、stage state 和 history。

- [ ] **Step 6: 运行单级回归**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_(single_stage_sim|coupled_integrator|integration)$" --output-on-failure
```

Expected: 首步 capacitor/thermal、首条时间戳、reset deterministic、peak amplitude 测试 PASS。

- [ ] **Step 7: 提交单级 Euler 里程碑**

```sh
git add include/coilgun/simulation/time_stepper.hpp \
  include/coilgun/simulation/single_stage_sim.hpp \
  src/simulation/single_stage_sim.cpp tests/test_single_stage_sim.cpp \
  tests/test_coupled_integrator.cpp
git commit -m "fix: make single-stage Euler use pre-step state"
```

---

## Task 5: 修正 CPU 多级生命周期、触发和 Euler

**Files:**
- Modify: `include/coilgun/simulation/multi_stage_sim.hpp`
- Modify: `src/simulation/multi_stage_sim.cpp`
- Modify: `include/coilgun/simulation/multi_stage_result.hpp`
- Test: `tests/test_multi_stage_sim.cpp`
- Test: `tests/test_coupled_integrator.cpp`

- [ ] **Step 1: 写 stage lifecycle 回归测试**

```cpp
TEST_CASE("excitation completion does not remove residual circuit current") {
    auto sim = make_waveform_stage_with_nonzero_current();
    const auto& step = sim.step();
    CHECK(step.coil_currents[0] != doctest::Approx(0.0));
    CHECK(step.cap_voltages[0] == doctest::Approx(0.0));
    CHECK(sim.stage_state(0).excitation_finished);
    CHECK(sim.stage_state(0).circuit_active);
    CHECK_FALSE(sim.stage_state(0).stage_completed);
}

TEST_CASE("completed stage current is explicitly zero") {
    auto sim = make_stage_that_decays_below_threshold();
    while (!sim.stage_state(0).stage_completed) sim.step();
    CHECK(sim.state().currents(0) == doctest::Approx(0.0));
    CHECK(sim.result().history.back().coil_currents[0] == doctest::Approx(0.0));
}

TEST_CASE("trigger position is recorded at the pre-step boundary") {
    auto sim = make_two_stage_position_trigger(0.05);
    while (sim.result().history.empty() || sim.result().history.back().state.time < 1e-3)
        sim.step();
    CHECK(sim.stage_summary(1).trigger_position >= 0.05);
}
```

Expected: 当前 `finished_` 同时表示 excitation/circuit completion，第一项或清零项失败。

- [ ] **Step 2: 替换多级 flags 为 lifecycle state**

将 `triggered_`、`finished_` 的业务判断集中到显式 stage state：

```cpp
std::vector<StageRuntimeState> stage_states_;
const StageRuntimeState& stage_state(std::size_t stage) const;
bool circuit_active(std::size_t stage) const noexcept;
```

矩阵 identity row/column 只对 `stage_completed` 或 `!triggered` 使用；`excitation_finished && circuit_active` 阶段仍参与 current decay、mutual matrix 和 force。

- [ ] **Step 3: 实现多级 pre-step Euler 顺序**

`MultiStageSim::step()` 必须按以下顺序执行：

1. 捕获所有 excitation snapshots、stage modes、physical state 和 pre-step resistance。
2. 用 pre-state 的 stage voltages 和 current 做 derivative/force。
3. 提交 currents、velocity、position。
4. 用 pre-step current 推进每个激励的连续状态。
5. 用 pre-step filament current/resistance 推进 thermal state。
6. 按事件优先级执行 crowbar、waveform finish、stage trigger、current decay 和 stage completion。
7. stage completion 时先清零 current，再设置 identity row/column。
8. 记录 `(step_count + 1) * dt_` 的 post-step history。

- [ ] **Step 4: 修正触发和完成判定**

触发检查使用 pre-step 到 post-step 的 crossing，而不是仅检查 post-state `>= value`。同一 event batch 中 stage index 从小到大处理；time-delay 使用上一 stage 的实际 trigger time。`+infinity` trigger 永不触发，但 finite trigger 在前一 stage 完成后仍保持 eligible。

- [ ] **Step 5: 修正 summary 统计**

逐 history point 判断 active；peak current 使用 `std::abs(step.coil_currents[i])`；`step_count_active` 只统计 `CircuitActive` 时间层；stage force 在 stage completed 后为零；`per_stage` 记录 lifecycle 结束状态。

- [ ] **Step 6: 运行多级测试**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_(multi_stage_sim|coupled_integrator)$" --output-on-failure
```

Expected: 单级/多级等价、crowbar、trigger、finish、current decay、summary 和 history timestamp 全部 PASS。

- [ ] **Step 7: 提交多级生命周期里程碑**

```sh
git add include/coilgun/simulation/multi_stage_sim.hpp \
  include/coilgun/simulation/multi_stage_result.hpp \
  src/simulation/multi_stage_sim.cpp tests/test_multi_stage_sim.cpp \
  tests/test_coupled_integrator.cpp
git commit -m "fix: separate excitation and stage circuit lifecycles"
```

---

## Task 6: 实现 event-aware CPU RK4

**Files:**
- Modify: `include/coilgun/simulation/time_stepper.hpp`
- Modify: `include/coilgun/simulation/single_stage_sim.hpp`
- Modify: `include/coilgun/simulation/multi_stage_sim.hpp`
- Modify: `src/simulation/single_stage_sim.cpp`
- Modify: `src/simulation/multi_stage_sim.cpp`
- Test: `tests/test_coupled_integrator.cpp`
- Test: `tests/test_single_stage_sim.cpp`
- Test: `tests/test_multi_stage_sim.cpp`

- [ ] **Step 1: 写 smooth coupled RK4 收敛测试**

使用固定电压 waveform、关闭 thermal、无 stage trigger 的受控系统，运行 `dt`, `dt/2`, `dt/4`，验证 RK4 误差比约为 16：

```cpp
TEST_CASE("coupled RK4 has fourth-order convergence on smooth excitation") {
    const auto r1 = run_controlled_rk4(1e-4);
    const auto r2 = run_controlled_rk4(5e-5);
    const auto r3 = run_controlled_rk4(2.5e-5);
    const double e1 = std::abs(r1.final_velocity - r3.final_velocity);
    const double e2 = std::abs(r2.final_velocity - r3.final_velocity);
    CHECK(e1 / e2 > 10.0);
}
```

- [ ] **Step 2: 扩展 RK4 API 以支持不可变 trial state**

保留现有普通 ODE API，并新增明确的 coupled overload：

```cpp
template<typename State, typename Derivative, typename Ops>
State advance_coupled(double dt, const State& initial,
                      Derivative&& derivative, Ops&& ops) const;
```

`Ops` 提供 `clone`、`add_scaled`、`weighted_sum` 和 `advance_snapshot`；每次 `k1` 到 `k4` 都从同一个 segment start 的 snapshot 派生，不能复用已被 event mutation 的真实 excitation。

- [ ] **Step 3: 实现单级 RK4 连续 trial**

每个 trial 包含 currents、position、velocity、temperature 和 excitation snapshot。`evaluate_derivatives()` 只读 trial state；resistance 按 trial temperature 派生。连续 RK4 结束后才处理 discrete events。

- [ ] **Step 4: 实现事件 bracket 和 protected bisection**

定义 `EventObservation { EventType type; double value; int stage; }`。对 capacitor zero crossing、waveform end、position/time trigger、current threshold 计算 segment endpoints；跨界时在 `[0, dt]` 上二分，最多 64 次，停止条件为时间宽度 `<= event_tolerance` 或 event function 绝对值 `<= value_tolerance`。失败抛出包含 type/stage/interval 的 `std::runtime_error`。

- [ ] **Step 5: 实现固定事件优先级**

按以下顺序应用：continuous clamp、crowbar on、waveform end、excitation finished、stage trigger、current decay/stage completed。同一 tolerance 内的事件合并；stage trigger 按 stage index 升序；零时长重复 event 计数超过 16 时抛错。

- [ ] **Step 6: 实现剩余时间段积分**

事件点提交 state/mode 后，对 `dt - event_time` 递归或循环执行新的 RK4 segment。每个 macro-step 的 event batch 和 segment 数设置上限，避免坏配置无限循环。

- [ ] **Step 7: 验证 RK4 和事件测试**

```sh
cmake --build --preset cpu-debug
ctest --preset cpu-debug -R "test_(coupled_integrator|single_stage_sim|multi_stage_sim)$" --output-on-failure
```

Expected: smooth convergence、capacitor zero crossing、crowbar、waveform end-time、trigger crossing、event priority、stage completion 和 reset determinism PASS。

- [ ] **Step 8: 提交 CPU RK4 里程碑**

```sh
git add include/coilgun/simulation/time_stepper.hpp \
  include/coilgun/simulation/single_stage_sim.hpp \
  include/coilgun/simulation/multi_stage_sim.hpp \
  src/simulation/single_stage_sim.cpp src/simulation/multi_stage_sim.cpp \
  tests/test_coupled_integrator.cpp tests/test_single_stage_sim.cpp \
  tests/test_multi_stage_sim.cpp
git commit -m "feat: implement event-aware coupled CPU RK4"
```

---

## Task 7: 修复 CUDA 资源、quadrature 和 adaptor 边界

**Files:**
- Modify: `include/coilgun/simulation/cuda/gpu_adaptor.hpp`
- Modify: `src/cuda/gpu_adaptor.cu`
- Modify: `include/coilgun/simulation/cuda/gpu_execution_context.hpp`
- Modify: `src/cuda/gpu_execution_context.cu`
- Modify: `include/coilgun/simulation/cuda/gpu_thermal.hpp`
- Modify: `src/cuda/gpu_thermal.cu`
- Modify: `src/cuda/gpu_mutual_pipeline.cu`
- Test: `tests/test_gpu_resource_contracts.cpp`
- Test: `tests/test_gpu_context_smoke.cpp`

- [ ] **Step 1: 写 adaptor capacity 失败测试**

在 CUDA build 下加入：

```cpp
TEST_CASE("GpuAdaptor rejects unconfigured and mismatched transfers") {
    GpuAdaptor adaptor;
    std::vector<double> out_m;
    std::vector<double> out_dm;
    CHECK_THROWS_AS(adaptor.upload_separation(std::vector<double>(1)),
                    std::invalid_argument);
    CHECK_THROWS_AS(adaptor.download_results(out_m, out_dm, -1),
                    std::invalid_argument);
}

TEST_CASE("GpuAdaptor enforces exact configured pair capacity") {
    auto [coils, armature] = make_gpu_geometry();
    GpuAdaptor adaptor;
    adaptor.setup(coils, armature, 9);
    const std::size_t pairs = coils.size() * armature.total_filaments();
    CHECK_THROWS_AS(adaptor.upload_separation(std::vector<double>(pairs + 1)),
                    std::invalid_argument);
}
```

测试实现不得使用泄漏的 `new`；正式代码中改为局部 `std::vector<double> out_m, out_dm`，上面仅表示 API 断言形状，落地时必须使用 RAII。

- [ ] **Step 2: 增加 adaptor 状态字段和 checked CUDA helper**

为 `GpuAdaptor` 增加 `configured_`、`device_id_`、`single_pair_capacity_`、`batch_pair_capacity_` 和 `checked_copy_*()`。所有 `cudaMalloc/cudaMemcpy/cudaFree/cudaGetLastError/cudaDeviceSynchronize` 调用都经过包含 operation name 的错误转换；size 计算先做 overflow 检查。

- [ ] **Step 3: 固定 CUDA 9-node quadrature contract**

`setup()` 和 pipeline launch 只接受 `n_nodes == 9`，其他值抛出 `std::invalid_argument("CUDA mutual pipeline requires n_nodes == 9")`。`GpuExecutionContext` 在 device 初始化时加载一次 immutable table；`gpu_mutual_pipeline.cu` 不再在普通 launch 或 Graph replay 前重写 constant symbols。

- [ ] **Step 4: 给 thermal workspace 增加 ownership key**

定义：

```cpp
struct ThermalWorkspaceKey {
    int device_id = -1;
    std::size_t table_version = 0;
    double min_temperature = 0.0;
    double max_temperature = 0.0;
    std::size_t table_length = 0;
    bool operator==(const ThermalWorkspaceKey&) const = default;
};
```

workspace reuse 只在 key 完全相同且当前 CUDA context device 相同的情况下允许；释放时恢复 owning device。移除隐式 thread-local workspace 作为跨 device ownership 来源。

- [ ] **Step 5: 运行 CPU-only 和 CUDA resource tests**

```sh
cmake --build --preset cuda-debug
ctest --preset cuda-debug -R "test_gpu_(resource_contracts|context_smoke)$" --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 \
  build/cuda-debug/tests/test_gpu_context_smoke
```

Expected: 传输 mismatch、错误 device、错误 table key 和 CUDA runtime failure 被明确拒绝；sanitizer 无 invalid access。无 GPU 环境时 resource unit tests 仍可测试 host validation，device smoke test 必须 skip 而不是 PASS。

- [ ] **Step 6: 提交 CUDA 资源里程碑**

```sh
git add include/coilgun/simulation/cuda/gpu_adaptor.hpp \
  include/coilgun/simulation/cuda/gpu_execution_context.hpp \
  include/coilgun/simulation/cuda/gpu_thermal.hpp \
  src/cuda/gpu_adaptor.cu src/cuda/gpu_execution_context.cu \
  src/cuda/gpu_thermal.cu src/cuda/gpu_mutual_pipeline.cu \
  tests/test_gpu_resource_contracts.cpp tests/test_gpu_context_smoke.cpp
git commit -m "fix(cuda): enforce resource ownership and transfer contracts"
```

---

## Task 8: 建立 GPU device-resident state 与 device assembly

**Files:**
- Modify: `include/coilgun/simulation/cuda/gpu_state_layout.hpp`
- Modify: `src/cuda/gpu_state_kernels.cu`
- Modify: `include/coilgun/simulation/cuda/gpu_engine.hpp`
- Modify: `src/cuda/gpu_engine.cu`
- Modify: `include/coilgun/simulation/cuda/gpu_solver.hpp`
- Modify: `src/cuda/gpu_solver.cu`
- Test: `tests/test_gpu_assembly.cpp`
- Test: `tests/test_gpu_resident_pipeline.cpp`

- [ ] **Step 1: 写 host/device assembly 对比测试**

使用同一 `GpuEngineGeometry`、同一 pre-step state 和同一 mask，分别调用 host reference assembly 与 device assembly；逐元素检查 matrix/RHS：

```cpp
TEST_CASE("device assembly matches host assembly before solve") {
    auto fixture = make_small_gpu_engine_fixture();
    const auto host = fixture.engine.assemble_reference_for_test();
    const auto device = fixture.engine.assemble_device_for_test();
    REQUIRE(host.matrix.size() == device.matrix.size());
    for (std::size_t i = 0; i < host.matrix.size(); ++i)
        CHECK(device.matrix[i] == doctest::Approx(host.matrix[i]).epsilon(1e-11));
    for (std::size_t i = 0; i < host.rhs.size(); ++i)
        CHECK(device.rhs[i] == doctest::Approx(host.rhs[i]).epsilon(1e-11));
}
```

覆盖 inactive batch、stage mask、mutual mask、thermal resistance、motional EMF 和 all-active matrix。

- [ ] **Step 2: 固定 device SoA layout 和 buffer ownership**

`GpuStateLayout` 固定：

- currents/derivatives/solution: `[batch][D]`
- stage masks/voltages: `[batch][S]`
- mutual and gradient: `[batch][S][F]`
- thermal state: `[batch][F]`

`GpuEngine::Resources` 持有所有 device pointer、capacity 和 owning device；`initial_state_` 是 host rollback snapshot；`state_` 只有在 observation boundary 更新。

- [ ] **Step 3: 上传 immutable geometry 和 fixed blocks 一次**

构造 engine 时上传 coil/filament geometry、fixed `M_cc`、fixed `M_aa`、self inductance、mass、reference resistance、material table 和 quadrature。step loop 禁止重新上传 geometry 或 fixed matrix。增加 allocation count 查询，测试两个 step 和 reset 期间地址不变。

- [ ] **Step 4: 实现 device matrix/RHS assembly kernel**

kernel 逻辑保持与 host reference 一致：

```text
if active[b] == 0: write identity(D), rhs = 0
for each stage s:
    if stage_mask[s] == 0 or trigger_mask[s] == 0: identity row
    else write Lcc, Mcc, stage voltage - R*I, and dynamic Mca
for each filament f: write Laa, Maa, -R(T)*I, and dynamic coupling
add velocity * dM/dx * current motional terms
```

第一版允许完整重建 matrix；不得在 device kernel 中读取 host `M1_`/`RHS_` scratch。

- [ ] **Step 5: 增加 device solver buffer API**

在 `GpuSolver` 增加：

```cpp
SolverStatus solve_device(const DeviceMatrixView& matrix,
                          const DeviceVectorView& rhs,
                          DeviceVectorView solution,
                          DeviceResidualView residual);
```

要求 matrix/RHS/solution 不经过 host row-major conversion。factorization failure、non-finite output 或 residual 超阈值返回失败状态，不提交 state。

- [ ] **Step 6: 接入 resident Euler pipeline**

步骤顺序固定为 mutual、matrix、solver、force、state、thermal、compact-status。用 pre-step device snapshot 或双 buffer 确保失败时可恢复。每一步只下载 compact status；final/periodic/every-step observation 按 policy 下载对应数据。

- [ ] **Step 7: 运行 GPU assembly 和 resident tests**

```sh
cmake --build --preset cuda-debug
ctest --preset cuda-debug -R "test_gpu_(assembly|resident_pipeline|engine_physics)$" --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 \
  build/cuda-debug/tests/test_gpu_resident_pipeline
```

Expected: Standard/Full host/device matrix/RHS 一致；连续 step 不增加 device allocation；fallback rollback 不提交失败 step；实际 GPU 运行报告 `gpu_executed=true`。

- [ ] **Step 8: 提交 resident/assembly 里程碑**

```sh
git add include/coilgun/simulation/cuda/gpu_state_layout.hpp \
  include/coilgun/simulation/cuda/gpu_engine.hpp \
  include/coilgun/simulation/cuda/gpu_solver.hpp \
  src/cuda/gpu_state_kernels.cu src/cuda/gpu_engine.cu src/cuda/gpu_solver.cu \
  tests/test_gpu_assembly.cpp tests/test_gpu_resident_pipeline.cpp
git commit -m "feat(cuda): add resident state and device circuit assembly"
```

---

## Task 9: 完成 CUDA thermal、Graph 和 batch control

**Files:**
- Modify: `include/coilgun/simulation/cuda/gpu_graph.hpp`
- Modify: `src/cuda/gpu_graph.cu`
- Modify: `src/cuda/gpu_engine.cu`
- Modify: `src/cuda/gpu_state_kernels.cu`
- Modify: `src/cuda/sim_batch.cu`
- Modify: `src/cuda/gpu_single_stage_sim.cu`
- Modify: `src/cuda/gpu_multi_stage_sim.cu`
- Test: `tests/test_gpu_graph_pipeline.cpp`
- Test: `tests/test_gpu_sim_batch.cpp`
- Test: `tests/test_gpu_precision.cpp`
- Test: `tests/test_gpu_thermal.cpp`

- [ ] **Step 1: 写 Graph variant key 测试**

验证以下任一变化都不复用旧 variant：batch capacity、layout dimension、stage/mutual mask topology、precision、thermal、solver；只改变 stage voltage 数值但不改变 topology 时允许复用：

```cpp
TEST_CASE("graph variant key separates topology and execution policy") {
    auto engine = make_small_gpu_engine_fixture().engine;
    const auto first = engine.graph_variant();
    engine.set_mutual_stage_mask(mask_without_stage(0));
    engine.select_graph_variant_at_boundary();
    CHECK(engine.graph_variant() != first);
    const auto second = engine.graph_variant();
    engine.set_stage_voltage(0, 12.0);
    engine.select_graph_variant_at_boundary();
    CHECK(engine.graph_variant() == second);
}
```

- [ ] **Step 2: 将 Graph capture 扩展到完整 pipeline**

固定 shape 下 capture：mutual、assembly、solver、force、motion/state、thermal、device status reduction。Graph replay 不读取 capture 外可变 constant symbol；variant key 与 allocation shape 完全匹配。Direct mode 保留给 dynamic boundary 和调试。

- [ ] **Step 3: 实现 Graph capture/replay 失败回退**

在 capture/replay 失败时：恢复 pre-step state、记录 `FallbackReason::RuntimeFailure`、锁定 Graph mode、下一步使用 Direct 或 CPU fallback。测试确认失败 step 的 currents/position/temperature/history 不变化。

- [ ] **Step 4: 实现 device thermal update**

thermal kernel 消耗 pre-step current、pre-step resistance、pre-step temperature，更新 temperature、resistivity、resistance、joule energy；同一 kernel 或明确的 ordered kernels 保证下一次 matrix assembly 读取 post-step resistance。CPU thermal fallback 与 GPU thermal 逐 filament 比较。

- [ ] **Step 5: 迁移 batch control**

对 `[B][S][F]` state 使用 mask，不压缩 member。device control 支持 position/time-delay trigger、excitation progression、quiet-stage completion、finished member status；复杂用户策略保留 host boundary。完成 member 继续占槽但不参与矩阵/force/thermal。

- [ ] **Step 6: 验证 precision 和 backend 等价性**

分别测试 Standard、Full、Aggressive；Standard/Full 使用独立 CPU reference tolerance，Aggressive 只使用其自身工程 tolerance。Graph、Direct、CPU fallback 对相同输入比较最终状态和每步 compact status。

- [ ] **Step 7: 运行 CUDA Graph/batch/thermal 测试**

```sh
cmake --build --preset cuda-debug
ctest --preset cuda-debug -L gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=1 \
  build/cuda-debug/tests/test_gpu_graph_pipeline
```

Expected: fixed-shape Graph 真实 replay；mask 变化触发 variant rebuild；capture failure 不污染 state；batch member 独立完成；GPU thermal 与 CPU thermal 在 tolerance 内一致。

- [ ] **Step 8: 提交完整 GPU pipeline 里程碑**

```sh
git add include/coilgun/simulation/cuda/gpu_graph.hpp \
  src/cuda/gpu_graph.cu src/cuda/gpu_engine.cu src/cuda/gpu_state_kernels.cu \
  src/cuda/sim_batch.cu src/cuda/gpu_single_stage_sim.cu \
  src/cuda/gpu_multi_stage_sim.cu tests/test_gpu_graph_pipeline.cpp \
  tests/test_gpu_sim_batch.cpp tests/test_gpu_precision.cpp tests/test_gpu_thermal.cpp
git commit -m "feat(cuda): complete resident graph and batch control pipeline"
```

---

## Task 10: 修复工具、文档、测试可审计性并做全量验证

**Files:**
- Modify: `tools/generate_t_table.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `docs/NumericalModel.md`
- Modify: `docs/API.md`
- Modify: `docs/API_cn.md`
- Modify: `README.md`
- Modify: `README_cn.md`
- Modify: `tests/CMakeLists.txt`
- Modify: `CMakePresets.json`
- Test: `tests/test_input_validation.cpp`
- Test: `tests/test_coupled_integrator.cpp`

- [ ] **Step 1: 修复 lookup-table generator worker count**

先在 `tests/test_t_table_generator.cpp` 写测试或可重复 helper：当 `hardware_concurrency()` 返回 `0`、`1`、`2`、`3` 时 worker count 不得下溢；实现使用有符号中间变量：

```cpp
const unsigned int hardware = std::thread::hardware_concurrency();
const unsigned int usable = hardware > 4 ? hardware - 4 : 1;
const unsigned int workers = std::min(usable, 32u);
```

将计算抽取为可测试的纯函数，并运行 generator 的 unit test。

在 `CMakeLists.txt` 增加 `COILGUN_BUILD_GENERATOR_TESTS`，并在 `tests/CMakeLists.txt` 注册：

```cmake
add_coilgun_test(test_t_table_generator)
set_tests_properties(test_t_table_generator PROPERTIES LABELS "fast" TIMEOUT 30)
```

- [ ] **Step 2: 同步数值模型文档**

在 `docs/NumericalModel.md` 明确：

- Euler capacitor/thermal 使用 pre-step current/resistance。
- 首条 post-step history 时间为 `dt`。
- excitation finished 不等于 circuit completed。
- stage completed 时 current 先清零再使用 identity row。
- RK4 事件优先级、bisection 保护和 unsupported GPU RK4。
- CPU `Reference/LookupTable/Full` 与 CUDA `Standard/Full/Aggressive` 的不同语义。

- [ ] **Step 3: 同步双语 API/README**

`API.md` 与 `API_cn.md` 必须逐功能同步：`std::optional<double>` position、snapshot interface、stage lifecycle、C++20、preset/test 命令、observation boundary、fallback report、partial Graph 变更和 GPU RK4 未实现。`README.md` 与 `README_cn.md` 删除不存在的 `docs/audit_report.md` 链接，改为 `docs/code_review_final_2026-07-21.md` 和最终设计文档。

- [ ] **Step 4: 修正测试报告语义**

文档不手工写死测试数量；使用 preset 和 CTest label 命令描述测试。CUDA 无设备时明确是 skip；GPU validation CI 必须检查：

```text
ExecutionReport.gpu_executed == true
ExecutionReport.backend != Fallback
```

- [ ] **Step 5: 运行完整 CPU 验证**

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
```

Expected: CPU Debug/Release 均配置、构建和测试通过；所有测试使用 C++20。

- [ ] **Step 6: 运行完整 CUDA 验证**

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug --output-on-failure
ctest --preset cuda-debug -L gpu --output-on-failure
```

Expected: 有 NVIDIA GPU 时 CUDA tests report actual `gpu_executed`；无 GPU 时 GPU tests 显式 skip，CPU tests 仍独立通过。

- [ ] **Step 7: 运行 sanitizer、故障注入和地址复用检查**

```sh
compute-sanitizer --tool memcheck --error-exitcode=1 \
  build/cuda-debug/tests/test_gpu_resident_pipeline
compute-sanitizer --tool racecheck --error-exitcode=1 \
  build/cuda-debug/tests/test_gpu_graph_pipeline
ctest --preset cuda-debug -R "test_gpu_(resource_contracts|paths|solver|graph_pipeline)$" \
  --output-on-failure
```

Expected: 无 invalid access、racecheck error、未捕获 CUDA error 或失败后 state 污染。

- [ ] **Step 8: 运行端到端 benchmark**

构建并运行 benchmark：

```sh
cmake --build --preset cuda-debug --target bench_gpu_engine
./build/cuda-debug/src/cuda/bench_gpu_engine
```

报告必须分开记录 total wall time、kernel、host assembly、solver、thermal、transfer、control/sync、Graph rebuild、steps/s 和 simulations/s；fallback 行不得纳入 GPU speedup。

- [ ] **Step 9: 做最终 diff 和文档一致性检查**

```sh
git diff --check
git status --short
git diff --stat
```

逐项核对：两份 API 文档内容等价；两份 README 链接存在；final design 的每个 success criterion 都有测试或 benchmark evidence；没有声称 GPU RK4、Persistent 或 structured solver 已实现。

- [ ] **Step 10: 提交文档与验证里程碑**

```sh
git add CMakeLists.txt CMakePresets.json tests/CMakeLists.txt \
  tools/generate_t_table.cpp docs/NumericalModel.md docs/API.md docs/API_cn.md \
  README.md README_cn.md tests/test_input_validation.cpp \
  tests/test_coupled_integrator.cpp
git commit -m "docs: synchronize integrated execution contracts"
```

---

## Task 11: 结构化 solver 与 GPU RK4 独立准入评审

**Files:**
- Create: `docs/benchmarks/2026-07-21-integrated-pipeline.md`
- Create: `docs/superpowers/specs/2026-07-21-gpu-rk4-feasibility-review.md`
- Create: `docs/superpowers/specs/2026-07-21-structured-solver-feasibility-review.md`
- Test: `tests/test_gpu_engine_physics.cpp`
- Test: `tests/test_coupled_integrator.cpp`

- [ ] **Step 1: 生成 dense solver baseline 数据**

对小、中、大 single 和 batch workload 记录 dense Eigen、device dense/batched solver 的 wall time、residual、condition indicator、memory footprint 和 precision drift。没有数值稳定性和 end-to-end speedup 证据时不改变 solver 默认策略。

- [ ] **Step 2: 写 structured solver feasibility 文档**

文档必须回答：固定 `L_aa` 能否复用、Schur complement 的 conditioning、residual 预算、GPU/CPU fallback、单实例和 batch crossover、在 Standard/Full/Aggressive 下的误差。结论只能是 `Not approved` 或 `Approved for prototype`，不得直接改 production solver。

- [ ] **Step 3: 写 GPU RK4 feasibility 文档**

文档必须回答：完整 continuous/discrete device state、excitation snapshot 表示、event localization、`k1..k4` buffer ownership、每次 trial 的 matrix assembly/solver placement、Graph capture boundary、single-stage prototype 和 CPU reference comparison。现有 GPU wrapper 继续对 RK4 抛出 `std::logic_error`，不得降级执行 Euler。

- [ ] **Step 4: 仅在 gate 通过后安排原型测试**

若且仅若文档结论为 `Approved for prototype`，才增加 single-stage prototype test；原型必须逐步比较 CPU coupled RK4 的 currents、position、velocity、temperature、excitation state 和 event time。gate 未通过时测试保持 explicit unsupported assertion。

- [ ] **Step 5: 提交独立评审结果**

```sh
git add docs/benchmarks/2026-07-21-integrated-pipeline.md \
  docs/superpowers/specs/2026-07-21-gpu-rk4-feasibility-review.md \
  docs/superpowers/specs/2026-07-21-structured-solver-feasibility-review.md \
  tests/test_gpu_engine_physics.cpp tests/test_coupled_integrator.cpp
git commit -m "docs: record solver and GPU RK4 feasibility gates"
```

---

## 最终验收清单

- [ ] CPU Euler 的 capacitor、thermal、force、motion 和 history 均明确使用 pre-step/post-step 契约。
- [ ] CPU RK4 积分 currents、motion、temperature 和 excitation continuous state；离散事件通过 bracket/bisection 处理。
- [ ] `ExcitationFinished`、`CircuitActive`、`StageCompleted` 不再共用一个布尔语义。
- [ ] Stage completed 时 public current/history current 显式为零。
- [ ] 所有公共物理、激励、仿真、trigger、sampling 输入在 derived calculation 前验证。
- [ ] explicit negative coil position 保留，`nullopt` 才使用默认位置。
- [ ] CPU cache 的非线程安全语义被明确记录，parallel hot path 不使用隐式 global cache。
- [ ] CUDA adaptor 的 configured/capacity/device/pair count 检查和 operation-specific CUDA errors 完成。
- [ ] quadrature table、thermal workspace、device allocation 绑定到明确 context/device 所有权。
- [ ] device-resident state、matrix/RHS assembly、solver、thermal 和 compact control status 完成并可回退。
- [ ] fixed-shape complete Graph replay 与 Direct/fallback 等价。
- [ ] batch 使用 mask 保持 logical index，completed member 不影响其他 member。
- [ ] Standard、Full、Aggressive 分别验证，Aggressive 不放宽 Standard/Full tolerance。
- [ ] CPU/CUDA preset、C++20、CTest labels/timeouts/GPU resource lock 一致。
- [ ] API.md/API_cn.md、README.md/README_cn.md 等价且链接存在。
- [ ] CPU-only、CUDA、sanitizer、failure injection 和 benchmark evidence 全部记录。
- [ ] GPU RK4 与 structured solver 仍受独立 feasibility gate 约束。

## 执行方式

本计划只允许当前 agent 顺序执行。执行前加载 `subagent-driven-development` 不符合本计划的执行约束；应直接按 Task 1 到 Task 11 在当前会话中实现，每完成一个 Task 运行其验证命令并等待结果，再进入下一个 Task。
