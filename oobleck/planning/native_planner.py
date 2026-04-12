from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REFERENCE_MICROBATCHES = 128


@dataclass(frozen=True)
class LayerExecutionResult:
    layer_index: int
    layer_name: str
    forward: float
    backward: float
    mem_required: int


@dataclass(frozen=True)
class StageExecutionResult:
    layers: tuple[int, int]
    layer_names: tuple[str, ...]
    forward: float
    backward: float
    mem_required: int

    @staticmethod
    def from_layers(layers: Sequence[LayerExecutionResult]) -> "StageExecutionResult":
        if not layers:
            raise ValueError("StageExecutionResult requires at least one layer.")
        return StageExecutionResult(
            layers=(layers[0].layer_index, layers[-1].layer_index + 1),
            layer_names=tuple(layer.layer_name for layer in layers),
            forward=sum(layer.forward for layer in layers),
            backward=sum(layer.backward for layer in layers),
            mem_required=sum(layer.mem_required for layer in layers),
        )

    def latency(self) -> float:
        return self.forward + self.backward

    def scaled_by_performance(self, performance: float) -> "StageExecutionResult":
        if performance <= 0:
            raise ValueError("Node performance must be greater than 0.")
        return StageExecutionResult(
            layers=self.layers,
            layer_names=self.layer_names,
            forward=self.forward / performance,
            backward=self.backward / performance,
            mem_required=self.mem_required,
        )

    def to_dict(self) -> dict:
        return {
            "layers": [self.layers[0], self.layers[1]],
            "layer_names": list(self.layer_names),
            "forward": self.forward,
            "backward": self.backward,
            "latency": self.latency(),
            "mem_required": self.mem_required,
        }


@dataclass(frozen=True)
class NativePipelineTemplate:
    model_name: str
    stages: tuple[StageExecutionResult, ...]
    t1: float
    t2: float
    t3: float
    kstar: int

    @staticmethod
    def from_stage(
        model_name: str, stage: StageExecutionResult
    ) -> "NativePipelineTemplate":
        latency = stage.latency()
        return NativePipelineTemplate(
            model_name=model_name,
            stages=(stage,),
            t1=latency,
            t2=2.0 * latency,
            t3=latency,
            kstar=0,
        )

    @staticmethod
    def from_stages(
        model_name: str, stages: Sequence[StageExecutionResult]
    ) -> "NativePipelineTemplate":
        if not stages:
            raise ValueError("NativePipelineTemplate requires at least one stage.")
        iterator = iter(stages)
        result = NativePipelineTemplate.from_stage(model_name, next(iterator))
        for stage in iterator:
            result = NativePipelineTemplate.merge(
                model_name,
                result,
                NativePipelineTemplate.from_stage(model_name, stage),
            )
        return result

    @staticmethod
    def merge(
        model_name: str,
        left: "NativePipelineTemplate",
        right: "NativePipelineTemplate",
    ) -> "NativePipelineTemplate":
        stages = left.stages + right.stages
        t1 = left.t1 + right.t1

        if left.stages[left.kstar].latency() > right.stages[right.kstar].latency():
            kstar = left.kstar
        else:
            kstar = len(left.stages) + right.kstar

        num_microbatches = 4 * len(stages)
        bottleneck_latency = stages[kstar].latency()
        t2 = (num_microbatches - len(stages) + kstar - 1) * bottleneck_latency

        if kstar == left.kstar:
            t3 = left.t3 + right.t1
        else:
            t3 = right.t3

        return NativePipelineTemplate(
            model_name=model_name,
            stages=stages,
            t1=t1,
            t2=t2,
            t3=t3,
            kstar=kstar,
        )

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def bottleneck_stage(self) -> StageExecutionResult:
        return self.stages[self.kstar]

    @property
    def bottleneck_latency(self) -> float:
        return self.bottleneck_stage.latency()

    @property
    def base_latency(self) -> float:
        return self.t1 + self.t2 + self.t3

    @property
    def linear_bias(self) -> float:
        return self.base_latency - 4 * self.num_stages * self.bottleneck_latency

    @property
    def mem_required(self) -> int:
        return sum(stage.mem_required for stage in self.stages)

    @property
    def modules_per_stage(self) -> tuple[tuple[str, ...], ...]:
        return tuple(stage.layer_names for stage in self.stages)

    def latency_with_mb(self, mb: int) -> float:
        return (
            self.base_latency
            + (mb - 4 * self.num_stages) * self.bottleneck_latency
        )

    def latency(self, mb: int | None = None) -> float:
        return self.base_latency if mb is None else self.latency_with_mb(mb)

    def comparison_key(self) -> tuple[float, int]:
        return (self.latency_with_mb(REFERENCE_MICROBATCHES), self.mem_required)

    def max_microbatches_for_iteration_time(self, iteration_time: float) -> int:
        if self.bottleneck_latency == 0:
            return math.inf if iteration_time >= self.base_latency else -1
        return math.floor(
            (iteration_time - self.linear_bias + 1e-9) / self.bottleneck_latency
        )

    def describe(
        self,
        num_instances: int | None = None,
        num_microbatches: int | None = None,
    ) -> dict:
        result = {
            "num_stages": self.num_stages,
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_slices": [[stage.layers[0], stage.layers[1]] for stage in self.stages],
            "modules_per_stage": [list(stage.layer_names) for stage in self.stages],
            "bottleneck_stage_index": self.kstar,
            "bottleneck_latency": self.bottleneck_latency,
            "base_latency": self.base_latency,
            "reference_latency_128mb": self.latency_with_mb(REFERENCE_MICROBATCHES),
            "mem_required": self.mem_required,
        }
        if num_instances is not None:
            result["num_instances"] = num_instances
        if num_microbatches is not None:
            result["num_microbatches_per_pipeline"] = num_microbatches
            result["iteration_time"] = self.latency(num_microbatches)
        return result

    def bind_node_performances(
        self, node_bindings: Sequence[tuple[int, float]]
    ) -> "BoundPipeline":
        if len(node_bindings) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} node bindings, got {len(node_bindings)}."
            )

        stage_order = sorted(
            range(self.num_stages),
            key=lambda stage_index: self.stages[stage_index].latency(),
            reverse=True,
        )
        node_order = sorted(node_bindings, key=lambda item: item[1], reverse=True)

        scaled_stages = list(self.stages)
        stage_node_indices = [0] * self.num_stages
        stage_node_performances = [0.0] * self.num_stages

        for stage_index, (node_index, performance) in zip(stage_order, node_order):
            scaled_stages[stage_index] = self.stages[stage_index].scaled_by_performance(
                performance
            )
            stage_node_indices[stage_index] = node_index
            stage_node_performances[stage_index] = performance

        return BoundPipeline(
            template=self,
            effective_template=NativePipelineTemplate.from_stages(
                self.model_name, tuple(scaled_stages)
            ),
            stage_node_indices=tuple(stage_node_indices),
            stage_node_performances=tuple(stage_node_performances),
        )


@dataclass(frozen=True)
class BoundPipeline:
    template: NativePipelineTemplate
    effective_template: NativePipelineTemplate
    stage_node_indices: tuple[int, ...]
    stage_node_performances: tuple[float, ...]

    @property
    def num_stages(self) -> int:
        return self.template.num_stages

    @property
    def bottleneck_latency(self) -> float:
        return self.effective_template.bottleneck_latency

    @property
    def base_latency(self) -> float:
        return self.effective_template.base_latency

    def latency(self, mb: int) -> float:
        return self.effective_template.latency(mb)

    def max_microbatches_for_iteration_time(self, iteration_time: float) -> int:
        return self.effective_template.max_microbatches_for_iteration_time(iteration_time)

    def describe(self, num_microbatches: int) -> dict:
        result = self.effective_template.describe(
            num_instances=1,
            num_microbatches=num_microbatches,
        )
        result["base_bottleneck_latency"] = self.template.bottleneck_latency
        result["base_stage_slices"] = [
            [stage.layers[0], stage.layers[1]] for stage in self.template.stages
        ]
        result["node_indices"] = list(self.stage_node_indices)
        result["node_performances"] = list(self.stage_node_performances)
        result["stage_bindings"] = [
            {
                "stage_index": stage_index,
                "node_index": self.stage_node_indices[stage_index],
                "node_performance": self.stage_node_performances[stage_index],
                "base_latency": self.template.stages[stage_index].latency(),
                "effective_latency": self.effective_template.stages[stage_index].latency(),
            }
            for stage_index in range(self.num_stages)
        ]
        return result


class PipelineTemplateGenerator:
    def __init__(
        self, model_name: str, profile_data: Sequence[LayerExecutionResult]
    ) -> None:
        self.model_name = model_name
        self.layer_execution_results = tuple(profile_data)
        self.stage_execution_results: dict[tuple[int, int], StageExecutionResult] = {}
        self.execution_result_cache: dict[
            tuple[int, int, int], NativePipelineTemplate | None
        ] = {}

    def divide_and_conquer(self, max_num_nodes: int) -> None:
        if self.stage_execution_results:
            return

        num_layers = len(self.layer_execution_results)
        if max_num_nodes > num_layers:
            raise ValueError("Invalid number of nodes.")

        for start in range(num_layers):
            for end in range(start + 1, num_layers + 1):
                stage = StageExecutionResult.from_layers(
                    self.layer_execution_results[start:end]
                )
                self.stage_execution_results[(start, end)] = stage
                self.execution_result_cache[(1, start, end)] = (
                    NativePipelineTemplate.from_stage(self.model_name, stage)
                )

        for num_stages in range(2, max_num_nodes + 1):
            for start in range(num_layers):
                for end in range(start + 1, num_layers + 1):
                    key = (num_stages, start, end)
                    if end - start < num_stages:
                        self.execution_result_cache[key] = None
                        continue

                    best_result: NativePipelineTemplate | None = None
                    for split in range(start + 1, end):
                        for left_num_stages in range(1, num_stages):
                            right_num_stages = num_stages - left_num_stages
                            left = self.execution_result_cache.get(
                                (left_num_stages, start, split)
                            )
                            right = self.execution_result_cache.get(
                                (right_num_stages, split, end)
                            )
                            if left is None or right is None:
                                continue
                            candidate = NativePipelineTemplate.merge(
                                self.model_name, left, right
                            )
                            if (
                                best_result is None
                                or candidate.comparison_key()
                                < best_result.comparison_key()
                            ):
                                best_result = candidate

                    self.execution_result_cache[key] = best_result

    def get_pipeline_template(self, num_nodes: int) -> NativePipelineTemplate:
        key = (num_nodes, 0, len(self.layer_execution_results))
        template = self.execution_result_cache.get(key)
        if template is None:
            raise ValueError(f"No pipeline template for {num_nodes} nodes.")
        return template


def create_pipeline_templates(
    model_name: str,
    profile_data: Sequence[LayerExecutionResult],
    num_nodes: Sequence[int],
) -> dict[int, NativePipelineTemplate]:
    if not num_nodes:
        raise ValueError("num_nodes must not be empty.")
    sorted_nodes = sorted(num_nodes)
    generator = PipelineTemplateGenerator(model_name, profile_data)
    generator.divide_and_conquer(sorted_nodes[-1])
    return {
        node_count: generator.get_pipeline_template(node_count)
        for node_count in sorted_nodes
    }


@dataclass
class InstantiationResult:
    num_instances: dict[NativePipelineTemplate, int]
    num_microbatches: dict[NativePipelineTemplate, int]
    iteration_time: float

    @property
    def total_num_nodes(self) -> int:
        return sum(
            template.num_stages * num_instances
            for template, num_instances in self.num_instances.items()
        )

    @property
    def total_num_microbatches(self) -> int:
        return sum(
            self.num_instances[template] * self.num_microbatches[template]
            for template in self.num_instances
        )

    @property
    def throughput(self) -> float:
        if self.iteration_time == 0:
            return math.inf
        return self.total_num_microbatches / self.iteration_time

    def describe(self) -> dict:
        templates = []
        for template in sorted(
            self.num_instances,
            key=lambda item: (
                item.num_stages,
                item.comparison_key(),
            ),
        ):
            templates.append(
                template.describe(
                    num_instances=self.num_instances[template],
                    num_microbatches=self.num_microbatches[template],
                )
            )
        return {
            "iteration_time": self.iteration_time,
            "throughput": self.throughput,
            "total_num_nodes": self.total_num_nodes,
            "total_num_microbatches": self.total_num_microbatches,
            "templates": templates,
        }


@dataclass(frozen=True)
class BoundPipelineAllocation:
    pipeline: BoundPipeline
    num_microbatches: int

    @property
    def iteration_time(self) -> float:
        return self.pipeline.latency(self.num_microbatches)


@dataclass
class BoundInstantiationResult:
    pipeline_allocations: tuple[BoundPipelineAllocation, ...]
    iteration_time: float

    @property
    def total_num_nodes(self) -> int:
        return sum(allocation.pipeline.num_stages for allocation in self.pipeline_allocations)

    @property
    def total_num_microbatches(self) -> int:
        return sum(
            allocation.num_microbatches for allocation in self.pipeline_allocations
        )

    @property
    def throughput(self) -> float:
        if self.iteration_time == 0:
            return math.inf
        return self.total_num_microbatches / self.iteration_time

    def describe(self) -> dict:
        return {
            "iteration_time": self.iteration_time,
            "throughput": self.throughput,
            "total_num_nodes": self.total_num_nodes,
            "total_num_microbatches": self.total_num_microbatches,
            "templates": [
                allocation.pipeline.describe(allocation.num_microbatches)
                for allocation in self.pipeline_allocations
            ],
        }


class PipelineInstantiator:
    def __init__(
        self,
        pipeline_templates: Mapping[int, NativePipelineTemplate],
        global_num_microbatches: int,
        fault_tolerance_threshold: int,
    ) -> None:
        self.pipeline_templates = dict(pipeline_templates)
        self.global_num_microbatches = global_num_microbatches
        self.fault_tolerance_threshold = fault_tolerance_threshold

    def instantiate(self, num_nodes: int) -> InstantiationResult:
        instantiation_options = self._enumerate_instantiation_options(num_nodes)
        best_option: dict[NativePipelineTemplate, int] | None = None
        best_distribution: tuple[float, dict[NativePipelineTemplate, int]] | None = None

        for option in instantiation_options:
            distribution = self.distribute_batch(option)
            if distribution is None:
                continue
            if best_distribution is None or distribution[0] < best_distribution[0]:
                best_option = option
                best_distribution = distribution

        if best_option is None or best_distribution is None:
            raise RuntimeError(
                f"Failed to find optimal batch distribution for {num_nodes} nodes."
            )

        return InstantiationResult(
            num_instances=best_option,
            num_microbatches=best_distribution[1],
            iteration_time=best_distribution[0],
        )

    @staticmethod
    def _template_binding_priority(
        template: NativePipelineTemplate,
    ) -> tuple[float, float, int]:
        return (
            template.bottleneck_latency,
            template.base_latency,
            template.num_stages,
        )

    def _bind_templates_to_nodes(
        self,
        num_pipelines: Mapping[NativePipelineTemplate, int],
        node_performances: Sequence[float],
    ) -> list[BoundPipeline]:
        if len(node_performances) != sum(
            template.num_stages * count for template, count in num_pipelines.items()
        ):
            raise ValueError(
                "The number of node performances must match the total number of nodes "
                "required by the selected templates."
            )

        template_instances = [
            template
            for template, count in num_pipelines.items()
            for _ in range(count)
        ]
        template_instances.sort(
            key=self._template_binding_priority,
            reverse=True,
        )

        remaining_nodes = sorted(
            list(enumerate(node_performances)),
            key=lambda item: item[1],
            reverse=True,
        )

        bound_pipelines: list[BoundPipeline] = []
        for template in template_instances:
            required_nodes = template.num_stages
            assigned_nodes = remaining_nodes[:required_nodes]
            del remaining_nodes[:required_nodes]
            bound_pipelines.append(template.bind_node_performances(assigned_nodes))

        return bound_pipelines

    def instantiate_with_node_performances(
        self, node_performances: Sequence[float]
    ) -> BoundInstantiationResult:
        if any(performance <= 0 for performance in node_performances):
            raise ValueError("All node performances must be greater than 0.")

        instantiation_options = self._enumerate_instantiation_options(
            len(node_performances)
        )
        best_result: BoundInstantiationResult | None = None

        for option in instantiation_options:
            bound_pipelines = self._bind_templates_to_nodes(option, node_performances)
            result = self.distribute_batch_to_bound_pipelines(bound_pipelines)
            if result is None:
                continue
            if best_result is None or result.iteration_time < best_result.iteration_time:
                best_result = result

        if best_result is None:
            raise RuntimeError(
                "Failed to find optimal batch distribution for the provided node "
                "performances."
            )

        return best_result

    def _enumerate_instantiation_options(
        self, num_nodes: int
    ) -> list[dict[NativePipelineTemplate, int]]:
        pipeline_templates = [
            self.pipeline_templates[num_stages]
            for num_stages in sorted(self.pipeline_templates)
        ]
        dp: list[list[list[dict[NativePipelineTemplate, int]]]] = [
            [[] for _ in range(num_nodes + 1)]
            for _ in range(len(pipeline_templates) + 1)
        ]

        for index in range(1, len(pipeline_templates) + 1):
            dp[index][0] = [defaultdict(int)]
            template = pipeline_templates[index - 1]
            for nodes in range(1, num_nodes + 1):
                dp[index][nodes] = [combo.copy() for combo in dp[index - 1][nodes]]
                if template.num_stages <= nodes:
                    for combo in dp[index][nodes - template.num_stages]:
                        new_combo = combo.copy()
                        new_combo[template] += 1
                        dp[index][nodes].append(new_combo)

        if not dp[-1][-1]:
            raise RuntimeError(
                f"Failed to find feasible sets of pipeline templates for {num_nodes} nodes."
            )

        return [
            dict(option)
            for option in dp[-1][-1]
            if sum(option.values()) >= self.fault_tolerance_threshold
        ]

    def distribute_batch(
        self,
        num_pipelines: Mapping[NativePipelineTemplate, int],
        need_all_pipelines_have_batch: bool = False,
    ) -> tuple[float, dict[NativePipelineTemplate, int]] | None:
        if sum(num_pipelines.values()) < self.fault_tolerance_threshold:
            raise AssertionError(
                "The number of pipelines should be greater than or equal to the fault "
                "tolerance threshold."
            )

        lower_bound = 1 if need_all_pipelines_have_batch else 0
        min_microbatches = sum(count * lower_bound for count in num_pipelines.values())
        if min_microbatches > self.global_num_microbatches:
            return None

        candidate_iteration_times: set[float] = set()
        for template, num_instances in num_pipelines.items():
            max_microbatches = self.global_num_microbatches // num_instances
            for microbatches in range(lower_bound, max_microbatches + 1):
                candidate_iteration_times.add(template.latency(microbatches))

        sorted_times = sorted(candidate_iteration_times)
        if not sorted_times:
            return None

        def assignment_for_iteration_time(
            iteration_time: float,
        ) -> dict[NativePipelineTemplate, int] | None:
            templates = list(num_pipelines.keys())
            upper_bounds: dict[NativePipelineTemplate, int] = {}
            for template in templates:
                max_microbatches = template.max_microbatches_for_iteration_time(
                    iteration_time
                )
                if max_microbatches < lower_bound:
                    return None
                upper_bounds[template] = min(
                    max_microbatches,
                    self.global_num_microbatches // num_pipelines[template],
                )

            target = self.global_num_microbatches - sum(
                num_pipelines[template] * lower_bound for template in templates
            )
            if target < 0:
                return None

            states: dict[int, dict[NativePipelineTemplate, int]] = {0: {}}
            for template in templates:
                count = num_pipelines[template]
                max_extra = upper_bounds[template] - lower_bound
                next_states: dict[int, dict[NativePipelineTemplate, int]] = {}
                for current_total, allocation in states.items():
                    for extra in range(max_extra + 1):
                        new_total = current_total + count * extra
                        if new_total > target:
                            break
                        if new_total in next_states:
                            continue
                        new_allocation = allocation.copy()
                        new_allocation[template] = lower_bound + extra
                        next_states[new_total] = new_allocation
                states = next_states
                if not states:
                    return None

            return states.get(target)

        low, high = 0, len(sorted_times) - 1
        best_time: float | None = None
        best_allocation: dict[NativePipelineTemplate, int] | None = None
        while low <= high:
            middle = (low + high) // 2
            iteration_time = sorted_times[middle]
            allocation = assignment_for_iteration_time(iteration_time)
            if allocation is None:
                low = middle + 1
            else:
                best_time = iteration_time
                best_allocation = allocation
                high = middle - 1

        if best_time is None or best_allocation is None:
            return None
        return best_time, best_allocation

    def distribute_batch_to_bound_pipelines(
        self,
        pipelines: Sequence[BoundPipeline],
        need_all_pipelines_have_batch: bool = False,
    ) -> BoundInstantiationResult | None:
        if len(pipelines) < self.fault_tolerance_threshold:
            raise AssertionError(
                "The number of pipelines should be greater than or equal to the fault "
                "tolerance threshold."
            )

        lower_bound = 1 if need_all_pipelines_have_batch else 0
        if len(pipelines) * lower_bound > self.global_num_microbatches:
            return None

        candidate_iteration_times: set[float] = set()
        for pipeline in pipelines:
            for microbatches in range(lower_bound, self.global_num_microbatches + 1):
                candidate_iteration_times.add(pipeline.latency(microbatches))

        sorted_times = sorted(candidate_iteration_times)
        if not sorted_times:
            return None

        def assignment_for_iteration_time(iteration_time: float) -> list[int] | None:
            upper_bounds: list[int] = []
            for pipeline in pipelines:
                max_microbatches = pipeline.max_microbatches_for_iteration_time(
                    iteration_time
                )
                if max_microbatches < lower_bound:
                    return None
                upper_bounds.append(min(max_microbatches, self.global_num_microbatches))

            target = self.global_num_microbatches - len(pipelines) * lower_bound
            states: dict[int, list[int]] = {0: []}
            for upper_bound in upper_bounds:
                next_states: dict[int, list[int]] = {}
                max_extra = upper_bound - lower_bound
                for current_total, allocation in states.items():
                    for extra in range(max_extra + 1):
                        new_total = current_total + extra
                        if new_total > target:
                            break
                        if new_total in next_states:
                            continue
                        next_states[new_total] = allocation + [lower_bound + extra]
                states = next_states
                if not states:
                    return None

            return states.get(target)

        low, high = 0, len(sorted_times) - 1
        best_time: float | None = None
        best_allocation: list[int] | None = None
        while low <= high:
            middle = (low + high) // 2
            iteration_time = sorted_times[middle]
            allocation = assignment_for_iteration_time(iteration_time)
            if allocation is None:
                low = middle + 1
            else:
                best_time = iteration_time
                best_allocation = allocation
                high = middle - 1

        if best_time is None or best_allocation is None:
            return None

        return BoundInstantiationResult(
            pipeline_allocations=tuple(
                BoundPipelineAllocation(pipeline=pipeline, num_microbatches=num_microbatches)
                for pipeline, num_microbatches in zip(pipelines, best_allocation)
            ),
            iteration_time=best_time,
        )


@dataclass
class FailureCaseResult:
    failed_nodes: int
    alive_nodes: int
    plan: InstantiationResult
    probability: float | None = None

    def describe(self) -> dict:
        result = {
            "failed_nodes": self.failed_nodes,
            "alive_nodes": self.alive_nodes,
            "plan": self.plan.describe(),
        }
        if self.probability is not None:
            result["probability"] = self.probability
            result["expected_throughput_contribution"] = (
                self.probability * self.plan.throughput
            )
        return result


class NativeOobleckPlanner:
    def __init__(
        self, model_name: str, profile_data: Sequence[LayerExecutionResult]
    ) -> None:
        self.model_name = model_name
        self.profile_data = tuple(profile_data)
        self.pipeline_templates: dict[int, NativePipelineTemplate] = {}

    @staticmethod
    def _resolve_min_count(
        min_num_nodes: int | None = None,
        min_num_stages: int | None = None,
    ) -> int:
        values = [
            value
            for value in (min_num_nodes, min_num_stages)
            if value is not None
        ]
        return max(values) if values else 1

    def ensure_templates(
        self,
        min_num_nodes: int | None,
        max_num_nodes: int,
        min_num_stages: int | None = None,
    ) -> dict[int, NativePipelineTemplate]:
        min_count = self._resolve_min_count(min_num_nodes, min_num_stages)
        required = list(range(min_count, max_num_nodes + 1))
        if any(node_count not in self.pipeline_templates for node_count in required):
            self.pipeline_templates.update(
                create_pipeline_templates(self.model_name, self.profile_data, required)
            )
        return {
            node_count: self.pipeline_templates[node_count] for node_count in required
        }

    def plan_for_num_nodes(
        self,
        num_nodes: int,
        global_num_microbatches: int,
        fault_tolerance_threshold: int = 1,
        min_num_nodes: int | None = None,
        min_num_stages: int | None = None,
    ) -> InstantiationResult:
        min_count = self._resolve_min_count(min_num_nodes, min_num_stages)
        templates = self.ensure_templates(min_count, num_nodes)
        instantiator = PipelineInstantiator(
            templates, global_num_microbatches, fault_tolerance_threshold
        )
        return instantiator.instantiate(num_nodes)

    def plan_for_node_performances(
        self,
        node_performances: Sequence[float],
        global_num_microbatches: int,
        fault_tolerance_threshold: int = 1,
        min_num_nodes: int | None = None,
        min_num_stages: int | None = None,
    ) -> BoundInstantiationResult:
        min_count = self._resolve_min_count(min_num_nodes, min_num_stages)
        if len(node_performances) < min_count:
            raise ValueError(
                f"len(node_performances)={len(node_performances)} is smaller than "
                f"the minimum required stage/node count {min_count}."
            )

        templates = self.ensure_templates(min_count, len(node_performances))
        instantiator = PipelineInstantiator(
            templates, global_num_microbatches, fault_tolerance_threshold
        )
        return instantiator.instantiate_with_node_performances(node_performances)

    def analyze_failure_cases(
        self,
        total_num_nodes: int,
        global_num_microbatches: int,
        failure_counts: Iterable[int],
        fault_tolerance_threshold: int = 1,
        min_num_nodes: int | None = None,
        min_num_stages: int | None = None,
    ) -> list[FailureCaseResult]:
        min_count = self._resolve_min_count(min_num_nodes, min_num_stages)
        results = []
        for failed_nodes in sorted(set(failure_counts)):
            alive_nodes = total_num_nodes - failed_nodes
            if alive_nodes < min_count:
                raise ValueError(
                    f"alive_nodes={alive_nodes} is smaller than the minimum "
                    f"required stage/node count {min_count}."
                )
            plan = self.plan_for_num_nodes(
                alive_nodes,
                global_num_microbatches=global_num_microbatches,
                fault_tolerance_threshold=fault_tolerance_threshold,
                min_num_nodes=min_count,
            )
            results.append(
                FailureCaseResult(
                    failed_nodes=failed_nodes,
                    alive_nodes=alive_nodes,
                    plan=plan,
                )
            )
        return results

    def analyze_failure_distribution(
        self,
        total_num_nodes: int,
        global_num_microbatches: int,
        failure_distribution: Mapping[int, float],
        fault_tolerance_threshold: int = 1,
        min_num_nodes: int | None = None,
        min_num_stages: int | None = None,
    ) -> dict:
        min_count = self._resolve_min_count(min_num_nodes, min_num_stages)
        cases = []
        expected_throughput = 0.0
        probability_sum = 0.0
        for failed_nodes, probability in sorted(failure_distribution.items()):
            alive_nodes = total_num_nodes - failed_nodes
            if alive_nodes < min_count:
                raise ValueError(
                    f"alive_nodes={alive_nodes} is smaller than the minimum "
                    f"required stage/node count {min_count}."
                )
            plan = self.plan_for_num_nodes(
                alive_nodes,
                global_num_microbatches=global_num_microbatches,
                fault_tolerance_threshold=fault_tolerance_threshold,
                min_num_nodes=min_count,
            )
            expected_throughput += probability * plan.throughput
            probability_sum += probability
            cases.append(
                FailureCaseResult(
                    failed_nodes=failed_nodes,
                    alive_nodes=alive_nodes,
                    plan=plan,
                    probability=probability,
                )
            )

        return {
            "probability_sum": probability_sum,
            "expected_throughput": expected_throughput,
            "cases": [case.describe() for case in cases],
        }


def reconfigure_like_oobleck(
    pipeline_templates: Mapping[int, NativePipelineTemplate],
    surviving_pipeline_stage_counts: Sequence[int],
    global_num_microbatches: int,
    fault_tolerance_threshold: int = 1,
) -> InstantiationResult:
    pipelines = [
        pipeline_templates[num_stages]
        for num_stages in surviving_pipeline_stage_counts
        if num_stages > 0
    ]
    if not pipelines:
        raise ValueError("No surviving pipelines.")

    instantiator = PipelineInstantiator(
        pipeline_templates, global_num_microbatches, fault_tolerance_threshold
    )
    num_instances = dict(Counter(pipelines))
    distribution = instantiator.distribute_batch(
        num_instances, need_all_pipelines_have_batch=True
    )
    if distribution is None:
        raise RuntimeError("Failed to distribute microbatches after reconfiguration.")

    return InstantiationResult(
        num_instances=num_instances,
        num_microbatches=distribution[1],
        iteration_time=distribution[0],
    )


def load_profile(profile_path: str | Path) -> tuple[str | None, list[LayerExecutionResult]]:
    data = json.loads(Path(profile_path).read_text())
    layers = [
        LayerExecutionResult(
            layer_index=layer["layer_index"],
            layer_name=layer["layer_name"],
            forward=layer["forward"],
            backward=layer["backward"],
            mem_required=layer["mem_required"],
        )
        for layer in data["layers"]
    ]
    return data.get("model_name"), layers


def _parse_int_list(value: str | None) -> list[int]:
    if value is None or value == "":
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str | None) -> list[float]:
    if value is None or value == "":
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _render_text_report(payload: dict) -> str:
    lines: list[str] = []
    if "probability_sum" in payload:
        lines.append(f"Probability sum: {payload['probability_sum']:.6f}")
        lines.append(f"Expected throughput: {payload['expected_throughput']:.6f}")
        lines.append("")
        cases = payload["cases"]
    elif "cases" in payload:
        cases = payload["cases"]
    else:
        cases = [payload]

    for case in cases:
        if "failed_nodes" in case:
            header = (
                f"Failure case: failed={case['failed_nodes']} alive={case['alive_nodes']}"
            )
            if "probability" in case:
                header += f" p={case['probability']:.6f}"
            lines.append(header)
            plan = case["plan"]
        else:
            lines.append("Plan")
            plan = case
        lines.append(
            f"  iteration_time={plan['iteration_time']:.6f} throughput={plan['throughput']:.6f}"
        )
        lines.append(
            f"  nodes={plan['total_num_nodes']} microbatches={plan['total_num_microbatches']}"
        )
        for template in plan["templates"]:
            lines.append(
                "  "
                f"template[{template['num_stages']} stages] "
                f"instances={template['num_instances']} "
                f"mb/pipeline={template['num_microbatches_per_pipeline']} "
                f"bottleneck={template['bottleneck_latency']:.6f} "
                f"iteration={template['iteration_time']:.6f}"
            )
            lines.append(f"    stage_slices={template['stage_slices']}")
            if "node_indices" in template:
                lines.append(
                    f"    node_indices={template['node_indices']} "
                    f"node_performances={template['node_performances']}"
                )
        lines.append("")

    return "\n".join(lines).rstrip()


def _iter_case_summaries(payload: dict) -> list[tuple[int, int, float]]:
    if "cases" not in payload:
        return []

    summaries: list[tuple[int, int, float]] = []
    for case in payload["cases"]:
        if "alive_nodes" not in case or "failed_nodes" not in case:
            continue
        plan = case.get("plan", {})
        if "iteration_time" not in plan:
            continue
        summaries.append(
            (
                int(case["alive_nodes"]),
                int(case["failed_nodes"]),
                float(plan["iteration_time"]),
            )
        )
    return summaries


def _render_case_summary(payload: dict) -> str:
    summaries = _iter_case_summaries(payload)
    if not summaries:
        return ""

    lines = []
    for alive_nodes, failed_nodes, iteration_time in summaries:
        lines.append(
            f"alive_nodes={alive_nodes} failed_nodes={failed_nodes} "
            f"iteration_time={int(round(iteration_time))}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pure Python Oobleck planner: generate native pipeline templates and "
            "micro-batch allocations from an Oobleck profile JSON."
        )
    )
    parser.add_argument("--profile", required=True, help="Path to profile JSON.")
    parser.add_argument(
        "--model-name",
        help="Override model_name from the profile JSON.",
    )
    parser.add_argument(
        "--total-nodes",
        type=int,
        required=True,
        help="Total cluster nodes before failures.",
    )
    parser.add_argument(
        "--global-num-microbatches",
        type=int,
        required=True,
        help="Global number of micro-batches.",
    )
    parser.add_argument(
        "--fault-tolerance-threshold",
        type=int,
        default=1,
        help="Minimum number of pipelines to keep.",
    )
    parser.add_argument(
        "--min-num-nodes",
        type=int,
        default=1,
        help="Smallest node count to pre-generate templates for.",
    )
    parser.add_argument(
        "--min-stages",
        type=int,
        help=(
            "Alias of --min-num-nodes. Useful when you want to forbid shallow "
            "pipeline templates such as 1-stage."
        ),
    )
    parser.add_argument(
        "--alive-nodes",
        help="Comma-separated alive-node counts to evaluate.",
    )
    parser.add_argument(
        "--node-performances",
        help=(
            "Comma-separated node performance factors for the current cluster state, "
            "for example 1,1,0.5,0.8. When provided, the planner binds stages to "
            "specific nodes and scales stage latencies by 1/performance."
        ),
    )
    parser.add_argument(
        "--failed-nodes",
        help="Comma-separated failed-node counts to evaluate.",
    )
    parser.add_argument(
        "--failure-distribution",
        help=(
            "JSON file mapping failed-node count to probability, e.g. "
            '{"0": 0.7, "1": 0.2, "2": 0.1}.'
        ),
    )
    parser.add_argument(
        "--surviving-pipeline-stage-counts",
        help=(
            "Comma-separated surviving stage counts per pipeline. "
            "This reproduces Oobleck's current reconfigure path."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the text report.",
    )
    parser.add_argument(
        "--output-json",
        help="Write the full planner result to this JSON file.",
    )
    args = parser.parse_args(argv)

    profile_model_name, profile_data = load_profile(args.profile)
    model_name = args.model_name or profile_model_name or "unknown-model"
    planner = NativeOobleckPlanner(model_name, profile_data)
    node_performances = _parse_float_list(args.node_performances)

    if node_performances:
        if args.alive_nodes or args.failure_distribution or args.surviving_pipeline_stage_counts:
            raise ValueError(
                "--node-performances currently supports planning for a single current "
                "cluster state only. Do not combine it with --alive-nodes, "
                "--failure-distribution, or --surviving-pipeline-stage-counts."
            )
        failed_node_counts = _parse_int_list(args.failed_nodes)
        if len(failed_node_counts) > 1:
            raise ValueError(
                "--node-performances can be combined with only a single value in "
                "--failed-nodes."
            )

        failed_nodes = failed_node_counts[0] if failed_node_counts else 0
        alive_nodes = args.total_nodes - failed_nodes
        if alive_nodes < 0:
            raise ValueError(
                f"failed_nodes={failed_nodes} exceeds total_nodes={args.total_nodes}."
            )
        if len(node_performances) != alive_nodes:
            raise ValueError(
                f"len(node_performances)={len(node_performances)} must match the "
                f"number of alive nodes {alive_nodes} "
                f"(total_nodes={args.total_nodes}, failed_nodes={failed_nodes})."
            )

        plan = planner.plan_for_node_performances(
            node_performances=node_performances,
            global_num_microbatches=args.global_num_microbatches,
            fault_tolerance_threshold=args.fault_tolerance_threshold,
            min_num_nodes=args.min_num_nodes,
            min_num_stages=args.min_stages,
        ).describe()
        if failed_node_counts:
            result = {
                "cases": [
                    {
                        "failed_nodes": failed_nodes,
                        "alive_nodes": alive_nodes,
                        "plan": plan,
                    }
                ]
            }
        else:
            result = plan
    elif args.surviving_pipeline_stage_counts:
        templates = planner.ensure_templates(
            args.min_num_nodes, args.total_nodes, args.min_stages
        )
        result = reconfigure_like_oobleck(
            templates,
            surviving_pipeline_stage_counts=_parse_int_list(
                args.surviving_pipeline_stage_counts
            ),
            global_num_microbatches=args.global_num_microbatches,
            fault_tolerance_threshold=args.fault_tolerance_threshold,
        ).describe()
    elif args.failure_distribution:
        failure_distribution = {
            int(key): float(value)
            for key, value in json.loads(Path(args.failure_distribution).read_text()).items()
        }
        result = planner.analyze_failure_distribution(
            total_num_nodes=args.total_nodes,
            global_num_microbatches=args.global_num_microbatches,
            failure_distribution=failure_distribution,
            fault_tolerance_threshold=args.fault_tolerance_threshold,
            min_num_nodes=args.min_num_nodes,
            min_num_stages=args.min_stages,
        )
    else:
        alive_nodes = _parse_int_list(args.alive_nodes)
        failed_nodes = _parse_int_list(args.failed_nodes)
        if alive_nodes:
            cases = []
            for current_alive_nodes in alive_nodes:
                failed = args.total_nodes - current_alive_nodes
                if failed < 0:
                    raise ValueError(
                        f"alive_nodes={current_alive_nodes} exceeds total_nodes={args.total_nodes}."
                    )
                plan = planner.plan_for_num_nodes(
                    current_alive_nodes,
                    global_num_microbatches=args.global_num_microbatches,
                    fault_tolerance_threshold=args.fault_tolerance_threshold,
                    min_num_nodes=args.min_num_nodes,
                    min_num_stages=args.min_stages,
                )
                cases.append(
                    FailureCaseResult(
                        failed_nodes=failed,
                        alive_nodes=current_alive_nodes,
                        plan=plan,
                    ).describe()
                )
            result = {"cases": cases}
        elif failed_nodes:
            result = {
                "cases": [
                    case.describe()
                    for case in planner.analyze_failure_cases(
                        total_num_nodes=args.total_nodes,
                        global_num_microbatches=args.global_num_microbatches,
                        failure_counts=failed_nodes,
                        fault_tolerance_threshold=args.fault_tolerance_threshold,
                        min_num_nodes=args.min_num_nodes,
                        min_num_stages=args.min_stages,
                    )
                ]
            }
        else:
            result = planner.plan_for_num_nodes(
                args.total_nodes,
                global_num_microbatches=args.global_num_microbatches,
                fault_tolerance_threshold=args.fault_tolerance_threshold,
                min_num_nodes=args.min_num_nodes,
                min_num_stages=args.min_stages,
            ).describe()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    summary = _render_case_summary(result)
    if summary:
        print(summary)

    if args.json:
        if not args.output_json:
            print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_render_text_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
