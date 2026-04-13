#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any


_LAYER_SUFFIX_RE = re.compile(r"(\d+)$")
_DEFAULT_TD_INTERCEPT = 0.00608446728438139
_DEFAULT_TD_LINEAR = 3.30146509
_DEFAULT_TD_QUADRATIC = 0.33615909


@dataclass(frozen=True)
class StageInstance:
    failed_nodes: int
    template_index: int
    instance_index: int
    stage_index: int
    layers: tuple[int, ...]

    @property
    def label(self) -> str:
        if not self.layers:
            return "[]"
        return f"[{self.layers[0]},{self.layers[-1] + 1})"

    @property
    def layer_count(self) -> int:
        return len(self.layers)


def _parse_layer_names(layer_names: list[str]) -> tuple[int, ...]:
    parsed: list[int] = []
    for name in layer_names:
        match = _LAYER_SUFFIX_RE.search(name)
        if not match:
            raise ValueError(f"Could not parse layer index from {name!r}")
        parsed.append(int(match.group(1)))
    parsed.sort()
    return tuple(parsed)


def _stage_layers(template: dict[str, Any]) -> list[tuple[int, ...]]:
    if "stage_slices" in template:
        return [tuple(range(start, end)) for start, end in template["stage_slices"]]
    if "modules_per_stage" in template:
        return [_parse_layer_names(stage) for stage in template["modules_per_stage"]]
    if "stages" in template:
        result: list[tuple[int, ...]] = []
        for stage in template["stages"]:
            if "layers" in stage:
                start, end = stage["layers"]
                result.append(tuple(range(start, end)))
            elif "layer_names" in stage:
                result.append(_parse_layer_names(stage["layer_names"]))
            else:
                raise ValueError("Stage does not contain layers or layer_names.")
        return result
    raise ValueError("Template does not contain stage_slices/modules_per_stage/stages.")


def expand_case(case: dict[str, Any]) -> list[StageInstance]:
    stages: list[StageInstance] = []
    for template_index, template in enumerate(case["plan"]["templates"]):
        stage_layers = _stage_layers(template)
        for instance_index in range(template["num_instances"]):
            for stage_index, layers in enumerate(stage_layers):
                stages.append(
                    StageInstance(
                        failed_nodes=case["failed_nodes"],
                        template_index=template_index,
                        instance_index=instance_index,
                        stage_index=stage_index,
                        layers=layers,
                    )
                )
    return stages


def transfer_cost(old_stage: StageInstance, new_stage: StageInstance) -> int:
    return len(set(new_stage.layers) - set(old_stage.layers))


def solve_min_transfer(
    survivors: list[StageInstance], new_stages: list[StageInstance]
) -> tuple[int, list[tuple[StageInstance, StageInstance, int]]]:
    if len(survivors) != len(new_stages):
        raise ValueError(
            f"Expected same number of stages after removing failures: "
            f"{len(survivors)=} {len(new_stages)=}"
        )

    @lru_cache(maxsize=None)
    def dp(new_index: int, used_mask: int) -> tuple[int, tuple[int, ...]]:
        if new_index == len(new_stages):
            return 0, ()

        best_cost = math.inf
        best_assignment: tuple[int, ...] | None = None
        for survivor_index, survivor in enumerate(survivors):
            if used_mask & (1 << survivor_index):
                continue
            this_cost = transfer_cost(survivor, new_stages[new_index])
            remaining_cost, remaining_assignment = dp(
                new_index + 1, used_mask | (1 << survivor_index)
            )
            total_cost = this_cost + remaining_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = (survivor_index,) + remaining_assignment

        if best_assignment is None:
            raise RuntimeError("Failed to find an assignment.")

        return int(best_cost), best_assignment

    total_cost, assignment = dp(0, 0)
    pairs: list[tuple[StageInstance, StageInstance, int]] = []
    for new_index, survivor_index in enumerate(assignment):
        survivor = survivors[survivor_index]
        new_stage = new_stages[new_index]
        pairs.append((survivor, new_stage, transfer_cost(survivor, new_stage)))
    return total_cost, pairs


def _removed_stage_key(stages: list[StageInstance]) -> tuple[str, ...]:
    return tuple(sorted(stage.label for stage in stages))


def analyze_transition(old_case: dict[str, Any], new_case: dict[str, Any]) -> dict[str, Any]:
    old_stages = expand_case(old_case)
    new_stages = expand_case(new_case)
    num_removed = len(old_stages) - len(new_stages)
    if num_removed <= 0:
        raise ValueError(
            f"Transition {old_case['failed_nodes']}->{new_case['failed_nodes']} "
            f"is not a shrink transition: {len(old_stages)=}, {len(new_stages)=}"
        )

    by_removed_signature: dict[
        tuple[str, ...], list[dict[str, Any]]
    ] = defaultdict(list)

    for removed_indices in combinations(range(len(old_stages)), num_removed):
        removed_index_set = set(removed_indices)
        removed_stages = [old_stages[index] for index in removed_indices]
        survivors = [
            stage for index, stage in enumerate(old_stages) if index not in removed_index_set
        ]
        total_transfer, pairs = solve_min_transfer(survivors, new_stages)
        by_removed_signature[_removed_stage_key(removed_stages)].append(
            {
                "removed_stages": [stage.label for stage in removed_stages],
                "transfer_layers": total_transfer,
                "matching": [
                    {
                        "from": survivor.label,
                        "to": new_stage.label,
                        "missing_layers": missing_layers,
                    }
                    for survivor, new_stage, missing_layers in pairs
                ],
            }
        )

    scenarios: list[dict[str, Any]] = []
    all_costs: list[int] = []
    for removed_signature, entries in sorted(by_removed_signature.items()):
        transfer_layers = sorted({entry["transfer_layers"] for entry in entries})
        exemplar = min(entries, key=lambda entry: entry["transfer_layers"])
        all_costs.extend(entry["transfer_layers"] for entry in entries)
        scenarios.append(
            {
                "removed_stage_multiset": list(removed_signature),
                "num_combinations": len(entries),
                "transfer_layers": transfer_layers,
                "best_matching": exemplar["matching"],
            }
        )

    return {
        "transition": f"{old_case['failed_nodes']}->{new_case['failed_nodes']}",
        "old_failed_nodes": old_case["failed_nodes"],
        "new_failed_nodes": new_case["failed_nodes"],
        "old_stage_count": len(old_stages),
        "new_stage_count": len(new_stages),
        "removed_stage_count": num_removed,
        "old_stage_histogram": dict(Counter(stage.label for stage in old_stages)),
        "new_stage_histogram": dict(Counter(stage.label for stage in new_stages)),
        "min_transfer_layers": min(all_costs),
        "max_transfer_layers": max(all_costs),
        "mean_transfer_layers": sum(all_costs) / len(all_costs),
        "scenarios": scenarios,
    }


def td_of_l(
    layers: float,
    intercept: float = _DEFAULT_TD_INTERCEPT,
    linear: float = _DEFAULT_TD_LINEAR,
    quadratic: float = _DEFAULT_TD_QUADRATIC,
) -> float:
    return (intercept + linear * layers + quadratic * layers * layers)*2


def analyze_file(
    path: Path,
    td_intercept: float = _DEFAULT_TD_INTERCEPT,
    td_linear: float = _DEFAULT_TD_LINEAR,
    td_quadratic: float = _DEFAULT_TD_QUADRATIC,
) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)

    cases = sorted(data["cases"], key=lambda case: case["failed_nodes"])
    transitions: list[dict[str, Any]] = [
        analyze_transition(old_case, new_case)
        for old_case, new_case in zip(cases, cases[1:], strict=False)
    ]
    for transition in transitions:
        transition["td_formula"] = {
            "intercept": td_intercept,
            "linear": td_linear,
            "quadratic": td_quadratic,
        }
        transition["mean_transfer_time_s"] = td_of_l(
            transition["mean_transfer_layers"],
            intercept=td_intercept,
            linear=td_linear,
            quadratic=td_quadratic,
        )
    return {"source": str(path), "transitions": transitions}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a planner result JSON and compute layer transfer counts between "
            "consecutive failure levels. This follows Oobleck's missing-layer semantics: "
            "for each new stage, count only layers that the reused old stage does not hold."
        )
    )
    parser.add_argument("result_json", type=Path, help="Path to result.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full analysis as JSON. Without this flag, print a compact summary.",
    )
    parser.add_argument(
        "--td-intercept",
        type=float,
        default=_DEFAULT_TD_INTERCEPT,
        help="Intercept term in TD(L) = a + bL + cL^2.",
    )
    parser.add_argument(
        "--td-linear",
        type=float,
        default=_DEFAULT_TD_LINEAR,
        help="Linear term in TD(L) = a + bL + cL^2.",
    )
    parser.add_argument(
        "--td-quadratic",
        type=float,
        default=_DEFAULT_TD_QUADRATIC,
        help="Quadratic term in TD(L) = a + bL + cL^2.",
    )
    args = parser.parse_args()

    analysis = analyze_file(
        args.result_json,
        td_intercept=args.td_intercept,
        td_linear=args.td_linear,
        td_quadratic=args.td_quadratic,
    )
    if args.json:
        print(json.dumps(analysis, indent=2))
        return

    print(f"Source: {analysis['source']}")
    for transition in analysis["transitions"]:
        print(
            f"{transition['transition']}: "
            f"min={transition['min_transfer_layers']} "
            f"max={transition['max_transfer_layers']} "
            f"mean={transition['mean_transfer_layers']:.3f} "
            f"mean_time_s={transition['mean_transfer_time_s']:.3f}"
        )
        for scenario in transition["scenarios"]:
            transfer_layers = ",".join(str(value) for value in scenario["transfer_layers"])
            removed = " + ".join(scenario["removed_stage_multiset"])
            print(
                f"  remove {removed}: transfer={transfer_layers} "
                f"(combinations={scenario['num_combinations']})"
            )


if __name__ == "__main__":
    main()
