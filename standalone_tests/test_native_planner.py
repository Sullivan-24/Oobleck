import contextlib
import io
import json
import unittest
from pathlib import Path

from oobleck.planning.native_planner import (
    LayerExecutionResult,
    NativeOobleckPlanner,
    main,
    reconfigure_like_oobleck,
)


def build_layers(num_layers: int) -> list[LayerExecutionResult]:
    return [
        LayerExecutionResult(
            layer_index=index,
            layer_name=f"layer{index}",
            forward=float(index + 1),
            backward=float(index + 1),
            mem_required=index + 1,
        )
        for index in range(num_layers)
    ]


class NativePlannerTest(unittest.TestCase):
    def test_create_pipeline_templates(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        templates = planner.ensure_templates(1, 3)

        self.assertEqual(sorted(templates), [1, 2, 3])
        for num_stages, template in templates.items():
            self.assertEqual(template.num_stages, num_stages)
            covered = []
            for stage in template.stages:
                covered.extend(range(stage.layers[0], stage.layers[1]))
            self.assertEqual(covered, list(range(6)))

    def test_plan_for_num_nodes(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        result = planner.plan_for_num_nodes(
            num_nodes=4,
            global_num_microbatches=12,
            fault_tolerance_threshold=1,
        )

        self.assertEqual(result.total_num_nodes, 4)
        self.assertEqual(result.total_num_microbatches, 12)
        self.assertGreater(result.iteration_time, 0)

    def test_reconfigure_like_oobleck(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        templates = planner.ensure_templates(1, 4)
        result = reconfigure_like_oobleck(
            pipeline_templates=templates,
            surviving_pipeline_stage_counts=[2, 1],
            global_num_microbatches=9,
            fault_tolerance_threshold=1,
        )

        self.assertEqual(result.total_num_nodes, 3)
        self.assertEqual(result.total_num_microbatches, 9)
        for num_microbatches in result.num_microbatches.values():
            self.assertGreaterEqual(num_microbatches, 1)

    def test_failure_distribution(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        result = planner.analyze_failure_distribution(
            total_num_nodes=4,
            global_num_microbatches=12,
            failure_distribution={0: 0.75, 1: 0.25},
            fault_tolerance_threshold=1,
        )

        self.assertAlmostEqual(result["probability_sum"], 1.0)
        self.assertGreater(result["expected_throughput"], 0)
        self.assertEqual(len(result["cases"]), 2)

    def test_min_num_stages_excludes_shallow_templates(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        result = planner.plan_for_num_nodes(
            num_nodes=4,
            global_num_microbatches=12,
            fault_tolerance_threshold=1,
            min_num_stages=2,
        )

        self.assertTrue(all(template.num_stages >= 2 for template in result.num_instances))

    def test_bind_node_performances_scales_stage_latency(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        template = planner.ensure_templates(2, 2)[2]
        bound = template.bind_node_performances([(0, 1.0), (1, 0.5)])

        self.assertCountEqual(bound.stage_node_indices, (0, 1))
        self.assertCountEqual(bound.stage_node_performances, (1.0, 0.5))
        for stage_index, performance in enumerate(bound.stage_node_performances):
            expected = template.stages[stage_index].latency() / performance
            self.assertEqual(bound.effective_template.stages[stage_index].latency(), expected)

    def test_plan_for_node_performances(self) -> None:
        planner = NativeOobleckPlanner("test-model", build_layers(6))
        result = planner.plan_for_node_performances(
            node_performances=[1.0, 0.5],
            global_num_microbatches=4,
            fault_tolerance_threshold=1,
            min_num_stages=2,
        )

        self.assertEqual(result.total_num_nodes, 2)
        self.assertEqual(result.total_num_microbatches, 4)
        self.assertEqual(len(result.pipeline_allocations), 1)
        allocation = result.pipeline_allocations[0]
        self.assertCountEqual(allocation.pipeline.stage_node_performances, (1.0, 0.5))
        for stage_index, performance in enumerate(
            allocation.pipeline.stage_node_performances
        ):
            if performance < 1.0:
                self.assertGreater(
                    allocation.pipeline.effective_template.stages[stage_index].latency(),
                    allocation.pipeline.template.stages[stage_index].latency(),
                )

    def test_cli_node_performances_with_failed_nodes(self) -> None:
        profile = {
            "model_name": "test-model",
            "microbatch_size": 1,
            "tp_size": 1,
            "precision": "bf16",
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "layer_name": layer.layer_name,
                    "forward": layer.forward,
                    "backward": layer.backward,
                    "mem_required": layer.mem_required,
                }
                for layer in build_layers(6)
            ],
        }

        profile_path = Path(__file__).resolve().parent / "_tmp_profile.json"
        try:
            profile_path.write_text(json.dumps(profile))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--profile",
                        str(profile_path),
                        "--total-nodes",
                        "4",
                        "--global-num-microbatches",
                        "4",
                        "--min-stages",
                        "2",
                        "--failed-nodes",
                        "1",
                        "--node-performances",
                        "1,0.5,1",
                        "--json",
                    ]
                )
        finally:
            if profile_path.exists():
                profile_path.unlink()

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        payload = json.loads(output[output.find("{") :])
        self.assertEqual(payload["cases"][0]["failed_nodes"], 1)
        self.assertEqual(payload["cases"][0]["alive_nodes"], 3)

    def test_cli_output_json_and_case_summary(self) -> None:
        profile = {
            "model_name": "test-model",
            "microbatch_size": 1,
            "tp_size": 1,
            "precision": "bf16",
            "layers": [
                {
                    "layer_index": layer.layer_index,
                    "layer_name": layer.layer_name,
                    "forward": layer.forward,
                    "backward": layer.backward,
                    "mem_required": layer.mem_required,
                }
                for layer in build_layers(6)
            ],
        }

        profile_path = Path(__file__).resolve().parent / "_tmp_profile.json"
        output_path = Path(__file__).resolve().parent / "_tmp_result.json"
        try:
            profile_path.write_text(json.dumps(profile))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--profile",
                        str(profile_path),
                        "--total-nodes",
                        "4",
                        "--global-num-microbatches",
                        "4",
                        "--failed-nodes",
                        "1",
                        "--output-json",
                        str(output_path),
                        "--json",
                    ]
                )
        finally:
            if profile_path.exists():
                profile_path.unlink()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_path.exists())
        payload = json.loads(output_path.read_text())
        self.assertEqual(payload["cases"][0]["failed_nodes"], 1)
        self.assertEqual(payload["cases"][0]["alive_nodes"], 3)
        self.assertIn("alive_nodes=3 failed_nodes=1 iteration_time=", stdout.getvalue())

        if output_path.exists():
            output_path.unlink()


if __name__ == "__main__":
    unittest.main()
