from __future__ import annotations

import copy
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from multiprocessing.synchronize import Condition
from pathlib import Path
from typing import Any

import click
import grpc
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    TP_AXIS,
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
)
from colossalai.shardformer.layer.parallel_module import ParallelModule
from cornstarch.process_group_mesh import PP_AXIS, HeterogeneousProcessGroupMesh
from cornstarch.shardformer.shard.shardformer import ModelSharder
from google.protobuf import empty_pb2
from loguru import logger
from torch.distributed import distributed_c10d
from torch.optim import Adam
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel

from oobleck.elastic import master_service_pb2, master_service_pb2_grpc
from oobleck.elastic.run import HostInfo, HostStatus, ScriptArguments
from oobleck.engine.configuration_engine import ConfigurationEngine
from oobleck.engine.execution_engine import ExecutionEngine
from oobleck.engine.plugin import OobleckPlugin
from oobleck.planning.profiler import JsonEncoder, LayerExecutionResult, ModelProfiler


_MODEL_LAYERS: list[str] = []
_RECONFIG_LOCK = threading.Lock()
_RECONFIG_METRICS: dict[str, Any] = {}
_RECONFIG_ACTIVE = False
_BASE_GPT2_CONFIG: dict[str, Any] = {
    "vocab_size": 32000,
    "n_positions": 4096,
    "n_ctx": 4096,
    "n_embd": 4096,
    "n_layer": 32,
    "n_head": 32,
    "n_inner": 11008,
    "activation_function": "silu",
    "initializer_range": 0.02,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 2,
    "tie_word_embeddings": False,
    "use_cache": False,
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
}
_CURRENT_GPT2_CONFIG: dict[str, Any] = dict(_BASE_GPT2_CONFIG)


def _load_gpt2_config(model_config_path: Path | None, seq_len: int) -> dict[str, Any]:
    config = dict(_BASE_GPT2_CONFIG)
    if model_config_path is not None:
        payload = json.loads(model_config_path.read_text())
        config.update(payload.get("gpt2_config", {}))

    config["n_positions"] = seq_len
    config["n_ctx"] = seq_len
    return config


def _set_current_gpt2_config(config: dict[str, Any]) -> None:
    global _CURRENT_GPT2_CONFIG
    _CURRENT_GPT2_CONFIG = dict(config)


def _current_gpt2_config() -> dict[str, Any]:
    return dict(_CURRENT_GPT2_CONFIG)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(payload, f)
        f.write("\n")


def _record_event(results_path: Path | None, payload: dict[str, Any]) -> None:
    if results_path is None:
        return
    event = {"ts": time.time(), **payload}
    _append_jsonl(results_path, event)


def _counter_to_stage_counts(pipelines: list[Any]) -> dict[str, int]:
    return {
        str(num_stages): count
        for num_stages, count in sorted(Counter(p.num_stages for p in pipelines).items())
    }


def _summarize_plan(
    pipelines: list[Any], num_microbatches: dict[Any, int]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = Counter(pipelines)
    templates: list[dict[str, Any]] = []
    for template, num_instances in sorted(counts.items(), key=lambda item: item[0].num_stages):
        templates.append(
            {
                "num_stages": template.num_stages,
                "num_instances": num_instances,
                "num_microbatches_per_pipeline": num_microbatches[template],
                "modules_per_stage": [len(stage) for stage in template.modules_per_stage],
            }
        )
    return _counter_to_stage_counts(pipelines), templates


def _reset_reconfig_metrics() -> None:
    global _RECONFIG_METRICS, _RECONFIG_ACTIVE
    with _RECONFIG_LOCK:
        _RECONFIG_METRICS = {}
        _RECONFIG_ACTIVE = False


def _update_reconfig_metrics(**kwargs: Any) -> None:
    with _RECONFIG_LOCK:
        _RECONFIG_METRICS.update(kwargs)


def _get_reconfig_metrics() -> dict[str, Any]:
    with _RECONFIG_LOCK:
        return copy.deepcopy(_RECONFIG_METRICS)


class SyntheticDataset(Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


def _build_collate_fn(vocab_size: int, seq_len: int):
    def collate_fn(batch: list[int]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device="cuda",
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids, device="cuda")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    return collate_fn


def _build_model(seq_len: int, gpt2_config: dict[str, Any]) -> GPT2LMHeadModel:
    global _MODEL_LAYERS

    config = GPT2Config(**gpt2_config)

    old_dtype = torch.get_default_dtype()
    old_device: str | None = None
    original_init_weights = GPT2PreTrainedModel._init_weights
    try:
        GPT2PreTrainedModel._init_weights = lambda self, module: None
        torch.set_default_dtype(torch.bfloat16)
        if torch.cuda.is_available():
            try:
                old_device = torch.get_default_device()
            except Exception:
                old_device = None
            try:
                torch.set_default_device("cuda")
            except Exception:
                old_device = None
        model = GPT2LMHeadModel(config)
    finally:
        GPT2PreTrainedModel._init_weights = original_init_weights
        torch.set_default_dtype(old_dtype)
        if old_device is not None:
            try:
                torch.set_default_device(old_device)
            except Exception:
                pass

    model = model.to(dtype=torch.bfloat16)
    _MODEL_LAYERS = list(model.transformer._modules.keys())
    _MODEL_LAYERS = [
        "transformer.wte",
        "transformer.wpe",
        "transformer.drop",
        *[f"transformer.h.{i}" for i in range(config.n_layer)],
        "transformer.ln_f",
        "lm_head",
    ]
    return model


def _model_memory_scale() -> float:
    config = _current_gpt2_config()
    return max(
        1.0,
        (
            (config["n_embd"] * config["n_inner"] * config["n_layer"])
            / (
                _BASE_GPT2_CONFIG["n_embd"]
                * _BASE_GPT2_CONFIG["n_inner"]
                * _BASE_GPT2_CONFIG["n_layer"]
            )
        ),
    )


def _profile_memory_multiplier(tp_size: int) -> float:
    # TP=1 should pessimistically account for optimizer state and activations.
    return _model_memory_scale() * (2.0 if tp_size == 1 else 1.0)


def _layer_profile(
    layer_name: str, layer_index: int, tp_size: int
) -> LayerExecutionResult:
    memory_multiplier = _profile_memory_multiplier(tp_size)

    if layer_name.startswith("transformer.h."):
        return LayerExecutionResult(
            layer_index=layer_index,
            layer_name=layer_name,
            forward=22.0,
            backward=44.0,
            mem_required=int(1_610_612_736 * memory_multiplier),
        )

    if layer_name in {"transformer.wte", "transformer.wpe"}:
        return LayerExecutionResult(
            layer_index=layer_index,
            layer_name=layer_name,
            forward=4.0,
            backward=8.0,
            mem_required=int(268_435_456 * memory_multiplier),
        )

    if layer_name == "transformer.drop":
        return LayerExecutionResult(
            layer_index=layer_index,
            layer_name=layer_name,
            forward=1.0,
            backward=1.0,
            mem_required=int(67_108_864 * memory_multiplier),
        )

    if layer_name == "transformer.ln_f":
        return LayerExecutionResult(
            layer_index=layer_index,
            layer_name=layer_name,
            forward=2.0,
            backward=4.0,
            mem_required=int(67_108_864 * memory_multiplier),
        )

    return LayerExecutionResult(
        layer_index=layer_index,
        layer_name=layer_name,
        forward=6.0,
        backward=12.0,
        mem_required=int(268_435_456 * memory_multiplier),
    )


def _patch_profiler() -> None:
    def synthetic_init_profile(self: ModelProfiler, inputs: dict[str, torch.Tensor]) -> None:
        configuration_engine = ConfigurationEngine.get_instance()
        if configuration_engine.agent_index != 0:
            return

        microbatch_size = inputs["input_ids"].shape[0]
        profile_path = ModelProfiler.get_profile_path(
            self.profile_dir, self.tp_size, microbatch_size, self.precision
        )
        if profile_path.exists():
            return

        data = {
            "model_name": self.model_name_or_path,
            "microbatch_size": microbatch_size,
            "tp_size": self.tp_size,
            "precision": self.precision,
            "layers": [
                asdict(_layer_profile(layer_name, index, self.tp_size))
                for index, layer_name in enumerate(_MODEL_LAYERS)
            ],
        }
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with profile_path.open("w") as f:
            json.dump(data, f, cls=JsonEncoder)

    def direct_load_profile(
        self: ModelProfiler, microbatch_size: int
    ) -> list[LayerExecutionResult]:
        profile_path = ModelProfiler.get_profile_path(
            self.profile_dir, self.tp_size, microbatch_size, self.precision
        )
        with profile_path.open() as f:
            data = json.load(f)
        return [
            LayerExecutionResult(
                layer_index=layer["layer_index"],
                layer_name=layer["layer_name"],
                forward=layer["forward"],
                backward=layer["backward"],
                mem_required=layer["mem_required"],
            )
            for layer in data["layers"]
        ]

    ModelProfiler.init_profile = synthetic_init_profile
    ModelProfiler.load_profile = direct_load_profile


def _patch_destroy() -> None:
    def patched_destroy(self: ExecutionEngine) -> None:
        configuration_engine = ConfigurationEngine.get_instance()
        t0 = time.perf_counter()

        try:
            if dist.is_initialized():
                dist.destroy_process_group(dist.GroupMember.WORLD)
        except Exception:
            pass
        finally:
            try:
                distributed_c10d._update_default_pg(None)
            except Exception:
                pass

            world = distributed_c10d._world
            for attr in [
                "pg_map",
                "pg_names",
                "pg_group_ranks",
                "pg_backend_config",
                "pg_to_tag",
                "tags_to_pg",
                "pg_coalesce_state",
                "pg_default_device",
            ]:
                obj = getattr(world, attr, None)
                if hasattr(obj, "clear"):
                    obj.clear()
            if hasattr(world, "group_count"):
                world.group_count = 0

        if configuration_engine.is_master:
            _update_reconfig_metrics(destroy_s=time.perf_counter() - t0)

    ExecutionEngine.on_receive_reconfiguration_notifiation = patched_destroy


def _patch_configuration_engine() -> None:
    original_init = ConfigurationEngine.init_distributed

    def patched_init(self: ConfigurationEngine) -> None:
        t0 = time.perf_counter()
        original_init(self)
        if self.is_master and _RECONFIG_ACTIVE:
            _update_reconfig_metrics(
                reinit_s=time.perf_counter() - t0,
                world_size=self.world_size,
            )

    ConfigurationEngine.init_distributed = patched_init


def _patch_torch_object_collectives_compat() -> None:
    original_object_to_tensor = distributed_c10d._object_to_tensor
    original_tensor_to_object = distributed_c10d._tensor_to_object

    def compat_object_to_tensor(
        obj: Any,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ):
        if group is None:
            group = dist.GroupMember.WORLD
        return original_object_to_tensor(obj, device, group)

    def compat_tensor_to_object(
        tensor: torch.Tensor,
        tensor_size: torch.Tensor,
        group: dist.ProcessGroup | None = None,
    ):
        if group is None:
            group = dist.GroupMember.WORLD
        return original_tensor_to_object(tensor, tensor_size, group)

    distributed_c10d._object_to_tensor = compat_object_to_tensor
    distributed_c10d._tensor_to_object = compat_tensor_to_object


def _patch_plugin_reconfigure() -> None:
    original_instantiate = OobleckPlugin._instantiate_pipelines

    def patched_instantiate(
        self: OobleckPlugin,
        pipeline_templates: dict[int, Any],
        global_num_microbatches: int,
        old_pg_mesh: list[Any] | None = None,
        old_rank_map: dict[HostInfo, list[int]] | None = None,
    ) -> tuple[list[Any], dict[Any, int]]:
        t0 = time.perf_counter()
        try:
            pipelines, num_microbatches = original_instantiate(
                self,
                pipeline_templates,
                global_num_microbatches,
                old_pg_mesh,
                old_rank_map,
            )
        except KeyError as exc:
            if old_pg_mesh is None or old_rank_map is None:
                raise
            logger.warning(
                "Preserving the previous pipeline shapes referenced a missing "
                f"template stage count {exc}. Falling back to a fresh global plan."
            )
            pipelines, num_microbatches = original_instantiate(
                self,
                pipeline_templates,
                global_num_microbatches,
                None,
                None,
            )

        configuration_engine = ConfigurationEngine.get_instance()
        if configuration_engine.is_master and _RECONFIG_ACTIVE:
            stage_counts, templates = _summarize_plan(pipelines, num_microbatches)
            _update_reconfig_metrics(
                instantiate_s=time.perf_counter() - t0,
                reconfigured_pipeline_stage_counts=stage_counts,
                reconfigured_templates=templates,
            )
        return pipelines, num_microbatches

    OobleckPlugin._instantiate_pipelines = patched_instantiate

    @torch.no_grad()
    def patched_reconfigure(
        self: OobleckPlugin,
        pipeline_templates: dict[int, Any],
        model: Any,
        optimizer: HybridParallelAMPOptimizer | HybridParallelNaiveOptimizer,
        dataloader: Any,
        lr_scheduler: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        global _RECONFIG_ACTIVE

        configuration_engine = ConfigurationEngine.get_instance()
        device = get_accelerator().get_current_device()
        total_start = time.perf_counter()

        layers = list(itertools.chain.from_iterable(self.pipelines[0].modules_per_stage))
        num_layers = len(layers)
        old_pg_mesh: np.ndarray = copy.deepcopy(self.pg_mesh.mesh)
        old_rank_map = copy.deepcopy(configuration_engine.rank_map)

        old_my_layers = torch.tensor(
            [
                index in [coord[PP_AXIS] for coord in self.pg_mesh.coords]
                for index in range(num_layers)
            ],
            dtype=torch.bool,
            device=device,
        )

        _RECONFIG_ACTIVE = True
        try:
            configuration_engine.get_host_update()
            del self.pg_mesh
            self.pg_mesh = None
            configuration_engine.init_distributed()

            def all_gather_layers_per_rank(data: torch.Tensor) -> dict[tuple[int, ...], np.ndarray]:
                data_list = torch.empty(
                    configuration_engine.world_size,
                    *data.shape,
                    device=device,
                    dtype=data.dtype,
                )
                dist.all_gather_into_tensor(data_list, data)

                data_list = data_list.numpy(force=True)
                result: dict[tuple[int, ...], np.ndarray] = {}
                for i in range(
                    0,
                    configuration_engine.world_size,
                    self.shard_config.tensor_parallel_size,
                ):
                    ranks = tuple(range(i, i + self.shard_config.tensor_parallel_size))
                    result[ranks] = data_list[i]
                return result

            old_layers_per_ranks = all_gather_layers_per_rank(old_my_layers)
            del old_my_layers

            new_pipelines, new_num_microbatches = self._instantiate_pipelines(
                pipeline_templates,
                self.global_batch_size // self.microbatch_size,
                old_pg_mesh,
                old_rank_map,
            )
            new_pg_mesh = HeterogeneousProcessGroupMesh(
                new_pipelines, self.shard_config.tensor_parallel_size
            )

            new_my_layers = torch.tensor(
                [
                    index in [coord[PP_AXIS] for coord in new_pg_mesh.coords]
                    for index in range(num_layers)
                ],
                dtype=torch.bool,
                device=device,
            )
            new_layers_per_ranks = all_gather_layers_per_rank(new_my_layers)

            transfer_start = time.perf_counter()
            layers_required_by_ranks: dict[int, list[tuple[int, ...]]] = defaultdict(list)
            for ranks, has_layers in old_layers_per_ranks.items():
                for layer_index, has_layer in enumerate(has_layers):
                    if not has_layer and new_layers_per_ranks[ranks][layer_index]:
                        layers_required_by_ranks[layer_index].append(ranks)

            layer_modules: dict[str, nn.Module] = {
                layer_name: module
                for name, module in model.module.named_modules()
                for layer_name in layers
                if name == layer_name
            }

            def get_layer_holder_ranks(layer_index: int) -> list[tuple[int, ...]]:
                holders = []
                for ranks, has_layers in old_layers_per_ranks.items():
                    if has_layers[layer_index]:
                        holders.append(ranks)
                return holders

            adds_to_param_info: dict[str, dict[Any, Any]] = {
                "param2id": {},
                "id2param": {},
                "param2shape": {},
            }
            removes_from_param_info: dict[str, list[Any]] = {
                "param2id": [],
                "id2param": [],
                "param2shape": [],
            }

            transferred_layers = 0
            transferred_parameters = 0

            for layer_index, ranks_need_layer_list in layers_required_by_ranks.items():
                layer_holder_ranks = get_layer_holder_ranks(layer_index)
                if not layer_holder_ranks:
                    raise RuntimeError(f"No one holds the layer: {layers[layer_index]}!")

                module = layer_modules[layers[layer_index]]

                for ranks_need_layer in ranks_need_layer_list:
                    if not layer_holder_ranks:
                        layer_holder_ranks = get_layer_holder_ranks(layer_index)

                    sender_ranks = layer_holder_ranks.pop(0)
                    if configuration_engine.rank in ranks_need_layer:
                        sender_rank = sender_ranks[
                            ranks_need_layer.index(configuration_engine.rank)
                        ]

                        for (
                            submodule,
                            name,
                            placeholder,
                        ) in ModelSharder.buffer_placeholders(
                            module, delete_placeholders_after=True
                        ):
                            setattr(submodule, name, placeholder.create())

                        for (
                            submodule,
                            name,
                            placeholder,
                        ) in ModelSharder.parameter_placeholders(
                            module, delete_placeholders_after=True
                        ):
                            size = torch.empty(1, dtype=torch.int64, device=device)
                            dist.recv(size, sender_rank)
                            tensor = torch.empty(
                                size.item(), dtype=torch.uint8, device=device
                            )
                            dist.recv(tensor, sender_rank)

                            buff = io.BytesIO(tensor.numpy(force=True))
                            tensor_obj: dict[str, Any] = torch.load(
                                buff, map_location=device
                            )

                            states = tensor_obj["states"]
                            master_tensor: torch.Tensor = tensor_obj["parameter"]

                            p = torch.nn.Parameter(
                                master_tensor.to(dtype=model.mixed_precision)
                                if isinstance(optimizer, MixedPrecisionOptimizer)
                                and master_tensor.dtype == torch.float32
                                else master_tensor
                            )
                            setattr(submodule, name, p)

                            optimizer.optim.state[master_tensor] = states
                            optimizer.optim.param_groups[0]["params"].append(master_tensor)

                            if isinstance(optimizer, MixedPrecisionOptimizer):
                                optimizer.master_to_working_map[master_tensor] = p
                                optimizer.working_to_master_map[p] = master_tensor

                            optim_param_index = optimizer.param_info["param2id"][
                                placeholder.param_id
                            ]
                            removes_from_param_info["param2id"].append(placeholder.param_id)
                            removes_from_param_info["id2param"].append(optim_param_index)
                            removes_from_param_info["param2shape"].append(placeholder.param_id)

                            adds_to_param_info["param2id"][id(p)] = optim_param_index
                            adds_to_param_info["id2param"][optim_param_index] = id(p)
                            adds_to_param_info["param2shape"][id(p)] = p.shape

                            transferred_parameters += 1
                        transferred_layers += 1

                    elif configuration_engine.rank in sender_ranks:
                        rank_need_layer = ranks_need_layer[
                            sender_ranks.index(configuration_engine.rank)
                        ]
                        for _, param in module.named_parameters():
                            buff = io.BytesIO()
                            master_param = (
                                optimizer.working_to_master_map[param]
                                if isinstance(optimizer, HybridParallelAMPOptimizer)
                                else param
                            )
                            states = optimizer.optim.state.get(master_param)
                            torch.save(
                                {"states": states, "parameter": master_param},
                                buff,
                            )
                            buff.seek(0)

                            tensor = torch.frombuffer(
                                buff.getbuffer(), dtype=torch.uint8
                            ).to(device)
                            size = torch.tensor(
                                [tensor.numel()], dtype=torch.int64, device=device
                            )
                            dist.send(size, rank_need_layer)
                            dist.send(tensor, rank_need_layer)

            my_layers = [
                layers[index] for index, has_layer in enumerate(new_my_layers) if has_layer
            ]
            tp_group = new_pg_mesh.get_group_along_axis(TP_AXIS)

            for layer in my_layers:
                module = layer_modules[layer]
                for submodule in module.modules():
                    if isinstance(submodule, ParallelModule):
                        for attr_name in dir(submodule):
                            attr = getattr(submodule, attr_name)
                            if isinstance(attr, dist.ProcessGroup):
                                setattr(submodule, attr_name, tp_group)

            for param_info_key, items in removes_from_param_info.items():
                for item in items:
                    del optimizer.param_info[param_info_key][item]

            for param_info_key, items in adds_to_param_info.items():
                optimizer.param_info[param_info_key].update(items)

            old_layers = next(
                held_layers
                for ranks, held_layers in old_layers_per_ranks.items()
                if configuration_engine.rank in ranks
            )
            new_layers = next(
                held_layers
                for ranks, held_layers in new_layers_per_ranks.items()
                if configuration_engine.rank in ranks
            )
            for index, (old_layer, new_layer) in enumerate(zip(old_layers, new_layers)):
                if old_layer and not new_layer:
                    module = layer_modules[layers[index]]
                    ModelSharder.set_tensors_to_placeholder(module)

            layer_transfer_s = time.perf_counter() - transfer_start
            if configuration_engine.is_master:
                _update_reconfig_metrics(
                    layer_transfer_s=layer_transfer_s,
                    layers_received=transferred_layers,
                    parameters_received=transferred_parameters,
                )

            configure_start = time.perf_counter()
            self.set_pipelines(new_pipelines, new_num_microbatches)
            model, optimizer, _, dataloader, lr_scheduler = self.configure(
                model, optimizer, None, dataloader, lr_scheduler, forced=True
            )

            if configuration_engine.is_master:
                _update_reconfig_metrics(
                    final_configure_s=time.perf_counter() - configure_start,
                    total_s=time.perf_counter() - total_start,
                )

            return model, optimizer, dataloader, lr_scheduler
        finally:
            _RECONFIG_ACTIVE = False

    OobleckPlugin.reconfigure = patched_reconfigure


def _apply_runtime_patches() -> None:
    _patch_profiler()
    _patch_destroy()
    _patch_configuration_engine()
    _patch_torch_object_collectives_compat()
    _patch_plugin_reconfigure()


class LocalMasterService(master_service_pb2_grpc.OobleckMasterServicer):
    def __init__(
        self,
        script_args: ScriptArguments,
        hosts: list[HostInfo],
        disconnect_condition: Condition,
    ):
        self.script_args = script_args
        self.hosts = hosts
        self.disconnect_condition = disconnect_condition
        self.master_port = 0

    def GetDistInfo(
        self,
        request: master_service_pb2.DistInfo,
        context: grpc.RpcContext,
    ) -> master_service_pb2.DistInfo:
        return master_service_pb2.DistInfo(
            hosts=[
                master_service_pb2.HostInfo(
                    ip=host.ip,
                    devices=host.devices,
                    port=host.port,
                    status=host.status.name,
                )
                for host in self.hosts
            ]
        )

    def GetCode(
        self,
        request: master_service_pb2.CodeInfo,
        context: grpc.RpcContext,
    ) -> master_service_pb2.CodeInfo:
        return master_service_pb2.CodeInfo(
            path=self.script_args.training_script.absolute().as_posix(),
            args=self.script_args.training_script_args,
        )

    def SetMasterRankPort(
        self, request: master_service_pb2.PortInfo, context: grpc.RpcContext
    ) -> empty_pb2.Empty:
        self.master_port = request.port
        return empty_pb2.Empty()

    def GetMasterRankPort(
        self, request: empty_pb2.Empty, context: grpc.RpcContext
    ) -> master_service_pb2.PortInfo:
        return master_service_pb2.PortInfo(port=self.master_port)

    def KillAgent(
        self, request: master_service_pb2.AgentInfo, context: grpc.RpcContext
    ) -> empty_pb2.Empty:
        self.hosts[request.agent_index].status = HostStatus.terminating
        with self.disconnect_condition:
            self.disconnect_condition.notify_all()
        return empty_pb2.Empty()

    def set_killed(self, agent_indices: list[int]) -> None:
        for agent_index in agent_indices:
            self.hosts[agent_index].status = HostStatus.killed

    def notify_reconfigure(self) -> None:
        with self.disconnect_condition:
            self.disconnect_condition.notify_all()

    def WatchReconfigurationNotification(
        self, request: empty_pb2.Empty, context: grpc.RpcContext
    ):
        with self.disconnect_condition:
            self.disconnect_condition.wait()

        if context.is_active():
            yield master_service_pb2.DistInfo(
                hosts=[
                    master_service_pb2.HostInfo(
                        ip=host.ip,
                        devices=host.devices,
                        port=host.port,
                        status=host.status.name,
                    )
                    for host in self.hosts
                ]
            )


def _start_agents(
    python_executable: str,
    master_port: int,
    tag: str,
    base_dir: Path,
    repo_root: Path,
    num_agents: int,
) -> tuple[list[subprocess.Popen[Any]], list[Any]]:
    processes: list[subprocess.Popen[Any]] = []
    log_files: list[Any] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")

    for agent_index in range(num_agents):
        log_path = base_dir / tag / f"agent{agent_index}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w")
        log_files.append(log_file)
        cmd = [
            python_executable,
            "-m",
            "oobleck.elastic.agent",
            "--master_ip",
            "127.0.0.1",
            "--master_port",
            str(master_port),
            "--agent_index",
            str(agent_index),
            "--tag",
            tag,
            "--base_dir",
            str(base_dir),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        processes.append(proc)
    return processes, log_files


def _parse_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--tag", required=True, type=str)
@click.option("--base-dir", type=click.Path(path_type=Path), default=Path("/tmp/oobleck_native_llama"))
@click.option("--model-config-path", type=click.Path(path_type=Path), default=None)
@click.option("--tp-size", type=int, default=2)
@click.option("--global-batch-size", type=int, default=16)
@click.option("--microbatch-size", type=int, default=1)
@click.option("--seq-len", type=int, default=4096)
@click.option("--num-agents", type=int, default=4)
@click.option("--num-gpus-per-agent", type=int, default=2)
@click.option("--inject-step", type=int, default=4)
@click.option("--post-failure-steps", type=int, default=3)
@click.option("--timeout-s", type=int, default=1800)
@click.option("--stall-timeout-s", type=int, default=180)
@click.option("--python-executable", type=str, default=sys.executable)
@click.option("--disable-flash-attn", is_flag=True, default=False)
@click.option("--fail-agent-indices", type=str, required=True)
def launch(
    tag: str,
    base_dir: Path,
    model_config_path: Path | None,
    tp_size: int,
    global_batch_size: int,
    microbatch_size: int,
    seq_len: int,
    num_agents: int,
    num_gpus_per_agent: int,
    inject_step: int,
    post_failure_steps: int,
    timeout_s: int,
    stall_timeout_s: int,
    python_executable: str,
    disable_flash_attn: bool,
    fail_agent_indices: str,
) -> None:
    repo_root = Path.cwd()
    job_dir = base_dir / tag
    trigger_path = job_dir / "inject_failure.trigger"
    results_path = job_dir / "results.jsonl"

    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    fail_indices = [int(index) for index in fail_agent_indices.split(",") if index]
    hosts = [
        HostInfo(
            ip="127.0.0.1",
            devices=",".join(
                str(gpu)
                for gpu in range(
                    agent_index * num_gpus_per_agent,
                    (agent_index + 1) * num_gpus_per_agent,
                )
            ),
            port=22,
            status=HostStatus.up,
        )
        for agent_index in range(num_agents)
    ]

    disconnect_condition = get_context("spawn").Condition()
    training_args = [
        "train",
        "--tag",
        tag,
        "--base-dir",
        str(base_dir),
        "--results-path",
        str(results_path),
        "--trigger-path",
        str(trigger_path),
        "--tp-size",
        str(tp_size),
        "--global-batch-size",
        str(global_batch_size),
        "--microbatch-size",
        str(microbatch_size),
        "--seq-len",
        str(seq_len),
        "--inject-step",
        str(inject_step),
        "--post-failure-steps",
        str(post_failure_steps),
    ]
    if model_config_path is not None:
        training_args.extend(["--model-config-path", str(model_config_path)])
    if disable_flash_attn:
        training_args.append("--disable-flash-attn")

    service = LocalMasterService(
        ScriptArguments(training_script=Path(__file__), training_script_args=training_args),
        hosts,
        disconnect_condition,
    )

    server = grpc.server(ThreadPoolExecutor(max_workers=16))
    master_service_pb2_grpc.add_OobleckMasterServicer_to_server(service, server)
    master_port = server.add_insecure_port("0.0.0.0:0")
    server.start()

    _record_event(
        results_path,
        {
            "type": "launch_config",
            "tag": tag,
            "fail_agent_indices": fail_indices,
            "inject_step": inject_step,
            "tp_size": tp_size,
            "global_batch_size": global_batch_size,
            "microbatch_size": microbatch_size,
            "seq_len": seq_len,
            "model_config_path": None if model_config_path is None else str(model_config_path),
            "disable_flash_attn": disable_flash_attn,
        },
    )

    processes, log_files = _start_agents(
        python_executable=python_executable,
        master_port=master_port,
        tag=tag,
        base_dir=base_dir,
        repo_root=repo_root,
        num_agents=num_agents,
    )

    injected = False
    last_growth = time.monotonic()
    last_event_count = 0
    interrupted_seen = False
    start_time = time.monotonic()

    try:
        while True:
            events = _parse_results(results_path)
            if len(events) != last_event_count:
                last_growth = time.monotonic()
                last_event_count = len(events)
                interrupted_seen = interrupted_seen or any(
                    event["type"] == "step_interrupted" for event in events
                )

            if not injected and trigger_path.exists():
                service.set_killed(fail_indices)
                _record_event(
                    results_path,
                    {
                        "type": "failure_injected",
                        "fail_agent_indices": fail_indices,
                        "step": inject_step,
                    },
                )
                service.notify_reconfigure()
                injected = True

            if all(proc.poll() is not None for proc in processes):
                _record_event(
                    results_path,
                    {
                        "type": "all_processes_exited",
                        "exit_codes": [proc.poll() for proc in processes],
                    },
                )
                break

            if time.monotonic() - start_time > timeout_s:
                _record_event(results_path, {"type": "master_timeout", "timeout_s": timeout_s})
                break

            if interrupted_seen and time.monotonic() - last_growth > stall_timeout_s:
                _record_event(
                    results_path,
                    {
                        "type": "master_stall_timeout",
                        "stall_timeout_s": stall_timeout_s,
                    },
                )
                break

            time.sleep(1.0)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
        for proc in processes:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        server.stop(grace=None)
        for log_file in log_files:
            log_file.close()


@cli.command()
@click.option("--tag", required=True, type=str)
@click.option("--base-dir", type=click.Path(path_type=Path), required=True)
@click.option("--model-config-path", type=click.Path(path_type=Path), default=None)
@click.option("--results-path", type=click.Path(path_type=Path), required=True)
@click.option("--trigger-path", type=click.Path(path_type=Path), required=True)
@click.option("--tp-size", type=int, required=True)
@click.option("--global-batch-size", type=int, required=True)
@click.option("--microbatch-size", type=int, required=True)
@click.option("--seq-len", type=int, required=True)
@click.option("--inject-step", type=int, required=True)
@click.option("--post-failure-steps", type=int, required=True)
@click.option("--disable-flash-attn", is_flag=True, default=False)
def train(
    tag: str,
    base_dir: Path,
    model_config_path: Path | None,
    results_path: Path,
    trigger_path: Path,
    tp_size: int,
    global_batch_size: int,
    microbatch_size: int,
    seq_len: int,
    inject_step: int,
    post_failure_steps: int,
    disable_flash_attn: bool,
) -> None:
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    _apply_runtime_patches()
    _reset_reconfig_metrics()
    gpt2_config = _load_gpt2_config(model_config_path, seq_len)
    _set_current_gpt2_config(gpt2_config)

    configuration_engine = ConfigurationEngine.get_instance()
    plugin = OobleckPlugin(
        tp_size=tp_size,
        global_batch_size=global_batch_size,
        microbatch_size=microbatch_size,
        precision="bf16",
        enable_fused_normalization=False,
        enable_flash_attention=not disable_flash_attn,
        fault_tolerance_threshold=1,
    )
    engine = ExecutionEngine(plugin)

    if configuration_engine.is_master:
        _record_event(
            results_path,
            {
                "type": "build_model_start",
                "rank": configuration_engine.rank,
                "agent_index": configuration_engine.agent_index,
                "model_config_path": None
                if model_config_path is None
                else str(model_config_path),
                "gpt2_config": gpt2_config,
            },
        )

    model = _build_model(seq_len, gpt2_config)
    model.gradient_checkpointing_enable()
    if configuration_engine.is_master:
        _record_event(
            results_path,
            {
                "type": "build_model_done",
                "rank": configuration_engine.rank,
                "agent_index": configuration_engine.agent_index,
            },
        )

    dataset = SyntheticDataset(length=4096)
    dataloader = plugin.prepare_dataloader(
        dataset,
        shuffle=False,
        drop_last=True,
        collate_fn=_build_collate_fn(vocab_size=gpt2_config["vocab_size"], seq_len=seq_len),
    )
    optimizer = Adam(model.parameters(), lr=1e-4, foreach=False)

    model, optimizer, _, dataloader, _ = engine.prepare(
        model,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=None,
    )

    if configuration_engine.is_master:
        stage_counts, templates = _summarize_plan(engine.plugin.pipelines, engine.plugin.num_microbatches)
        _record_event(
            results_path,
            {
                "type": "prepare_done",
                "rank": configuration_engine.rank,
                "agent_index": configuration_engine.agent_index,
            },
        )
        _record_event(
            results_path,
            {
                "type": "initial_plan",
                "rank": configuration_engine.rank,
                "agent_index": configuration_engine.agent_index,
                "pipeline_stage_counts": stage_counts,
                "templates": templates,
            },
        )

    model.train()
    optimizer.zero_grad()
    dataloader_iter = iter(dataloader)
    is_pp_last_stage = engine.plugin.stage_manager.is_last_stage()
    reconfigured = False
    total_steps = inject_step + post_failure_steps + 1

    for step in range(1, total_steps + 1):
        if configuration_engine.is_master:
            _record_event(
                results_path,
                {
                    "type": "step_start",
                    "rank": configuration_engine.rank,
                    "agent_index": configuration_engine.agent_index,
                    "step": step,
                    "reconfigured": reconfigured,
                },
            )
        start = time.perf_counter()
        try:
            outputs = engine.execute(
                dataloader_iter,
                model,
                criterion=lambda outputs, inputs: outputs.loss,
                optimizer=optimizer,
                return_loss=True,
                return_outputs=False,
            )
        except Exception as exc:
            exc_text = str(exc).lower()
            is_runtime_failure = isinstance(exc, RuntimeError) and any(
                token in exc_text
                for token in [
                    "illegal memory access",
                    "communicator was aborted",
                    "nccl",
                    "connection reset by peer",
                ]
            )
            if is_runtime_failure:
                if configuration_engine.is_master:
                    _record_event(
                        results_path,
                        {
                            "type": "runtime_failure_caught",
                            "rank": configuration_engine.rank,
                            "agent_index": configuration_engine.agent_index,
                            "step": step,
                            "reconfigured": reconfigured,
                            "error": repr(exc),
                        },
                    )
                setattr(dataloader_iter, "invalidated", True)
                try:
                    if dist.is_initialized():
                        engine.on_receive_reconfiguration_notifiation()
                except Exception:
                    pass
                engine.need_reconfiguration = True
                outputs = None
            else:
                if configuration_engine.is_master:
                    _record_event(
                        results_path,
                        {
                            "type": "step_exception",
                            "rank": configuration_engine.rank,
                            "agent_index": configuration_engine.agent_index,
                            "step": step,
                            "reconfigured": reconfigured,
                            "error": repr(exc),
                        },
                    )
                raise

        if outputs is None:
            if configuration_engine.is_master:
                _record_event(
                    results_path,
                    {
                        "type": "step_interrupted",
                        "rank": configuration_engine.rank,
                        "agent_index": configuration_engine.agent_index,
                        "step": step,
                        "reconfigured": reconfigured,
                    },
                )
            try:
                model, optimizer, dataloader = engine.reconfigure(
                    model, optimizer, dataloader
                )
            except Exception as exc:
                if configuration_engine.is_master:
                    _record_event(
                        results_path,
                        {
                            "type": "reconfigure_failed",
                            "rank": configuration_engine.rank,
                            "agent_index": configuration_engine.agent_index,
                            "step": step,
                            "error": repr(exc),
                        },
                    )
                raise

            dataloader_iter = iter(dataloader)
            reconfigured = True
            if configuration_engine.is_master:
                metrics = _get_reconfig_metrics()
                stage_counts, templates = _summarize_plan(
                    engine.plugin.pipelines, engine.plugin.num_microbatches
                )
                metrics.setdefault("reconfigured_pipeline_stage_counts", stage_counts)
                metrics.setdefault("reconfigured_templates", templates)
                _record_event(
                    results_path,
                    {
                        "type": "reconfigure_metrics",
                        "rank": configuration_engine.rank,
                        "agent_index": configuration_engine.agent_index,
                        "step": step,
                        **metrics,
                    },
                )
            continue

        optimizer.step()
        optimizer.zero_grad()
        step_s = time.perf_counter() - start

        if configuration_engine.is_master:
            loss = None
            if is_pp_last_stage and isinstance(outputs, dict) and "loss" in outputs:
                loss = float(outputs["loss"])
            _record_event(
                results_path,
                {
                    "type": "step_done",
                    "rank": configuration_engine.rank,
                    "agent_index": configuration_engine.agent_index,
                    "step": step,
                    "reconfigured": reconfigured,
                    "loss": loss,
                    "step_s": step_s,
                    "throughput_samples_s": global_batch_size / step_s,
                    "throughput_tokens_s": (global_batch_size * seq_len) / step_s,
                },
            )

            if step == inject_step and not reconfigured:
                trigger_path.parent.mkdir(parents=True, exist_ok=True)
                trigger_path.write_text("inject\n")

        if step == inject_step and not reconfigured:
            # Hold all ranks at a step boundary until the watcher thread receives
            # the reconfiguration signal, so no rank starts the next pipeline step
            # with a soon-to-be-invalid communicator.
            wait_deadline = time.monotonic() + 5.0
            while time.monotonic() < wait_deadline and not engine.need_reconfiguration:
                time.sleep(0.05)


if __name__ == "__main__":
    cli()
