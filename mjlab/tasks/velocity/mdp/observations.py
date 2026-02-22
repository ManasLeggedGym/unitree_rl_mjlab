from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor, BuiltinSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

def height_map(env: ManagerBasedRlEnv, sensor_name: str)-> torch.Tensor:
  sensor: RayCastSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.normals_w is not None
  map = sensor_data.normals_w.flatten(start_dim=1)
  return torch.sign(map) *  torch.log1p(torch.abs(map))

def external_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  # import pdb; pdb.set_trace()
  robot = env.scene.entities["robot"]
  assert robot.data.body_external_force is not None
  assert robot.data.body_external_torque is not None
  f = robot.data.body_external_force.flatten(start_dim=-1)
  t = robot.data.body_external_torque.flatten(start_dim=-1)
  force = torch.sign(f) * torch.log1p(torch.abs(f))
  torque = torch.sign(t)* torch.log1p(torch.abs(t))
  return torch.cat((force,torque),dim=-1)

def orientation(env: ManagerBasedRlEnv):
  robot = env.scene.entities["robot"]
  assert robot.data.root_link_quat_w3
  pose = robot.data.root_link_pose_w
  return torch.sign(pose)* torch.log1p(torch.abs(pose))
