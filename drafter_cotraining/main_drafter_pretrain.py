# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Draft-model pretraining entry point. Mirrors main_drafter_ct.py but for
the offline pretrain trainer (no rollout / RL loop).

Usage:
    python -m recipe.drafter_cotraining.main_drafter_pretrain
"""

import hydra

from verl.utils.device import auto_set_device

from recipe.drafter_cotraining.trainer.pretrain_trainer import run_draft_model_pretrain


@hydra.main(config_path="config", config_name="draft_model_pretrain_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_draft_model_pretrain(config)


if __name__ == "__main__":
    main()
