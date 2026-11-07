#!/usr/bin/env bash
# 验证共享环境 GPU、模板解析，不启动结构预测。
set -uo pipefail
export PATH="/public_bme2/Share200T/管吉松/envs/af3/bin:$PATH"
PYTHON=/public_bme2/Share200T/管吉松/envs/af3/bin/python
cd "/public_bme2/Share200T/管吉松/张天夫/Optimize-peptide-sequences" || exit 2
"$PYTHON" - <<'PY' || exit $?
import json
import jax
import jax.numpy as jnp
from alphafold3.common import folding_input
from 小肽公共库 import SEQ_202, ASIC1A_SEQ, create_af3_json
from 小肽模板 import target_options
print('JAX_VERSION', jax.__version__, flush=True)
print('JAX_DEVICES', jax.devices(), flush=True)
x = jnp.arange(1000, dtype=jnp.float32)
print('GPU_COMPUTE', float(jnp.dot(x, x)), flush=True)
job = create_af3_json(SEQ_202, ASIC1A_SEQ, 'asic_template_check',
                      **target_options('ASIC1A'))
payload = json.dumps(job)
fold = folding_input.Input.from_json(payload)
proteins = [c for c in fold.chains if isinstance(c, folding_input.ProteinChain)]
print('TEMPLATE_CHAINS', [(c.id, len(c.sequence), len(c.templates or ())) for c in proteins], flush=True)
print('AF3_SHARED_ENV_OK', flush=True)
PY
