# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The push connector under the standard ``NixlConnector`` name.

Serving frameworks that only accept ``kv_connector="NixlConnector"`` (e.g. Dynamo's PD protocol registry) can run
push-mode transfer with ``"kv_connector": "NixlConnector", "kv_connector_module_path":
"vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_as_nixl"``. Request and response shapes are the same as
the pull connector's.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector as NixlConnector,
)

__all__ = ["NixlConnector"]
