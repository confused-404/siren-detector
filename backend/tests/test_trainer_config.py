from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("tensorflow")

from helpers.create_model import SpecCnnDropoutConfig
from trainer import build_dropout_config


def test_build_dropout_config_reads_cli_values() -> None:
    args = argparse.Namespace(
        dropout_conv1=0.15,
        dropout_conv2=0.25,
        dropout_conv3=0.35,
        dropout_dense=0.45,
    )

    assert build_dropout_config(args) == SpecCnnDropoutConfig(
        conv_block_1=0.15,
        conv_block_2=0.25,
        conv_block_3=0.35,
        dense_layer=0.45,
    )
