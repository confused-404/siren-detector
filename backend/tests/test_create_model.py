from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from helpers.create_model import SpecCnnDropoutConfig, create_spec_cnn_with_custom_dropouts


def test_create_spec_cnn_uses_explicit_dropout_config() -> None:
    model = create_spec_cnn_with_custom_dropouts(
        input_shape=(8, 8, 1),
        num_classes=3,
        dropout_config=SpecCnnDropoutConfig(
            conv_block_1=0.11,
            conv_block_2=0.22,
            conv_block_3=0.33,
            dense_layer=0.44,
        ),
    )

    dropout_rates = [layer.rate for layer in model.layers if hasattr(layer, "rate")]
    assert dropout_rates == [0.11, 0.22, 0.33, 0.44]
