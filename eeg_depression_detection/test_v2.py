#!/usr/bin/env python3
import sys
import traceback

sys.path.insert(0, '.')

try:
    print("Importing BiLSTM...")
    from models.branches.bilstm_encoder import EEGBiLSTMEncoder
    print("  OK")

    print("Importing ThreeWayFusion...")
    from models.fusion.three_way_fusion import ThreeWayAttentionFusion
    print("  OK")

    print("Importing Full Model V2...")
    from models.full_model_v2 import AdvancedEEGDepressionDetectorV2, ModelConfigV2, model_summary_v2
    print("  OK")

    print("\nCreating model...")
    import torch
    model = AdvancedEEGDepressionDetectorV2(ModelConfigV2())

    s = model_summary_v2(model)
    print(f"\nV2 Model Summary:")
    print(f"  Total params: {s['total_parameters']:,}")
    print(f"  - Transformer: {s['transformer_parameters']:,}")
    print(f"  - Bi-LSTM: {s['bilstm_parameters']:,}")
    print(f"  - GNN: {s['gnn_parameters']:,}")
    print(f"  - Fusion: {s['fusion_parameters']:,}")
    print("\nALL TESTS PASSED!")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
