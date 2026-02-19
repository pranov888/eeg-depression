"""Test model forward pass."""
import torch
import traceback

def test_gnn():
    """Test GNN encoder."""
    print("Testing GNN encoder...")
    from models.branches.gnn_encoder import EEGGraphAttentionNetwork, create_eeg_graph_batch

    gnn = EEGGraphAttentionNetwork(
        node_feat_dim=576,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.3,
        output_dim=128
    ).cuda()
    print("  GNN created successfully")

    batch_size = 4
    wpd_features = torch.randn(batch_size, 19, 576)
    x, edge_index, batch = create_eeg_graph_batch(wpd_features)
    x = x.cuda()
    edge_index = edge_index.cuda()
    batch = batch.cuda()

    print(f"  x: {x.shape}, edge_index: {edge_index.shape}, batch: {batch.shape}")

    with torch.no_grad():
        out = gnn(x, edge_index, batch)
    print(f"  GNN output: {out.shape}")
    print("  GNN test PASSED!")
    return True


def test_transformer():
    """Test Transformer encoder."""
    print("\nTesting Transformer encoder...")
    from models.branches.transformer_encoder import EEGTransformerEncoder

    transformer = EEGTransformerEncoder(
        img_size=(64, 128),
        patch_size=(8, 16),
        in_channels=1,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_ff=512,
        dropout=0.1,
        use_cls_token=True
    ).cuda()
    print("  Transformer created successfully")

    batch_size = 4
    scalograms = torch.randn(batch_size, 1, 64, 128).cuda()

    with torch.no_grad():
        out = transformer(scalograms)
    print(f"  Transformer output: {out.shape}")
    print("  Transformer test PASSED!")
    return True


def test_full_model():
    """Test full model."""
    print("\nTesting full model...")
    from models.full_model import AdvancedEEGDepressionDetector, ModelConfig
    from models.branches.gnn_encoder import create_eeg_graph_batch

    config = ModelConfig()
    model = AdvancedEEGDepressionDetector(config).cuda()
    model.eval()
    print("  Full model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    batch_size = 4
    scalograms = torch.randn(batch_size, 1, 64, 128).cuda()
    wpd_features = torch.randn(batch_size, 19, 576)

    x, edge_index, batch = create_eeg_graph_batch(wpd_features)
    x = x.cuda()
    edge_index = edge_index.cuda()
    batch = batch.cuda()

    with torch.no_grad():
        outputs = model(scalograms, x, edge_index, batch)
    print(f"  Full model output keys: {outputs.keys()}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Probs shape: {outputs['probs'].shape}")
    print(f"  Sample predictions: {outputs['probs'][:2].squeeze().tolist()}")
    print("  Full model test PASSED!")
    return True


if __name__ == "__main__":
    try:
        test_gnn()
        test_transformer()
        test_full_model()
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
