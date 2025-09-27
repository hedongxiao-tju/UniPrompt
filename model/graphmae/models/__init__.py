from .edcoder import PreModel


def build_model(num_heads = 4, num_out_heads = 1, num_hidden = 512, num_layers = 2, residual = False, attn_drop = 0.1, in_drop = 0.2, norm = None, negative_slope = 0.2, encoder_type = "gat", decoder_type = "gat", mask_rate = 0.5, drop_edge_rate = 0.0, replace_rate = 0.0, activation = "prelu", loss_fn = "sce", alpha_l = 2, num_features = 1433, concat_hidden = False):
    model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model
