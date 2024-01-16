# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea


def build(model_config):
    output_shape = model_config['output_shape'] # [480, 360, 32]
    num_class = model_config['num_class'] # 20
    num_input_features = model_config['num_input_features'] # 16
    use_norm = model_config['use_norm'] # True
    init_size = model_config['init_size'] # 32
    fea_dim = model_config['fea_dim'] # 9
    out_fea_dim = model_config['out_fea_dim'] # 256
    
    img_fea_dim = model_config['img_fea_dim'] # 3    

    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape, # [480, 360, 32]
        use_norm=use_norm, # True
        num_input_features=num_input_features, # 16
        init_size=init_size, # 32
        nclasses=num_class # 20
    )
 
    cy_fea_net = cylinder_fea(
                              grid_size=output_shape, # [480, 360, 32]
                              fea_dim=fea_dim, # 9
                              out_pt_fea_dim=out_fea_dim, # 256
                              fea_compre=num_input_features, # 16 
                              img_fea_dim=img_fea_dim
                              )

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape # [480, 360, 32]
    )

    return model
