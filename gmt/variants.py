define_variants = {
    'tiny':   {'d_model':128,  'num_layers':2, 'ff_dim':256,  'dropout':0.1,  'nhead':4},
    'small':  {'d_model':256,  'num_layers':3, 'ff_dim':512,  'dropout':0.15, 'nhead':8},
    'medium': {'d_model':512,  'num_layers':4, 'ff_dim':1024, 'dropout':0.2,  'nhead':8},
    'large':  {'d_model':768,  'num_layers':5, 'ff_dim':1536, 'dropout':0.2,  'nhead':8},
}
MODEL_NAME = 'GMT'