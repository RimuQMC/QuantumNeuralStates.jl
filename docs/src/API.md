
# API

```@meta
CurrentModule = QuantumNeuralStates
```

```@docs
_backprop!
layer_inputs
LayerRange
make_buffers
make_buffer
_flatten!
_flatten_ln!
_fill_JW_Jb!
_conv_JW_kernel!
_conv_Jb_kernel!
_conv_dx_kernel!
_pool_back_uniform_kernel!
_pool_back_extremum_kernel!
_pool_uniform!
_pool_extremum!
_pool_backward!

chain_signature
_signature
_write_architecture
_write_weights
_write_addrs
_write_input_scale
_format_addr

_actname
_typename
_funcname
_devname
_flag_mem
_group
_fmt_bytes
_mem_array
memory_estimate

SCALE_FUNCTIONS
_set_params!
addrs_random
final_elocs_statistics!
safe_denom
select_device

center_order
grid_dims
build_truncation
change_truncation!

push_epoch!
_get_burnin
_build_opt_buffer
_run_epoch

_check_device

hasparams
_init_std
_lookup_deriv
MultiForwardLayer
_padleft
_lout
_src
_tap_inv
_check_input
_input_shape
_pool_forward_kernel!
LayerNorm_multiforward

_activation_health
_print_stats

cg_matvec!

elocs_clamping!

_ansatz_first_modify!
_ansatz_modify_new!
```

## Index

```@index
Pages = ["API.md"]
```

