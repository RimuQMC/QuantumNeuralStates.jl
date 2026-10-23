
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
encoding_signature
nn_signature
_signature
_write_architecture
_write_weights
_write_addrs
_write_input_scale
_write_mf
parse_encoding
_format_addr

_actname
_typename
_funcname
_devname
_flag_mem
_group
_fmt_bytes
_mem_array

_set_params!
addrs_random
safe_denom
select_device
RunningMean
is_outlier!

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
_prepare_input_kernel!
_prepare_input_single_kernel!
_prepare_input_stream_kernel!
_find_column
_column_start

cg_matvec!

weighted_variance_kernel!
weighted_sum_kernel!

_ansatz_first_modify!
_ansatz_modify_new!

_mf_addrs_kernel!
_mf_offdiag_kernel!
_to_backend

build_model

solve_self_consistent_eta
calc_omega_denominator
calc_f
strong_coupling_amplitudes
_mf_func_froehlich
```

## Index

```@index
Pages = ["API.md"]
```

