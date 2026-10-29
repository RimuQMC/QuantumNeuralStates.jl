
# Neural Ansatz

```@meta
CurrentModule = QuantumNeuralStates
```
## Wave-function ansatze
```@docs
AnsatzType
LogPsi
LogPsiSignTanh

psi!
log_psi!
init_gradient_seed!
```
## Neural Ansatz
```@docs
NeuralAnsatz
MeanField
_choose_meanfield_function
TruncationBuffer
violates_truncation

SCALE_FUNCTIONS

prepare_input!
compute_logψ
compute_ψ_64
multi_compute_logψ!

neuron_statistics
jacobian_statistics
```

## Index

```@index
Pages = ["ansatz.md"]
```
