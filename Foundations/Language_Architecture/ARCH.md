# _A New Belief System_
_Random Matrix Theory Applied to Deep Belief Signaling Networks_

---
## [A Formal Belief System](arch_start/ARCH_0/ARCH_0.md)

<img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;\mbox{Unity&space;of&space;Knowledge}&space;=&space;I(ABB_{openAI_{GYM}}(GAN;&space;NAS;&space;ME;&space;MLE;&space;AI))\\&space;Inference&space;=&space;Belief(I(ABB_{openAI_{GYM}}(GAN;&space;NAS;&space;ME;&space;MLE;&space;AI)))&space;\\&space;Z&space;\rightarrow&space;Inference&space;\\&space;\mbox{Then,&space;to&space;learn&space;on&space;the&space;inference:}&space;\\&space;min_{T}{\sum_{i}&space;\sum_{j}&space;T_{ij}&space;Z_{ij}}&space;\sum_{i}&space;\sum_{j}&space;T_{ij}Z_{ij}&space;&plus;&space;\dfrac{a}{2}&space;||T||^{2}_{F}&space;&plus;&space;\dfrac{a}{2}&space;||T||^{2}_{2}s.t.&space;||T||&space;=&space;n&space;\end{align}" title="\begin{align} \mbox{Unity of Knowledge} = I(ABB_{openAI_{GYM}}(GAN; NAS; ME; MLE; AI))\\ Inference = Belief(I(ABB_{openAI_{GYM}}(GAN; NAS; ME; MLE; AI))) \\ Z \rightarrow Inference \\ \mbox{Then, to learn on the inference:} \\ min_{T}{\sum_{i} \sum_{j} T_{ij} Z_{ij}} \sum_{i} \sum_{j} T_{ij}Z_{ij} + \dfrac{a}{2} ||T||^{2}_{F} + \dfrac{a}{2} ||T||^{2}_{2}s.t. ||T|| = n \end{align}" />


## Python and Qiskit Implementation
```python
# Run inference on information shared between random populations of...
belief_prop = bp.random(population, environments, neural_architectures: neural_ode, gan, cnn, rnn; depth: multi, ...)

# Analyze intersection of neural architectures and environments(graph signal processing)
GSP.engine(analysis(union for belief_prop), algo_seq: [forward, backward, forward])
```

---
`GSP.engine` can be further optimized through quantum topological search:

```python
# Initialization
import matplotlib.pyplot as plt
import numpy as np

# Importing Qiskit
from qiskit import IBMQ, Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.providers.ibmq import least_busy
from qiskit.quantum_info import Statevector

# Import basic plot tools
from qiskit.visualization import plot_histogram

# Initialize quantum components
n = 2 # qubits
grover_circuit = QuantumCircuit(n)
grover_circuit = initialize_s(grover_circuit, [0,1])
grover_circuit.draw()

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    for q in qubits:
        qc.h(q)
    return qc
```

![circuit](circuit.png)
.
.
.
.
. 
And so on

_See references for quantum computing, graph signal processing, and belief propagation:_
[📖](references.md)