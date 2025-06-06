import pennylane as qml
import torch.nn as nn
import torch
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = "256"

n_layers = 1  # Number of circuit layers
n_qubits_1 = 4  # Number of qubits for the upper part
n_qubits_2 = 5  # Number of qubits for the lower part

dev_1 = qml.device('lightning.qubit', wires=n_qubits_1, batch_obs = True) # Device for the upper part
dev_2 = qml.device('lightning.qubit', wires=n_qubits_2, batch_obs = True) # Device for the lower part

theta1 = np.pi/3
theta2 = -np.pi/3

# Construct the encoding circuit
def embedding_circuit(inputs, n_qubits):
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)

def mps_upper(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits-1):
        
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index] * theta1, wires=i)
        qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(np.pi / 2, wires=i)

        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index + 3] * theta2, wires=i)
        qml.RY(weights[layer, param_index + 4] * theta2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 5] * theta2, wires=(i + 1) % n_qubits)
        param_index += 6

def mps_lower(layer, n_qubits, weights):
    param_index = 0
    for i in range(n_qubits-1):
        
        qml.RX(-np.pi / 2, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(weights[layer, param_index] * theta1, wires=i)
        qml.RY(weights[layer, param_index + 1] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
        qml.RY(weights[layer, param_index + 2] * theta1, wires=(i + 1) % n_qubits)
        qml.CNOT(wires=[(i + 1) % n_qubits, i])
        qml.RX(np.pi / 2, wires=i)

        param_index += 3 

# Construct circuit 1 (without measurement, last qubit is the split point)
def build_circuits1(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        mps_upper(layer, qubits, weights)

# Construct circuit 2 (without measurement, last qubit is the split point)
def build_circuits2(inputs, weights, qubits):
    embedding_circuit(inputs, qubits)
    for layer in range(n_layers):
        mps_lower(layer, qubits, weights)

############################################################################
# Measurement
@qml.qnode(dev_1, interface="torch")
def circuit_front_1(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    return qml.probs(wires=n_qubits_1 - 1)


@qml.qnode(dev_1, interface="torch")
def circuit_front_2(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    qml.Hadamard(wires=n_qubits_1 - 1)
    return qml.probs(wires=n_qubits_1 - 1)


@qml.qnode(dev_1, interface="torch")
def circuit_front_3(inputs, weights):
    build_circuits1(inputs, weights, n_qubits_1)
    qml.RX(np.pi / 2, wires=n_qubits_1 - 1)
    return qml.probs(wires=n_qubits_1 - 1)

############################################################################
# Preparation
# Define quantum nodes
@qml.qnode(dev_2, interface="torch")
def circuit_back_1(inputs, weights):
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_2(inputs, weights):
    qml.PauliX(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_3(inputs, weights):
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_4(inputs, weights):
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_5(inputs, weights):
    qml.RX(np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


@qml.qnode(dev_2, interface="torch")
def circuit_back_6(inputs, weights):
    qml.RX(-np.pi / 2, wires=0)
    build_circuits2(inputs, weights, n_qubits_2)
    return qml.probs(wires=[i for i in range(n_qubits_2)])


def calculate_probabilities(tensor):
    # Assuming tensor shape [batch_size, num_probs]
    # Even indices are for event 0, odd for event 1
    prob_event_0 = tensor[:, 0::2].sum(dim=1)  # Sum over even indices
    prob_event_1 = tensor[:, 1::2].sum(dim=1)  # Sum over odd indices

    # Stack results into [batch_size, 2]
    result_tensor = torch.stack([prob_event_0,prob_event_1], dim=1)
    return result_tensor


def combine_outputs(probabilities_front_1, probabilities_front_2, probabilities_front_3, output_back_1, output_back_2, output_back_3, 
                    output_back_4, output_back_5, output_back_6):
    # Stack all back-end outputs
    res2 = torch.stack([output_back_1, output_back_2, output_back_3,
                        output_back_4, output_back_5, output_back_6], dim=1) # torch.Size([2**n_qubits_2]) => torch.Size([batch_size, 6, 2**n_qubits_2])

    result_1 = torch.cat((probabilities_front_1, probabilities_front_2, probabilities_front_3), dim=1) # torch.Size([batch_size, 6])

    # Calculate nz, nx, ny
    nz = result_1[:, 0] - result_1[:, 1] # torch.Size([batch_size])
    nx = result_1[:, 2] - result_1[:, 3]
    ny = result_1[:, 4] - result_1[:, 5]

    # Calculate c coefficients
    c0 = (1 + nz) / 2 # torch.Size([batch_size])
    c1 = (1 - nz) / 2
    c2 = nx / 2
    c3 = -c2
    c4 = ny / 2
    c5 = -c4

    c_list = torch.stack([c0, c1, c2, c3, c4, c5], dim=1) # torch.Size([batch_size, 6])

    result_list = torch.zeros_like(output_back_1) # torch.Size([batch_size, 2**n_qubits_2])

    for j in range(c_list.shape[1]):
        result_list += c_list[:, j].unsqueeze(1) * res2[:, j, :]

    return result_list

        
class LearnableScaledLayer(nn.Module):
    def __init__(self, initial_scale=5.0):
        super(LearnableScaledLayer, self).__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, x):
        return x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * self.scale


class QNNSplited(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim=512, with_shortcut=False, drop_out = 0):
        super(QNNSplited, self).__init__()

        self.mlp_input = nn.Sequential(nn.Linear(input_shape,hidden_dim), nn.Linear(hidden_dim, n_qubits_1 + n_qubits_2 - 1))
        self.drop_out = nn.Dropout(drop_out)
        self.mlp_output = nn.Linear(2**n_qubits_2, output_shape)
        self.mlp_withshortcut = nn.Linear(2**n_qubits_2 + input_shape, output_shape)
        # Define shared weight tensor
        self.shared_weights_1 = nn.Parameter(torch.rand((n_layers, (n_qubits_1 - 1) * 6)), requires_grad=True)
        weight_shapes_1 = {"weights": (n_layers, 2 * (n_qubits_1 - 1) * 3)} # Specify weights shape
        self.QLayer_front_1 = qml.qnn.TorchLayer(circuit_front_1, weight_shapes_1)  # QNode to TorchLayer
        self.QLayer_front_2 = qml.qnn.TorchLayer(circuit_front_2, weight_shapes_1)  # QNode to TorchLayer
        self.QLayer_front_3 = qml.qnn.TorchLayer(circuit_front_3, weight_shapes_1)  # QNode to TorchLayer
        self.QLayer_front_1.weights.data = self.shared_weights_1
        self.QLayer_front_2.weights.data = self.shared_weights_1
        self.QLayer_front_3.weights.data = self.shared_weights_1

        self.shared_weights_2 = nn.Parameter(torch.rand((n_layers, (n_qubits_2 - 1) * 3)), requires_grad=True)
        weight_shapes_2 = {"weights": (n_layers, (n_qubits_2 - 1) * 3)} # Specify weights shape
        self.QLayer_back_1 = qml.qnn.TorchLayer(circuit_back_1, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_2 = qml.qnn.TorchLayer(circuit_back_2, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_3 = qml.qnn.TorchLayer(circuit_back_3, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_4 = qml.qnn.TorchLayer(circuit_back_4, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_5 = qml.qnn.TorchLayer(circuit_back_5, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_6 = qml.qnn.TorchLayer(circuit_back_6, weight_shapes_2)  # QNode to TorchLayer
        self.QLayer_back_1.weights.data = self.shared_weights_2
        self.QLayer_back_2.weights.data = self.shared_weights_2
        self.QLayer_back_3.weights.data = self.shared_weights_2
        self.QLayer_back_4.weights.data = self.shared_weights_2
        self.QLayer_back_5.weights.data = self.shared_weights_2
        self.QLayer_back_6.weights.data = self.shared_weights_2

        self.tanh = nn.Tanh()
        self.activate = nn.Sigmoid()
        self.scaler = LearnableScaledLayer()
        self.scaler_input = LearnableScaledLayer()
        self.with_shortcut = with_shortcut

    def forward(self, inputs): # inputs = stack(batch_size, x], dim=0)
        
        if self.with_shortcut:
            shortcut_x = inputs
        
        inputs = self.mlp_input(inputs) #torch.Size([batch_size, Qbits])
        inputs = self.drop_out(inputs)


        inputs = self.scaler_input(inputs)
        output_front_1 =[]
        output_front_2 =[]
        output_front_3 =[]

        output_back_1 =[]
        output_back_2 =[]
        output_back_3 =[]
        output_back_4 =[]
        output_back_5 =[]
        output_back_6 =[]

        for i in inputs:
            output_front_1.append(self.QLayer_front_1(i[:n_qubits_1]))
            output_front_2.append(self.QLayer_front_2(i[:n_qubits_1]))
            output_front_3.append(self.QLayer_front_3(i[:n_qubits_1]))
            output_back_1.append(self.QLayer_back_1(i[n_qubits_1-1:]))
            output_back_2.append(self.QLayer_back_2(i[n_qubits_1-1:]))
            output_back_3.append(self.QLayer_back_3(i[n_qubits_1-1:]))
            output_back_4.append(self.QLayer_back_4(i[n_qubits_1-1:]))
            output_back_5.append(self.QLayer_back_5(i[n_qubits_1-1:]))
            output_back_6.append(self.QLayer_back_6(i[n_qubits_1-1:]))
            
        output_front_1 = torch.stack(output_front_1, dim=0)
        output_front_2 = torch.stack(output_front_2, dim=0)
        output_front_3 = torch.stack(output_front_3, dim=0)
        output_back_1 = torch.stack(output_back_1, dim=0)
        output_back_2 = torch.stack(output_back_2, dim=0)
        output_back_3 = torch.stack(output_back_3, dim=0)
        output_back_4 = torch.stack(output_back_4, dim=0)
        output_back_5 = torch.stack(output_back_5, dim=0)
        output_back_6 = torch.stack(output_back_6, dim=0)

        combined_outputs = combine_outputs(output_front_1, output_front_2, output_front_3,
                                                output_back_1, output_back_2, output_back_3,
                                                output_back_4, output_back_5, output_back_6).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        combined_outputs = self.scaler(combined_outputs)

        # Apply final linear layer and activation
        
        if self.with_shortcut:
            outputs = torch.cat((shortcut_x, combined_outputs), dim=1)
            output_tensor = self.mlpwithshortcut(outputs)            
        else:
            combined_outputs = self.drop_out(combined_outputs)
            output_tensor = self.mlp_output(combined_outputs)
        
        result = output_tensor
        return result
