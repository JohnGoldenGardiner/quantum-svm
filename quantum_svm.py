###############################################################################
# Quantum SVM Builder
#
# Author: John Gardiner <johngoldengardiner@gmail.com>
#
# Description: Provides the class QuantumSVM which builds an SVM that uses
#     a quantum circuit to calculate the kernel function.
#
# Usage: See qsvm_demo.ipynb in this repository for usage examples.
###############################################################################


from typing import Callable
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit import execute
from qiskit.primitives import Sampler
from quadprog import solve_qp


class QuantumSVM:
    """
    Implements a quantum SVM. In this case that is an SVM whose feature
    maps are described by quantum circuits and whose kernel entries are 
    obtained via a quantum computation. More precisely, each datapoint 
    :math:`\\vec{x}` is mapped to a quantum state 
    :math:`U(\\vec{x})\left|0\\right>`, and the kernel is given by
    .. math::

        K(\\vec{x}_i, \\vec{x}_j) = \left| \left<0\\right| U(\\vec{x}_i)^\dag 
        U(\\vec{x}_j) \left|0\\right> \\right|^2.

    The kernel elements are computed by sampling from the probability 
    distribution resulting from acting on $\left|0\\right>$ with the 
    circuit $U(\\vec{x}_i)^\dag U(\\vec{x}_j)$. The circuit is implemented 
    either by simulation or on a quantum computer.
    """

    def __init__(self, num_qubits=4, num_layers=2, diagonal_gates=None,
                 parameter_map=None, sampler=None):
        """
        Args:
            num_qubits: Number of qubits in feature map circuit
            num_layers: Number of repetitions of Hadamard change of basis
                followed by diagonal parametrized unitary
            diagonal_gates: list of lists. Each entry in the list is a
                list of qubits on which to apply a e^{-i theta/2 Z...Z} gate
            parameter_map: A function that takes as input a datapoint and
                outputs rotation angles to be used as parameters theta of
                the gates listed in `diagonal_map`
            sampler: a Qiskit or Qiskit Runtime sampler primitive used to
                find kernel entries from circuits
        """

        self.num_qubits = num_qubits

        # Specify default diagonal gates if none provided
        if diagonal_gates is None:
            diagonal_gates = [[i] for i in range(num_qubits)]
            diagonal_gates += [
                [i, i + 1]
                for i in range(num_qubits - 1)
                if i % 2 == 0
            ]
            diagonal_gates += [
                [i, i + 1]
                for i in range(num_qubits - 1)
                if i % 2 == 1
            ]
        self.diagonal_gates = diagonal_gates

        # Specify a default map from datapoints to parameters if none provided
        if parameter_map is None:

            def parameter_map(x):
                dimension = len(x)
                angles = []
                for i, gate in enumerate(diagonal_gates):
                    angle = 1
                    for j in range(len(gate)):
                        angle *= (0.5 - x[(i + j) % dimension])
                    angles.append(angle)
                angles = 2*np.pi*np.array(angles)
                return angles
        
        self.parameter_map = parameter_map

        # Specify default sampler if none provided
        if sampler is None:
            sampler = Sampler(options={'shots': 1e12})
        self.sampler = sampler

        # Build ansatz circuit
        theta = ParameterVector('theta', len(diagonal_gates))
        ansatz = QuantumCircuit(num_qubits)
        for _ in range(num_layers):
           
            # Hadamard change of basis
            for qubit in range(num_qubits):
                ansatz.h(qubit)
           
            # Add e^{-i \frac{\theta_j}{2} Z...Z} gates
            for j, gate in enumerate(diagonal_gates):
                for i in range(1, len(gate)):
                    ansatz.cx(gate[i - 1], gate[i])
                ansatz.rz(theta[j], gate[-1])
                for i in range(1, len(gate)):
                    ansatz.cx(gate[-i - 1], gate[-i])
                
        self.ansatz = ansatz


    def feature_map_circuit(self, x):
        """
        Takes as input a datapoint and outputs the circuit preparing the
        corresponding point in feature space.

        Args:
            x: datapoint as an ndarray

        Returns:
            QuantumCircuit
        """
        parameter_map = self.parameter_map
        angles = parameter_map(x)
        if len(angles) != self.ansatz.num_parameters:
            raise IndexError(
                'Number of parameters must match number of parametrized gates'
            )
        return self.ansatz.bind_parameters(angles)


    def build_kernel(self, *args):
        """
        Takes as input either one or two ndarrays describing datapoints.
        If two arguments X1 and X2 then outputs an array containing the 
        kernel entries between points in X1 and points in X2. If only one
        argument X is given then outputs kernel entries between any two 
        points in X.
        """

        if len(args) not in [1, 2]:
            raise TypeError(
                f'build_kernel takes 1 or 2 arrays as positional arguments '
                f'but {len(args)} were given'
            )
        
        if len(args) == 2:

            X1, X2 = args

            classical_register = ClassicalRegister(self.num_qubits)
            circuit_list = []
            entry = 1
            num_entries = X1.shape[0]*X2.shape[0]
            for x1 in X1:
                for x2 in X2:
                    print(f'Preparing circuit {entry} of {num_entries}', end='\r')
                    circuit1 = self.feature_map_circuit(x1)
                    circuit2 = self.feature_map_circuit(x2).inverse()
                    qubits = circuit1.qubits
                    circuit = circuit1.compose(circuit2, qubits=qubits)
                    circuit.add_register(classical_register)
                    circuit.measure(qubits, classical_register)
                    circuit_list.append(circuit)
                    entry += 1
            
            print('Running circuits' + 30*' ', end='\r')
            job = self.sampler.run(circuit_list)
            dists = job.result().quasi_dists
            kernel = np.array([dist.get(0) or 0 for dist in dists])
            kernel = kernel.reshape((X1.shape[0], X2.shape[0]))
            print('Job completed' + 30*' ', end='\r')

        elif len(args) == 1:

            X = args[0]

            classical_register = ClassicalRegister(self.num_qubits)
            circuit_list = []
            entry = 1
            num_entries = X.shape[0]*(X.shape[0] - 1)//2
            for i in range(X.shape[0]):
                for j in range(i):
                    print(f'Preparing circuit {entry} of {num_entries}', end='\r')
                    circuit1 = self.feature_map_circuit(X[i])
                    circuit2 = self.feature_map_circuit(X[j]).inverse()
                    qubits = circuit1.qubits
                    circuit = circuit1.compose(circuit2, qubits=qubits)
                    circuit.add_register(classical_register)
                    circuit.measure(qubits, classical_register)
                    circuit_list.append(circuit)
                    entry += 1

            print('Running circuits' + 30*' ', end='\r')
            job = self.sampler.run(circuit_list)
            dists = job.result().quasi_dists
            probs = [dist.get(0) or 0 for dist in dists]

            kernel = np.eye(X.shape[0])/2
            ind = 0
            for i in range(X.shape[0]):
                for j in range(i):
                    kernel[i, j] = probs[ind]
                    ind += 1
            kernel = kernel + kernel.T

            print('Job completed' + 30*' ', end='\r')
        
        return kernel


    def train(self, X, Y, regularizer=10.0):
        """
        Args:
            X: ndarray of training datapoints
            Y: 1d ndarray of classifications, either -1 or 1.
            regularizer: An upper constraint on coefficient values. 
                Correspond to allowing slack variables where the penalty 
                for a misclassified training point is proportional to 
                regularizer.
        """
        self.X_train = X
        self.Y_train = Y
        self.regularizer = regularizer
        self.kernel_train = self.build_kernel(X)

        # Make kernel positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.kernel_train)
        tol = 5e-15
        eigenvals[eigenvals < tol] = tol
        kernel_posdef = eigenvecs*eigenvals @ eigenvecs.conj().T

        # quadprog.solve_qp minimizes 1/2 x^T Q x - q ^ T x subject to
        # A^T x >= b where >= is equality on the first meq
        Q = Y[:, np.newaxis]*kernel_posdef*Y
        q = np.ones(len(Y))
        A = np.concatenate(
            (Y[:, np.newaxis], np.eye(len(Y)), -np.eye(len(Y))),
            axis=1
        )
        b = np.concatenate((np.zeros(1 + len(Y)), -regularizer*np.ones(len(Y))))
        
        solution = solve_qp(Q, q, A, b, meq=1)
        self.coeffs = solution[0]

        # Find support vectors
        tol = 1e-7
        self.support_inds = [
            i for i, coeff in enumerate(self.coeffs)
            if coeff > tol
        ]
        self.margin_inds = [
            i for i, coeff in enumerate(self.coeffs)
            if coeff > tol and coeff < regularizer - tol
        ]
        self.support_coeffs = self.coeffs[self.support_inds]
        self.support_vectors = self.X_train[self.support_inds]

        self.bias = (
            np.sum(
                self.kernel_train[self.margin_inds][:, self.support_inds]
                *self.Y_train[self.support_inds]*self.support_coeffs
            )
            - np.sum(self.Y_train[self.margin_inds])
        )/len(self.margin_inds)


    def predict(self, X, score=False):
        """
        Args:
            X: points to be classified
            score: If set to False this returns the classification as 1 
                or -1 for each datapoint. If set to True the raw score 
                (distance from the decision boundary in feature-space) 
                is returned for each datapoint.

        Returns:
            Classification, either 1 or -1 if score=False. If score=True
                returns the signed distance from the decision boundary 
                in feature-space.
        """
        
        kernel = self.build_kernel(X, self.support_vectors)
        Y_predicted = np.sum(
            kernel*self.support_coeffs*self.Y_train[self.support_inds], axis=1
        ) - self.bias

        if score == False:
            Y_predicted = np.array([1 if y > 0 else -1 for y in Y_predicted])

        return Y_predicted