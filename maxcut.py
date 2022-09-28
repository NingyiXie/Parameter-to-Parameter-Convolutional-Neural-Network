import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, execute
from qaoalib.utils import maxcut_brute
from qaoalib.math import fast_kron


def qaoa_circuit(G, params):

    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]

    q = QuantumRegister(len(G.nodes))
    qc = QuantumCircuit(q)

    qc.h(q)
    for p in range(depth):
        for u, v, d in G.edges(data=True):
            qc.barrier()
            qc.cx(u, v)
            if 'weight' in d.keys():
                w = d['weight']
            else:
                w = 1
            qc.rz(-gamma[p]*w, v)
            qc.cx(u, v)
        qc.barrier()
        qc.rx(2*beta[p], q)

    return qc

def get_stete(params, graph):
    backend = Aer.get_backend('statevector_simulator')
    state = execute(qaoa_circuit(
        graph, params).reverse_bits(), backend).result().get_statevector()
    state = state.reshape((-1, 1))
    return state


def FQAOA(params,graph):  # calculate expected value
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
        
    right = get_stete(params, graph)
    sum_ = 0
    for u, v, d in graph.edges(data=True):
        edge = (u, v)
        kron_list = [Z if i in edge else I for i in range(len(graph.nodes))]
        sum_ += d.get("weight", 1) * (1 - (right.conj().T @ fast_kron(kron_list, right)).item().real)
        
    return sum_/2

def Cmax(graph):  # using brute approach solve the maxcut
    true_obj,_ = maxcut_brute(graph)
    return true_obj

class QAOA_MaxCut:
    def __init__(self, graph, optimizer='L-BFGS-B', gamma_bounds=(0, np.pi), beta_bounds=(0, np.pi/2)):
        
        self.graph = graph

        self.gamma_bounds = gamma_bounds
        self.beta_bounds = beta_bounds
        self.optimizer = optimizer

        self.optimized_params = []
        self.optimized_expected_value = 0

    def optimize_fun(self):
        def f(params):
            return -FQAOA(params,self.graph)
        return f

    def run(self,p,init_param):
        bnds=[]
        for _ in range(p):
            bnds.insert(0,self.gamma_bounds)
            bnds.append(self.beta_bounds)

        if len(init_param)//2 == p:
            print("Optimizing p = ",p,'    ......')
            x0 = np.asarray(init_param)
            optimized_res = minimize(self.optimize_fun(), x0, method=self.optimizer, bounds=bnds)
            self.optimized_expected_value = -optimized_res.fun
            self.optimized_params = list(optimized_res.x)

