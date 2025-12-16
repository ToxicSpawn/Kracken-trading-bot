"""Quantum-ready optimizations."""
from typing import Dict, List, Optional
from strategies.base_strategy import BaseStrategy
from utils.logger import logger

try:
    from qiskit import Aer
    from qiskit.algorithms import QAOA, VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.utils import QuantumInstance
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available, quantum strategies will be disabled")

try:
    from qiskit_finance.applications.optimization import PortfolioOptimization
    QISKIT_FINANCE_AVAILABLE = True
except ImportError:
    QISKIT_FINANCE_AVAILABLE = False


class QuantumPortfolioOptimizer(BaseStrategy):
    """Quantum portfolio optimizer using QAOA."""
    
    def __init__(self, config: Dict, assets: List[str], expected_returns: List[float], 
                 covariance_matrix: List[List[float]]):
        super().__init__(config, "Quantum_Portfolio_Optimizer")
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum strategies")
        
        self.assets = assets
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        quantum_config = config.get("strategies", {}).get("quantum", {})
        self.risk_factor = quantum_config.get("risk_factor", 0.5)
        self.budget = quantum_config.get("budget", 1)
        self.quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    
    def optimize(self) -> Optional[Dict]:
        """Use QAOA to optimize portfolio allocation."""
        if not QISKIT_AVAILABLE or not QISKIT_FINANCE_AVAILABLE:
            return None
        
        try:
            # Define the portfolio optimization problem
            problem = PortfolioOptimization(
                expected_returns=self.expected_returns,
                covariances=self.covariance_matrix,
                risk_factor=self.risk_factor,
                budget=self.budget
            )
            
            # Convert to QUBO
            qubo = problem.to_quadratic_program()
            
            # Set up QAOA
            optimizer = COBYLA(maxiter=100)
            qaoa = QAOA(optimizer=optimizer, reps=3, quantum_instance=self.quantum_instance)
            
            # Run QAOA
            result = qaoa.compute_minimum_eigenvalue(qubo)
            
            # Interpret the result
            optimal_weights = problem.interpret(result)
            
            self.log(f"Quantum-optimized portfolio weights: {optimal_weights}")
            
            return {
                "action": "portfolio_optimize",
                "weights": dict(zip(self.assets, optimal_weights)),
                "confidence": 1.0
            }
        except Exception as e:
            self.log(f"Error in quantum portfolio optimization: {e}")
            return None
    
    async def run(self, _=None) -> Optional[Dict]:
        """Run the quantum portfolio optimizer."""
        return self.optimize()


class QuantumArbitrageOptimizer(BaseStrategy):
    """Quantum arbitrage optimizer using VQE."""
    
    def __init__(self, config: Dict, arbitrage_opportunities: List[Dict]):
        super().__init__(config, "Quantum_Arbitrage_Optimizer")
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum strategies")
        
        self.arbitrage_opportunities = arbitrage_opportunities
        self.quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'))
    
    def _build_qubo(self) -> Dict:
        """Convert arbitrage problem to QUBO."""
        num_opportunities = len(self.arbitrage_opportunities)
        qubo = {}
        
        # Add terms for each arbitrage opportunity
        for i, opp in enumerate(self.arbitrage_opportunities):
            qubo[(i, i)] = -opp.get("profit_pct", 0)  # Reward for taking profitable paths
        
        # Add penalty for taking too many paths
        for i in range(num_opportunities):
            for j in range(i+1, num_opportunities):
                qubo[(i, j)] = 10.0  # Penalty for taking multiple paths
        
        return qubo
    
    def optimize(self) -> Optional[Dict]:
        """Use VQE to find the best arbitrage path."""
        if not QISKIT_AVAILABLE:
            return None
        
        try:
            # Build QUBO
            qubo = self._build_qubo()
            
            # Convert QUBO to Ising Hamiltonian
            from qiskit.opflow import PauliSumOp
            num_qubits = len(self.arbitrage_opportunities)
            
            pauli_op = None
            for (i, j), coeff in qubo.items():
                if i == j:
                    pauli_str = "I" * i + "Z" + "I" * (num_qubits - i - 1)
                    if pauli_op is None:
                        pauli_op = PauliSumOp.from_list([(pauli_str, coeff)])
                    else:
                        pauli_op += PauliSumOp.from_list([(pauli_str, coeff)])
            
            if pauli_op is None:
                return None
            
            # Set up VQE
            optimizer = SPSA(maxiter=100)
            ansatz = RealAmplitudes(reps=3)
            vqe = VQE(ansatz, optimizer, quantum_instance=self.quantum_instance)
            
            # Run VQE
            result = vqe.compute_minimum_eigenvalue(pauli_op)
            
            # Interpret the result
            best_path_indices = self._interpret_result(result)
            best_paths = [self.arbitrage_opportunities[i] for i in best_path_indices]
            
            self.log(f"Quantum-optimized arbitrage paths: {best_paths}")
            
            return {
                "action": "quantum_arbitrage",
                "paths": best_paths,
                "confidence": 1.0
            }
        except Exception as e:
            self.log(f"Error in quantum arbitrage optimization: {e}")
            return None
    
    def _interpret_result(self, result) -> List[int]:
        """Convert VQE result to arbitrage path indices."""
        try:
            optimal_state = result.eigenstate
            best_indices = []
            
            # Find which qubits are in the |1> state
            for i, amplitude in enumerate(optimal_state):
                if abs(amplitude) > 0.1:  # Threshold for considering a qubit as |1>
                    best_indices.append(i)
            
            return best_indices
        except Exception:
            return []
    
    async def run(self, _=None) -> Optional[Dict]:
        """Run the quantum arbitrage optimizer."""
        return self.optimize()

