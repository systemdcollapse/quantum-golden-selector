# quantum_golden_selector.py
"""
Quantum Golden Selector - Reverse engineering Version
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import sys
import traceback
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# For IBM hardware
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
    from qiskit_ibm_runtime.accounts.exceptions import AccountNotFoundError
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# Local simulator
try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


class QuantumGoldenSelector:
    """
    Quantum circuit that selects the golden apex from a superposition of candidates.
    Text-only version with enhanced precision and corrected anomalies.
    """
    
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio constant
    
    def __init__(self, use_hardware: bool = False, backend_name: Optional[str] = None):
        """
        Initialize the Quantum Golden Selector.
        
        Args:
            use_hardware: Whether to use real quantum hardware
            backend_name: Specific backend name (e.g., 'ibm_sherbrooke')
        """
        self.use_hardware = use_hardware
        self.service = None
        self.backend_name = backend_name
        self.backend = None
        
        load_dotenv()
        
        if self.use_hardware:
            if not IBM_AVAILABLE:
                raise ImportError(
                    "IBM Quantum Runtime not installed.\n"
                    "Install with: pip install qiskit-ibm-runtime"
                )
            self._initialize_ibm_runtime()
            
    def _initialize_ibm_runtime(self):
        """Initialize connection to IBM Quantum Runtime with robust fallback logic."""
        token = os.getenv('IBM_QUANTUM_TOKEN')
        
        print("🔌 Connecting to IBM Quantum...")
        
        # Try saved credentials first
        try:
            self.service = QiskitRuntimeService()
            print("✅ Connected using saved credentials.")
            return
        except AccountNotFoundError:
            print("   No saved credentials found. Will try connecting with API token.")
        except Exception as e:
            print(f"   Warning: {e}")

        if not token:
            raise ValueError(
                "IBM_QUANTUM_TOKEN not found in .env file.\n"
                "Please add: IBM_QUANTUM_TOKEN=your_token_here"
            )
            
        # Try different channels in order of preference
        connection_configs = [
            {"channel": "ibm_cloud", "name": "IBM Cloud"},
            {"channel": "ibm_quantum", "name": "IBM Quantum (Legacy)"}
        ]
        
        connection_errors = []
        for config in connection_configs:
            channel = config["channel"]
            name = config["name"]
            print(f"   Attempting connection via '{name}' channel...")
            try:
                self.service = QiskitRuntimeService(channel=channel, token=token)
                # Save for future use
                self.service.save_account(channel=channel, token=token, overwrite=True)
                print(f"✅ Successfully connected and saved credentials for '{name}' channel.")
                return
            except Exception as e:
                error_detail = str(e)
                print(f"   ❌ Failed via '{name}': {error_detail}")
                connection_errors.append(f"'{name}': {error_detail}")
        
        raise ConnectionError(
            "Could not connect to IBM Quantum.\n\n"
            "Connection attempts:\n" + 
            "\n".join(f"- {err}" for err in connection_errors) +
            "\n\nTroubleshooting:\n"
            "1. Verify token at: https://quantum.ibm.com/account\n"
            "2. Check account status and service availability\n"
            "3. Try regenerating your API token"
        )

    def generate_golden_candidates(self, sequence: List[int], n_candidates: int = 15) -> List[Tuple[int, float]]:
        """
        Generate candidate apexes including ones that approximate φ.
        Enhanced to ensure better distribution around golden ratio.
        """
        if not sequence:
            raise ValueError("Sequence cannot be empty")
            
        reference = sequence[0]
        
        # Calculate the ideal golden apex with higher precision
        golden_apex_exact = reference * self.PHI
        golden_apex_int = int(round(golden_apex_exact))
        
        # Generate candidates with better distribution
        candidates_set = {golden_apex_int}
        
        # Add candidates near the golden apex
        for offset in [-2, -1, 1, 2]:
            candidate = golden_apex_int + offset
            if candidate > 0:
                candidates_set.add(candidate)
        
        # Add random candidates in a reasonable range
        min_apex = int(reference * 1.3)
        max_apex = int(reference * 2.0)
        
        attempts = 0
        while len(candidates_set) < n_candidates and attempts < n_candidates * 3:
            attempts += 1
            if max_apex > min_apex:
                apex = np.random.randint(min_apex, max_apex + 1)
                candidates_set.add(apex)
        
        # Ensure we have exactly n_candidates
        candidates_list = list(candidates_set)
        if len(candidates_list) < n_candidates:
            # Add sequential values if needed
            for i in range(n_candidates - len(candidates_list)):
                candidates_list.append(max_apex + i + 1)
        
        candidates = candidates_list[:n_candidates]
        np.random.shuffle(candidates)
        
        # Calculate deviations with high precision
        results = []
        for apex in candidates:
            ratio = apex / reference
            deviation = abs(ratio - self.PHI)
            results.append((apex, deviation))
            
        return results
    
    def create_grover_circuit(self, candidates: List[Tuple[int, float]], 
                              tolerance: float = 0.01) -> Tuple[QuantumCircuit, List[int]]:
        """
        Create a Grover circuit with enhanced precision.
        Reduced default tolerance for better golden ratio selection.
        """
        if not candidates:
            raise ValueError("No candidates provided")
            
        n_candidates = len(candidates)
        # Proper calculation of required qubits
        n_qubits = max(2, (n_candidates - 1).bit_length())
        
        # Ensure we don't exceed the search space
        if n_candidates > 2**n_qubits:
            n_qubits += 1
        
        # Sort by deviation for target selection
        sorted_candidates = sorted(enumerate(candidates), key=lambda x: x[1][1])
        
        # Select targets with tighter tolerance
        target_indices = []
        if sorted_candidates:
            best_deviation = sorted_candidates[0][1][1]
            # Include all candidates within tolerance
            for idx, (apex, deviation) in sorted_candidates:
                if deviation <= best_deviation + tolerance:
                    target_indices.append(idx)
                    if len(target_indices) >= min(4, n_candidates // 2):
                        break
        
        if not target_indices:
            # Fallback: at least include the best candidate
            target_indices = [sorted_candidates[0][0]]
        
        print(f"\n🎯 Quantum Oracle Configuration:")
        print(f"   Search space: {2**n_qubits} states ({n_qubits} qubits)")
        print(f"   Valid candidates: {n_candidates}")
        print(f"   Tolerance: {tolerance:.3f}")
        print(f"   Target indices: {target_indices}")
        print(f"   Target apexes: {[candidates[i][0] for i in target_indices]}")
        print(f"   Target deviations: {[f'{candidates[i][1]:.6f}' for i in target_indices]}")
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        
        # Initial superposition
        qc.h(range(n_qubits))
        
        # Calculate optimal Grover iterations
        M = len(target_indices)
        N = 2**n_qubits
        
        if M > 0 and M < N:
            # Precise calculation of iterations
            theta = np.arcsin(np.sqrt(M/N))
            optimal_iterations = int(np.pi / (4 * theta))
            # Ensure at least 1 iteration but cap at reasonable limit
            iterations = max(1, min(optimal_iterations, int(np.sqrt(N))))
        else:
            iterations = 1
        
        print(f"   Grover iterations: {iterations} (optimal for M={M}, N={N})")
        
        # Apply Grover operator
        for i in range(iterations):
            qc.barrier(label=f"Iteration_{i+1}")
            self._apply_oracle(qc, target_indices, n_qubits)
            self._apply_diffusion(qc, n_qubits)
        
        # Measurement
        qc.measure_all()
        
        return qc, target_indices
    
    def _apply_oracle(self, qc: QuantumCircuit, targets: List[int], n_qubits: int):
        """
        Apply oracle with consistent bit ordering.
        Fixed bit string handling for consistency.
        """
        for target in targets:
            if target >= 2**n_qubits:
                continue
                
            # Use consistent bit ordering (MSB to LSB)
            target_bits = format(target, f'0{n_qubits}b')
            
            # Flip qubits where bit is 0
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    qc.x(i)
            
            # Multi-controlled Z gate
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                qc.cz(0, 1)
            else:
                # Use multi-controlled Z
                qc.h(n_qubits - 1)
                if n_qubits > 2:
                    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                else:
                    qc.cx(0, 1)
                qc.h(n_qubits - 1)
            
            # Restore qubits
            for i, bit in enumerate(target_bits):
                if bit == '0':
                    qc.x(i)

    def _apply_diffusion(self, qc: QuantumCircuit, n_qubits: int):
        """Apply the Grover diffusion operator with proper implementation."""
        # Apply Hadamard gates
        qc.h(range(n_qubits))
        
        # Apply X gates
        qc.x(range(n_qubits))
        
        # Multi-controlled Z gate
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            if n_qubits > 2:
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            else:
                qc.cx(0, 1)
            qc.h(n_qubits - 1)
        
        # Restore with X gates
        qc.x(range(n_qubits))
        
        # Final Hadamard gates
        qc.h(range(n_qubits))
    
    def run_quantum_selection(self, sequence: List[int], 
                              n_candidates: int = 15,
                              shots: int = 8192,
                              tolerance: float = 0.01) -> int:
        """
        Run the complete quantum golden selection process.
        Text-only version without visualizations.
        """
        print("="*60)
        print("🌌 QUANTUM GOLDEN SELECTOR - TEXT ONLY")
        print("="*60)
        print(f"Sequence: {sequence}")
        print(f"Golden ratio φ = {self.PHI:.10f}")
        print(f"Target: Find apex ≈ {sequence[0]} × φ = {sequence[0] * self.PHI:.2f}")
        
        # Generate candidates
        candidates = self.generate_golden_candidates(sequence, n_candidates)
        
        print(f"\n📊 Generated {len(candidates)} candidates:")
        print("   Top 5 by golden ratio deviation:")
        for i, (apex, dev) in enumerate(sorted(candidates, key=lambda x: x[1])[:5]):
            ratio = apex / sequence[0]
            print(f"   {i+1}. Apex {apex}: ratio={ratio:.6f}, φ-deviation={dev:.6f}")
        
        # Create quantum circuit
        qc, target_indices = self.create_grover_circuit(candidates, tolerance)
        
        if not target_indices:
            print("\n⚠️  Warning: No target indices selected. Check tolerance setting.")
        
        # Show circuit details
        print("\n📐 Circuit Details:")
        print(f"   Qubits: {qc.num_qubits}")
        print(f"   Classical bits: {qc.num_clbits}")
        print(f"   Depth: {qc.depth()}")
        print(f"   Gate count: {dict(qc.count_ops())}")
        print(f"   Circuit size: {qc.size()}")
        
        # Show circuit diagram
        print("\n📐 Circuit Diagram:")
        print(qc.draw('text'))
        
        # Execute
        if self.use_hardware:
            counts = self._run_on_hardware(qc, shots, target_indices)
        else:
            counts = self._run_on_simulator(qc, shots)
        
        # Process results
        print(f"\n📊 Quantum Measurement Results:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Map measurements to candidates
        candidate_measurements = {}
        for bitstring, count in sorted_counts:
            # Ensure consistent bit order interpretation
            index = int(bitstring, 2)
            if index < len(candidates):
                candidate_measurements[index] = candidate_measurements.get(index, 0) + count
        
        # Show top results
        print("   Top measurements:")
        shown = 0
        for index, count in sorted(candidate_measurements.items(), 
                                  key=lambda x: x[1], reverse=True):
            if shown >= 5:
                break
            apex, dev = candidates[index]
            is_target = "🎯" if index in target_indices else "  "
            percentage = (count / shots) * 100
            print(f"   {is_target} Index {index:2d}: {count:5d} times ({percentage:5.1f}%) "
                  f"→ apex {apex:3d}, φ-dev={dev:.6f}")
            shown += 1
        
        # Select result
        if not candidate_measurements:
            raise RuntimeError("No valid measurements obtained")
        
        # Select the most measured candidate
        selected_index = max(candidate_measurements, key=candidate_measurements.get)
        selected_apex = candidates[selected_index][0]
        selected_dev = candidates[selected_index][1]
        
        print(f"\n✨ Selected Golden Apex: {selected_apex}")
        print(f"   Measurement frequency: {candidate_measurements[selected_index]/shots:.1%}")
        print(f"   Ratio: {selected_apex}/{sequence[0]} = {selected_apex/sequence[0]:.10f}")
        print(f"   φ-deviation: {selected_dev:.10f}")
        print(f"   Completes sequence: {[selected_apex] + sequence}")
        
        # Show detailed measurement statistics
        self._show_measurement_statistics(counts, candidates, target_indices, shots)
        
        return selected_apex
    
    def _show_measurement_statistics(self, counts: Dict[str, int], 
                                   candidates: List[Tuple[int, float]],
                                   target_indices: List[int], 
                                   shots: int):
        """Show detailed measurement statistics in text format."""
        print(f"\n📈 Detailed Measurement Statistics:")
        
        # Calculate hit rates
        target_hits = 0
        total_valid_measurements = 0
        
        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            if index < len(candidates):
                total_valid_measurements += count
                if index in target_indices:
                    target_hits += count
        
        if target_indices:
            target_hit_rate = target_hits / shots
            expected_random_rate = len(target_indices) / (2**len(format(len(candidates)-1, 'b')))
            
            print(f"   Oracle hit rate: {target_hit_rate:.1%} ({target_hits}/{shots})")
            print(f"   Expected random: {expected_random_rate:.1%}")
            print(f"   Quantum advantage: {target_hit_rate/expected_random_rate:.2f}x")
        
        print(f"   Valid measurements: {total_valid_measurements}/{shots} ({total_valid_measurements/shots:.1%})")
        
        # Show distribution of all measurements
        print(f"\n📊 Full Measurement Distribution:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bitstring, count in sorted_counts[:10]:  # Show top 10
            index = int(bitstring, 2)
            percentage = (count / shots) * 100
            
            if index < len(candidates):
                apex, dev = candidates[index]
                marker = "🎯" if index in target_indices else "  "
                print(f"   {marker} |{bitstring}⟩ → Index {index:2d} (apex {apex:3d}): "
                      f"{count:4d} ({percentage:4.1f}%)")
            else:
                print(f"      |{bitstring}⟩ → Invalid index {index}: {count:4d} ({percentage:4.1f}%)")
    
    def _run_on_simulator(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Run circuit on local Aer simulator."""
        if not AER_AVAILABLE:
            raise ImportError("AerSimulator not available. Install qiskit-aer.")
            
        print(f"\n⚡ Executing on local simulator with {shots} shots...")
        backend = AerSimulator()
        
        # Transpile for optimization even on simulator
        qc_transpiled = transpile(qc, backend, optimization_level=1)
        
        print(f"   Transpiled circuit depth: {qc_transpiled.depth()}")
        print(f"   Transpiled gate count: {dict(qc_transpiled.count_ops())}")
        
        job = backend.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        print(f"   ✅ Simulation completed")
        return counts
    
    def _select_best_backend(self, min_qubits: int):
        """
        Select the best available backend.
        Properly handles backend_name if specified.
        """
        print("\n🔬 Selecting quantum backend...")
        
        # If specific backend requested, try to use it
        if self.backend_name:
            try:
                backend = self.service.backend(self.backend_name)
                status = backend.status()
                config = backend.configuration()
                
                if not status.operational:
                    raise RuntimeError(f"Backend {self.backend_name} is not operational")
                
                if config.n_qubits < min_qubits:
                    raise RuntimeError(
                        f"Backend {self.backend_name} has only {config.n_qubits} qubits, "
                        f"but {min_qubits} are required"
                    )
                
                print(f"✅ Using requested backend: {self.backend_name}")
                print(f"   Status: Operational, Queue: {status.pending_jobs} jobs")
                print(f"   Qubits: {config.n_qubits}, Max shots: {config.max_shots}")
                return backend
                
            except Exception as e:
                print(f"⚠️  Could not use requested backend {self.backend_name}: {e}")
                print("   Falling back to automatic selection...")
        
        # Automatic backend selection
        backends = self.service.backends(simulator=False, operational=True)
        
        suitable_backends = []
        for backend in backends:
            try:
                status = backend.status()
                config = backend.configuration()
                
                if not status.operational or config.n_qubits < min_qubits:
                    continue
                
                # Simple scoring based on queue and size
                score = status.pending_jobs + (config.n_qubits - min_qubits) * 0.1
                suitable_backends.append((score, backend))
                
            except Exception:
                continue
        
        if not suitable_backends:
            raise RuntimeError(f"No suitable backends found with at least {min_qubits} qubits")
        
        # Select backend with lowest score
        _, best_backend = min(suitable_backends, key=lambda x: x[0])
        print(f"✅ Selected backend: {best_backend.name}")
        
        return best_backend

    def _run_on_hardware(self, qc: QuantumCircuit, shots: int, 
                        target_indices: List[int]) -> Dict[str, int]:
        """Run circuit on IBM Quantum hardware."""
        backend = self._select_best_backend(qc.num_qubits)
        
        print(f"\n⚡ Executing on {backend.name}...")
        print(f"   Backend qubits: {backend.configuration().n_qubits}")
        print(f"   Backend basis gates: {backend.configuration().basis_gates}")
        
        # Transpile
        print("   Transpiling circuit...")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"   Original depth: {qc.depth()}")
        print(f"   Transpiled depth: {qc_transpiled.depth()}")
        print(f"   Original gates: {dict(qc.count_ops())}")
        print(f"   Transpiled gates: {dict(qc_transpiled.count_ops())}")
        
        # Execute
        try:
            with Session(backend=backend) as session:
                sampler = SamplerV2()
                
                print(f"   Submitting job with {shots} shots...")
                job = sampler.run([qc_transpiled], shots=shots)
                print(f"   Job ID: {job.job_id()}")
                print("   Waiting for results...")
                
                result = job.result()
                
                # Extract counts
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                
                print("   ✅ Hardware execution completed")
                
                # Performance analysis
                if target_indices:
                    target_counts = sum(
                        counts.get(format(idx, f'0{qc.num_qubits}b'), 0) 
                        for idx in target_indices
                    )
                    hit_rate = target_counts / shots
                    expected_rate = len(target_indices) / (2**qc.num_qubits)
                    
                    print(f"\n📈 Hardware Performance Analysis:")
                    print(f"   Oracle hit rate: {hit_rate:.1%}")
                    print(f"   Expected random: {expected_rate:.1%}")
                    print(f"   Quantum advantage: {hit_rate/expected_rate:.2f}x")
                
                return counts
                
        except Exception as e:
            raise RuntimeError(f"Hardware execution failed: {e}") from e


def main():
    """Main demo with enhanced error handling and text-only output."""
    # Parse arguments
    use_hardware = '--hardware' in sys.argv
    
    # Verify dependencies
    if not IBM_AVAILABLE and use_hardware:
        print("❌ IBM Quantum Runtime not installed.")
        print("   Install with: pip install qiskit-ibm-runtime")
        return 1
    
    if not AER_AVAILABLE and not use_hardware:
        print("❌ qiskit-aer not installed.")
        print("   Install with: pip install qiskit-aer")
        return 1
    
    # Run demonstration
    sequence = [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]
    
    try:
        print(f"\n🔧 Initializing Quantum Golden Selector...")
        print(f"   Mode: {'IBM Quantum Hardware' if use_hardware else 'Local Simulator'}")
        
        # Check for backend specification
        backend_name = None
        for arg in sys.argv:
            if arg.startswith('--backend='):
                backend_name = arg.split('=')[1]
                print(f"   Requested backend: {backend_name}")
        
        selector = QuantumGoldenSelector(use_hardware=use_hardware, 
                                       backend_name=backend_name)
        
        # Run with tight tolerance for precision
        golden_apex = selector.run_quantum_selection(
            sequence, 
            n_candidates=16,
            shots=8192,
            tolerance=0.005  # Tighter tolerance for better precision
        )
        
        print("\n" + "="*60)
        print("🎉 Quantum computation complete!")
        print(f"The next number in the sequence is: {golden_apex}")
        print(f"Expected (classical): {int(round(sequence[0] * selector.PHI))}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())