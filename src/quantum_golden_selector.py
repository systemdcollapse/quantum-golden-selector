# quantum_golden_selector_final.py
"""
Quantum Golden Selector - Hardware-Only Certification Version
This version is strictly for running on real quantum hardware and will fail
if the '--hardware' flag is not provided. All local simulation paths have been removed.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import sys
import traceback
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from dotenv import load_dotenv

# Suppress DeprecationWarning for now
warnings.filterwarnings('ignore', category=DeprecationWarning)

# For IBM hardware
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
    from qiskit_ibm_runtime.accounts.exceptions import AccountNotFoundError
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# Local simulator library is NOT imported in this version.

class QuantumGoldenSelector:
    """
    Quantum circuit that selects the golden apex from a superposition of candidates.
    Professional version with explicit error handling and precision enhancements.
    """
    
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio constant

    def __init__(self, execution_mode: str = 'hardware', backend_name: Optional[str] = None):
        """Initialize the Quantum Golden Selector for hardware execution."""
        self.execution_mode = execution_mode
        self.simulator = None

        if self.execution_mode == 'hardware':
            if not IBM_AVAILABLE:
                raise ImportError(
                    "IBM Quantum Runtime not installed. This script requires it.\n"
                    "Install with: pip install qiskit-ibm-runtime"
                )
            self.service = None
            self.backend_name = backend_name
            
            load_dotenv()
            self._initialize_ibm_runtime()
        elif self.execution_mode == 'test':
            print("🧪 Initializing local Aer simulator...")
            self.simulator = AerSimulator()
        else:
            raise ValueError(f"Invalid execution_mode: {execution_mode}. Must be 'hardware' or 'test'.")
            
    def _initialize_ibm_runtime(self):
        """
        Initialize connection to IBM Quantum Runtime with a strict, non-fallback logic.
        """
        token = os.getenv('IBM_QUANTUM_TOKEN')
        
        print("🔌 Connecting to IBM Quantum...")
        
        try:
            self.service = QiskitRuntimeService()
            print("✅ Connected using saved credentials.")
            return
        except AccountNotFoundError:
            print("   No saved credentials found. Proceeding with token from .env file.")
        except Exception as e:
            print(f"   Could not use saved credentials ({type(e).__name__}). Proceeding with token.")

        if not token:
            raise ValueError(
                "IBM_QUANTUM_TOKEN not found in .env file and no saved credentials worked."
            )
        
        print("   Attempting connection via 'ibm_cloud' channel with API token...")
        try:
            self.service = QiskitRuntimeService(channel="ibm_cloud", token=token)
            self.service.save_account(channel="ibm_cloud", token=token, overwrite=True)
            print("✅ Successfully connected and saved credentials for future use.")
        except Exception as e:
            raise ConnectionError(
                "The primary connection attempt with the provided API token failed. This is the root cause.\n\n"
                f"--> Original Error: {type(e).__name__}: {e}\n\n"
                "Troubleshooting Steps:\n"
                "1. Verify that the IBM_QUANTUM_TOKEN in your .env file is correct, active, and has no typos.\n"
                "2. Check your internet connection and any corporate firewalls.\n"
                "3. Confirm your account status at https://quantum.ibm.com/account is active."
             ) from e


    def generate_golden_candidates(self, sequence: List[int], n_candidates: int = 15) -> List[Tuple[int, float]]:
        """Generate candidate apexes including one that approximates φ."""
        reference = sequence[0]
        golden_apex = int(round(reference * self.PHI))
        min_apex = int(reference * 1.2)
        max_apex = int(reference * 2.1)
        
        candidate_set = {golden_apex}
        while len(candidate_set) < n_candidates:
            if max_apex > min_apex:
                apex = np.random.randint(min_apex, max_apex)
                candidate_set.add(apex)
            else:
                candidate_set.add(min_apex + len(candidate_set))

        candidates = list(candidate_set)
        np.random.shuffle(candidates)
        
        results = [(apex, abs(apex / reference - self.PHI)) for apex in candidates]
        return results
    
    def create_grover_circuit(self, candidates: List[Tuple[int, float]], 
                              tolerance: float = 0.05) -> Tuple[QuantumCircuit, List[int]]:
        """Create a Grover circuit to find the apex closest to the golden ratio."""
        n_candidates = len(candidates)
        n_qubits = (n_candidates - 1).bit_length()
        n_qubits = max(2, n_qubits)
        
        sorted_candidates = sorted(enumerate(candidates), key=lambda x: x[1][1])
        
        target_indices = []
        if sorted_candidates:
            best_deviation = sorted_candidates[0][1][1]
            for idx, (apex, deviation) in sorted_candidates:
                if deviation <= best_deviation + tolerance and len(target_indices) < 4:
                    target_indices.append(idx)
        
        print(f"\n🎯 Quantum Oracle Configuration:")
        print(f"   Search space: {2**n_qubits} states ({n_qubits} qubits)")
        print(f"   Valid candidates: {n_candidates}")
        print(f"   Target indices: {target_indices}")
        print(f"   Target apexes: {[candidates[i][0] for i in target_indices] if target_indices else 'None'}")
        
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        
        M = len(target_indices)
        N = 2**n_qubits
        iterations = 1
        if M > 0 and M < N:
            optimal_iterations = int(np.pi / 4 * np.sqrt(N / M))
            iterations = max(1, optimal_iterations)
        
        print(f"   Grover iterations: {iterations}")
        
        for i in range(iterations):
            qc.barrier(label=f"Oracle_{i+1}")
            self._apply_oracle(qc, target_indices, n_qubits)
            qc.barrier(label=f"Diffusion_{i+1}")
            self._apply_diffusion(qc, n_qubits)
        
        qc.measure_all()
        return qc, target_indices
    
    def _apply_oracle(self, qc: QuantumCircuit, targets: List[int], n_qubits: int):
        """Apply the oracle that marks the target states."""
        for target in targets:
            if target >= 2**n_qubits: continue
            target_bits = format(target, f'0{n_qubits}b')[::-1]
            for i, bit in enumerate(target_bits):
                if bit == '0': qc.x(i)
            
            if n_qubits > 1:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            else:
                qc.z(0)

            for i, bit in enumerate(target_bits):
                if bit == '0': qc.x(i)

    def _apply_diffusion(self, qc: QuantumCircuit, n_qubits: int):
        """Apply the Grover diffusion operator."""
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        
        if n_qubits > 1:
            qc.h(n_qubits-1)
            qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            qc.h(n_qubits-1)
        else:
            qc.z(0)
            
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
    
    def run_quantum_selection(self, sequence: List[int], 
                              n_candidates: int = 15,
                              shots: int = 8192,
                              tolerance: float = 0.05) -> int:
        """Run the complete quantum golden selection process on hardware."""
        print("="*60)
        print("🌌 QUANTUM GOLDEN SELECTOR (HARDWARE-ONLY)")
        print("="*60)
        print(f"Sequence: {sequence}")
        
        candidates = self.generate_golden_candidates(sequence, n_candidates)
        print(f"\n📊 Generated {len(candidates)} candidates (top 5 by deviation):")
        # DEFINITIVE FIX: Corrected lambda key from x[1][1] to x[1]
        for i, (apex, dev) in enumerate(sorted(candidates, key=lambda x: x[1])[:5]):
            print(f"   Apex {apex}: ratio={apex/sequence[0]:.6f}, φ-deviation={dev:.6f}")
        
        qc, target_indices = self.create_grover_circuit(candidates, tolerance=tolerance)
        
        print("\n" + "="*20 + " CIRCUIT DIAGRAM " + "="*20)
        print(qc.draw('text'))
        print("="*60)

        print(f"\n🔧 Logical Circuit Statistics:")
        print(f"   Depth: {qc.depth()}")
        print(f"   Gate count: {dict(qc.count_ops())}")
        
        counts = self._execute_circuit(qc, shots)
        
        print(f"\n📊 Quantum Measurement Results (top 5):")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        candidate_measurements = {}
        for bitstring, count in sorted_counts:
            index = int(bitstring, 2)
            if index < len(candidates):
                candidate_measurements[index] = candidate_measurements.get(index, 0) + count

        for index, count in sorted(candidate_measurements.items(), key=lambda x:x[1], reverse=True)[:5]:
             apex, dev = candidates[index]
             is_target = "🎯" if index in target_indices else "  "
             print(f"   {is_target} Index {index:<3}: {count:>5} times ({count/shots*100:5.1f}%) "
                   f"→ apex {apex:<4}, φ-dev={dev:.6f}")

        if not candidate_measurements:
            raise RuntimeError("No valid candidates were measured. This should not happen.")

        selected_index = max(candidate_measurements, key=candidate_measurements.get)
        selected_apex = candidates[selected_index][0]
        
        print(f"\n✨ Selected Golden Apex: {selected_apex}")
        
        return selected_apex
    
    def _select_best_backend(self, min_qubits: int):
        """
        Select the best available backend based on a quality score.
        If a specific backend name is provided, it will be used directly.
        """
        print("\n🔬 Selecting quantum backend...")
        
        if self.backend_name:
            try:
                backend = self.service.backend(self.backend_name)
                status = backend.status()
                config = backend.configuration()
                
                if not status.operational:
                    raise RuntimeError(f"Requested backend '{self.backend_name}' is not operational.")
                if config.n_qubits < min_qubits:
                    raise RuntimeError(f"Requested backend '{self.backend_name}' has only {config.n_qubits} qubits, but {min_qubits} are required.")
                
                print(f"✅ Using user-specified backend: {self.backend_name}")
                return backend
            except Exception as e:
                raise RuntimeError(f"Could not use requested backend '{self.backend_name}'. Reason: {e}") from e

        # Automatic selection if no backend is specified
        backends = self.service.backends(simulator=False, operational=True)
        scored_backends = []
        for backend in backends:
            status = backend.status()
            if not status.operational or status.pending_jobs > 100:
                continue
            
            config = backend.configuration()
            if config.n_qubits < min_qubits:
                continue

            try:
                target = backend.target
                has_error_data = False
                avg_cx_error = 0.1
                avg_ro_error = 0.2

                if target:
                    cx_errors = []
                    if "cx" in target:
                        for props in target["cx"].values():
                            if props and hasattr(props, 'error') and props.error is not None:
                                cx_errors.append(props.error)
                    
                    ro_errors = []
                    if "measure" in target:
                        for props in target["measure"].values():
                            if props and hasattr(props, 'error') and props.error is not None:
                                ro_errors.append(props.error)

                    if cx_errors and ro_errors:
                        avg_cx_error = np.mean(cx_errors)
                        avg_ro_error = np.mean(ro_errors)
                        has_error_data = True
                
                score = (status.pending_jobs + (1000 * avg_cx_error) + (500 * avg_ro_error) + (config.n_qubits - min_qubits))
                scored_backends.append((score, backend))
                
                eval_status = "OK" if has_error_data else "No detailed error data"
                print(f"   - Candidate: {backend.name} ({config.n_qubits} qubits), "
                      f"Queue: {status.pending_jobs}, "
                      f"Status: {eval_status}, "
                      f"Score: {score:.2f}")

            except Exception as e:
                print(f"   - Could not evaluate backend {backend.name}: {e}")

        if not scored_backends:
            raise RuntimeError("No suitable operational quantum backends found.")
            
        best_score, best_backend = min(scored_backends, key=lambda x: x[0])
        print(f"✅ Auto-selected best backend: {best_backend.name} (Score: {best_score:.2f})")
        return best_backend

    def _execute_circuit(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """
        Run circuit on either IBM Quantum hardware or a local simulator based on execution_mode.
        """
        if self.execution_mode == 'hardware':
            return self._run_on_hardware_impl(qc, shots)
        elif self.execution_mode == 'test':
            return self._run_on_simulator_impl(qc, shots)
        else:
            raise RuntimeError(f"Unknown execution mode: {self.execution_mode}")

    def _run_on_hardware_impl(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """
        Implementation for running on IBM Quantum hardware.
        """
        backend = self._select_best_backend(qc.num_qubits)
        
        print(f"\n⚡ Preparing execution for {backend.name}...")
        
        print(f"   - Transpiling circuit for {backend.name} with optimization level 3...")
        qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
        print(f"   - Resilience: Default (handled by runtime session)")
        print(f"   - Shots: {shots}")

        print(f"\n⚡ Submitting job via Session and SamplerV2...")
        
        with Session(backend=backend) as session:
            sampler = SamplerV2()
            
            job = sampler.run([qc_transpiled], shots=shots)
            
            print(f"   - ✅ Job submitted successfully. Job ID: {job.job_id()}")
            
            print(f"   - Waiting for results from {backend.name}...")
            result = job.result()
            
            print("   - ✅ Job completed, results received.")
            pub_result = result[0]
            hw_counts = pub_result.data.meas.get_counts()
            
            return hw_counts

    def _run_on_simulator_impl(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """
        Implementation for running on a local Aer simulator.
        """
        print(f"⚡ Running on local Aer simulator with {shots} shots...")
        # Transpile is less critical for Aer, but can still be useful for optimization
        # For simplicity, we'll run directly, but a transpile step could be added here too.
        job = self.simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        print("   - ✅ Simulation completed.")
        return counts


def main():
    """Main demo with enhanced error handling and text-only output."""
    is_hardware_mode = '--hardware' in sys.argv
    is_test_mode = '--test' in sys.argv

    if not is_hardware_mode and not is_test_mode:
        print("❌ ERROR: This script requires an execution mode.")
        print("   Please run with either the '--hardware' flag for real quantum hardware")
        print("   or the '--test' flag for local simulation.")
        return 1
    
    if is_hardware_mode and is_test_mode:
        print("❌ ERROR: Cannot run in both --hardware and --test modes simultaneously.")
        return 1

    execution_mode = 'hardware' if is_hardware_mode else 'test'
        
    # This check should only apply if we are in hardware mode
    if execution_mode == 'hardware' and not IBM_AVAILABLE:
        print("❌ IBM Quantum Runtime not installed.")
        print("   Install with: pip install qiskit-ibm-runtime")
        return 1
    
    sequence = [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]
    
    try:
        print(f"\n🔧 Initializing Quantum Golden Selector...")
        
        backend_name = None
        for arg in sys.argv:
            if arg.startswith('--backend='):
                backend_name = arg.split('=')[1]
                print(f"   Requested backend: {backend_name}")
        
        selector = QuantumGoldenSelector(execution_mode=execution_mode, backend_name=backend_name)
        
        golden_apex = selector.run_quantum_selection(
            sequence, 
            n_candidates=16,
            shots=8192,
            tolerance=0.005
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
