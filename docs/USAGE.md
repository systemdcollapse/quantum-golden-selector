📖 Usage Guide - Quantum Golden Selector
This comprehensive guide will show you how to use the Quantum Golden Selector in all its aspects, from basic functionalities to advanced usage.
🚀 Basic Usage
Simple Execution
# Standard run with local simulator
python src/quantum_golden_selector.py

# Expected output:
# 🌌 QUANTUM GOLDEN SELECTOR - TEXT ONLY
# ============================================================
# Sequence: [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]
# Golden ratio φ = 1.6180339887
# Target: Find apex ≈ 55 × φ = 89.02
# ...
# ✨ Selected Golden Apex: 89
Command-Line Parameters
# IBM Quantum hardware
python src/quantum_golden_selector.py --hardware

# Specify backend
python src/quantum_golden_selector.py --hardware --backend=ibm_sherbrooke

# Number of measurements (shots)
python src/quantum_golden_selector.py --shots=4096

# Number of candidates
python src/quantum_golden_selector.py --candidates=20

# Parameter combination
python src/quantum_golden_selector.py --hardware --backend=ibm_kyoto --shots=8192 --candidates=16
🔧 Advanced Configuration
Configuration File (.env)
# IBM Quantum configuration
IBM_QUANTUM_TOKEN=your_token_here
IBM_QUANTUM_BACKEND=ibm_sherbrooke

# Algorithm configurations
DEFAULT_SHOTS=8192
DEFAULT_CANDIDATES=15
DEFAULT_TOLERANCE=0.01

# Logging configurations
LOG_LEVEL=INFO
LOG_FILE=quantum_selector.log
ENABLE_VERBOSE_OUTPUT=true

# Performance configurations
QISKIT_PARALLEL=true
OPTIMIZATION_LEVEL=3
Algorithm Parameters
Parameter
Description
Default Value
Recommended Range
shots
Number of quantum measurements
8192
1024-16384
candidates
Number of generated candidates
15
10-30
tolerance
Deviation tolerance for φ
0.01
0.001-0.1
optimization_level
Circuit optimization level
3
0-3
📊 Execution Modes
1. Local Simulator (Default)
from src.quantum_golden_selector import QuantumGoldenSelector

# Initialize with simulator
selector = QuantumGoldenSelector(use_hardware=False)

# Sequence to extend
sequence = [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]

# Run selection
result = selector.run_quantum_selection(
    sequence=sequence,
    n_candidates=16,
    shots=8192,
    tolerance=0.005
)

print(f"Next number: {result}")
2. IBM Quantum Hardware
# Initialize with hardware
selector = QuantumGoldenSelector(
    use_hardware=True,
    backend_name="ibm_sherbrooke"  # Optional
)

# Run with hardware-optimized parameters
result = selector.run_quantum_selection(
    sequence=[55, 34, 21, 13, 8, 5, 3, 2, 1, 1],
    n_candidates=12,  # Fewer candidates to reduce qubits
    shots=4096,       # Fewer shots to reduce cost
    tolerance=0.01    # Wider tolerance for noise
)

🔍 Detailed Results Analysis
Output Interpretation
🎯 Quantum Oracle Configuration:
   Search space: 16 states (4 qubits)          # Quantum search space
   Valid candidates: 16                        # Generated valid candidates
   Tolerance: 0.005                           # Tolerance for target selection
   Target indices: [0, 1]                     # Target candidate indices
   Grover iterations: 3                       # Optimal Grover iterations

📊 Quantum Measurement Results:
   🎯 Index  0: 1920 times (23.4%) → apex  89  # Target hit with high probability
      Index  7:  654 times ( 8.0%) → apex  92  # Other candidates

📋 Best Practices
1. Parameter Selection
# For maximum accuracy (slower)
high_accuracy_params = {
    'shots': 16384,
    'n_candidates': 20,
    'tolerance': 0.001
}

# For fast execution (lower accuracy)
fast_execution_params = {
    'shots': 2048,
    'n_candidates': 12,
    'tolerance': 0.02
}

# For quantum hardware (balanced)
hardware_params = {
    'shots': 4096,
    'n_candidates': 15,
    'tolerance': 0.01
}


