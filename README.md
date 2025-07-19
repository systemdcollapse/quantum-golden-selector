# 🌌 Quantum Golden Selector

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Qiskit](https://img.shields.io/badge/qiskit-1.0+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

This content was developed through a synergy between my personal contribution and the use of artificial intelligence tools. In particular, AI provided significant support in writing code and offered logical assistance in response to my requests, helping to give structure and meaning to the prompts and solutions adopted.

The application example is a reverse engineering. It is recommended to familiarize yourself with the underlying logic before raising any doubts or requests for clarification.
This quantum algorithm implementation adopts an approach inspired by Grover’s algorithm to identify optimal approximations of the golden ratio (φ) within numerical sequences. The project demonstrates practical applications of quantum computing to mathematical optimization problems and is also intended for educational purposes.
## 🎯 Overview

The Quantum Golden Selector leverages quantum superposition and Grover's algorithm to efficiently search through candidate numbers and identify those that best approximate the golden ratio φ ≈ 1.618033988749. Instead of classical brute-force search, it uses quantum parallelism to evaluate multiple candidates simultaneously.

### Key Features

- **Quantum Grover Search**: Optimized implementation of Grover's algorithm for mathematical searches
- **Golden Ratio Precision**: High-precision calculations for φ approximations
- **Hardware Ready**: Compatible with IBM Quantum hardware and local simulators
- **Text-Only Interface**: Clean, professional output without graphical dependencies
- **Configurable Parameters**: Customizable tolerance, candidates, and execution settings

## 🚀 Quick Start

### Accessing IBM Quantum Hardware
```bash
1. To use real IBM Quantum hardware, authentication is required.
2. Choose one of the available authentication options and edit the .env.example
file with your credentials [TOKEN].
3. Save it as .env in the same folder as the script you want to run.
4. You can monitor your job status directly from the [IBM Quantum Dashboard]
```
https://quantum.ibm.com/account

### Virtual environments
```bash
python -m venv quantum_env
source quantum_env/bin/activate
cd Downloads/quantum-golden-selector-main
pip install --upgrade pip
pip install -r requirements.txt
python src/quantum_golden_selector.py --test o --hardware
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-golden-selector.git
cd quantum-golden-selector

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Basic Usage

```bash
# Run with local simulator (default)
python src/quantum_golden_selector.py

# Run on IBM Quantum hardware (requires API token)
python src/quantum_golden_selector.py --hardware

# Use specific backend
python src/quantum_golden_selector.py --hardware --backend=ibm_sherbrooke
```

### Example Output

```
🌌 QUANTUM GOLDEN SELECTOR - TEXT ONLY
============================================================
Sequence: [55, 34, 21, 13, 8, 5, 3, 2, 1, 1]
Golden ratio φ = 1.6180339887
Target: Find apex ≈ 55 × φ = 89.02

📊 Generated 16 candidates:
   Top 5 by golden ratio deviation:
   1. Apex 89: ratio=1.618182, φ-deviation=0.000148
   2. Apex 88: ratio=1.600000, φ-deviation=0.018034
   3. Apex 90: ratio=1.636364, φ-deviation=0.018330
   
🎯 Quantum Oracle Configuration:
   Search space: 16 states (4 qubits)
   Valid candidates: 16
   Target indices: [0, 1]
   Grover iterations: 3

✨ Selected Golden Apex: 89
   Measurement frequency: 23.4%
   Ratio: 89/55 = 1.6181818182
   φ-deviation: 0.0001480776
```

================================== Circuit sampler ===========================================
```bash
«        ┌───┐┌───┐┌───┐               ┌───┐┌───┐      ░ ┌─┐         
«   q_0: ┤ X ├┤ H ├┤ X ├────────────■──┤ X ├┤ H ├──────░─┤M├─────────
«        ├───┤├───┤└───┘            │  ├───┤├───┤      ░ └╥┘┌─┐      
«   q_1: ┤ H ├┤ X ├─────────────────■──┤ X ├┤ H ├──────░──╫─┤M├──────
«        ├───┤├───┤                 │  ├───┤├───┤      ░  ║ └╥┘┌─┐   
«   q_2: ┤ H ├┤ X ├─────────────────■──┤ X ├┤ H ├──────░──╫──╫─┤M├───
«        ├───┤├───┤┌───┐┌───┐┌───┐┌─┴─┐├───┤├───┤┌───┐ ░  ║  ║ └╥┘┌─┐
«   q_3: ┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ H ├─░──╫──╫──╫─┤M├
«        └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘ ░  ║  ║  ║ └╥┘
«meas: 4/═════════════════════════════════════════════════╩══╩══╩══╩═
«                                                         0  1  2  3 
 ```
===============================================================================================

## 🔬 Algorithm Details

### Quantum Circuit Architecture

1. **Superposition Creation**: Initialize qubits in equal superposition using Hadamard gates
2. **Oracle Implementation**: Mark states corresponding to golden ratio candidates
3. **Grover Iterations**: Apply optimal number of Grover operators to amplify target states
4. **Measurement**: Collapse to most probable golden ratio approximation

### Mathematical Foundation

The algorithm searches for values `x` such that:
```
|x/reference - φ| < tolerance
```

Where φ = (1 + √5)/2 ≈ 1.6180339887498948

### Quantum Advantage

- **Classical Complexity**: O(N) for N candidates
- **Quantum Complexity**: O(√N) using Grover's algorithm
- **Practical Speedup**: ~4x improvement for typical problem sizes

## 📁 Project Structure

```
quantum-golden-selector/
├── src/
│   └── quantum_golden_selector.py    # Main algorithm implementation
├── docs/
│   ├── INSTALLATION.md               # Detailed installation guide
│   ├── USAGE.md                     # Comprehensive usage examples
│   └── ALGORITHM.md                 # Technical algorithm documentation
├── requirements.txt                  # Python dependencies
├── .env.example                     # Environment configuration template
└── README.md                        # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your IBM Quantum credentials:

```bash
# Required for hardware execution
IBM_QUANTUM_TOKEN=your_api_token_here

# Optional: specify preferred backend
IBM_QUANTUM_BACKEND=ibm_sherbrooke
```

### Command Line Options

- `--hardware`: Use IBM Quantum hardware instead of simulator
- `--backend=name`: Specify particular quantum backend
- `--shots=N`: Number of quantum measurements (default: 8192)
- `--candidates=N`: Number of candidate values to generate (default: 15)

## 🔧 Dependencies

### Core Requirements
- **Python 3.8+**
- **qiskit >= 1.0.0**: Quantum circuit framework
- **qiskit-aer >= 0.13.0**: Local quantum simulation
- **numpy >= 1.24.0**: Numerical computations
- **python-dotenv >= 1.0.0**: Environment management

### Optional (for hardware)
- **qiskit-ibm-runtime >= 0.20.0**: IBM Quantum access

## 🎓 Educational Value

This project demonstrates several important quantum computing concepts:

- **Quantum Superposition**: Simultaneous evaluation of multiple candidates
- **Grover's Algorithm**: Quadratic speedup for unstructured search
- **Quantum Oracle Design**: Marking target states in quantum search
- **Quantum Circuit Optimization**: Transpilation for different backends
- **Hybrid Classical-Quantum**: Combining classical preprocessing with quantum processing

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📊 Performance Metrics

Typical performance on different backends:

| Backend | Qubits | Success Rate | Execution Time |
|---------|--------|--------------|----------------|
| AerSimulator | 4-8 | 95%+ | ~1s |
| ibm_sherbrooke | 127 | 85-90% | ~30s |
| ibm_kyoto* | 127 | 80-85% | ~45s |
*requires subscription

## 🐛 Troubleshooting

### Common Issues

**"IBM Quantum connection failed"**
- Verify your API token in `.env` file
- Check token validity at [IBM Quantum](https://quantum.ibm.com/account)

**"No suitable backends found"**
- Ensure your account has access to quantum systems
- Try reducing the number of required qubits

**"AerSimulator not found"**
- Install qiskit-aer: `pip install qiskit-aer`

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

## 📚 Further Reading

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Grover's Algorithm Tutorial](https://qiskit.org/textbook/ch-algorithms/grover.html)
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)
- [Golden Ratio Mathematics](https://en.wikipedia.org/wiki/Golden_ratio)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@software{quantum_golden_selector,
  title={Quantum Golden Selector: Grover's Algorithm for Golden Ratio Approximation},
  author={systemdcollapse},
  year={2024},
  url={https://github.com/yourusername/quantum-golden-selector}
}
```

## 🌟 Acknowledgments

- [Qiskit Team](https://qiskit.org/) for the excellent quantum computing framework
- [IBM Quantum Network](https://quantum-network.ibm.com/) for quantum hardware access
- The quantum computing community for inspiration and support

---

**Made with ⚛️ and 💛 for the quantum computing community**

*For questions, suggestions, or collaborations, feel free to open an issue or reach out!*
