# 🔧 Installation Guide - Quantum Golden Selector

This guide walks you through the full installation of Quantum Golden Selector on various operating systems.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 2GB for all dependencies
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### Recommended
- **Python**: 3.9+ for best performance
- **RAM**: 16GB for complex simulations
- **CPU**: Multi-core for parallel simulations
- **Internet**: Required for IBM Quantum access

## 🐍 Python Installation

### Windows

1. **Download Python** from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to:
   - ✅ Check "Add Python to PATH"
   - ✅ Select "Install for all users"
   - ✅ Go to "Customize installation" → "Advanced Options" → "Add Python to environment variables"

3. **Verify installation**:
```cmd
python --version
pip --version
macOS
Option 1: Homebrew (Recommended)

bash
Copia
Modifica
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
Option 2: Direct Download

Download from python.org

Run the downloaded .pkg file

Linux (Ubuntu/Debian)
bash
Copia
Modifica
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev -y

# Optional: Create symlinks
sudo ln -sf /usr/bin/python3.11 /usr/bin/python
sudo ln -sf /usr/bin/pip3 /usr/bin/pip
Linux (CentOS/RHEL/Fedora)
bash
Copia
Modifica
# CentOS/RHEL
sudo yum install python3.11 python3.11-pip python3.11-devel -y

# Fedora
sudo dnf install python3.11 python3.11-pip python3.11-devel -y
📦 Project Installation
Method 1: Clone from GitHub (Recommended)
bash
Copia
Modifica
git clone https://github.com/yourusername/quantum-golden-selector.git
cd quantum-golden-selector

python -m venv quantum_env
# Activate virtual environment
# Windows:
quantum_env\Scripts\activate
# macOS/Linux:
source quantum_env/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
Method 2: Download ZIP
Go to GitHub and click "Code" → "Download ZIP"

Extract the archive

Follow the virtual environment steps above

🔧 Virtual Environment Configuration
Why Use a Virtual Environment?
Isolates project dependencies

Avoids conflicts with other Python projects

Eases deployment and sharing

Creating and Managing
bash
Copia
Modifica
python -m venv quantum_env

# Activation
# Windows PowerShell:
quantum_env\Scripts\Activate.ps1
# Windows CMD:
quantum_env\Scripts\activate.bat
# macOS/Linux:
source quantum_env/bin/activate

# Check active environment
which python

# Deactivate when done
deactivate
📚 Dependency Installation
Core Dependencies (Required)
bash
Copia
Modifica
pip install qiskit>=1.0.0
pip install numpy>=1.24.0
pip install python-dotenv>=1.0.0

# For local simulation
pip install qiskit-aer>=0.13.0
IBM Hardware Dependencies (Optional)
bash
Copia
Modifica
pip install qiskit-ibm-runtime>=0.20.0
Install All via requirements.txt
bash
Copia
Modifica
pip install -r requirements.txt
pip list | grep qiskit
🔑 IBM Quantum Configuration
Get Your API Token
Sign up at IBM Quantum

Log in to your account

Go to "Account" → "API Token"

Copy your token

Local Setup
bash
Copia
Modifica
cp .env.example .env
# Edit .env with your favorite editor
# Windows:
notepad .env
# macOS:
open -a TextEdit .env
# Linux:
nano .env
Sample .env file:

env
Copia
Modifica
# Your IBM Quantum token
IBM_QUANTUM_TOKEN=your_actual_token_here

# Preferred backend (optional)
# IBM_QUANTUM_BACKEND=ibm_sherbrooke
Test Connection
python
Copia
Modifica
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService
import os

load_dotenv()

try:
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f\"✅ Connected! Backends available: {len(backends)}\")
    for backend in backends[:3]:
        print(f\"   - {backend.name}\")
except Exception as e:
    print(f\"❌ Connection error: {e}\")
🧪 Verify Installation
Quick Test
bash
Copia
Modifica
# Local simulator test
python src/quantum_golden_selector.py

# IBM connection test (if configured)
python src/quantum_golden_selector.py --test
