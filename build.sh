#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "Build complete!"
echo ""
echo "1. Copy .env.example to .env and fill in your API keys:"
echo "   cp .env.example .env"
echo ""
echo "2. Run the server:"
echo "   source venv/bin/activate && python main.py"
echo ""
echo "Agents: planner, developer, tester, code_reviewer, deployer, reporter"
echo "API docs: http://localhost:8000/docs"
