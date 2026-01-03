#!/bin/bash
# Setup git hooks for fdars project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks..."

# Install pre-commit hook
cp "$SCRIPT_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"

echo "✓ Pre-commit hook installed (cargo fmt + clippy)"
echo ""
echo "Hooks installed successfully!"
echo "The pre-commit hook will run 'cargo fmt --check' and 'cargo clippy' before each commit."
