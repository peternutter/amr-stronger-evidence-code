#!/bin/bash

# --- CONFIGURATION ---
HOURS=8
MEM=8192
CORES=4
TUNNEL_NAME="tunnel-$USER"
PROJECT_DIR="$HOME/amr-stronger-evidence-code/deception_experiments"

# Paths
CODE_BIN="$HOME/code"
AUTH_DIR="$HOME/.vscode-cli-data"
# ---------------------

# 1. DISABLE ENCRYPTION (Correct Variable Name)
# This forces the CLI to store the token in plain text JSON,
# allowing compute nodes to read it without a keyring service.
export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=true

echo "----------------------------------------------------------------"
echo " 🚀 VS Code Tunnel Manager (No Keyring)"
echo " 📂 Project: $PROJECT_DIR"
echo "----------------------------------------------------------------"

# ==============================================================================
# STEP 1: CHECK/INSTALL SERVER
# ==============================================================================
if [ ! -f "$CODE_BIN" ]; then
    echo "⬇️  VS Code Server not found. Downloading..."
    cd $HOME
    curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-linux-x64' --output vscode_cli.tar.gz
    tar -xf vscode_cli.tar.gz
    rm vscode_cli.tar.gz
    echo "✅ Installed to $HOME/code"
fi

# ==============================================================================
# STEP 2: VERIFICATION CHECK (Interactive Login)
# ==============================================================================
# We check if the plain-text token exists.
if [ ! -f "$AUTH_DIR/code_tunnel.json" ]; then
    echo "----------------------------------------------------------------"
    echo "⚠️  AUTHENTICATION REQUIRED (First Run or Reset)"
    echo "   Please authorize via the link below."
    echo "----------------------------------------------------------------"

    # We run the login command HERE (on the Login Node).
    # Because 'VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT' is exported above,
    # it will save the token as plain text in $AUTH_DIR.
    "$CODE_BIN" tunnel user login --provider github --cli-data-dir "$AUTH_DIR"

    echo "✅ Credentials saved securely to $AUTH_DIR"
    echo "----------------------------------------------------------------"
fi

# ==============================================================================
# STEP 3: LAUNCH COMPUTE JOB
# ==============================================================================
echo " 🚀 Requesting Compute Node..."

TOOLS_PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Remote Payload
REMOTE_CMD="
    # Ensure the remote node also knows to ignore keyrings
    export VSCODE_CLI_DISABLE_KEYCHAIN_ENCRYPT=true

    export PATH='$TOOLS_PATH'

    # Load Modules (Generic)
    module load gcc python zsh > /dev/null 2>&1

    # Navigate
    cd $PROJECT_DIR || echo '⚠️ Could not find project dir'

    echo '✅ Node Ready. Starting Tunnel...'
    echo '👉 Connection Name: $TUNNEL_NAME'

    # Start Tunnel
    # It will read the token file we created in Step 2
    '$CODE_BIN' tunnel --name $TUNNEL_NAME --accept-server-license-terms --cli-data-dir '$AUTH_DIR'
"

# Submit Job
srun --ntasks=1 \
     --cpus-per-task=$CORES \
     --mem-per-cpu=$MEM \
     --time=0${HOURS}:00:00 \
     --pty \
     bash -l -c "$REMOTE_CMD"
