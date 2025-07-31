#!/bin/bash
# Network operations wrapper with retry logic

MAX_RETRIES=3
RETRY_DELAY=5

retry_command() {
    local cmd="$1"
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt/$MAX_RETRIES: $cmd"
        
        if eval "$cmd"; then
            echo "✅ Command succeeded on attempt $attempt"
            return 0
        else
            echo "❌ Command failed on attempt $attempt"
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    echo "❌ Command failed after $MAX_RETRIES attempts"
    return 1
}

# Export function for use in other scripts
export -f retry_command
