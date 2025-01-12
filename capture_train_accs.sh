#!/bin/bash

# Function to capture the latest train_acc values from each screen session
capture_train_accs() {
    # List all active screen sessions
    screen_sessions=$(screen -ls | awk '/\t/ {print $1}')

    if [ -z "$screen_sessions" ]; then
        echo "No active screen sessions found."
        return 1
    fi

    declare -A train_accs

    echo "Capturing train_accs from active screen sessions..."

    # Loop through each screen session
    for session in $screen_sessions; do
        # Create a temporary file to store the output
        tmpfile=$(mktemp)

        # Attach to the screen session, dump last 50 lines, and detach
        screen -S "$session" -X hardcopy "$tmpfile"

        if [ -f "$tmpfile" ]; then
            # Extract the latest train_acc value
            latest_train_acc=$(grep -oP 'train_acc=\K[0-9.]+(?=,)' "$tmpfile" | tail -1)

            if [ -n "$latest_train_acc" ]; then
                train_accs[$session]=$latest_train_acc
            else
                train_accs[$session]=0
            fi

            # Clean up the temporary file
            rm -f "$tmpfile"
        else
            echo "Failed to retrieve output for session: $session"
            train_accs[$session]=0
        fi
    done

    # Sort and display the top 10 screen sessions by train_acc
    echo "\nTop 10 screen sessions with the highest train_accs:"
    for session in "${!train_accs[@]}"; do
        echo "$session ${train_accs[$session]}"
    done | sort -k2 -nr | head -10
}

# Main function
main() {
    capture_train_accs
}

# Execute main function
main
