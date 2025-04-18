"""
Helper script to start MLFlow tracking server.

Run this script before executing your training or scoring flows to ensure
that MLFlow tracking server is running.
"""
import os
import subprocess
import time
import signal
import sys

def is_mlflow_running():
    """Check if MLFlow server is already running on port 5000."""
    import socket
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('localhost', 5000))
        s.shutdown(2)
        return True
    except:
        return False
    finally:
        s.close()

def start_mlflow_server():
    """Start MLFlow tracking server."""
    # Create directories if they don't exist
    os.makedirs('mlruns', exist_ok=True)
    
    if is_mlflow_running():
        print("MLFlow server is already running on port 5000")
        return None
    
    print("Starting MLFlow tracking server...")
    # Start the MLFlow server
    process = subprocess.Popen(
        ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it a moment to start
    time.sleep(2)
    
    if process.poll() is not None:
        # Process has terminated
        stdout, stderr = process.communicate()
        print("Failed to start MLFlow server:")
        print(stderr)
        return None
    
    print(f"MLFlow server started with PID {process.pid}")
    return process

def stop_mlflow_server(process):
    """Stop the MLFlow tracking server."""
    if process is not None:
        print(f"Stopping MLFlow server (PID {process.pid})...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("MLFlow server stopped")
        except subprocess.TimeoutExpired:
            print("MLFlow server did not terminate, forcing...")
            process.kill()

def handle_shutdown(signum, frame, process=None):
    """Handle shutdown signals."""
    if process:
        stop_mlflow_server(process)
    sys.exit(0)

if __name__ == "__main__":
    # Start MLFlow server
    server_process = start_mlflow_server()
    
    if server_process:
        # Set up signal handlers to gracefully shutdown
        signal.signal(signal.SIGINT, lambda s, f: handle_shutdown(s, f, server_process))
        signal.signal(signal.SIGTERM, lambda s, f: handle_shutdown(s, f, server_process))
        
        print("MLFlow tracking server is running. Press Ctrl+C to stop.")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            stop_mlflow_server(server_process)
    else:
        print("MLFlow server is already running or failed to start.")