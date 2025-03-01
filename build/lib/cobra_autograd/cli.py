def check_installation():
    """Verify GPU support availability"""
    try:
        import cupy as cp # type: ignore
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA Available: {cp.is_available()}")
        print(f"CUDA Runtime Version: {cp.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        print("CuPy not installed. GPU features unavailable.")
    except Exception as e:
        print(f"GPU check failed: {str(e)}")