from ctypes import *

dll_path = "C:/Users/ttres/Desktop/3A/S2/MachineLearning/ml_library/cmake-build-debug/ml_library.dll"
mylib = cdll.LoadLibrary(dll_path)

def wrap_function(lib, funcname, argtypes, restype):
    """
    Simplify wrapping ctypes functions
    """
    func = lib.__getattr__(funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func