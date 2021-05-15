from ctypes import *

dll_path = "/Users/redamaizate/Documents/3IABD/Projet-Annuel/ml_library/cmake-build-debug/libml_library.dylib"
mylib = cdll.LoadLibrary(dll_path)

def wrap_function(lib, funcname, argtypes, restype):
    """
    Simplify wrapping ctypes functions
    """
    func = lib.__getattr__(funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func