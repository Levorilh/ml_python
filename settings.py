from ctypes import *
from env import *

dll_path = DLL_PATH
mylib = cdll.LoadLibrary(dll_path)

def wrap_function(lib, funcname, argtypes, restype):
    """
    Simplify wrapping ctypes functions
    """
    func = lib.__getattr__(funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func