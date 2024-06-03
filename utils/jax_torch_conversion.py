import jax.numpy
import torch
from dlpack import asdlpack

def jax2torch(array_jax):
    return torch.from_dlpack(asdlpack(array_jax))

def torch2jax(array_torch):
    return jax.numpy.from_dlpack(asdlpack(array_torch))

def test():
    x = jax.numpy.array([1, 2, 3])
    y = jax2torch(x)
    z = torch2jax(y)
    print(x, y, z)

if __name__ == "__main__":
    test()
    test()
    test()
