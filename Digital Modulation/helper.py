import numpy as np

def convert_to_bits(file_path):
    with open('file_path','rb') as f:
        file_bytes=f.read()
    bits=np.unpackbits(np.frombuffer(file_bytes,dtype=np.unit8))
    return bits