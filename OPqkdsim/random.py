import os
import secrets

class CRNG:
    def __init__(self, method="hardware"):
        """
        Initialize QRNG with a chosen method:
        - "hardware": Uses os.urandom() for true randomness.
        - "crypto": Uses secrets.randbits() for cryptographic randomness.
        """
        self.method = method

    def random_bit(self):
        """Generates a single random bit (0 or 1) using the chosen method."""
        if self.method == "hardware":
            return ord(os.urandom(1)) % 2  # Uses hardware-based entropy
        elif self.method == "crypto":
            return secrets.randbits(1)  # Uses cryptographic randomness
        else:
            raise ValueError("Invalid QRNG method. Choose 'hardware' or 'crypto'.")

    def generate_bits(self, num_bits=1):
        """Generates a list of random bits of specified length."""
        return [self.random_bit() for _ in range(num_bits)]

    def generate_basis(self, num_bases=1):
        """Generates a list of random basis choices (0: Rectilinear, 1: Diagonal)."""
        return [self.random_bit() for _ in range(num_bases)]
