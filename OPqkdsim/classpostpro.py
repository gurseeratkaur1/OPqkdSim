import numpy as np
import scipy.sparse as sp
import scipy.special as sps
from scipy.linalg import hadamard
from .timeline import Timeline

class SiftedKey(Timeline):
    """
    Computes the sifted key for Alice after receiving Bob's bases.
    
    Args:
        timeline (Timeline): Event-driven simulation framework.
        alice_bits (np.ndarray): Alice's original bits.
        alice_bases (np.ndarray): Alice's chosen bases.
        sifted_key (list or np.ndarray): Mutable array to store the sifted key.
    """
    def __init__(self, timeline, alice_bits, alice_bases, sifted_key=None):
        self.timeline = timeline
        self.alice_bits = np.array(alice_bits)  # Ensure NumPy array
        self.alice_bases = np.array(alice_bases)

        # Validate user-provided storage
        if sifted_key is None:
            raise ValueError("A sifted_key list or NumPy array must be provided to store the key.")
        
        self.sifted_key = sifted_key  # Reference to external storage

    def receive_bob_bases(self, event_time, message):
        """
        Receives Bob's bases from the classical channel and computes the sifted key.
        """
        if isinstance(message, dict):
            bob_bases_received = np.array(message["basis"])  # Ensure NumPy array
        else:
            bob_bases_received = np.array(message)
        
        # Identify matching bases
        matching_indices = np.where(bob_bases_received == self.alice_bases)[0]
        
        # Compute sifted key
        sifted_key_values = self.alice_bits[matching_indices]

        # Store in user-provided array (supports both lists and NumPy arrays)
        if isinstance(self.sifted_key, list):
            self.sifted_key.clear()  # Clear existing values
            self.sifted_key.extend(sifted_key_values.tolist())  # Append new values
        elif isinstance(self.sifted_key, np.ndarray):
            self.sifted_key[:] = sifted_key_values  # Modify NumPy array in place
        else:
            raise TypeError("sifted_key must be a mutable list or NumPy array.")

        print(f"Alice computed sifted key of length {len(self.sifted_key)}.")

        # Publish sifted key for next steps (e.g., LDPC error reconciliation)
        # self.timeline.publish(self, self.sifted_key)



class LDPCReconciliation:
    """
    Implements LDPC Error Reconciliation for Bob's sifted key.
    
    Args:
        sifted_key (list): Bob's sifted key as a binary list.
        code_rate (float): LDPC code rate (e.g., 0.8 means 80% information, 20% redundancy).
    """
    def __init__(self, sifted_key, code_rate=0.8):
        self.sifted_key = np.array(sifted_key)
        self.n = len(self.sifted_key)  # Total code length
        self.k = int(self.n * code_rate)  # Number of information bits
        self.H, self.G = self.generate_ldpc_matrices(self.n, self.k)  # Generate parity-check and generator matrices
    
    def generate_ldpc_matrices(self, n, k):
        """Generates a simple LDPC-like parity-check matrix H and generator matrix G."""
        np.random.seed(42)  # Ensure reproducibility
        
        # Create a random sparse parity-check matrix H
        H = np.random.randint(0, 2, size=(n-k, n))  # (n-k) parity equations for n bits
        
        # Create a generator matrix G (must satisfy H * G^T = 0)
        G = np.eye(k, n, dtype=int)  # Start with an identity matrix for message bits
        for i in range(n-k):  
            G = np.vstack((G, H[i]))  # Append parity check rows
        
        return H, G

    def compute_parity_bits(self):
        """Encodes Bob's sifted key using LDPC and extracts parity bits."""
        codeword = np.dot(self.G, self.sifted_key) % 2  # Generate codeword
        parity_bits = codeword[self.k:]  # Extract parity bits (redundant bits)
        return parity_bits

    def correct_errors(self, alice_sifted_key):
        """
        Alice corrects errors using iterative decoding (simple parity-check method).
        
        Args:
            alice_sifted_key (list): Alice's sifted key before error correction.
            received_parity (list): Parity bits received from Bob over the classical channel.
        
        Returns:
            np.ndarray: Corrected sifted key.
        """
        alice_sifted_key = np.array(alice_sifted_key)
        max_iter = 10  # Maximum number of decoding iterations

        for _ in range(max_iter):
            syndrome = np.dot(self.H, alice_sifted_key) % 2  # Compute syndrome
            if np.all(syndrome == 0):  
                break  # No errors detected
            
            error_indices = np.where(syndrome == 1)[0]
            alice_sifted_key[error_indices] ^= 1  # Flip bits to correct errors

        return alice_sifted_key

    def publish_parity(self, parity_bits):
        """Simulates sending the parity bits through the classical channel."""
        print(f"Bob sent parity bits: {parity_bits.tolist()}")


class UhashPA:
    """
    Implements Privacy Amplification using Universal Hashing (Toeplitz matrices).
    """

    def __init__(self, output_length=256):
        self.output_length = output_length

    def generate_toeplitz_matrix(self, input_length):
        """
        Generates a Toeplitz matrix for universal hashing.
        """
        rand_values = np.random.randint(0, 2, input_length + self.output_length - 1)
        toeplitz_matrix = sp.diags(
            [rand_values[i:input_length + i] for i in range(self.output_length)], offsets=np.arange(self.output_length)
        ).toarray()
        return toeplitz_matrix % 2  # Binary matrix

    def apply_hashing(self, key):
        """
        Applies Toeplitz matrix hashing for privacy amplification.
        """
        input_length = len(key)
        toeplitz_matrix = self.generate_toeplitz_matrix(input_length)
        return (toeplitz_matrix @ key) % 2

    def amplify(self, reconciled_key):
        """
        Reduces key length while improving secrecy.
        """
        return self.apply_hashing(reconciled_key)


class QBERCalculator:
    """
    Class to calculate the Quantum Bit Error Rate (QBER) and statistical error.
    """

    def __init__(self, sifted_key_length, sample_bits, confidence=0.99):
        self.sifted_key_length = sifted_key_length  # Total bits in sifted key
        self.sample_bits = sample_bits  # Number of sample bits to estimate QBER
        self.confidence = confidence  # Confidence level (S)
        self.qber = None
        self.statistical_error = None
        self.approximated_qber = None

    def calculate_qber(self, alice_key, bob_key):
        """
        Computes the QBER by comparing a random sample of Alice and Bob's keys.
        """
        sample_indices = np.random.choice(self.sifted_key_length, self.sample_bits, replace=False)
        sample_errors = np.sum(alice_key[sample_indices] != bob_key[sample_indices])

        self.qber = sample_errors / self.sample_bits
        return self.qber

    def calculate_statistical_error(self):
        """
        Computes the statistical error in QBER estimation using Equation 2.4.
        """
        #self.statistical_error = np.sqrt(self.qber * (1 - self.qber) / (self.sample_bits * (1 - self.confidence)))
        self.statistical_error = np.sqrt(self.qber * (1 - self.qber) / self.sample_bits)

        return self.statistical_error

    def calculate_approximated_qber(self):
        """
        Computes the approximated QBER using Equation 2.5.
        """
        if self.qber is None or self.statistical_error is None:
            raise ValueError("QBER and statistical error must be calculated first.")
        
        self.approximated_qber = self.qber + self.statistical_error
        print(f"approximated_qber = {self.qber} ± {self.statistical_error}")
        return self.approximated_qber


class SecretKeyRateCalculator:
    """
    Class to calculate the Secret Key Rate (SKR) based on the amount of information revealed to Eve.
    """

    def __init__(self, sifted_key_length, total_time_slots, error_correction_bits, eve_allowed_fraction=0.01):
        self.sifted_key_length = sifted_key_length  # LBits
        self.total_time_slots = total_time_slots  # N
        self.error_correction_bits = error_correction_bits  # LParity
        self.eve_allowed_fraction = eve_allowed_fraction  # EAllowed
        self.final_key_length = None
        self.secret_key_rate = None

    def calculate_bits_revealed_to_eve(self, qber, dark_count_probability, pns_attack_bits):
        """
        Computes the total number of bits revealed to Eve.
        """
        # Intercept-resend attack (LIR)
        intercept_resend_bits = np.ceil(2 * (qber - dark_count_probability) * self.sifted_key_length)

        # Total bits revealed to Eve (LEve)
        total_eve_bits = intercept_resend_bits + pns_attack_bits + self.error_correction_bits
        return total_eve_bits

    def calculate_final_key_length(self, total_eve_bits):
        """
        Computes the final key length after privacy amplification.
        """
        privacy_amplification_bits = np.ceil((total_eve_bits - self.eve_allowed_fraction * self.sifted_key_length) / 
                                             (1 - self.eve_allowed_fraction))
        
        self.final_key_length = self.sifted_key_length - privacy_amplification_bits
        return self.final_key_length

    def calculate_secret_key_rate(self):
        """
        Computes the secret key rate (SKR).
        """
        if self.final_key_length is None:
            raise ValueError("Final key length must be calculated first.")

        self.secret_key_rate = self.final_key_length / self.total_time_slots
        return self.secret_key_rate