import numpy as np
import scipy.sparse as sp
import scipy.special as sps
from scipy.linalg import hadamard

class lpdcER:
    """
    Implements LDPC-based error reconciliation for QKD.
    Uses Alice's and Bob's bit lists along with their basis choices.
    """

    def __init__(self, key_length=10**6, code_rate=0.8):
        self.key_length = key_length
        self.code_rate = code_rate  # Rate R of LDPC code
        self.threshold_ber = self.get_threshold_ber()

        # Generate LDPC Parity-check matrix
        self.parity_matrix = self.generate_ldpc_parity_matrix()

    def get_threshold_ber(self):
        """
        Determines the maximum tolerable Bit Error Rate (BER) for a given LDPC rate.
        """
        rate_threshold_map = {
            0.9: 0.0109, 0.85: 0.0199, 0.8: 0.0298, 0.75: 0.0396, 
            0.7: 0.0504, 0.65: 0.0633, 0.6: 0.0766, 0.55: 0.0904, 0.5: 0.1071
        }
        return rate_threshold_map.get(self.code_rate, 0.05)

    def generate_ldpc_parity_matrix(self):
        """
        Generates a random sparse parity-check matrix for LDPC coding.
        """
        n = self.key_length
        k = int(n * self.code_rate)
        m = n - k  # Number of parity bits
        density = 0.01  # Sparsity level

        return sp.random(m, n, density=density, format='csr')

    def match_bases(self, alice_bits, alice_bases, bob_bits, bob_bases):
        """
        Filters out mismatched bases and returns the sifted key.
        """
        matching_indices = np.where(alice_bases == bob_bases)[0]
        alice_sifted = alice_bits[matching_indices]
        bob_sifted = bob_bits[matching_indices]
        return alice_sifted, bob_sifted

    def encode_parity_bits(self, key):
        """
        Computes parity bits for a given key using the parity-check matrix.
        """
        return (self.parity_matrix @ key) % 2

    def decode(self, received_key, received_parity):
        """
        Performs iterative decoding using belief propagation.
        """
        max_iterations = 50
        key = received_key.copy()

        for _ in range(max_iterations):
            parity_check = (self.parity_matrix @ key) % 2
            error_positions = np.where(parity_check != received_parity)[0]

            if error_positions.size == 0:
                break  # No more errors

            # Flip bits corresponding to errors
            key[error_positions] = 1 - key[error_positions]

        return key

    def reconcile(self, alice_bits, alice_bases, bob_bits, bob_bases):
        """
        Performs LDPC-based error reconciliation after basis matching.
        """
        alice_sifted, bob_sifted = self.match_bases(alice_bits, alice_bases, bob_bits, bob_bases)

        # Encode Bob's key to generate parity bits
        bob_parity = self.encode_parity_bits(bob_sifted)

        # Alice corrects her key using the received parity bits
        alice_corrected_key = self.decode(alice_sifted, bob_parity)

        return alice_corrected_key

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
        self.statistical_error = np.sqrt(self.qber * (1 - self.qber) / (self.sample_bits * (1 - self.confidence)))
        return self.statistical_error

    def calculate_approximated_qber(self):
        """
        Computes the approximated QBER using Equation 2.5.
        """
        if self.qber is None or self.statistical_error is None:
            raise ValueError("QBER and statistical error must be calculated first.")
        
        self.approximated_qber = self.qber + self.statistical_error
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