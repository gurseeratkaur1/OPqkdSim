from django.shortcuts import render
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ThOPqkdsim.sim7 import *
from ThOPqkdsim.sim8 import *
from ThOPqkdsim.sim9 import *
# Create your views here.

def home(request):
    return render(request, "home.html", {})

def get_plot_base64(plt):
    """
    Convert a matplotlib figure to a base64 encoded string for embedding in HTML
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Encode the PNG image to base64 string
    return base64.b64encode(image_png).decode('utf-8')


def modified_plot_qber_vs_mu(mu_values, time_window, distance,
                      alice_detector_efficiency, bob_detector_efficiency,
                      alice_channel_base_efficiency, bob_channel_base_efficiency,
                      dark_count_rate):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    qber_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'bo-', linewidth=2)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Mean Photon Number')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values

def modified_plot_skr_vs_mu(mu_values, time_window, key_length, distance,
                     alice_detector_efficiency, bob_detector_efficiency,
                     alice_channel_base_efficiency, bob_channel_base_efficiency,
                     dark_count_rate, repetition_rate):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    skr_values = []
    
    for mu in mu_values:
        simulator = BB84Simulator(
            mu=mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance
        )
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'go-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Mean Photon Number (μ)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Mean Photon Number')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values

def modified_plot_qber_vs_distance(distance_values, time_window, mu,
                           alice_detector_efficiency, bob_detector_efficiency,
                           alice_channel_base_efficiency, bob_channel_base_efficiency,
                           dark_count_rate):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    qber_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'ro-', linewidth=2)
    plt.grid(True)
    plt.axhline(y=5, color='magenta', linestyle='--', label='5% threshold')
    plt.axhline(y=7, color='orange', linestyle='--', label='7% threshold')
    plt.axhline(y=11, color='red', linestyle='--', label='11% limit')
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('Quantum Bit Error Rate vs Distance')
    plt.legend()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64

def modified_plot_skr_vs_distance(distance_values, time_window, key_length, mu,
                          alice_detector_efficiency, bob_detector_efficiency,
                          alice_channel_base_efficiency, bob_channel_base_efficiency,
                          dark_count_rate,repetition_rate):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    skr_values = []
    
    simulator = BB84Simulator(
        mu=mu,
        alice_detector_efficiency=alice_detector_efficiency,
        bob_detector_efficiency=bob_detector_efficiency,
        alice_channel_base_efficiency=alice_channel_base_efficiency,
        bob_channel_base_efficiency=bob_channel_base_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        distance=0
    )
    
    for distance in distance_values:
        simulator.update_distance(distance)
        skr_per_pulse = simulator.calculate_skr(key_length)
        skr_per_second = skr_per_pulse * repetition_rate  # Convert to bits/second
        skr_values.append(skr_per_second)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'mo-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (bits/s)')
    plt.title('Secret Key Rate vs Distance')
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64



def bb84(request):
    """
    View function to handle the BB84 simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data
        mu = float(request.POST.get('mu', 0.1))
        distance = float(request.POST.get('distance', 10))
        attenuation = float(request.POST.get('attenuation', 0.2))
        alice_detector_efficiency = float(request.POST.get('alice_detector_efficiency', 0.8))
        bob_detector_efficiency = float(request.POST.get('bob_detector_efficiency', 0.8))
        alice_channel_base_efficiency = float(request.POST.get('alice_channel_base_efficiency', 1.0))
        bob_channel_base_efficiency = float(request.POST.get('bob_channel_base_efficiency', 0.3913))
        dark_count_rate = float(request.POST.get('dark_count_rate', 1000))
        time_window = float(request.POST.get('time_window', 1)) * 1e-9  # Convert ns to seconds
        repetition_rate = float(request.POST.get('repetition_rate', 1000000))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        distance_max_qber = float(request.POST.get('distance_max_qber', 300))
        distance_max_skr = float(request.POST.get('distance_max_skr', 210))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
        # Generate plots
        qber_vs_mu_plot, qber_values = modified_plot_qber_vs_mu(
            mu_values, time_window, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate
        )
        
        skr_vs_mu_plot, _ = modified_plot_skr_vs_mu(
            mu_values, time_window, 1000000, distance,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate
        )
        
        # Identify optimal μ value
        optimal_indices = [i for i, qber in enumerate(qber_values) if 5 <= qber <= 7]
        if optimal_indices:
            optimal_mu_index = optimal_indices[len(optimal_indices)//2]
            optimal_mu = mu_values[optimal_mu_index]
        else:
            optimal_mu_index = np.argmin(np.abs(np.array(qber_values) - 6))
            optimal_mu = mu_values[optimal_mu_index]
        
        # Generate distance plots with optimal μ
        qber_vs_distance_plot = modified_plot_qber_vs_distance(
            distance_values_qber, time_window, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate
        )
        
        skr_vs_distance_plot = modified_plot_skr_vs_distance(
            distance_values_skr, time_window, 10000, optimal_mu,
            alice_detector_efficiency, bob_detector_efficiency,
            alice_channel_base_efficiency, bob_channel_base_efficiency,
            dark_count_rate, repetition_rate
        )
        
        # Calculate additional metrics
        simulator = BB84Simulator(
            mu=optimal_mu,
            alice_detector_efficiency=alice_detector_efficiency,
            bob_detector_efficiency=bob_detector_efficiency,
            alice_channel_base_efficiency=alice_channel_base_efficiency,
            bob_channel_base_efficiency=bob_channel_base_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            distance=distance,
            attenuation=attenuation
        )
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr(1000)
        
        # Find max distance where QBER ≤ 11%
        max_distance = 0
        for d in np.arange(0, 500, 1):
            simulator.update_distance(d)
            qber = simulator.calculate_qber()
            if qber <= 11:
                max_distance = d
            else:
                break
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot
        }
        
        return render(request, 'bb84.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.6f}",
            'max_distance': f"{max_distance:.1f}",
            'repetition_rate': repetition_rate 
        })
    
    # If GET request, just render the form
    return render(request, 'bb84.html')




def modified_plot_key_rate_vs_distance(qkd, max_distance=150):
    """Modified version that returns the base64 encoded plot"""
    distances, key_rates, _, _, _, _, _ = analyze_distance_dependence(qkd, max_distance)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(distances, key_rates)
    plt.xlabel('Distance (km)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title('Secure Key Rate vs Distance')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def decoy_modified_plot_qber_vs_distance(qkd, max_distance=150):
    """Modified version that returns the base64 encoded plot"""
    distances, _, qbers, _, _, _, _ = analyze_distance_dependence(qkd, max_distance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [qber * 100 for qber in qbers])  # Convert to percentage
    plt.xlabel('Distance (km)')
    plt.ylabel('QBER (%)')
    plt.title('QBER vs Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt)

def modified_plot_key_rate_vs_mu(qkd, distance=50):
    """Modified version that returns the base64 encoded plot and key rates"""
    mu_values, mu_key_rates, _ = analyze_mu_dependence(qkd, distance=distance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, mu_key_rates)
    plt.xlabel('Signal State Intensity (μ)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title(f'Secure Key Rate vs Signal State Intensity at {distance} km')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt), mu_values, mu_key_rates

def modified_plot_key_rate_vs_nu1(qkd, distance=50):
    """Modified version that returns the base64 encoded plot and key rates"""
    nu1_values, nu1_key_rates = analyze_decoy_state_intensity(qkd, distance=distance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(nu1_values, nu1_key_rates)
    plt.xlabel('Decoy State Intensity (ν₁)')
    plt.ylabel('Secure Key Rate (bits/s)')
    plt.title(f'Secure Key Rate vs Decoy State Intensity at {distance} km')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return get_plot_base64(plt), nu1_values, nu1_key_rates

def analyze_distance_dependence(qkd, max_distance=150, step=1):
    """Analyze how key rate and QBER change with distance"""
    distances = np.arange(0, max_distance + step, step)
    key_rates = []
    qbers = []
    gains = []
    Y1_Ls = []
    e1_Us = []
    Q1_Ls = []
    
    for d in distances:
        rate, qber, gain, Y1_L, e1_U, Q1_L = qkd.key_rate(d)
        key_rates.append(rate)
        qbers.append(qber)
        gains.append(gain)
        Y1_Ls.append(Y1_L)
        e1_Us.append(e1_U)
        Q1_Ls.append(Q1_L)
    
    return distances, key_rates, qbers, gains, Y1_Ls, e1_Us, Q1_Ls

def analyze_mu_dependence(qkd, distance=50, mu_range=(0.1, 1.0), step=0.05):
    """Analyze how key rate and QBER change with signal state intensity mu"""
    mu_values = np.arange(mu_range[0], mu_range[1] + step, step)
    key_rates = []
    qbers = []
    
    original_mu = qkd.mu
    for mu in mu_values:
        qkd.mu = mu
        rate, qber, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
        qbers.append(qber)
    
    qkd.mu = original_mu
    return mu_values, key_rates, qbers

def analyze_decoy_state_intensity(qkd, distance=50, nu1_range=(0.01, 0.3), step=0.02):
    """Analyze how key rate changes with decoy state intensity nu1"""
    nu1_values = np.arange(nu1_range[0], nu1_range[1] + step, step)
    key_rates = []
    
    original_nu1 = qkd.nu1
    for nu1 in nu1_values:
        qkd.nu1 = nu1
        rate, _, _, _, _, _ = qkd.key_rate(distance)
        key_rates.append(rate)
    
    qkd.nu1 = original_nu1
    return nu1_values, key_rates

def decoy_bb84(request):
    """
    View function to handle the Decoy State BB84 simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data for the DecoyStateQKD parameters
        wavelength = float(request.POST.get('wavelength', 1550))  # nm
        alpha = float(request.POST.get('alpha', 0.21))  # dB/km (fiber loss coefficient)
        e_detector = float(request.POST.get('e_detector', 0.033))  # detector error probability
        Y0 = float(request.POST.get('Y0', 1.7e-6))  # background rate
        eta_bob = float(request.POST.get('eta_bob', 0.045))  # Bob's side efficiency
        mu = float(request.POST.get('mu', 0.5))  # signal state intensity
        nu1 = float(request.POST.get('nu1', 0.1))  # decoy state 1 intensity
        nu2 = float(request.POST.get('nu2', 0.0))  # decoy state 2 intensity (vacuum)
        f = float(request.POST.get('f', 1.22))  # error correction efficiency
        q = float(request.POST.get('q', 0.5))  # protocol efficiency factor
        rep_rate = float(request.POST.get('rep_rate', 2e6))  # repetition rate in Hz
        
        # Get plot range parameters
        max_distance = float(request.POST.get('max_distance', 150))
        distance_for_mu = float(request.POST.get('distance_for_mu', 50))
        mu_min = float(request.POST.get('mu_min', 0.1))
        mu_max = float(request.POST.get('mu_max', 1.0))
        mu_step = float(request.POST.get('mu_step', 0.05))
        nu1_min = float(request.POST.get('nu1_min', 0.01))
        nu1_max = float(request.POST.get('nu1_max', 0.3))
        nu1_step = float(request.POST.get('nu1_step', 0.02))
        
        # Create the QKD object with the parameters
        qkd = DecoyStateQKD(
            wavelength=wavelength,
            alpha=alpha,
            e_detector=e_detector,
            Y0=Y0,
            eta_bob=eta_bob,
            mu=mu,
            nu1=nu1,
            nu2=nu2,
            f=f,
            q=q,
            rep_rate=rep_rate
        )
        
        # Generate plots
        key_rate_vs_distance_plot = modified_plot_key_rate_vs_distance(qkd, max_distance)
        qber_vs_distance_plot = decoy_modified_plot_qber_vs_distance(qkd, max_distance)
        
        key_rate_vs_mu_plot, mu_values, mu_key_rates = modified_plot_key_rate_vs_mu(
            qkd, 
            distance=distance_for_mu,
            # mu_range=(mu_min, mu_max), 
            # step=mu_step
        )
        
        key_rate_vs_nu1_plot, nu1_values, nu1_key_rates = modified_plot_key_rate_vs_nu1(
            qkd, 
            distance=distance_for_mu,
            # nu1_range=(nu1_min, nu1_max), 
            # step=nu1_step
        )
        
        # Find optimal mu (signal intensity) at the given distance
        optimal_mu_index = np.argmax(mu_key_rates)
        optimal_mu = mu_values[optimal_mu_index]
        
        # Find optimal nu1 (decoy state intensity) at the given distance
        optimal_nu1_index = np.argmax(nu1_key_rates)
        optimal_nu1 = nu1_values[optimal_nu1_index]
        
        # Calculate maximum achievable distance
        distances, key_rates, _, _, _, _, _ = analyze_distance_dependence(qkd, max_distance=300, step=1)
        max_achievable_distance = 0
        for i, rate in enumerate(key_rates):
            if rate > 0:
                max_achievable_distance = distances[i]
        
        # Calculate key rate at the specified distance
        current_key_rate, current_qber, _, _, _, _ = qkd.key_rate(distance_for_mu)
        
        # Package results for the template
        plots = {
            'key_rate_vs_distance': key_rate_vs_distance_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'key_rate_vs_mu': key_rate_vs_mu_plot,
            'key_rate_vs_nu1': key_rate_vs_nu1_plot
        }
        
        return render(request, 'decoybb84.html', {
            'plots': plots,
            'wavelength': wavelength,
            'alpha': alpha,
            'e_detector': e_detector,
            'Y0': Y0,
            'eta_bob': eta_bob,
            'mu': mu,
            'nu1': nu1,
            'nu2': nu2,
            'f': f,
            'q': q,
            'rep_rate': rep_rate,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_nu1': f"{optimal_nu1:.4f}",
            'max_achievable_distance': f"{max_achievable_distance:.1f}",
            'current_key_rate': f"{current_key_rate:.2f}",
            'current_qber': f"{current_qber * 100:.2f}",  # Convert to percentage
            'distance_for_mu': distance_for_mu
        })
    
    # If GET request, just render the form with default values
    return render(request, 'decoybb84.html')



def cow_modified_plot_qber_vs_mu(mu_values, distance, 
                             detector_efficiency, dark_count_rate,
                             time_window, channel_base_efficiency,
                             attenuation, data_line_ratio,
                             decoy_probability, repetition_rate):
    """
    Modified version of plot_qber_vs_mu that returns the base64 encoded plot
    and the QBER values
    """
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(
        distance=distance,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate QBER for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, qber_values, 'b-', linewidth=2)
    plt.plot(mu_values, qber_values, 'bo', markersize=6)
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('QBER (%)', fontsize=12)
    plt.title(f'Quantum Bit Error Rate vs Mean Photon Number in COW Protocol (Distance: {distance} km)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, qber_values


def cow_modified_plot_skr_vs_mu(mu_values, distance, 
                            detector_efficiency, dark_count_rate,
                            time_window, channel_base_efficiency,
                            attenuation, data_line_ratio,
                            decoy_probability, repetition_rate):
    """
    Modified version of plot_skr_vs_mu that returns the base64 encoded plot
    and the SKR values
    """
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(
        distance=distance,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate SKR for each mu value
    for mu in mu_values:
        simulator.update_mu(mu)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mu_values, skr_values, 'g-', linewidth=2)
    plt.plot(mu_values, skr_values, 'go', markersize=6)
    plt.grid(True, alpha=0.7)
    plt.xlabel('Mean Photon Number (μ)', fontsize=12)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Mean Photon Number in COW Protocol (Distance: {distance} km)', fontsize=14)
    
    # Mark optimal mean photon number
    if max(skr_values) > 0:
        optimal_mu_index = np.argmax(skr_values)
        optimal_mu = mu_values[optimal_mu_index]
        optimal_skr = skr_values[optimal_mu_index]
        
        plt.plot(optimal_mu, optimal_skr, 'ro', markersize=8)
        plt.annotate(f'Optimal μ ≈ {optimal_mu:.2f}\nSKR ≈ {optimal_skr:.2e} bits/s',
                     xy=(optimal_mu, optimal_skr), 
                     xytext=(optimal_mu + 0.1, optimal_skr * 0.8),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64, skr_values


def cow_modified_plot_qber_vs_distance(distance_values, mu, 
                                  detector_efficiency, dark_count_rate,
                                  time_window, channel_base_efficiency,
                                  attenuation, data_line_ratio,
                                  decoy_probability, repetition_rate):
    """
    Modified version of plot_qber_vs_distance that returns the base64 encoded plot
    """
    qber_values = []
    
    # Create simulator
    simulator = COWProtocol(
        mu=mu,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate QBER for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        qber = simulator.calculate_qber()
        qber_values.append(qber)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, qber_values, 'r-', linewidth=2)
    plt.plot(distance_values, qber_values, 'ro', markersize=6)
    plt.axhline(y=11, color='r', linestyle='--', label='Security threshold (11%)')
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('QBER (%)', fontsize=12)
    plt.title(f'Quantum Bit Error Rate vs Distance in COW Protocol (μ = {mu})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cow_modified_plot_skr_vs_distance(distance_values, mu, 
                                 detector_efficiency, dark_count_rate,
                                 time_window, channel_base_efficiency,
                                 attenuation, data_line_ratio,
                                 decoy_probability, repetition_rate):
    """
    Modified version of plot_skr_vs_distance that returns the base64 encoded plot
    """
    skr_values = []
    
    # Create simulator
    simulator = COWProtocol(
        mu=mu,
        detector_efficiency=detector_efficiency,
        dark_count_rate=dark_count_rate,
        time_window=time_window,
        channel_base_efficiency=channel_base_efficiency,
        attenuation=attenuation,
        data_line_ratio=data_line_ratio,
        decoy_probability=decoy_probability,
        repetition_rate=repetition_rate
    )
    
    # Calculate SKR for each distance value
    for distance in distance_values:
        simulator.update_distance(distance)
        skr = simulator.calculate_skr()
        skr_values.append(skr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distance_values, skr_values, 'm-', linewidth=2)
    plt.plot(distance_values, skr_values, 'mo', markersize=6)
    plt.grid(True, alpha=0.7)
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Secret Key Rate (bits/s)', fontsize=12)
    plt.title(f'Secret Key Rate vs Distance in COW Protocol (μ = {mu})', fontsize=14)
    
    # Find maximum distance with positive SKR
    positive_indices = np.where(np.array(skr_values) > 0)[0]
    if len(positive_indices) > 0:
        max_dist_idx = positive_indices[-1]
        max_dist = distance_values[max_dist_idx]
        plt.axvline(x=max_dist, color='r', linestyle=':', alpha=0.7)
        plt.text(max_dist + 5, max(skr_values)/10, f'Max distance: {max_dist:.1f} km', color='r')
    
    plt.tight_layout()
    
    plot_base64 = get_plot_base64(plt)
    
    return plot_base64


def cowqkd(request):
    """
    View function to handle the COW QKD simulator form and run simulations.
    """
    if request.method == 'POST':
        # Get form data
        mu = float(request.POST.get('mu', 0.5))
        distance = float(request.POST.get('distance', 50))
        attenuation = float(request.POST.get('attenuation', 0.2))
        detector_efficiency = float(request.POST.get('detector_efficiency', 0.2))
        dark_count_rate = float(request.POST.get('dark_count_rate', 500))
        time_window = float(request.POST.get('time_window', 1)) * 1e-9  # Convert ns to seconds
        channel_base_efficiency = float(request.POST.get('channel_base_efficiency', 0.8))
        data_line_ratio = float(request.POST.get('data_line_ratio', 0.9))
        decoy_probability = float(request.POST.get('decoy_probability', 0.1))
        repetition_rate = float(request.POST.get('repetition_rate', 500e6))
        
        # Get plot range parameters
        mu_min = float(request.POST.get('mu_min', 0.01))
        mu_max = float(request.POST.get('mu_max', 1.0))
        distance_max_qber = float(request.POST.get('distance_max_qber', 200))
        distance_max_skr = float(request.POST.get('distance_max_skr', 200))
        
        # Define plot ranges
        mu_values = np.linspace(mu_min, mu_max, 20)
        distance_values_qber = np.linspace(0, distance_max_qber, 50)
        distance_values_skr = np.linspace(0, distance_max_skr, 50)
        
        # Generate plots
        qber_vs_mu_plot, qber_values = cow_modified_plot_qber_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        skr_vs_mu_plot, skr_values = cow_modified_plot_skr_vs_mu(
            mu_values, distance,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        # Identify optimal μ value for maximum SKR
        if max(skr_values) > 0:
            optimal_mu_index = np.argmax(skr_values)
            optimal_mu = mu_values[optimal_mu_index]
        else:
            # If no positive SKR found, use a default value for μ
            optimal_mu = 0.3  # A common value for COW protocol
        
        # Generate distance plots with optimal μ
        qber_vs_distance_plot = cow_modified_plot_qber_vs_distance(
            distance_values_qber, optimal_mu,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        skr_vs_distance_plot = cow_modified_plot_skr_vs_distance(
            distance_values_skr, optimal_mu,
            detector_efficiency, dark_count_rate,
            time_window, channel_base_efficiency,
            attenuation, data_line_ratio,
            decoy_probability, repetition_rate
        )
        
        # Calculate additional metrics
        simulator = COWProtocol(
            mu=optimal_mu,
            distance=distance,
            detector_efficiency=detector_efficiency,
            dark_count_rate=dark_count_rate,
            time_window=time_window,
            channel_base_efficiency=channel_base_efficiency,
            attenuation=attenuation,
            data_line_ratio=data_line_ratio,
            decoy_probability=decoy_probability,
            repetition_rate=repetition_rate
        )
        
        optimal_qber = simulator.calculate_qber()
        optimal_skr = simulator.calculate_skr()
        
        # Find max distance where QBER ≤ 11% (security threshold for COW)
        max_distance = 0
        for d in np.arange(0, 300, 1):
            simulator.update_distance(d)
            qber = simulator.calculate_qber()
            if qber <= 11:
                max_distance = d
            else:
                break
        
        # Package results for the template
        plots = {
            'qber_vs_mu': qber_vs_mu_plot,
            'skr_vs_mu': skr_vs_mu_plot,
            'qber_vs_distance': qber_vs_distance_plot,
            'skr_vs_distance': skr_vs_distance_plot
        }
        
        return render(request, 'cowqkd.html', {
            'plots': plots,
            'optimal_mu': f"{optimal_mu:.4f}",
            'optimal_qber': f"{optimal_qber:.4f}",
            'optimal_skr': f"{optimal_skr:.4e}",
            'max_distance': f"{max_distance:.1f}",
            'repetition_rate': f"{repetition_rate:.2e}"
        })
    
    # If GET request, just render the form
    return render(request, 'cowqkd.html')
