
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

# --- Configuration ---
SAMPLING_RATE = 64  # Hz (Standard for low-power wearables)
NOISE_LEVEL = 0.05  # Base variance for Gaussian noise
ARTIFACT_PROB = 0.005 # Probability of a massive spike (artifact)

def generate_noise(length, magnitude=1.0):
    """Generates random Gaussian noise and occasional outliers (artefacts)."""
    # Gaussian background noise
    noise = np.random.normal(0, 0.1 * magnitude, length)
    
    # Add random artifacts (spikes typical in loose sensors)
    num_artifacts = int(length * ARTIFACT_PROB)
    artifact_indices = np.random.choice(length, num_artifacts, replace=False)
    noise[artifact_indices] += np.random.choice([-1, 1], num_artifacts) * (magnitude * 5)
    
    return noise

def get_emotion_params(emotion):
    """
    Returns physiological base parameters for specific emotions.
    Logic:
    - Sad: Low HR, Low Movement, Steady EDA, Lower Temp
    - Happy: Med HR, Med Movement, Spikey EDA
    - Aggressive: High HR, High Movement, High EDA, High Temp
    """
    if emotion == 'Sad':
        return {
            'hr_mean': 60, 'hr_std': 5,
            'eda_trend': -0.01, 'eda_volatility': 0.1,
            'move_mag': 0.2, 'temp_base': 36.5
        }
    elif emotion == 'Happy':
        return {
            'hr_mean': 85, 'hr_std': 10,
            'eda_trend': 0.02, 'eda_volatility': 0.5,
            'move_mag': 2.0, 'temp_base': 37.0
        }
    elif emotion == 'Aggressive':
        return {
            'hr_mean': 110, 'hr_std': 15,
            'eda_trend': 0.05, 'eda_volatility': 1.5,
            'move_mag': 8.0, 'temp_base': 37.5
        }
    return None

def generate_signals():
    # 1. User Input
    try:
        total_duration_sec = int(input("Enter total simulation duration (seconds): "))
    except ValueError:
        print("Invalid input. Defaulting to 60 seconds.")
        total_duration_sec = 60

    print(f"Generating {total_duration_sec} seconds of data...")

    # Data containers
    timestamps = []
    bvp_data = []
    eda_data = []
    temp_data = []
    ibi_data = []
    gyro_x, gyro_y, gyro_z = [], [], []
    emotion_labels = []

    current_time = 0
    
    # Initial baseline values
    current_eda = 5.0  # microsiemens
    current_temp = 37.0 # Celsius

    while current_time < total_duration_sec:
        # Random emotion selection
        emotion = random.choice(['Happy', 'Sad', 'Aggressive'])
        
        # Random duration for this emotion state (between 5s and 20s)
        # Clip if it exceeds total remaining time
        state_duration = random.randint(5, 20)
        if current_time + state_duration > total_duration_sec:
            state_duration = total_duration_sec - current_time

        params = get_emotion_params(emotion)
        num_samples = state_duration * SAMPLING_RATE
        
        # --- Time Vector for this chunk ---
        t_chunk = np.linspace(current_time, current_time + state_duration, num_samples, endpoint=False)
        timestamps.extend(t_chunk)
        
        # --- 1. BVP & IBI Generation ---
        # Simulate heart rate variability for this chunk
        current_hr = np.random.normal(params['hr_mean'], params['hr_std'])
        freq_hz = current_hr / 60.0
        
        # BVP is a sine wave + harmonics
        bvp_chunk = np.sin(2 * np.pi * freq_hz * t_chunk) 
        # Add dicrotic notch (secondary wave)
        bvp_chunk += 0.5 * np.sin(2 * np.pi * (freq_hz * 2) * t_chunk)
        # Add noise
        bvp_chunk += generate_noise(num_samples, magnitude=0.5)
        bvp_data.extend(bvp_chunk)

        # IBI (Inter-beat Interval in ms) is inverse of freq
        # We add jitter to simulate HRV
        ibi_val = (1000 / freq_hz) 
        ibi_chunk = np.random.normal(ibi_val, 20, num_samples) # 20ms jitter
        ibi_data.extend(ibi_chunk)

        # --- 2. EDA Generation ---
        # Random walk (drift) + Phasic bursts (spikes)
        eda_drift = np.linspace(0, params['eda_trend'] * state_duration, num_samples)
        eda_noise = generate_noise(num_samples, magnitude=params['eda_volatility'])
        # Create the chunk
        eda_chunk = current_eda + eda_drift + eda_noise
        eda_chunk = np.maximum(eda_chunk, 0.1) # EDA can't be negative
        eda_data.extend(eda_chunk)
        current_eda = eda_chunk[-1] # Update baseline for next loop

        # --- 3. Temperature Generation ---
        # Very slow shift towards target temp
        temp_target = params['temp_base']
        temp_chunk = np.linspace(current_temp, temp_target, num_samples)
        temp_chunk += np.random.normal(0, 0.05, num_samples) # Sensor noise
        temp_data.extend(temp_chunk)
        current_temp = temp_chunk[-1]

        # --- 4. Gyroscope (3 Axis) ---
        # White noise modulated by movement magnitude
        mag = params['move_mag']
        gx = np.random.normal(0, mag, num_samples) + generate_noise(num_samples, mag)
        gy = np.random.normal(0, mag, num_samples) + generate_noise(num_samples, mag)
        gz = np.random.normal(0, mag, num_samples) + generate_noise(num_samples, mag)
        gyro_x.extend(gx)
        gyro_y.extend(gy)
        gyro_z.extend(gz)

        # Track emotion label for reference
        emotion_labels.extend([emotion] * num_samples)

        current_time += state_duration

    # Create DataFrame
    df = pd.DataFrame({
        'Time': timestamps,
        'BVP': bvp_data,
        'EDA': eda_data,
        'Temp': temp_data,
        'IBI': ibi_data,
        'Gyro_X': gyro_x,
        'Gyro_Y': gyro_y,
        'Gyro_Z': gyro_z,
        'Emotion_Label': emotion_labels
    })

    return df

def visualize_data(df):
    print("Visualizing data...")
    
    # 1. Individual Signals Plot
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('Synthetic Physiological Signals (Noisy/Wearable Simulation)', fontsize=16)

    # Plot BVP
    axes[0].plot(df['Time'], df['BVP'], color='red', linewidth=0.8)
    axes[0].set_ylabel('BVP (Amplitude)')
    axes[0].set_title('Blood Volume Pulse (BVP)')

    # Plot EDA
    axes[1].plot(df['Time'], df['EDA'], color='purple')
    axes[1].set_ylabel('Conductance (\u00b5S)')
    axes[1].set_title('Electrodermal Activity (EDA)')

    # Plot Gyro
    axes[2].plot(df['Time'], df['Gyro_X'], label='X', alpha=0.6)
    axes[2].plot(df['Time'], df['Gyro_Y'], label='Y', alpha=0.6)
    axes[2].plot(df['Time'], df['Gyro_Z'], label='Z', alpha=0.6)
    axes[2].set_ylabel('dps')
    axes[2].set_title('3-Axis Gyroscope')
    axes[2].legend(loc='upper right')

    # Plot Temp
    axes[3].plot(df['Time'], df['Temp'], color='orange')
    axes[3].set_ylabel('Temp (°C)')
    axes[3].set_title('Skin Temperature')

    # Plot Emotion Labels (as a step chart background)
    # We map emotions to numeric levels for visualization
    emo_map = {'Sad': 1, 'Happy': 2, 'Aggressive': 3}
    df['Emo_Num'] = df['Emotion_Label'].map(emo_map)
    
    axes[4].step(df['Time'], df['Emo_Num'], where='post', color='blue')
    axes[4].set_yticks([1, 2, 3])
    axes[4].set_yticklabels(['Sad', 'Happy', 'Aggressive'])
    axes[4].set_ylabel('Emotion State')
    axes[4].set_xlabel('Time (s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 2. Combined Graph (Normalized to fit on one plot)
    # We normalize columns just for this visualization
    plt.figure(figsize=(12, 6))
    cols_to_plot = ['BVP', 'EDA', 'Temp', 'Gyro_X']
    
    for col in cols_to_plot:
        normalized = (df[col] - df[col].mean()) / df[col].std()
        plt.plot(df['Time'], normalized, label=col, alpha=0.7)
        
    plt.title("Combined Normalized Signals Overlay")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude (Z-Score)")
    plt.legend()
    plt.show()

def save_data(df):
    # Create directory if not exists
    if not os.path.exists('output_data'):
        os.makedirs('output_data')

    print("Saving files to /output_data folder...")

    # Save Combined
    df.to_csv('output_data/combined_signals.csv', index=False)
    
    # Save Individual
    df[['Time', 'BVP']].to_csv('output_data/signal_bvp.csv', index=False)
    df[['Time', 'EDA']].to_csv('output_data/signal_eda.csv', index=False)
    df[['Time', 'Temp']].to_csv('output_data/signal_temp.csv', index=False)
    df[['Time', 'IBI']].to_csv('output_data/signal_ibi.csv', index=False)
    df[['Time', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']].to_csv('output_data/signal_gyro.csv', index=False)
    
    print("Data saved successfully.")

if __name__ == "__main__":
    data = generate_signals()
    save_data(data)
    visualize_data(data)
