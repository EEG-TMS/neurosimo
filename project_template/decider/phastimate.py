import numpy as np
from scipy.signal import firls, filtfilt, hilbert
from spectrum import aryule

import csv

class Decider:
    def __init__(self, num_of_eeg_channels, num_of_emg_channels, sampling_frequency):
        """Initialize the Decider with parameters and filter design."""
        self.sampling_frequency = sampling_frequency

        # Parameters for the phastimate function
        self.hilbert_window = 64
        self.edge = 35
        self.ar_order = 15

        self.downsample_ratio = 10  # Downsampling factor

        self.processing_interval_in_seconds = 1  # Process every 1 second
        self.buffer_size_in_seconds = 1         # Buffer 1 second of data

        self.processing_interval_in_samples = int(self.processing_interval_in_seconds * self.sampling_frequency)
        self.buffer_size_in_samples = int(self.buffer_size_in_seconds * self.sampling_frequency)

        self.bpf = np.array([-0.0001, -0.0022, -0.0045, -0.0069, -0.0094, -0.0118, -0.0142, -0.0164, -0.0184, -0.0202, -0.0217, -0.0228, -0.0235, -0.0237, -0.0235, -0.0228,
                    -0.0216, -0.0199, -0.0177, -0.0152, -0.0122, -0.0088, -0.0053, -0.0015, 0.0025, 0.0064, 0.0104, 0.0142, 0.0177, 0.0210, 0.0240, 0.0264,
                    0.0284, 0.0299, 0.0308, 0.0311, 0.0308, 0.0299, 0.0284, 0.0264, 0.0240, 0.0210, 0.0177, 0.0142, 0.0104, 0.0064, 0.0025, -0.0015,
                    -0.0053, -0.0088, -0.0122, -0.0152, -0.0177, -0.0199, -0.0216, -0.0228, -0.0235, -0.0237, -0.0235, -0.0228, -0.0217, -0.0202, -0.0184, -0.0164,
                    -0.0142, -0.0118, -0.0094, -0.0069, -0.0045, -0.0022, -0.0001])

        # Target phase for synchronization
        self.target_phase = np.pi  # Use numpy's pi

        # Maximum number of future samples to consider (configurable)
        self.max_future_samples = int(self.edge / 2)

        self.tolerance = 0.05  # Tolerance for phase difference

    def get_configuration(self):
        """Return the configuration for the processing interval and sample window."""
        return {
            'processing_interval_in_samples': self.processing_interval_in_samples,
            'process_on_trigger': False,
            'sample_window': [-(self.buffer_size_in_samples - 1), 0],
            'events': [],
        }

    def process(self, current_time, timestamps, valid_samples, eeg_buffer, emg_buffer,
                current_sample_index, ready_for_trial, is_trigger, is_event, event_type):
        """Process the EEG data to estimate phase and schedule a trigger."""
        if not ready_for_trial:
            return

        if not np.all(valid_samples):
            return

        # Extract and reference C3 data directly
        try:
            data = eeg_buffer[:, 4] - 0.25 * np.sum(eeg_buffer[:, [20, 22, 24, 26]], axis=1)
        except IndexError:
            # Handle the case where the EEG buffer does not have the expected number of channels
            return

        # Demean the data in-place
        data -= np.mean(data)

        # Downsample the data
        downsampled_data = data[::self.downsample_ratio]

        # Run phastimate to estimate future phases
        estimated_phases, _ = self.phastimate(
            downsampled_data,
            self.bpf, [1.0], self.edge, self.ar_order, self.hilbert_window)

        if estimated_phases is None:
            return  # Not enough data in phastimate

        # Process estimated phases
        num_estimated_samples = estimated_phases.shape[0]
        future_samples = estimated_phases[int(num_estimated_samples / 2):]

        # Limit the future samples to the maximum number specified
        future_samples = future_samples[:self.max_future_samples]

        # Compute the angular difference between future samples and target phase
        phase_difference = np.angle(np.exp(1j * (future_samples - self.target_phase)))

        # Find the index where the phase difference is smallest
        index_of_peak = np.argmin(np.abs(phase_difference))

        if np.abs(phase_difference[index_of_peak]) > self.tolerance:
            print('Phase difference exceeds tolerance: {:.3f}'.format(phase_difference[index_of_peak]))
            return

        # Compute execution time for the trigger
        time_from_now = (index_of_peak * self.downsample_ratio) / self.sampling_frequency
        execution_time = current_time + time_from_now

        print(f'Execution time from now: {time_from_now:.3f} s')

        # Issue the timed trigger
        return {'timed_trigger': execution_time}

    def phastimate(self, data, filter_b, filter_a, edge, ar_order, hilbert_window,
                   offset_correction=0, iterations=None, armethod='aryule'):
        """Estimate the phase of the EEG signal using autoregressive modeling and Hilbert transform."""
        if iterations is None:
            iterations = edge + int(np.ceil(hilbert_window / 2))

        # Ensure data length is sufficient for filtering
        padlen = 3 * (max(len(filter_a), len(filter_b)) - 1)
        if data.shape[0] <= padlen:
            return None, None  # Not enough data

        # Apply band-pass filter (BPF)
        data_filtered = filtfilt(filter_b, filter_a, data)

        # Remove edge samples to mitigate filter transients
        if data_filtered.shape[0] <= 2 * edge:
            return None, None  # Not enough data after removing edge samples
        data_no_edge = data_filtered[edge:-edge]

        # Determine AR parameters
        x = data_no_edge
        if len(x) < ar_order:
            return None, None  # Not enough data for AR model

        if armethod == 'aryule':
            a, _, _ = aryule(x, ar_order)
            actual_ar_order = len(a)  # Actual AR order used
            coefficients = -1 * a[::-1]  # Flip and negate all coefficients
        else:
            raise ValueError('Unknown AR method')

        # Prepare vector for forward prediction
        total_samples = len(data_no_edge) + iterations
        data_predicted = np.zeros(total_samples)
        data_predicted[:len(data_no_edge)] = data_no_edge

        # Extend the data array for forward prediction
        data_predicted = np.concatenate((data_no_edge, np.ones(iterations, dtype=np.float64)))

        # Run the forward prediction
        for i in range(iterations):
            idx = len(data_no_edge) + i
            data_segment = data_predicted[idx - actual_ar_order:idx]
            data_predicted[idx] = np.sum(coefficients * data_segment)

        # Extract the last hilbert_window samples for the Hilbert transform
        if data_predicted.shape[0] < hilbert_window:
            return None, None  # Not enough data for Hilbert transform
        hilbert_window_data = data_predicted[-hilbert_window:]

        # Compute the analytic signal and phase
        analytic_signal = hilbert(hilbert_window_data)
        phase = np.angle(analytic_signal)
        amplitude = np.abs(analytic_signal)

        return phase, amplitude
