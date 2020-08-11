import serial
import time
import numpy as np
from sklearn.cluster import MeanShift

# Constants
OBJ_STRUCT_SIZE_BYTES = 12
BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
MMWDEMO_UART_MSG_DETECTED_POINTS = 1
MMWDEMO_UART_MSG_RANGE_PROFILE = 2
max_buffer_size = 2 ** 15
magic_word = [2, 1, 4, 3, 6, 5, 8, 7]

# word array to convert 4 bytes to a 32 bit number
word_32 = [1, 2 ** 8, 2 ** 16, 2 ** 24]

# word array to convert 4 bytes to a 16 bit number
word_16 = [1, 2 ** 8]

class MMWave:
    def __init__(self, 
                 config_file_name = '1443_3d.cfg', 
                 cli_port = '/dev/ttyACM0', 
                 data_port = '/dev/ttyACM1',
                 cli_baud = 115200,
                 data_baud = 921600,
                 num_rx_ant = 4,
                 num_tx_ant = 3):
        self.config_file_name = config_file_name
        self.cli_port = serial.Serial(cli_port, cli_baud)
        self.data_port = serial.Serial(data_port, data_baud)

        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant

        self.byte_buffer = np.zeros(2 ** 15, dtype='uint8')
        self.byte_buffer_length = 0

        self.serial_config()
        self.parse_config_file()

    # Function to configure the serial ports and send the data from
    # the configuration file to the radar
    def serial_config(self):
        config = [line.rstrip('\r\n') for line in open(self.config_file_name)]
        for line in config:
            self.cli_port.write((line + '\n').encode())
            print(line)
            time.sleep(0.01)

    # Function to parse the data inside the configuration file
    def parse_config_file(self):
        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(self.config_file_name)]

        for line in config:
            # Split the line
            split_words = line.split(' ')

            # Get the information about the profile configuration
            if 'profileCfg' in split_words[0]:
                start_freq = int(float(split_words[2]))
                idle_time = int(split_words[3])
                ramp_end_time = float(split_words[5])
                freq_slope_const = float(split_words[8])
                num_adc_samples = int(split_words[10])
                num_adc_samples_round_to_2 = 1

                while num_adc_samples > num_adc_samples_round_to_2:
                    num_adc_samples_round_to_2 = num_adc_samples_round_to_2 * 2

                dig_out_sample_rate = int(split_words[11])

            # Get the information about the frame configuration
            elif 'frameCfg' in split_words[0]:
                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                num_frames = int(split_words[4])
                frame_periodicity = int(split_words[5])

        # Combine the read data to obtain the configuration parameters
        num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
        num_doppler_bins     = num_chirps_per_frame / self.num_tx_ant

        self.config_parameters = {
            'num_doppler_bins'       : num_doppler_bins,
            'num_range_bins'         : num_adc_samples_round_to_2,
            'range_resolution_meters': (3e8 * dig_out_sample_rate * 1e3) / \
                                       (2 * freq_slope_const * 1e12 * num_adc_samples),
            'range_idx_to_meters'    : (3e8 * dig_out_sample_rate * 1e3) / \
                                       (2 * freq_slope_const * 1e12 * num_adc_samples_round_to_2),
            'doppler_resolution_mps' : 3e8 / (2 * start_freq * 1e9 * (idle_time + ramp_end_time) * \
                                              1e-6 * num_doppler_bins * self.num_tx_ant),
            'max_range'              : (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3),
            'max_velocity'           : 3e8 / (4 * start_freq * 1e9 * (idle_time + ramp_end_time) * \
                                              1e-6 * self.num_tx_ant),
            'num_frames'             : num_frames,
            'frame_periodicity'      : frame_periodicity
        }

    # Function to read and parse the incoming data
    def _read_data(self):
        # Initialize variables
        magic_ok = 0  # Checks if magic number has been read
        data_ok = 0  # Checks if the data has been read correctly
        frame_number = 0
        det_obj = {}

        read_buffer = self.data_port.read(self.data_port.in_waiting)
        byte_vec = np.frombuffer(read_buffer, dtype='uint8')
        byte_count = len(byte_vec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (self.byte_buffer_length + byte_count) < max_buffer_size:
            self.byte_buffer[self.byte_buffer_length:self.byte_buffer_length + byte_count] = \
                byte_vec[:byte_count]

            self.byte_buffer_length += byte_count

        # Check that the buffer has some data
        if self.byte_buffer_length > 16:

            # Check for all possible locations of the magic word
            possible_locs = np.where(self.byte_buffer == magic_word[0])[0]

            # Confirm that is the beginning of the magic word and store the index in start_idx
            start_idx = []
            for loc in possible_locs:
                check = self.byte_buffer[loc:loc + 8]
                if np.all(check == magic_word):
                    start_idx.append(loc)

            # Check that start_idx is not empty
            if start_idx:

                # Remove the data before the first start index
                if start_idx[0] > 0:
                    self.byte_buffer[:self.byte_buffer_length - start_idx[0]] = \
                        self.byte_buffer[start_idx[0]:self.byte_buffer_length]

                    self.byte_buffer_length -= start_idx[0]

                # Check that there have no errors with the byte buffer length
                if self.byte_buffer_length < 0:
                    self.byte_buffer_length = 0

                # Read the total packet length
                total_pac_len = np.matmul(self.byte_buffer[12:12 + 4], word_32)

                # Check that all the packet has been read
                if (self.byte_buffer_length >= total_pac_len) and (self.byte_buffer_length != 0):
                    magic_ok = 1

        # If magic_ok is equal to 1 then process the message
        if magic_ok:
            

            # Initialize the pointer index
            idx = 0

            # Read the header
            magic_number = self.byte_buffer[idx:idx + 8]
            idx += 8
            version = format(np.matmul(self.byte_buffer[idx:idx + 4], word_32), 'x')
            idx += 4
            total_pac_len = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
            idx += 4
            platform = format(np.matmul(self.byte_buffer[idx:idx + 4], word_32), 'x')
            idx += 4
            frame_number = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
            idx += 4
            time_cpu_cycles = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
            idx += 4
            num_detected_obj = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
            idx += 4
            num_tlvs = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
            idx += 4

            # Read the TLV messages
            for tlv_idx in range(num_tlvs):

                # Check the header of the TLV message
                tlv_type = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
                idx += 4
                tlv_length = np.matmul(self.byte_buffer[idx:idx + 4], word_32)
                idx += 4

                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    tlv_num_obj = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                    idx += 2
                    tlv_xyz_q_format = 2 ** np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                    idx += 2

                    # Initialize the arrays
                    range_idx = np.zeros(tlv_num_obj, dtype='int16')
                    doppler_idx = np.zeros(tlv_num_obj, dtype='int16')
                    peak_val = np.zeros(tlv_num_obj, dtype='int16')
                    x = np.zeros(tlv_num_obj, dtype='int16')
                    y = np.zeros(tlv_num_obj, dtype='int16')
                    z = np.zeros(tlv_num_obj, dtype='int16')

                    for object_num in range(tlv_num_obj):
                        # Read the data for each object
                        range_idx[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2
                        doppler_idx[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2
                        peak_val[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2
                        x[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2
                        z[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2
                        y[object_num] = np.matmul(self.byte_buffer[idx:idx + 2], word_16)
                        idx += 2

                    # Make the necessary corrections and calculate the rest of the data
                    range_val = range_idx * self.config_parameters['range_idx_to_meters']
                    doppler_idx[doppler_idx > (self.config_parameters['num_doppler_bins'] / 2 - 1)] = \
                        doppler_idx[doppler_idx > (self.config_parameters['num_doppler_bins'] / 2 - 1)] - 65535
                    doppler_val = doppler_idx * self.config_parameters['doppler_resolution_mps']
                    # x[x > 32767] = x[x > 32767] - 65536
                    # y[y > 32767] = y[y > 32767] - 65536
                    # z[z > 32767] = z[z > 32767] - 65536
                    x = x / tlv_xyz_q_format
                    y = y / tlv_xyz_q_format
                    z = z / tlv_xyz_q_format

                    # Store the data in the det_obj dictionary
                    det_obj = {'num_obj': tlv_num_obj, 
                               'range_idx': range_idx, 
                               'range': range_val, 
                               'doppler_idx': doppler_idx,
                               'doppler': doppler_val, 
                               'peak_val': peak_val, 
                               'x': -x, 'y': y, 'z': z
                            }

                    data_ok = 1

                elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                    idx += tlv_length

            # Remove already processed data
            if idx > 0 and data_ok == 1:
                shift_size = idx

                self.byte_buffer[:self.byte_buffer_length - shift_size] = \
                    self.byte_buffer[shift_size:self.byte_buffer_length]

                self.byte_buffer_length -= shift_size

                # Check that there are no errors with the buffer length
                if self.byte_buffer_length < 0:
                    self.byte_buffer_length = 0

        if not data_ok:
            self.byte_buffer = np.zeros(2 ** 15, dtype='uint8')
            self.byte_buffer_length = 0
            self.data_port.flushOutput()

        return data_ok, frame_number, det_obj
    
    def update(self):
        data = None
        samples = 3
        retry = 8
        while samples > 0 and retry >= 0:
            try:
                time.sleep(self.config_parameters['frame_periodicity'] / 1000 * 1.1)
                data_ok, _, det_obj = self._read_data()
                if data_ok:
                    x = det_obj['x']
                    y = det_obj['y']
                    z = det_obj['z']
                    r = det_obj['range']
                    samples -= 1
                    for i in range(len(x)):
                        datapt = [[x[i], y[i], z[i], r[i]]]
                        if data is None:
                            data = np.array(datapt)
                        else:
                            data = np.append(data, datapt, axis=0)
                else:
                    retry -= 1
            except Exception:
                self.byte_buffer = np.zeros(2 ** 15, dtype='uint8')
                self.byte_buffer_length = 0
                retry -= 1
        print(retry)

        if data is None:
            return None

        clusters = MeanShift(bandwidth=0.25).fit(data)
        centroids = clusters.cluster_centers_

        data = []
        for centroid in centroids:
            data.append({'x': centroid[0],
                         'y': centroid[1],
                         'z': centroid[2],
                         'r': centroid[3]
                        })
        return data

    def cleanup(self):
        self.cli_port.write('sensorStop\n'.encode())
        self.cli_port.close()
        self.data_port.close()

if __name__ == '__main__':
    mmwave = MMWave()
    time.sleep(0.5)
    while True:
        try:
            print(mmwave.update())
            time.sleep(0.5)
        except KeyboardInterrupt:
            mmwave.cleanup()
            break