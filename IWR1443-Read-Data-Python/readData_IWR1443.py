import serial
import time
import numpy as np

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui

# Change the configuration file name
CONFIG_FILE_NAME = 'profile2_091319.cfg'

CLIport = {}
Dataport = {}
byte_buffer = np.zeros(2 ** 15, dtype='uint8')
byte_buffer_length = 0


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serial_config(config_file_name):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    #    CLIport = serial.Serial('COM3', 115200)
    #    Dataport = serial.Serial('COM4', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(config_file_name)]
    for line in config:
        CLIport.write((line + '\n').encode())
        print(line)
        time.sleep(0.01)

    return CLIport, Dataport


# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parse_config_file(config_file_name):
    config_parameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(config_file_name)]
    for line in config:

        # Split the line
        split_words = line.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        num_Rx_ant = 4
        num_Tx_ant = 3

        # Get the information about the profile configuration
        if "profileCfg" in split_words[0]:
            startFreq = int(float(split_words[2]))
            idleTime = int(split_words[3])
            rampEndTime = float(split_words[5])
            freqSlopeConst = float(split_words[8])
            numAdcSamples = int(split_words[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(split_words[11])

        # Get the information about the frame configuration
        elif "frameCfg" in split_words[0]:

            chirpStartIdx = int(split_words[1])
            chirpEndIdx = int(split_words[2])
            numLoops = int(split_words[3])
            numFrames = int(split_words[4])
            framePeriodicity = int(split_words[5])

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    config_parameters["numDopplerBins"] = numChirpsPerFrame / num_Tx_ant
    config_parameters["numRangeBins"] = numAdcSamplesRoundTo2
    config_parameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    config_parameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * config_parameters["numRangeBins"])
    config_parameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * config_parameters["numDopplerBins"] * num_Tx_ant)
    config_parameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    config_parameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * num_Tx_ant)

    return config_parameters


# ------------------------------------------------------------------

# Function to read and parse the incoming data
def read_and_parse_data_14xx(Dataport, config_parameters):
    global byte_buffer, byte_buffer_length

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    max_buffer_size = 2 ** 15
    magic_word = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magic_ok = 0  # Checks if magic number has been read
    data_ok = 0  # Checks if the data has been read correctly
    frame_number = 0
    det_obj = {}

    read_buffer = Dataport.read(Dataport.in_waiting)
    byte_vec = np.frombuffer(read_buffer, dtype='uint8')
    byte_count = len(byte_vec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byte_buffer_length + byte_count) < max_buffer_size:
        byte_buffer[byte_buffer_length:byte_buffer_length + byte_count] = byte_vec[:byte_count]
        byte_buffer_length += byte_count

    # Check that the buffer has some data
    if byte_buffer_length > 16:

        # Check for all possible locations of the magic word
        possible_locs = np.where(byte_buffer == magic_word[0])[0]

        # Confirm that is the beginning of the magic word and store the index in start_idx
        start_idx = []
        for loc in possible_locs:
            check = byte_buffer[loc:loc + 8]
            if np.all(check == magic_word):
                start_idx.append(loc)

        # Check that start_idx is not empty
        if start_idx:

            # Remove the data before the first start index
            if start_idx[0] > 0:
                byte_buffer[:byte_buffer_length - start_idx[0]] = byte_buffer[start_idx[0]:byte_buffer_length]
                byte_buffer_length = byte_buffer_length - start_idx[0]

            # Check that there have no errors with the byte buffer length
            if byte_buffer_length < 0:
                byte_buffer_length = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byte_buffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byte_buffer_length >= totalPacketLen) and (byte_buffer_length != 0):
                magic_ok = 1

    # If magic_ok is equal to 1 then process the message
    if magic_ok:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byte_buffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byte_buffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byte_buffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byte_buffer[idX:idX + 4], word), 'x')
        idX += 4
        frame_number = np.matmul(byte_buffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byte_buffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byte_buffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byte_buffer[idX:idX + 4], word)
        idX += 4

        # Read the TLV messages
        for tlv_idx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byte_buffer[idX:idX + 4], word)
            idX += 4
            tlv_length = np.matmul(byte_buffer[idX:idX + 4], word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # word array to convert 4 bytes to a 16 bit number
                word = [1, 2 ** 8]
                tlv_numObj = np.matmul(byte_buffer[idX:idX + 2], word)
                idX += 2
                tlv_xyzQFormat = 2 ** np.matmul(byte_buffer[idX:idX + 2], word)
                idX += 2

                # Initialize the arrays
                range_idx = np.zeros(tlv_numObj, dtype='int16')
                doppler_idx = np.zeros(tlv_numObj, dtype='int16')
                peak_val = np.zeros(tlv_numObj, dtype='int16')
                x = np.zeros(tlv_numObj, dtype='int16')
                y = np.zeros(tlv_numObj, dtype='int16')
                z = np.zeros(tlv_numObj, dtype='int16')

                for objectNum in range(tlv_numObj):
                    # Read the data for each object
                    range_idx[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2
                    doppler_idx[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2
                    peak_val[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2
                    x[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2
                    y[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2
                    z[objectNum] = np.matmul(byte_buffer[idX:idX + 2], word)
                    idX += 2

                # Make the necessary corrections and calculate the rest of the data
                range_val = range_idx * config_parameters["rangeIdxToMeters"]
                doppler_idx[doppler_idx > (config_parameters["numDopplerBins"] / 2 - 1)] = \
                    doppler_idx[doppler_idx > (config_parameters["numDopplerBins"] / 2 - 1)] - 65535
                doppler_val = doppler_idx * config_parameters["dopplerResolutionMps"]
                # x[x > 32767] = x[x > 32767] - 65536
                # y[y > 32767] = y[y > 32767] - 65536
                # z[z > 32767] = z[z > 32767] - 65536
                x = x / tlv_xyzQFormat
                y = y / tlv_xyzQFormat
                z = z / tlv_xyzQFormat

                # Store the data in the det_obj dictionary
                det_obj = {"numObj": tlv_numObj, "range_idx": range_idx, "range": range_val, "doppler_idx": doppler_idx,
                          "doppler": doppler_val, "peak_val": peak_val, "x": x, "y": y, "z": z
                          }

                data_ok = 1

                print(det_obj['range'].mean())

            elif tlv_type == MMWDEMO_UART_MSG_RANGE_PROFILE:
                idX += tlv_length

        # Remove already processed data
        if idX > 0 and data_ok == 1:
            shift_size = idX

            byte_buffer[:byte_buffer_length - shift_size] = byte_buffer[shift_size:byte_buffer_length]
            byte_buffer_length = byte_buffer_length - shift_size

            # Check that there are no errors with the buffer length
            if byte_buffer_length < 0:
                byte_buffer_length = 0

    return data_ok, frame_number, det_obj


# ------------------------------------------------------------------

# Function to update the data and display in the plot
def update(config_parameters):
    data_ok = 0
    global det_obj
    x = []
    y = []

    # Read and parse the received data
    data_ok, frame_number, det_obj = read_and_parse_data_14xx(Dataport, config_parameters)

    if data_ok:
        # print(det_obj)
        x = -det_obj["x"]
        y = det_obj["y"]
        print(f'x: {x}, y: {y}')
    # s.setData(x,y)
    # QtGui.QApplication.processEvents()

    return data_ok


# -------------------------    MAIN   -----------------------------------------
def main():
    # Configurate the serial port
    CLIport, Dataport = serial_config(CONFIG_FILE_NAME)

    # Get the configuration parameters from the configuration file
    config_parameters = parse_config_file(CONFIG_FILE_NAME)

    # START QtAPP for the plot
    # app = QtGui.QApplication([])

    # Set the plot
    # pg.setConfigOption('background','w')
    # win = pg.GraphicsWindow(title="2D scatter plot")
    # p = win.addPlot()
    # p.setXRange(-2,2)
    # p.setYRange(0,4)
    # p.setLabel('left',text = 'Y position (m)')
    # p.setLabel('bottom', text= 'X position (m)')
    # s = p.plot([],[],pen=None,symbol='o')


    # Main loop
    det_obj = {}
    frame_data = {}
    current_index = 0
    while True:
        try:
            # Update the data and check if the data is okay
            data_ok = update(config_parameters)

            if data_ok:
                # Store the current frame into frame_data
                frame_data[current_index] = det_obj
                current_index += 1
                print(f'Current index = {current_index}')

            time.sleep(0.2)  # Sampling frequency of ___

        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write('sensorStop\n'.encode())
            CLIport.close()
            Dataport.close()
            break


if __name__ == '__main__':
    main()
