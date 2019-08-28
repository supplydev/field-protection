from struct import unpack
import numpy as np, numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DetObj:
    """Class for detected objects that exceeded RCS and CFAR thresholds."""

    def __init__(self, slice, Q):
        # Q is the number of fractional bits in fixed point representation
        # populate the attributes of an object, given a 12-byte data slice
        (self.range, DopplerIdx, self.peakVal, x, y, z) = \
            unpack('<HhHhhh', slice)

        # TODO (determine correct scaling factor)
        self.Doppler = .01 * DopplerIdx

        # convert from Q format to meters
        self.x = x / (1 << Q)
        self.y = y / (1 << Q)
        self.z = z / (1 << Q)
        return


def parse(frm):
    """Parse a single frame's byte string into substrings for the detected objects."""
    # remove the header from the frame
    frm = frm[36:]

    # check if it's type 1 (detected object) data
    tlv_t, tlv_l = unpack('<II', frm[0:8])
    if tlv_t != 1:
        # No detected objects in this frame
        return
    # remove TLV
    frm = frm[8:]

    # remove everything after detObj
    frm = frm[0:tlv_l]
    n_objects, q = unpack('<HH', frm[0:4])

    # remove num objects and format
    frm = frm[4:]

    len_slice = len(frm) // n_objects
    if len_slice != 12:
        print("Warning: Object slices are not 12 bytes long!")

    # return a list of data slices, each corresponding to a single detObj
    slices = [frm[len_slice * i:len_slice * (i + 1)] for i in range(n_objects)]
    return (slices, q)


# ----------------------------------------------------------------------
# Begin main process
# ----------------------------------------------------------------------

fileStr = "Trial1.dat"
with open(fileStr, mode='rb') as file:  # rb is read binary
    stream = file.read()

# array of frames, with each beginning at magic word. The frame
# indices can be used as evenly spaced time markers
separated_frames = []

while len(stream) > 0:
    # get the length of the the current frame
    frame_len, = unpack('<I', stream[12:16])

    # add current frame to the list of frames
    separated_frames.append(stream[0:frame_len])

    # remove this frame from the beginning of the stream
    stream = stream[frame_len:]

n_frames = len(separated_frames)

# list containing a list of DetObj's at each time index
targets_list = [[] for _ in range(n_frames)]

for i in range(n_frames):
    parse_out = parse(separated_frames[i])
    if parse_out is None:
        print("No detected objects in frame " + str(i))
    else:
        slices_i, q_i = parse_out
        targets_list[i] = [DetObj(slices_i[j], q_i) for j in range(len(slices_i))]

n_targets = [len(targets_list[i]) for i in range(len(targets_list))]
max_n_targets = max(n_targets)

# create the mask (False corresponds to valid target)
mask = np.array([[i >= n_targets[j] for i in range(max_n_targets)] for j in range(n_frames)])
mask = np.tile(np.reshape(mask, mask.shape + (1,)), (1, 1, 2))

# nd array of all the x & y components of the targets in every frame
targets_array = np.zeros((n_frames,max_n_targets,2))
for i in range(n_frames):
    for tar_j in range(len(targets_list[i])):
        targets_array[i,tar_j,:] = np.array([getattr(targets_list[i][tar_j],coor) for coor in ["x","y"]])

# mask the array according to valid targets
targets_array = ma.masked_array(targets_array, mask)

# make another array which can be referenced by the coordinate string
targets_array2 = np.ma.zeros((n_frames, max_n_targets), dtype=[('x', float, (1,)), ('y', float, (1,))])
targets_array2['x'] = targets_array[:,:,0:1]
targets_array2['y'] = targets_array[:,:,1:]

centroids = np.ma.mean(targets_array, axis=1)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis([-2, 2, 0, 4])
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')
scat = ax.scatter(np.zeros(max_n_targets), np.zeros(max_n_targets), marker='o')


def update(i):

    offsets = np.ma.vstack((targets_array[i,:,:], centroids[i,:]))
    scat.set_offsets(offsets)
    scat.set_facecolors(['b']*(np.shape(offsets)[0] - 1) + ['r'])
    return

# stop after n_frames
animation = FuncAnimation(fig, update, frames=n_frames, interval=200, repeat=True)
plt.show()


print(len(frames))
