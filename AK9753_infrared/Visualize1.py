import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

ser = serial.Serial('COM4')

nPoints = 300
nSensors = 4
temp = 0
t = np.arange(0, nPoints)
ys = np.zeros((nPoints, nSensors))
ysDiff = np.diff(ys,1,0)
fig, (ax1, ax2) = plt.subplots(2, sharex='col')
sen1, = ax1.plot(t,ys[:,0])
sen2, = ax1.plot(t,ys[:,1])
sen3, = ax1.plot(t,ys[:,2])
sen4, = ax1.plot(t,ys[:,3])
diff1, = ax2.plot(t[1:],ysDiff[:,0])
diff2, = ax2.plot(t[1:],ysDiff[:,1])
diff3, = ax2.plot(t[1:],ysDiff[:,2])
diff4, = ax2.plot(t[1:],ysDiff[:,3])
ax1.axis([0,nPoints,-5000,9000])
ax2.axis([0,nPoints,-400,400])
tempText = ax1.text(10,8000,"Sensor temp = " + str(temp) + " C")


count = 0
def updateData():
    global count, ys, temp
    while ser.in_waiting:
        rline = str(ser.readline())
        lineSplit = rline.split("[")
        reading = np.zeros(6)
        for i in range(6):
            reading[i] = lineSplit[i+1].split("]")[0]
        for i in range(nSensors):
            ys[count,i] = reading[i]
        temp = reading[4]
        count += 1
        print(reading[0])
        if count == nPoints:
            count = 0

def animate(i):
    global count, ys, temp
    updateData()
    ysCat = np.roll(ys,-1*count,0)
    ysDiff = np.diff(ysCat,1,0)
    sen1.set_ydata(ysCat[:,0])
    sen2.set_ydata(ysCat[:,1])
    sen3.set_ydata(ysCat[:,2])
    sen4.set_ydata(ysCat[:,3])
    diff1.set_ydata(ysDiff[:,0])
    diff2.set_ydata(ysDiff[:,1])
    diff3.set_ydata(ysDiff[:,2])
    diff4.set_ydata(ysDiff[:,3])

    tempText.set_text("Sensor temp = " + str(temp) + " C")
    return

ani = animation.FuncAnimation(fig, animate, interval = 100)
plt.show()