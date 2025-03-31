import asyncio
import struct
import csv
import numpy as np 

from bleak import BleakScanner, BleakClient, BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

SENSOR_SERVICE_UUID = "0000beef-0000-1000-8000-00805f9b34fb"
SENSOR_CHARACTERISTIC_UUID = '0000beef-0000-1000-8000-00805f9b34fb'

class fixed_size_buffer:
    """
    A fixed-size buffer implemented using a NumPy array, suitable for sliding window operations.
    """

    def __init__(self, size, dimensions, dtype=float):
        """
        Initializes the buffer.

        Args:
            size (int): The maximum size of the buffer (window size).
            dtype (type): The data type of the buffer elements.
        """
        self.size = size
        self.buffer = np.zeros((size, dimensions), dtype=dtype)
        self.index = 0  # Current insertion index
        self.is_full = False

    def append(self, value):
        """
        Appends a value to the buffer, overwriting the oldest value if full.

        Args:
            value: The value to append.
        """
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0 and not self.is_full:
            self.is_full = True

    def get(self):
        """
        Retrieves the current contents of the buffer.

        Returns:
            numpy.ndarray: The buffer contents. If the buffer is not full, the returned array
                           contains only the appended values. If full, it contains all values,
                           with the oldest values appearing first.
        """
        if self.is_full:
            return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))
        else:
            return self.buffer[:self.index]

    def __len__(self):
        """
        Returns the current number of elements in the buffer.
        """
        if self.is_full:
            return self.size
        else:
            return self.index
    def clear(self):
        """
        Clears the buffer, resetting it to its initial state.
        """
        self.buffer.fill(0)
        self.index = 0
        self.is_full = False

def detect_peak(buffer):
    # detect peak for buffer 
    
    mps = 2.2369362920544
    v = np.sqrt((0.385 * (peak_acc * 0.250)**2)/0.160)
    s = v * mps
    if s > 30:
        print(f'peak {s:.2f} mph')

async def sensor_connect(dev: BLEDevice, done: asyncio.Event):
    #f = open("data-stick.csv", mode ="w")
    #writer = csv.writer(f)
    # accumulate the last 250 ms 
    bucket_size = 250 
    bucket_dim = 8
    bucket = np.zeros((bucket_size, bucket_dim), dtype=np.float64)
    bucket_pos = 0

    # window keeping the last three seconds aggregates in 250ms 
    # buckets, sliding by 250ms 
    wnd_size = 12
    wnd_dim = 8
    wnd_pos = 0
    wnd = fixed_size_buffer(wnd_size, wnd_dim) 

    # remember the millisecond of the last peak so we do not 
    # output for the same point acceleration multiple times
    peak_ms = 0

    def disconnected(c: BleakClient):
        print('disconnected from ', c.address)
        done.set()

    def on_data(c: BleakGATTCharacteristic, data: bytearray):
        nonlocal wnd_pos, bucket_pos, peak_ms

        m = struct.unpack("<Qffffff", data)

        # read measurements 
        data = list(m)
        ts, gyro, lin_acc = data[0], data[1:4], data[4:7]

        #writer.writerow(data)
        #print('ts: %s, gyro: %s, lin acc: %s' % (ts, gyro, lin_acc))

        # calculate milliseconds and acceleration
        ms = np.int64(ts/1000)
        acc = np.sqrt(lin_acc[0]**2+lin_acc[1]**2+lin_acc[2]**2)

        # store new measurement 
        measurement = [ms, acc] + lin_acc + gyro
        bucket[wnd_pos] = measurement 
        wnd_pos += 1

        # check if 250ms buffer was filled 
        if wnd_pos >= bucket_size:
            # get max across all metrics and add it to window
            max_measurement = bucket.max(axis=0)
            wnd.append(max_measurement)

            # clear the buffer to collect the next 250 points 
            wnd_pos = 0
            bucket.fill(0)

            # check if the window was filled, if it was calculate 
            # peak acceleration 
            if len(wnd) >= wnd_size:
                data = wnd.get()

                # get the max acc and remember position 
                # acceleration is the second column
                index = np.argmax(data[:, 1])
                ms = data[index, 0]
                acc = data[index, 1]

                # check this point is different than the last one
                if peak_ms != ms:
                    # calculate miles per hour 
                    mps = 2.2369362920544
                    v = np.sqrt((0.385 * (acc * 0.250)**2)/0.160)
                    s = v * mps
                    if s > 20:
                        # update last peak timestamp and print value
                        peak_ms = ms 
                        print(f'peak at {ms} - {s:.2f} mph')                        

        # ts, gx, gy, gz, ax, ay, az = m
        # print('{} ->  ({}, {}, {}) ({}, {}, {})'.format(*m))
        # print('Gyro  x {:5.0f} y {:5.0f} z {:5.0f}'.format(gx, gy, gz))
        # print('Accel x {:5.1f} y {:5.1f} z {:5.1f}'.format(ax, ay, az))

    async with BleakClient(dev.address, disconnected_callback=disconnected) as client:
        print('connecting to ', dev.address)
        await client.connect()

        await client.start_notify(SENSOR_SERVICE_UUID, on_data)
        print('waiting on disconnect')
        await done.wait()

    f.close()



def is_sensor_device(d: BLEDevice, a: AdvertisementData):
    return SENSOR_SERVICE_UUID in a.service_uuids


async def main():
    scanner = BleakScanner()

    while True:
        dev = await scanner.find_device_by_filter(is_sensor_device)
        if dev:
            done = asyncio.Event()
            await sensor_connect(dev, done)
            await done.wait()
        await asyncio.sleep(10)


asyncio.run(main())
