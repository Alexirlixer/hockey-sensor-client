import asyncio
import struct
from detect import *

from bleak import BleakScanner, BleakClient, BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

SENSOR_SERVICE_UUID = "0000beef-0000-1000-8000-00805f9b34fb"
SENSOR_CHARACTERISTIC_UUID = '0000beef-0000-1000-8000-00805f9b34fb'

async def sensor_connect(dev: BLEDevice, done: asyncio.Event): 
    detector = shot_detect()

    def disconnected(c: BleakClient):
        print('disconnected from ', c.address)
        done.set()

    def on_data(c: BleakGATTCharacteristic, data: bytearray):
        m = struct.unpack("<Qffffff", data)

        # read measurements 
        data = list(m)

        # attempt shot detection 
        detector.on_measurement(data)
                  
        # ts, gyro, lin_acc = data[0], data[1:4], data[4:7]
        # print('ts: %s, gyro: %s, lin acc: %s' % (ts, gyro, lin_acc))

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
