"""Varint tests: exact bytes for hand-worked values, decoding mid-stream, streams, round trips."""
import random

import pytest

from engine_py.varint import decode_varint, encode_varint


def encode(n):
    buf=bytearray()
    encode_varint(n,buf)
    return bytes(buf)

def decode_stream(data):
    values=[]
    i=0
    while i<len(data):
        value,i=decode_varint(data,i)
        values.append(value)
    return values


# ---- exact bytes ----

@pytest.mark.parametrize("n,EXPECTED", [
    (0,   b"\x00"),          # zero still takes one byte
    (5,   b"\x05"),          # [0|0000101]
    (127, b"\x7f"),          # largest 1-byte value
    (128, b"\x80\x01"),      # [1|0000000] [0|0000001]: smallest 2-byte value
    (300, b"\xac\x02"),      # [1|0101100] [0|0000010]
])
def test_varint_encode_bytes(n,EXPECTED):
    assert encode(n) == EXPECTED

@pytest.mark.parametrize("n,EXPECTED", [(127,1),(128,2),(16383,2),(16384,3),(2**21-1,3),(2**21,4)])
def test_varint_byte_sizes(n,EXPECTED):
    assert len(encode(n)) == EXPECTED

def test_varint_encode_appends_to_existing_buffer():
    buf=bytearray(b"\x01")
    encode_varint(300,buf)
    assert bytes(buf) == b"\x01\xac\x02"


# ---- decoding ----

def test_varint_decode_returns_next_index():
    data=b"\xac\x02\x05"
    EXPECTED=(300,2)
    assert decode_varint(data,0) == EXPECTED

def test_varint_decode_mid_stream():
    data=b"\xac\x02\x05"
    EXPECTED=(5,3)
    assert decode_varint(data,2) == EXPECTED

def test_varint_decode_multibyte_mid_stream():
    # a 2-byte number that doesn't start at index 0: shift must restart at 0 for each number
    data=b"\x05\xac\x02"
    EXPECTED=(300,3)
    assert decode_varint(data,1) == EXPECTED

def test_varint_decode_stream():
    # [0|0000011] [1|0000000] [0|0000001] [0|0000101]  ->  3, 128, 5
    data=bytes([0b00000011,0b10000000,0b00000001,0b00000101])
    EXPECTED=[3,128,5]
    assert decode_stream(data) == EXPECTED


# ---- round trips ----

@pytest.mark.parametrize("n", [0,1,127,128,16383,16384,5182,2**31,10**12])
def test_varint_round_trip(n):
    data=encode(n)
    assert decode_varint(data,0) == (n,len(data))

def test_varint_round_trip_random_stream():
    rng=random.Random(42)
    numbers=[rng.randrange(0,2**rng.randrange(1,40)) for _ in range(10000)]
    buf=bytearray()
    for n in numbers:
        encode_varint(n,buf)
    assert decode_stream(bytes(buf)) == numbers
