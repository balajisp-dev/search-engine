"""Variable-length integers (LEB128 style). YOU implement this.

Each byte holds 7 bits of the number, lowest group first. The top bit (0x80) is a flag:
1 = another byte follows, 0 = this is the last byte of the number.

    5    -> [0|0000101]                  b'\x05'
    128  -> [1|0000000] [0|0000001]      b'\x80\x01'
    300  -> [1|0101100] [0|0000010]      b'\xac\x02'

Toolkit: n & 0x7F (lowest 7 bits), n >> 7 (drop them), b | 0x80 (set flag),
b & 0x80 (is flag set?), (b & 0x7F) << shift (place a group back).
"""


def encode_varint(n: int, buf: bytearray) -> None:
    """Append the varint bytes of n (n >= 0) to buf."""
    while True:
        last_7_bits=n & 0x7F
        n=n>>7
        if n>0:
            buf.append(last_7_bits | 0x80)
        else:
            buf.append(last_7_bits)
            break



     
def decode_varint(data: bytes, i: int) -> tuple[int, int]:
    """Read one varint starting at data[i]. Return (value, index of the byte after it)."""
    result=0
    flag=1
    shift=0
    while flag:
        result=result | ((data[i] & 0x7F)<<shift)
        flag=data[i] & 0x80
        if flag:i=i+1
        shift+=7
    return (result,i+1)



