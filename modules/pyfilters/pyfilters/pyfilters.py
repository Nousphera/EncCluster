# Modified 2026-10-15: stable client/server hashing; source provenance in PROVENANCE.md.
from ._xorfilter import lib, ffi
from ctypes import c_ulonglong
import struct
import hashlib
import json
import numpy as np

def _stable_hash(item):
    # Modified 2026-10-15: Python string hash is salted independently per process.
    # Canonical JSON provides the same client/server prehash on every host.
    if not isinstance(item, (str, int, tuple, list)):
        raise TypeError("Filter keys must be strings, integers, or integer pairs")
    raw = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8, person=b"enccluster-bf").digest(), "little")

class Xor8:

	def __init__(self, size):
		self.__filter = ffi.new("xor8_t *")
		status = lib.xor8_allocate(size, self.__filter)
		self.size = size
		if not status:
			print("Unable to allocate memory for 8 bit filter")

	def __repr__(self):
		return "Xor8 object with size(in bytes):{}".format(self.size_in_bytes())

	def __getitem__(self, item):
		return self.contains(item)

	def __del__(self):
		lib.xor8_free(self.__filter)

	def populate(self, data: list):
		data = list(map(lambda x: _stable_hash(x), data))
		return lib.xor8_buffered_populate(data, len(data), self.__filter)

	def contains(self, item):
		item = _stable_hash(item)
		return lib.xor8_contain(item, self.__filter)

	def query(self, data: list):
		_idxs = [x for x in data if self.contains(x)]
		return _idxs

	def size_in_bytes(self):
		return lib.xor8_size_in_bytes(self.__filter)

	def serialize(self):
		buffer = bytearray(int(self.size_in_bytes()+100))
		buffer_ptr = ffi.from_buffer(buffer)
		lib.xor8_serialize(self.__filter, buffer_ptr)
		return buffer[:16+3*int(struct.unpack('QQ', buffer[:16])[-1])]

	def deserialize(self, buffer):
		buffer_ptr = ffi.from_buffer(buffer)
		lib.xor8_deserialize(buffer_ptr, self.__filter)

	def modify(self, entries):
		if set(entries.keys()).issubset(['seed','size','fingerprints']):
			# Extract all info
			buffer = bytearray(int(self.size_in_bytes()+100))
			lib.xor8_serialize(self.__filter, ffi.from_buffer(buffer),)
			buffer = buffer[:16+3*int(struct.unpack('QQ', buffer[:16])[-1])]
			if 'seed' in list(entries.keys()): # Fill bytes_arr [:8] bytes with entries[seed] element (uint64)
				struct.pack_into('Q', buffer, 0, entries['seed'])    
			if 'fingerprints' in list(entries.keys()): # Fill bytes_arr [16:] bytes with entries['fingerprints'] (array of uint16)
				struct.pack_into(f"{len(entries['fingerprints'])}B", buffer, 16, *entries['fingerprints'])
			lib.xor8_deserialize(ffi.from_buffer(buffer), self.__filter)
		else:
			raise ValueError("Invalid keys in the entries dictionary")

	@property
	def fingerprints(self):
		buffer = self.serialize()
		_, blocks_size = struct.unpack('QQ', buffer[:16])
		fingerprints = list(struct.unpack_from(f"{int(3*blocks_size)}B", buffer, 16))
		return np.asarray(fingerprints, dtype=np.uint8)

	@property
	def seed(self):
		buffer = self.serialize()
		(seed,) = struct.unpack('Q', buffer[:8])
		return seed

class Xor16:

	def __init__(self, size):
		self.__filter = ffi.new("xor16_t *")
		status = lib.xor16_allocate(size, self.__filter)
		self.size = size
		if not status:
			print("Unable to allocate memory for 16 bit filter")

	def __repr__(self):
		return "Xor16 object with size(in bytes):{}".format(self.size_in_bytes())

	def __getitem__(self, item):
		return self.contains(item)

	def __del__(self):
		lib.xor16_free(self.__filter)

	def populate(self, data):
		data = list(map(lambda x: _stable_hash(x), data))
		return lib.xor16_buffered_populate(data, len(data), self.__filter)

	def contains(self, item):
		item = _stable_hash(item)
		return lib.xor16_contain(item, self.__filter)

	def query(self, data: list):
		_idxs = [x for x in data if self.contains(x)]
		return _idxs

	def size_in_bytes(self):
		return lib.xor16_size_in_bytes(self.__filter)

	def serialize(self):
		buffer = bytearray(int(self.size_in_bytes()+100))
		buffer_ptr = ffi.from_buffer(buffer)
		lib.xor16_serialize(self.__filter, buffer_ptr)
		return buffer[:16+2*3*int(struct.unpack('QQ', buffer[:16])[-1])]

	def deserialize(self, buffer):
		buffer_ptr = ffi.from_buffer(buffer)
		lib.xor16_deserialize(buffer_ptr, self.__filter)

	def modify(self, entries):
		if set(entries.keys()).issubset(['seed','size','fingerprints']):
			# Extract all info
			buffer = bytearray(int(self.size_in_bytes()+100))
			lib.xor16_serialize(self.__filter, ffi.from_buffer(buffer))
			buffer = buffer[:16+2*3*int(struct.unpack('QQ', buffer[:16])[-1])]
			if 'seed' in list(entries.keys()): # Fill bytes_arr [:8] bytes with entries[seed] element (uint64)
				struct.pack_into('Q', buffer, 0, entries['seed'])    
			if 'fingerprints' in list(entries.keys()): # Fill bytes_arr [16:] bytes with entries['fingerprints'] (array of uint16)
				struct.pack_into(f"{len(entries['fingerprints'])}H", buffer, 16, *entries['fingerprints'])
			lib.xor16_deserialize(ffi.from_buffer(buffer), self.__filter)
		else:
			raise ValueError("Invalid keys in the entries dictionary")

	@property
	def fingerprints(self):
		buffer = self.serialize()
		_, blocks_size = struct.unpack('QQ', buffer[:16])
		fingerprints = list(struct.unpack_from(f"{int(3*blocks_size)}H", buffer, 16))
		return np.asarray(fingerprints, dtype=np.uint16)

	@property
	def seed(self):
		buffer = self.serialize()
		(seed,) = struct.unpack('Q', buffer[:8])
		return seed

class Fuse8:

	def __init__(self, size, r=1.075):
		self.__filter = ffi.new("binary_fuse8_t *")
		status = lib.binary_fuse8_allocate(int(r*size), self.__filter)
		self.size = size
		if not status:
			print("Unable to allocate memory for 8 bit filter")

	def __repr__(self):
		return "Fuse8 object with size(in bytes):{}".format(self.size_in_bytes())

	def __getitem__(self, item):
		return self.contains(item)

	def __del__(self):
		lib.binary_fuse8_free(self.__filter)

	def populate(self, data: list):
		data = list(map(lambda x: _stable_hash(x), data))
		return lib.binary_fuse8_populate(data, len(data), self.__filter)

	def contains(self, item):
		item = _stable_hash(item)
		return lib.binary_fuse8_contain(item, self.__filter)

	def query(self, data: list):
		_idxs = [x for x in data if self.contains(x)]
		return _idxs

	def size_in_bytes(self):
		return lib.binary_fuse8_size_in_bytes(self.__filter)

	def serialize(self):
		buffer = bytearray(int(self.size_in_bytes()+100))
		buffer_ptr = ffi.from_buffer(buffer)
		lib.binary_fuse8_serialize(self.__filter, buffer_ptr)
		return buffer[:28+int(struct.unpack('QIIIII', buffer[:28])[-1])]

	def deserialize(self, buffer):
		buffer_ptr = ffi.from_buffer(buffer)
		lib.binary_fuse8_deserialize(buffer_ptr, self.__filter)

	def modify(self, entries):
		if set(entries.keys()).issubset(['seed','size','fingerprints']):
			# Extract all info
			buffer = bytearray(int(self.size_in_bytes()+100))
			lib.binary_fuse8_serialize(self.__filter, ffi.from_buffer(buffer))
			buffer = buffer[:28+2*int(struct.unpack('QIIIII', buffer[:28])[-1])]
			if 'seed' in list(entries.keys()): # Fill bytes_arr [:8] bytes with entries[seed] element (uint64)
				struct.pack_into('Q', buffer, 0, entries['seed'])    
			if 'fingerprints' in list(entries.keys()): # Fill bytes_arr [28:] bytes with entries['fingerprints'] (array of uint16)
				struct.pack_into(f"{len(entries['fingerprints'])}B", buffer, 28, *entries['fingerprints'])
			lib.binary_fuse8_deserialize(ffi.from_buffer(buffer), self.__filter)
		else:
			raise ValueError("Invalid keys in the entries dictionary")

	@property
	def fingerprints(self):
		buffer = self.serialize()
		_,_,_,_,_, blocks_size = struct.unpack('QIIIII', buffer[:28])
		fingerprints = list(struct.unpack_from(f"{int(blocks_size)}B", buffer, 28))
		return np.asarray(fingerprints, dtype=np.uint8)

	@property
	def seed(self):
		buffer = self.serialize()
		(seed,) = struct.unpack('Q', buffer[:8])
		return seed

class Fuse16:

	def __init__(self, size, r=1.075):
		self.__filter = ffi.new("binary_fuse16_t *")
		status = lib.binary_fuse16_allocate(int(r*size), self.__filter)
		self.size = size
		if not status:
			print("Unable to allocate memory for 16 bit filter")

	def __repr__(self):
		return "Fuse16 object with size(in bytes):{}".format(self.size_in_bytes())

	def __getitem__(self, item):
		return self.contains(item)

	def __del__(self):
		lib.binary_fuse16_free(self.__filter)

	def populate(self, data: list):
		data = list(map(lambda x: _stable_hash(x), data))
		return lib.binary_fuse16_populate(data, len(data), self.__filter)

	def contains(self, item):
		item = _stable_hash(item)
		return lib.binary_fuse16_contain(item, self.__filter)

	def query(self, data: list):
		_idxs = [x for x in data if self.contains(x)]
		return _idxs

	def size_in_bytes(self):
		return lib.binary_fuse16_size_in_bytes(self.__filter)

	def serialize(self):
		buffer = bytearray(int(self.size_in_bytes()+100))
		buffer_ptr = ffi.from_buffer(buffer)
		lib.binary_fuse16_serialize(self.__filter, buffer_ptr)
		return buffer[:28+2*int(struct.unpack('QIIIII', buffer[:28])[-1])]

	def deserialize(self, buffer):
		buffer_ptr = ffi.from_buffer(buffer)
		lib.binary_fuse16_deserialize(buffer_ptr, self.__filter)

	def modify(self, entries):
		if set(entries.keys()).issubset(['seed','size','fingerprints']):
			# Extract all info
			buffer = bytearray(int(self.size_in_bytes()+100))
			lib.binary_fuse16_serialize(self.__filter, ffi.from_buffer(buffer))
			buffer = buffer[:28+2*int(struct.unpack('QIIIII', buffer[:28])[-1])]
			if 'seed' in list(entries.keys()): # Fill bytes_arr [:8] bytes with entries[seed] element (uint64)
				struct.pack_into('Q', buffer, 0, entries['seed'])    
			if 'fingerprints' in list(entries.keys()): # Fill bytes_arr [28:] bytes with entries['fingerprints'] (array of uint16)
				struct.pack_into(f"{len(entries['fingerprints'])}H", buffer, 28, *entries['fingerprints'])
			lib.binary_fuse16_deserialize(ffi.from_buffer(buffer), self.__filter)
		else:
			raise ValueError("Invalid keys in the entries dictionary")

	@property
	def fingerprints(self):
		buffer = self.serialize()
		_,_,_,_,_, blocks_size = struct.unpack('QIIIII', buffer[:28])
		fingerprints = list(struct.unpack_from(f"{int(blocks_size)}H", buffer, 28))
		return np.asarray(fingerprints, dtype=np.uint16)

	@property
	def seed(self):
		buffer = self.serialize()
		(seed,) = struct.unpack('Q', buffer[:8])
		return seed
