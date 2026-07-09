package cuda

import "unsafe"

func float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
