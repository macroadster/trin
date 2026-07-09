package cuda

import (
	"sync"
	"sync/atomic"
)

// Lightweight integer handles for DeviceBuffer so the language surface can
// retain buffers across calls (mirrors weight sessions).

var (
	bufMu   sync.Mutex
	bufs    = map[int32]*DeviceBuffer{}
	bufSeq  int32
)

// RegisterBuffer stores b and returns a positive handle id.
func RegisterBuffer(b *DeviceBuffer) int32 {
	if b == nil {
		return 0
	}
	id := atomic.AddInt32(&bufSeq, 1)
	bufMu.Lock()
	bufs[id] = b
	bufMu.Unlock()
	return id
}

// GetBuffer looks up a handle.
func GetBuffer(id int32) (*DeviceBuffer, error) {
	bufMu.Lock()
	b := bufs[id]
	bufMu.Unlock()
	if b == nil {
		return nil, errMsg("cuda buffer handle not found")
	}
	return b, nil
}

// FreeBuffer frees and forgets a handle (idempotent).
func FreeBuffer(id int32) {
	bufMu.Lock()
	b := bufs[id]
	delete(bufs, id)
	bufMu.Unlock()
	if b != nil {
		b.Free()
	}
}

// BufferCount is live registered buffers (tests/status).
func BufferCount() int {
	bufMu.Lock()
	n := len(bufs)
	bufMu.Unlock()
	return n
}
