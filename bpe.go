package tiktoken

import (
	"container/heap"
	"sort"
)

type rank = uint32

const maxRank rank = ^rank(0)

type merge struct {
	start int
	rank  rank
}

type mergeHeap []merge

func (h mergeHeap) Len() int           { return len(h) }
func (h mergeHeap) Less(i, j int) bool { return h[i].rank < h[j].rank }
func (h mergeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *mergeHeap) Push(x any) {
	*h = append(*h, x.(merge))
}

func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type state struct {
	prev     int
	end      int
	nextEnd  int
	nextRank rank
	curRank  rank
}

func bytePairEncode(piece []byte, ranks map[string]rank) []rank {
	pieceLen := len(piece)
	if pieceLen == 1 {
		return []rank{ranks[string(piece)]}
	}
	if pieceLen < 100 {
		return bytePairMerge(ranks, piece)
	}
	return bytePairMergeLarge(ranks, piece)
}

func bytePairMerge(ranks map[string]rank, piece []byte) []rank {
	pieceLen := len(piece)
	if pieceLen == 1 {
		return []rank{ranks[string(piece)]}
	}

	parts := make([]struct {
		start int
		r     rank
	}, 0, pieceLen+1)
	minRank := struct {
		r   rank
		idx int
	}{maxRank, -1}

	for i := 0; i < pieceLen-1; i++ {
		key := string(piece[i : i+2])
		r := maxRank
		if val, ok := ranks[key]; ok {
			r = val
		}
		parts = append(parts, struct {
			start int
			r     rank
		}{i, r})
		if r < minRank.r {
			minRank = struct {
				r   rank
				idx int
			}{r, i}
		}
	}
	parts = append(parts, struct {
		start int
		r     rank
	}{pieceLen - 1, maxRank})
	parts = append(parts, struct {
		start int
		r     rank
	}{pieceLen, maxRank})

	getRank := func(i int) rank {
		if i+3 < len(parts) {
			key := string(piece[parts[i].start:parts[i+3].start])
			if val, ok := ranks[key]; ok {
				return val
			}
		}
		return maxRank
	}

	for minRank.r != maxRank {
		i := minRank.idx

		if i > 0 {
			parts[i-1].r = getRank(i - 1)
		}
		parts[i].r = getRank(i)
		parts = append(parts[:i+1], parts[i+2:]...)

		minRank = struct {
			r   rank
			idx int
		}{maxRank, -1}
		for i = 0; i < len(parts)-1; i++ {
			if parts[i].r < minRank.r {
				minRank = struct {
					r   rank
					idx int
				}{parts[i].r, i}
			}
		}
	}

	result := make([]rank, 0, len(parts)-1)
	for i := 0; i < len(parts)-1; i++ {
		key := string(piece[parts[i].start:parts[i+1].start])
		result = append(result, ranks[key])
	}
	return result
}

func bytePairMergeLarge(ranks map[string]rank, piece []byte) []rank {
	pieceLen := len(piece)

	stateArr := make([]state, 0, pieceLen)
	stateArr = append(stateArr, state{
		prev:     -1,
		end:      1,
		nextEnd:  2,
		nextRank: maxRank,
		curRank:  maxRank,
	})

	h := make(mergeHeap, 0, pieceLen)
	for i := 0; i < pieceLen-1; i++ {
		key := string(piece[i : i+2])
		if r, ok := ranks[key]; ok {
			heap.Push(&h, merge{start: i, rank: r})
			if len(stateArr) > i {
				stateArr[i].nextRank = r
			}
		}
		stateArr = append(stateArr, state{
			prev:     i,
			end:      i + 2,
			nextEnd:  i + 3,
			nextRank: maxRank,
			curRank:  maxRank,
		})
	}

	potentialMerge := func(stateArr *[]state, h *mergeHeap, start int, nextEndItem int) {
		(*stateArr)[start].nextEnd = nextEndItem
		(*stateArr)[start].nextRank = maxRank
		if nextEndItem <= pieceLen {
			key := string(piece[start:nextEndItem])
			if r, ok := ranks[key]; ok {
				heap.Push(h, merge{start: start, rank: r})
				(*stateArr)[start].nextRank = r
			}
		}
	}

	for len(h) > 0 {
		left := heap.Pop(&h).(merge)
		if left.rank == maxRank {
			break
		}
		if left.rank != stateArr[left.start].nextRank {
			continue
		}

		leftStart := left.start
		rightStart := stateArr[leftStart].end
		rightEnd := stateArr[leftStart].nextEnd

		stateArr[leftStart].curRank = stateArr[leftStart].nextRank
		stateArr[leftStart].end = rightEnd
		potentialMerge(&stateArr, &h, leftStart, stateArr[rightStart].nextEnd)

		if rightEnd < len(stateArr) {
			stateArr[rightEnd].prev = leftStart
		}
		if leftStart > 0 {
			prevStart := stateArr[leftStart].prev
			potentialMerge(&stateArr, &h, prevStart, rightEnd)
		}
		stateArr[rightStart].nextRank = maxRank
	}

	result := make([]rank, 0)
	i := 0
	for i < len(stateArr) {
		if stateArr[i].curRank != maxRank {
			result = append(result, stateArr[i].curRank)
		} else {
			key := string(piece[i:stateArr[i].end])
			result = append(result, ranks[key])
		}
		i = stateArr[i].end
	}
	return result
}

type sortedTokenBytes [][]byte

func (s sortedTokenBytes) Len() int           { return len(s) }
func (s sortedTokenBytes) Less(i, j int) bool { return string(s[i]) < string(s[j]) }
func (s sortedTokenBytes) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func sortedTokenBytesFromEncoder(encoder map[string]rank) [][]byte {
	keys := make([][]byte, 0, len(encoder))
	for k := range encoder {
		keys = append(keys, []byte(k))
	}
	sort.Sort(sortedTokenBytes(keys))
	return keys
}
