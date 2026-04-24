package tiktoken

import (
	"errors"
	"fmt"
	"sort"

	"github.com/dlclark/regexp2"
)

var ErrInvalidToken = errors.New("invalid token for decoding")

type DecodeKeyError struct {
	Token rank
}

func (e *DecodeKeyError) Error() string {
	return fmt.Sprintf("Invalid token for decoding: %d", e.Token)
}

type CoreBPE struct {
	encoder              map[string]rank
	specialTokensEncoder map[string]rank
	decoder              map[rank][]byte
	specialTokensDecoder map[rank][]byte
	regex                *regexp2.Regexp
	specialRegex         *regexp2.Regexp
}

func NewCoreBPE(encoder map[string]rank, specialTokensEncoder map[string]rank, pattern string) (*CoreBPE, error) {
	regex, err := regexp2.Compile(pattern, 0)
	if err != nil {
		return nil, err
	}

	decoder := make(map[rank][]byte, len(encoder))
	for k, v := range encoder {
		decoder[v] = []byte(k)
	}

	if len(decoder) != len(encoder) {
		return nil, errors.New("encoder and decoder must be of equal length")
	}

	specialTokensDecoder := make(map[rank][]byte, len(specialTokensEncoder))
	for k, v := range specialTokensEncoder {
		specialTokensDecoder[v] = []byte(k)
	}

	var specialRegex *regexp2.Regexp
	if len(specialTokensEncoder) > 0 {
		var sb string
		first := true
		for s := range specialTokensEncoder {
			if first {
				sb = regexp2.Escape(s)
				first = false
			} else {
				sb += "|" + regexp2.Escape(s)
			}
		}
		specialRegex, err = regexp2.Compile(sb, 0)
		if err != nil {
			return nil, err
		}
	}

	return &CoreBPE{
		encoder:              encoder,
		specialTokensEncoder: specialTokensEncoder,
		decoder:              decoder,
		specialTokensDecoder: specialTokensDecoder,
		regex:                regex,
		specialRegex:         specialRegex,
	}, nil
}

func (c *CoreBPE) DecodeBytes(tokens []rank) ([]byte, error) {
	ret := make([]byte, 0, len(tokens)*2)
	for _, token := range tokens {
		tokenBytes, ok := c.decoder[token]
		if !ok {
			tokenBytes, ok = c.specialTokensDecoder[token]
			if !ok {
				return nil, &DecodeKeyError{Token: token}
			}
		}
		ret = append(ret, tokenBytes...)
	}
	return ret, nil
}

func (c *CoreBPE) EncodeOrdinary(text string) []rank {
	ret := make([]rank, 0)

	match, err := c.regex.FindStringMatch(text)
	for match != nil && err == nil {
		piece := []byte(match.String())
		if token, ok := c.encoder[string(piece)]; ok {
			ret = append(ret, token)
		} else {
			ret = append(ret, bytePairEncode(piece, c.encoder)...)
		}
		match, err = c.regex.FindNextMatch(match)
	}

	return ret
}

func (c *CoreBPE) Encode(text string, allowedSpecial map[string]struct{}) ([]rank, error) {
	if c.specialRegex == nil {
		return c.EncodeOrdinary(text), nil
	}

	ret := make([]rank, 0)
	start := 0

	for {
		var nextSpecial *regexp2.Match
		startFind := start

		for {
			m, err := c.specialRegex.FindStringMatch(text[startFind:])
			if err != nil {
				return nil, err
			}
			if m == nil {
				break
			}
			actualMatch := m
			actualMatch.Index += startFind
			if _, allowed := allowedSpecial[actualMatch.String()]; allowed {
				nextSpecial = actualMatch
				break
			}
			startFind = actualMatch.Index + 1
		}

		end := len(text)
		if nextSpecial != nil {
			end = nextSpecial.Index
		}

		if start < end {
			subText := text[start:end]
			match, err := c.regex.FindStringMatch(subText)
			for match != nil && err == nil {
				piece := []byte(match.String())
				if token, ok := c.encoder[string(piece)]; ok {
					ret = append(ret, token)
				} else {
					ret = append(ret, bytePairEncode(piece, c.encoder)...)
				}
				match, err = c.regex.FindNextMatch(match)
			}
			if err != nil {
				return nil, err
			}
		}

		if nextSpecial == nil {
			break
		}

		token, ok := c.specialTokensEncoder[nextSpecial.String()]
		if !ok {
			return nil, fmt.Errorf("unknown special token")
		}
		ret = append(ret, token)
		start = nextSpecial.Index + nextSpecial.Length
	}

	return ret, nil
}

func (c *CoreBPE) DecodeSingleTokenBytes(token rank) ([]byte, error) {
	tokenBytes, ok := c.decoder[token]
	if !ok {
		tokenBytes, ok = c.specialTokensDecoder[token]
		if !ok {
			return nil, &DecodeKeyError{Token: token}
		}
	}
	return tokenBytes, nil
}

func (c *CoreBPE) EncodeSingleToken(textOrBytes []byte) (rank, error) {
	if token, ok := c.encoder[string(textOrBytes)]; ok {
		return token, nil
	}
	return 0, fmt.Errorf("token not found")
}

func (c *CoreBPE) TokenByteValues() [][]byte {
	tokens := make([]rank, 0, len(c.decoder))
	for t := range c.decoder {
		tokens = append(tokens, t)
	}
	sort.Slice(tokens, func(i, j int) bool { return tokens[i] < tokens[j] })

	result := make([][]byte, len(tokens))
	for i, t := range tokens {
		result[i] = c.decoder[t]
	}
	return result
}

func (c *CoreBPE) SpecialTokensSet() map[string]struct{} {
	result := make(map[string]struct{}, len(c.specialTokensEncoder))
	for k := range c.specialTokensEncoder {
		result[k] = struct{}{}
	}
	return result
}
