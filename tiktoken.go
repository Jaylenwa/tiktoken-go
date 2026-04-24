package tiktoken

import "fmt"

type Encoding struct {
	name           string
	patStr         string
	mergeableRanks map[string]rank
	specialTokens  map[string]rank
	coreBPE        *CoreBPE
}

type EncodingParams struct {
	Name           string
	PatStr         string
	MergeableRanks map[string]rank
	SpecialTokens  map[string]rank
	ExplicitNVocab int
}

func NewEncoding(params EncodingParams) (*Encoding, error) {
	coreBPE, err := NewCoreBPE(params.MergeableRanks, params.SpecialTokens, params.PatStr)
	if err != nil {
		return nil, err
	}

	return &Encoding{
		name:           params.Name,
		patStr:         params.PatStr,
		mergeableRanks: params.MergeableRanks,
		specialTokens:  params.SpecialTokens,
		coreBPE:        coreBPE,
	}, nil
}

func (e *Encoding) Name() string {
	return e.name
}

func (e *Encoding) Encode(text string) ([]int, error) {
	tokens, err := e.coreBPE.Encode(text, e.SpecialTokensSet())
	if err != nil {
		return nil, err
	}
	result := make([]int, len(tokens))
	for i, t := range tokens {
		result[i] = int(t)
	}
	return result, nil
}

func (e *Encoding) EncodeOrdinary(text string) ([]int, error) {
	tokens := e.coreBPE.EncodeOrdinary(text)
	result := make([]int, len(tokens))
	for i, t := range tokens {
		result[i] = int(t)
	}
	return result, nil
}

func (e *Encoding) EncodeSingleToken(textOrBytes []byte) (int, error) {
	token, err := e.coreBPE.EncodeSingleToken(textOrBytes)
	return int(token), err
}

func (e *Encoding) DecodeBytes(tokens []int) ([]byte, error) {
	rankTokens := make([]rank, len(tokens))
	for i, t := range tokens {
		rankTokens[i] = rank(t)
	}
	return e.coreBPE.DecodeBytes(rankTokens)
}

func (e *Encoding) Decode(tokens []int) (string, error) {
	bytes, err := e.DecodeBytes(tokens)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func (e *Encoding) DecodeSingleTokenBytes(token int) ([]byte, error) {
	return e.coreBPE.DecodeSingleTokenBytes(rank(token))
}

func (e *Encoding) DecodeTokensBytes(tokens []int) [][]byte {
	result := make([][]byte, len(tokens))
	for i, t := range tokens {
		bytes, _ := e.DecodeSingleTokenBytes(t)
		if bytes != nil {
			result[i] = bytes
		} else {
			result[i] = []byte{}
		}
	}
	return result
}

func (e *Encoding) TokenByteValues() [][]byte {
	return e.coreBPE.TokenByteValues()
}

func (e *Encoding) SpecialTokensSet() map[string]struct{} {
	result := make(map[string]struct{}, len(e.specialTokens))
	for k := range e.specialTokens {
		result[k] = struct{}{}
	}
	return result
}

func (e *Encoding) IsSpecialToken(token int) bool {
	_, ok := e.coreBPE.specialTokensDecoder[rank(token)]
	return ok
}

func (e *Encoding) EOTToken() int {
	if token, ok := e.specialTokens["<|endoftext|>"]; ok {
		return int(token)
	}
	return -1
}

func (e *Encoding) NVocab() int {
	maxToken := rank(0)
	for _, v := range e.mergeableRanks {
		if v > maxToken {
			maxToken = v
		}
	}
	for _, v := range e.specialTokens {
		if v > maxToken {
			maxToken = v
		}
	}
	return int(maxToken) + 1
}

func (e *Encoding) EncodeBatch(texts []string) ([][]int, error) {
	results := make([][]int, len(texts))
	for i, text := range texts {
		tokens, err := e.Encode(text)
		if err != nil {
			return nil, err
		}
		results[i] = tokens
	}
	return results, nil
}

func EncodingForModel(modelName string) (*Encoding, error) {
	encName, ok := modelToEncoding[modelName]
	if !ok {
		return nil, fmt.Errorf("unknown model: %s", modelName)
	}

	constructor, ok := ENCODING_CONSTRUCTORS[encName]
	if !ok {
		return nil, fmt.Errorf("unknown encoding: %s", encName)
	}

	return constructor()
}
