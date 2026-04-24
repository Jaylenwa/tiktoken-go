package tiktoken

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	EncodingNameR50kBase     = "r50k_base"
	EncodingNameP50kBase     = "p50k_base"
	EncodingNameP50kEdit     = "p50k_edit"
	EncodingNameCl100kBase   = "cl100k_base"
	EncodingNameO200kBase    = "o200k_base"
	EncodingNameO200kHarmony = "o200k_harmony"

	r50kBPEURL  = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
	r50kBPEHash = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"

	p50kBPEURL  = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken"
	p50kBPEHash = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"

	cl100kBPEURL  = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
	cl100kBPEHash = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"

	o200kBPEURL  = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
	o200kBPEHash = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
)

var r50kPatStr = `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$|\s+(?!\S)|\s`

var cl100kPatStr = `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s`

var o200kPatStr = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

var cacheDir = os.Getenv("TIKTOKEN_CACHE_DIR")

func init() {
	if cacheDir == "" {
		cacheDir = filepath.Join(os.TempDir(), "tiktoken-cache")
	}
}

func readFile(path string) ([]byte, error) {
	if !strings.Contains(path, "://") {
		return os.ReadFile(path)
	}

	resp, err := http.Get(path)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func checkHash(data []byte, expectedHash string) bool {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]) == expectedHash
}

func loadFromCache(path, expectedHash string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if expectedHash == "" || checkHash(data, expectedHash) {
		return data, nil
	}
	return nil, fmt.Errorf("hash mismatch")
}

func saveToCache(path string, data []byte) error {
	os.MkdirAll(cacheDir, 0755)
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func loadTiktokenBPEFromURL(url, expectedHash string) (map[string]rank, error) {
	hashKey := sha256Hash(url)
	cachePath := filepath.Join(cacheDir, hashKey)

	if data, err := loadFromCache(cachePath, expectedHash); err == nil {
		return parseTiktokenBPE(data), nil
	}

	data, err := readFile(url)
	if err != nil {
		return nil, err
	}

	if expectedHash != "" && !checkHash(data, expectedHash) {
		return nil, fmt.Errorf("hash mismatch for %s", url)
	}

	saveToCache(cachePath, data)

	return parseTiktokenBPE(data), nil
}

func sha256Hash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func parseTiktokenBPE(data []byte) map[string]rank {
	ret := make(map[string]rank)
	for _, line := range strings.Split(string(data), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		token, err := base64.StdEncoding.DecodeString(parts[0])
		if err != nil {
			continue
		}
		r, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		ret[string(token)] = rank(r)
	}
	return ret
}

func LoadR50kBase() (*Encoding, error) {
	ranks, err := loadTiktokenBPEFromURL(r50kBPEURL, r50kBPEHash)
	if err != nil {
		return nil, err
	}
	return NewEncoding(EncodingParams{
		Name:           "r50k_base",
		PatStr:         r50kPatStr,
		MergeableRanks: ranks,
		SpecialTokens:  map[string]rank{"<|endoftext|>": 50256},
		ExplicitNVocab: 50257,
	})
}

func LoadP50kBase() (*Encoding, error) {
	ranks, err := loadTiktokenBPEFromURL(p50kBPEURL, p50kBPEHash)
	if err != nil {
		return nil, err
	}
	return NewEncoding(EncodingParams{
		Name:           "p50k_base",
		PatStr:         r50kPatStr,
		MergeableRanks: ranks,
		SpecialTokens:  map[string]rank{"<|endoftext|>": 50256},
		ExplicitNVocab: 50281,
	})
}

func LoadP50kEdit() (*Encoding, error) {
	ranks, err := loadTiktokenBPEFromURL(p50kBPEURL, p50kBPEHash)
	if err != nil {
		return nil, err
	}
	return NewEncoding(EncodingParams{
		Name:           "p50k_edit",
		PatStr:         r50kPatStr,
		MergeableRanks: ranks,
		SpecialTokens: map[string]rank{
			"<|endoftext|>":  50256,
			"<|fim_prefix|>": 50281,
			"<|fim_middle|>": 50282,
			"<|fim_suffix|>": 50283,
		},
	})
}

func LoadCl100kBase() (*Encoding, error) {
	ranks, err := loadTiktokenBPEFromURL(cl100kBPEURL, cl100kBPEHash)
	if err != nil {
		return nil, err
	}
	return NewEncoding(EncodingParams{
		Name:           "cl100k_base",
		PatStr:         cl100kPatStr,
		MergeableRanks: ranks,
		SpecialTokens: map[string]rank{
			"<|endoftext|>":   100257,
			"<|fim_prefix|>":  100258,
			"<|fim_middle|>":  100259,
			"<|fim_suffix|>":  100260,
			"<|endofprompt|>": 100276,
		},
		ExplicitNVocab: 100280,
	})
}

func LoadO200kBase() (*Encoding, error) {
	ranks, err := loadTiktokenBPEFromURL(o200kBPEURL, o200kBPEHash)
	if err != nil {
		return nil, err
	}
	return NewEncoding(EncodingParams{
		Name:           "o200k_base",
		PatStr:         o200kPatStr,
		MergeableRanks: ranks,
		SpecialTokens: map[string]rank{
			"<|endoftext|>":   199999,
			"<|endofprompt|>": 200018,
		},
		ExplicitNVocab: 200019,
	})
}

func LoadO200kHarmony() (*Encoding, error) {
	baseEnc, err := LoadO200kBase()
	if err != nil {
		return nil, err
	}

	specialTokens := make(map[string]rank)
	for k, v := range baseEnc.specialTokens {
		specialTokens[k] = v
	}

	specialTokens["<|startoftext|>"] = 199998
	specialTokens["<|endoftext|>"] = 199999
	specialTokens["<|reserved_200000|>"] = 200000
	specialTokens["<|reserved_200001|>"] = 200001
	specialTokens["<|return|>"] = 200002
	specialTokens["<|constrain|>"] = 200003
	specialTokens["<|reserved_200004|>"] = 200004
	specialTokens["<|channel|>"] = 200005
	specialTokens["<|start|>"] = 200006
	specialTokens["<|end|>"] = 200007
	specialTokens["<|message|>"] = 200008
	specialTokens["<|reserved_200009|>"] = 200009
	specialTokens["<|reserved_200010|>"] = 200010
	specialTokens["<|reserved_200011|>"] = 200011
	specialTokens["<|call|>"] = 200012

	for i := 200013; i < 201088; i++ {
		specialTokens[fmt.Sprintf("<|reserved_%d|>", i)] = rank(i)
	}

	return NewEncoding(EncodingParams{
		Name:           "o200k_harmony",
		PatStr:         baseEnc.patStr,
		MergeableRanks: baseEnc.mergeableRanks,
		SpecialTokens:  specialTokens,
	})
}

var ENCODING_CONSTRUCTORS = map[string]func() (*Encoding, error){
	EncodingNameR50kBase:     LoadR50kBase,
	EncodingNameP50kBase:     LoadP50kBase,
	EncodingNameP50kEdit:     LoadP50kEdit,
	EncodingNameCl100kBase:   LoadCl100kBase,
	EncodingNameO200kBase:    LoadO200kBase,
	EncodingNameO200kHarmony: LoadO200kHarmony,
}

var modelToEncoding = map[string]string{
	"gpt-4":              EncodingNameCl100kBase,
	"gpt-4-0314":         EncodingNameCl100kBase,
	"gpt-4-0613":         EncodingNameCl100kBase,
	"gpt-4-32k":          EncodingNameCl100kBase,
	"gpt-4-32k-0314":     EncodingNameCl100kBase,
	"gpt-4-32k-0613":     EncodingNameCl100kBase,
	"gpt-3.5-turbo":      EncodingNameCl100kBase,
	"gpt-3.5-turbo-0301": EncodingNameCl100kBase,
	"gpt-3.5-turbo-0613": EncodingNameCl100kBase,
	"gpt-3.5-turbo-16k":  EncodingNameCl100kBase,
	"text-davinci-003":   EncodingNameP50kBase,
	"text-davinci-002":   EncodingNameP50kBase,
	"code-davinci-002":   EncodingNameP50kBase,
	"code-davinci-001":   EncodingNameP50kBase,
	"text-curie-001":     EncodingNameR50kBase,
	"text-babbage-001":   EncodingNameR50kBase,
	"text-ada-001":       EncodingNameR50kBase,
	"davinci":            EncodingNameR50kBase,
	"curie":              EncodingNameR50kBase,
	"babbage":            EncodingNameR50kBase,
	"ada":                EncodingNameR50kBase,
	"gpt-4o":             EncodingNameO200kBase,
	"gpt-4o-2024-05-13":  EncodingNameO200kBase,
	"chatgpt-4o-latest":  EncodingNameO200kBase,
}
