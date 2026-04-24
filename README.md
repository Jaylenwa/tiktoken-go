# tiktoken-go

A fast BPE (Byte Pair Encoding) tokenizer implementation in pure Go, compatible with OpenAI's [tiktoken](https://github.com/openai/tiktoken).

## Features

- **Pure Go** - No C bindings or external dependencies beyond `regexp2`
- **Fast** - Heap-based BPE algorithm for efficient tokenization of long texts
- **Compatible** - Supports all OpenAI encodings and model aliases
- **Caching** - Automatic caching of downloaded BPE files with hash verification

## Installation

```bash
go get github.com/jaylen/tiktoken-go
```

## Supported Encodings

| Encoding | Description |
|----------|-------------|
| `r50k_base` | GPT-3 (ada, babbage, curie, davinci) |
| `p50k_base` | Code GPT-3 / text-davinci |
| `p50k_edit` | GPT-3.5 editing model |
| `cl100k_base` | GPT-4 / GPT-3.5-turbo |
| `o200k_base` | GPT-4o |
| `o200k_harmony` | GPT-4o with extended special tokens |

## Usage

### By Model Name

```go
package main

import (
    "fmt"
    "github.com/jaylen/tiktoken-go"
)

func main() {
    enc, err := tiktoken.EncodingForModel("gpt-4")
    if err != nil {
        panic(err)
    }

    text := "Hello, world!"
    tokens := enc.Encode(text)
    fmt.Println("Tokens:", tokens)

    decoded := enc.Decode(tokens)
    fmt.Println("Decoded:", decoded)
}
```

### By Encoding Name

```go
enc, err := tiktoken.LoadCl100kBase()
```

### Batch Encoding

```go
texts := []string{"Hello", "World"}
tokens, err := enc.EncodeBatch(texts)
```

## API Reference

### Encoding

- `EncodingForModel(model string)` - Get encoding by model name
- `LoadCl100kBase()`, `LoadO200kBase()`, etc. - Load specific encoding
- `Encode(text string)` - Encode text to token IDs
- `Decode(tokens []int)` - Decode token IDs back to text
- `EncodeSingleToken(text []byte)` - Encode a single token
- `DecodeSingleTokenBytes(token int)` - Decode single token to bytes
- `IsSpecialToken(token int)` - Check if token is special
- `EOTToken()` - Get end-of-text token ID

### Cache Directory

By default, BPE files are cached to `os.TempDir()/tiktoken-cache`. Set custom path:

```go
os.Setenv("TIKTOKEN_CACHE_DIR", "/path/to/cache")
```

## License

[MIT](LICENSE)
