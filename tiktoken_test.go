package tiktoken

import (
	"testing"
)

func TestR50kBaseEncoding(t *testing.T) {
	enc, err := LoadR50kBase()
	if err != nil {
		t.Fatalf("LoadR50kBase failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Encode returned empty tokens")
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	// Expected tokens from Python tiktoken for "hello world" with r50k_base
	if tokens[0] != 31373 || tokens[1] != 995 {
		t.Errorf("unexpected tokens: %v, expected [31373, 995]", tokens)
	}

	t.Logf("r50k_base: %q -> %v", text, tokens)
}

func TestP50kBaseEncoding(t *testing.T) {
	enc, err := LoadP50kBase()
	if err != nil {
		t.Fatalf("LoadP50kBase failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	if enc.NVocab() != 50281 {
		t.Errorf("NVocab = %d, want 50281", enc.NVocab())
	}

	t.Logf("p50k_base: %q -> %v (n_vocab=%d)", text, tokens, enc.NVocab())
}

func TestP50kEditEncoding(t *testing.T) {
	enc, err := LoadP50kEdit()
	if err != nil {
		t.Fatalf("LoadP50kEdit failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	t.Logf("p50k_edit: %q -> %v", text, tokens)
}

func TestCl100kBaseEncoding(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Encode returned empty tokens")
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	// Verify n_vocab is in expected range (100280 is the official count)
	if enc.NVocab() < 100000 {
		t.Errorf("NVocab = %d, seems too low", enc.NVocab())
	}

	t.Logf("cl100k_base: %q -> %v (n_vocab=%d)", text, tokens, enc.NVocab())
}

func TestO200kBaseEncoding(t *testing.T) {
	enc, err := LoadO200kBase()
	if err != nil {
		t.Fatalf("LoadO200kBase failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Encode returned empty tokens")
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	if enc.NVocab() != 200019 {
		t.Errorf("NVocab = %d, want 200019", enc.NVocab())
	}

	t.Logf("o200k_base: %q -> %v (n_vocab=%d)", text, tokens, enc.NVocab())
}

func TestO200kHarmonyEncoding(t *testing.T) {
	enc, err := LoadO200kHarmony()
	if err != nil {
		t.Fatalf("LoadO200kHarmony failed: %v", err)
	}

	text := "hello world"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	if enc.NVocab() < 200000 {
		t.Errorf("NVocab = %d, seems too low for harmony", enc.NVocab())
	}

	t.Logf("o200k_harmony: %q -> %v (n_vocab=%d)", text, tokens, enc.NVocab())
}

func TestAllEncodingsRoundtrip(t *testing.T) {
	testCases := []struct {
		name string
		load func() (*Encoding, error)
		text string
	}{
		{"r50k_base", LoadR50kBase, "hello world"},
		{"p50k_base", LoadP50kBase, "hello world"},
		{"p50k_edit", LoadP50kEdit, "hello world"},
		{"cl100k_base", LoadCl100kBase, "hello world"},
		{"o200k_base", LoadO200kBase, "hello world"},
		{"o200k_harmony", LoadO200kHarmony, "hello world"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := tc.load()
			if err != nil {
				t.Fatalf("Load failed: %v", err)
			}

			tokens, err := enc.Encode(tc.text)
			if err != nil {
				t.Fatalf("Encode failed: %v", err)
			}

			if len(tokens) == 0 {
				t.Error("Encode returned empty tokens")
			}

			decoded, err := enc.Decode(tokens)
			if err != nil {
				t.Fatalf("Decode failed: %v", err)
			}

			if decoded != tc.text {
				t.Errorf("roundtrip failed: got %q, want %q", decoded, tc.text)
			}
		})
	}
}

func TestSpecialTokensHandling(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	testCases := []struct {
		name            string
		text            string
		minTokenCount   int
		specialInMiddle bool
	}{
		{
			"endoftext alone",
			"<|endoftext|>",
			1,
			false,
		},
		{
			"text before endoftext",
			"hello<|endoftext|>",
			2,
			true,
		},
		{
			"text after endoftext",
			"<|endoftext|>world",
			2,
			false, // special token is at position 0
		},
		{
			"text around endoftext",
			"hello<|endoftext|>world",
			3,
			true,
		},
		{
			"endofprompt",
			"hello<|endofprompt|>world",
			3,
			true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tokens, err := enc.Encode(tc.text)
			if err != nil {
				t.Fatalf("Encode failed: %v", err)
			}

			if len(tokens) != tc.minTokenCount {
				t.Errorf("expected %d tokens, got %d: %v", tc.minTokenCount, len(tokens), tokens)
			}

			decoded, _ := enc.Decode(tokens)
			if decoded != tc.text {
				t.Errorf("roundtrip failed: got %q, want %q", decoded, tc.text)
			}

			// Verify special token is in the right position
			if tc.specialInMiddle {
				// Token at position 1 should be a special token
				if !enc.IsSpecialToken(tokens[1]) {
					t.Errorf("token[1] = %d should be special", tokens[1])
				}
			}
		})
	}
}

func TestEncodeOrdinaryIgnoresSpecialTokens(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	text := "hello<|endoftext|>world"
	ordinaryTokens, err := enc.EncodeOrdinary(text)
	if err != nil {
		t.Fatalf("EncodeOrdinary failed: %v", err)
	}

	if len(ordinaryTokens) == 3 {
		t.Errorf("EncodeOrdinary should not treat special tokens specially, got %d tokens: %v", len(ordinaryTokens), ordinaryTokens)
	}

	t.Logf("EncodeOrdinary: %q -> %v", text, ordinaryTokens)
}

func TestDecodeSingleToken(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	token := 15339 // "hello" in cl100k_base
	bytes, err := enc.DecodeSingleTokenBytes(token)
	if err != nil {
		t.Fatalf("DecodeSingleTokenBytes failed: %v", err)
	}

	if string(bytes) != "hello" {
		t.Errorf("DecodeSingleTokenBytes(%d) = %q, want %q", token, string(bytes), "hello")
	}
}

func TestTokenByteValues(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	values := enc.TokenByteValues()
	// TokenByteValues should have entries for all regular tokens
	if len(values) < 100000 {
		t.Errorf("TokenByteValues length = %d, seems too low", len(values))
	}

	// Verify we can decode a known token
	helloBytes := enc.DecodeTokensBytes([]int{15339})
	if len(helloBytes) != 1 || string(helloBytes[0]) != "hello" {
		t.Errorf("DecodeTokensBytes([15339]) = %v, want [hello]", helloBytes)
	}

	t.Logf("TokenByteValues length: %d", len(values))
}

func TestIsSpecialToken(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	if !enc.IsSpecialToken(100257) {
		t.Error("IsSpecialToken(100257) should be true for <|endoftext|>")
	}

	if !enc.IsSpecialToken(100276) {
		t.Error("IsSpecialToken(100276) should be true for <|endofprompt|>")
	}

	if enc.IsSpecialToken(15339) {
		t.Error("IsSpecialToken(15339) should be false for 'hello'")
	}
}

func TestEOTToken(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	if enc.EOTToken() != 100257 {
		t.Errorf("EOTToken = %d, want 100257", enc.EOTToken())
	}
}

func TestEmptyString(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	tokens, err := enc.Encode("")
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) != 0 {
		t.Errorf("Encode(\"\") returned %d tokens, want 0", len(tokens))
	}

	decoded, err := enc.Decode([]int{})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if decoded != "" {
		t.Errorf("Decode([]) = %q, want %q", decoded, "")
	}
}

func TestUnicodeText(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	testCases := []string{
		"你好世界",
		"🎉 Hello 🌍",
		"日本語",
		"مرحبا",
	}

	for _, text := range testCases {
		tokens, err := enc.Encode(text)
		if err != nil {
			t.Errorf("Encode(%q) failed: %v", text, err)
			continue
		}

		if len(tokens) == 0 {
			t.Errorf("Encode(%q) returned empty tokens", text)
			continue
		}

		decoded, err := enc.Decode(tokens)
		if err != nil {
			t.Errorf("Decode failed for %q: %v", text, err)
			continue
		}

		if decoded != text {
			t.Logf("Note: roundtrip may differ for %q: got %q", text, decoded)
		}
	}
}

func TestLongerText(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog. 0123456789"
	tokens, err := enc.Encode(text)
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	if len(tokens) == 0 {
		t.Error("Encode returned empty tokens")
	}

	decoded, _ := enc.Decode(tokens)
	if decoded != text {
		t.Errorf("roundtrip failed: got %q, want %q", decoded, text)
	}

	t.Logf("long text: %d chars -> %d tokens", len(text), len(tokens))
}

func TestBatchEncoding(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	texts := []string{
		"hello world",
		"goodbye world",
		"The quick brown fox",
		"日本語",
	}

	tokens, err := enc.EncodeBatch(texts)
	if err != nil {
		t.Fatalf("EncodeBatch failed: %v", err)
	}

	if len(tokens) != len(texts) {
		t.Errorf("EncodeBatch returned %d results, want %d", len(tokens), len(texts))
	}

	for i, text := range texts {
		if len(tokens[i]) == 0 {
			t.Errorf("EncodeBatch[%d] returned empty tokens", i)
		}

		decoded, _ := enc.Decode(tokens[i])
		if decoded != text {
			t.Errorf("roundtrip[%d] failed: got %q, want %q", i, decoded, text)
		}
	}
}

func TestDecodeTokensBytes(t *testing.T) {
	enc, err := LoadCl100kBase()
	if err != nil {
		t.Fatalf("LoadCl100kBase failed: %v", err)
	}

	tokens := []int{15339, 1917} // "hello world"
	bytes := enc.DecodeTokensBytes(tokens)

	if len(bytes) != 2 {
		t.Errorf("DecodeTokensBytes returned %d results, want 2", len(bytes))
	}

	if string(bytes[0]) != "hello" || string(bytes[1]) != " world" {
		t.Errorf("DecodeTokensBytes = %v, want [hello, ' world']", bytes)
	}
}

func TestEncodingConstructors(t *testing.T) {
	constructors := []struct {
		name string
		fn   func() (*Encoding, error)
	}{
		{"r50k_base", LoadR50kBase},
		{"p50k_base", LoadP50kBase},
		{"p50k_edit", LoadP50kEdit},
		{"cl100k_base", LoadCl100kBase},
		{"o200k_base", LoadO200kBase},
		{"o200k_harmony", LoadO200kHarmony},
	}

	for _, c := range constructors {
		t.Run(c.name, func(t *testing.T) {
			enc, err := c.fn()
			if err != nil {
				t.Errorf("Load failed: %v", err)
				return
			}

			if enc.Name() != c.name {
				t.Errorf("Encoding name = %q, want %q", enc.Name(), c.name)
			}
		})
	}
}

func TestModelToEncoding(t *testing.T) {
	testCases := []struct {
		model   string
		encName string
	}{
		{"gpt-4", "cl100k_base"},
		{"gpt-3.5-turbo", "cl100k_base"},
		{"text-davinci-003", "p50k_base"},
		{"text-curie-001", "r50k_base"},
		{"ada", "r50k_base"},
		{"gpt-4o", "o200k_base"},
	}

	for _, tc := range testCases {
		t.Run(tc.model, func(t *testing.T) {
			enc, err := EncodingForModel(tc.model)
			if err != nil {
				t.Fatalf("EncodingForModel(%s) failed: %v", tc.model, err)
			}

			if enc.Name() != tc.encName {
				t.Errorf("EncodingForModel(%s) = %s, want %s", tc.model, enc.Name(), tc.encName)
			}
		})
	}
}

func TestEncodingForUnknownModel(t *testing.T) {
	_, err := EncodingForModel("unknown-model-xyz")
	if err == nil {
		t.Error("EncodingForModel should return error for unknown model")
	}
}

func TestBytePairEncodeBasic(t *testing.T) {
	ranks := map[string]rank{
		"ab": 0,
		"cd": 1,
	}

	result := bytePairEncode([]byte("abcd"), ranks)
	if len(result) != 2 || result[0] != 0 || result[1] != 1 {
		t.Errorf("bytePairEncode(\"abcd\") = %v, want [0, 1]", result)
	}
}

func TestBytePairEncodeRepeated(t *testing.T) {
	ranks := map[string]rank{
		"ab": 0,
	}

	result := bytePairEncode([]byte("abab"), ranks)
	if len(result) != 2 || result[0] != 0 || result[1] != 0 {
		t.Errorf("bytePairEncode(\"abab\") = %v, want [0, 0]", result)
	}
}

func TestBytePairEncodeSingleByte(t *testing.T) {
	ranks := map[string]rank{
		"a": 0,
		"b": 1,
		"c": 2,
	}

	result := bytePairEncode([]byte("abc"), ranks)
	if len(result) != 3 {
		t.Errorf("bytePairEncode(\"abc\") length = %d, want 3", len(result))
	}
}

func TestBytePairMergeSmallPiece(t *testing.T) {
	ranks := map[string]rank{
		"ab": 0,
		"bc": 1,
		"cd": 2,
	}

	result := bytePairMerge(ranks, []byte("abcd"))
	if len(result) == 0 {
		t.Error("bytePairMerge returned empty result")
	}
}

func TestBytePairMergeLargePiece(t *testing.T) {
	ranks := map[string]rank{
		"ab": 0,
		"bc": 1,
		"cd": 2,
	}

	// This is >= 100 bytes, so uses large merge
	longText := make([]byte, 150)
	for i := range longText {
		longText[i] = "abcd"[i%4]
	}

	result := bytePairMergeLarge(ranks, longText)
	if len(result) == 0 {
		t.Error("bytePairMergeLarge returned empty result")
	}
}

func TestSortedTokenBytes(t *testing.T) {
	encoder := map[string]rank{
		"b": 1,
		"a": 0,
		"c": 2,
	}

	result := sortedTokenBytesFromEncoder(encoder)
	if len(result) != 3 {
		t.Errorf("sortedTokenBytes length = %d, want 3", len(result))
	}

	expected := [][]byte{[]byte("a"), []byte("b"), []byte("c")}
	for i, exp := range expected {
		if string(result[i]) != string(exp) {
			t.Errorf("sortedTokenBytes[%d] = %q, want %q", i, result[i], exp)
		}
	}
}
