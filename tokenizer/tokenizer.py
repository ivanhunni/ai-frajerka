from tokenizers import Tokenizer

class CustomTokenizer:
    def __init__(self, model_path="tokenizer.json"):
        # Načítanie vytrénovaného tokenizera zo súboru
        self.tokenizer = Tokenizer.from_file(model_path)

    def encode(self, text: str):
        """Prevedie text na ID tokenov."""
        output = self.tokenizer.encode(text)
        return output.ids, output.tokens

    def decode(self, ids: list[int]):
        """Prevedie zoznam ID tokenov späť na text."""
        return self.tokenizer.decode(ids)

if __name__ == "__main__":
    # Testovacia ukážka použitia
    t = CustomTokenizer()
    
    test_text = "Umenie a vedecká metóda."
    ids, tokens = t.encode(test_text)
    
    print(f"Pôvodný text: {test_text}")
    print(f"Tokens:       {tokens}")
    print(f"IDs:          {ids}")
    print(f"Dekódované:   {t.decode(ids)}")