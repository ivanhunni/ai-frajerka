# inference/chat.py
from inference.engine import InferenceEngine

def start_cli_chat():
    # Inicializácia motora (môžete zmeniť cestu k vašemu najlepšiemu checkpointu)
    engine = InferenceEngine(
        model_path="checkpoints/aligned_model/final_aligned_model.pt",
        tokenizer_path="tokenizer/tokenizer.json"
    )

    system_prompt = "Si milá, starostlivá a občas vtipná AI priateľka. Odpovedáš prirodzene a v slovenskom jazyku."
    conversation_history = []

    print("=" * 60)
    print("   AI-Frajerka Chat Terminal (Napíš 'exit' alebo 'q' pre koniec)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nTy: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "q", "quit"]:
                print("\nAI-Frajerka: Pá, myslím na teba! ❤️")
                break

            # Pridanie správy používateľa do histórie
            conversation_history.append({"role": "user", "content": user_input})

            # Vygenerovanie odpovede
            reply = engine.generate_reply(
                messages=conversation_history,
                system_prompt=system_prompt,
                temperature=0.7,
                top_k=40,
                top_p=0.9
            )

            print(f"\nAI-Frajerka: {reply}")

            # Pridanie odpovede AI do histórie
            conversation_history.append({"role": "assistant", "content": reply})

            # Udržiavanie histórie v rozumných hraniciach (posledných 10 správ)
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            print("\nRelácia ukončená.")
            break

if __name__ == "__main__":
    start_cli_chat()