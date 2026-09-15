import argparse

def main():
    parser = argparse.ArgumentParser(description="AI-Frajerka Core Management Script")
    parser.add_argument("mode", choices=["tokenizer", "pretrain", "instruction", "conversation", "align", "chat", "api", "eval"])
    args = parser.parse_args()

    if args.mode == "tokenizer":
        from tokenizer.train_tokenizer import train
        train()
    elif args.mode == "pretrain":
        from pretraining.run_pretrain import main as run
        run()
    elif args.mode == "instruction":
        from instruction_training.train_instruction import train_instruction_tuning
        train_instruction_tuning()
    elif args.mode == "conversation":
        from conversation_training.train_conversation import train_conversation_tuning
        train_conversation_tuning()
    elif args.mode == "align":
        from alignment.train_alignment import train_dpo_alignment
        train_dpo_alignment()
    elif args.mode == "chat":
        from inference.chat import start_cli_chat
        start_cli_chat()
    elif args.mode == "api":
        import uvicorn
        from api.app import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == "eval":
        from evaluation.evaluate_personality import evaluate_model
        evaluate_model()

if __name__ == "__main__":
    main()
