Dataset
   ↓
Tokenizer
   ↓
Transformer architecture
   ↓
Pretraining
   ↓
Instruction training
   ↓
Conversation training
   ↓
Alignment
   ↓
Inference engine
   ↓
Tvoja AI


dataset/
│
├── pretraining/
│   ├── general_text
│   └── dialogue
│
├── instruction/
│   ├── question_answer
│   ├── conversation
│   └── reasoning
│
├── personality/
│   ├── casual.jsonl
│   ├── humor.jsonl
│   ├── emotional.jsonl
│   ├── romantic.jsonl
│   └── conflict.jsonl
│
└── evaluation/
    ├── personality_tests
    └── conversation_tests
jsonl