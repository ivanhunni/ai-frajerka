# Rozšírený slovenský dataset

Dataset je generovaný pre tento projekt a používa iba JSON súbory, aby zodpovedal aktuálnym loaderom v projekte.

Obsah:
- pretraining/general_text: súvislé slovenské texty
- pretraining/dialogue: dialógový text uložený v poli `text`
- instruction/question_answer: `instruction` + `output`
- instruction/conversation: `system` + `messages`
- instruction/reasoning: `problem` + `reasoning` + `output`
- personality: `system` + `messages`
- evaluation/conversation_tests/alignment_train.json: `prompt` + `chosen` + `rejected` pre DPO loader

Poznámka: personality súbory sú zámerne `.json`, nie `.jsonl`, pretože aktuálny ConversationDataset používa `json.load()` a celý súbor musí byť validný JSON.
