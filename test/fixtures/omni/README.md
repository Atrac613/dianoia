# LFM3-Omni fixtures

`config.json` is metadata from Liquid AI's LFM3-Omni-2B checkpoint at revision
`21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`. No model weights are included.

Source: https://huggingface.co/LiquidAI/LFM3-Omni-2B/blob/21d75123f86c1f7ffb4bef1ea202710c4e7ebe67/config.json

Checkpoint license: LFM Open License v1.0
https://huggingface.co/LiquidAI/LFM3-Omni-2B/blob/main/LICENSE

`prompt_oracle.json` contains prompts rendered by the pinned
`chat_template.jinja` and tokenized with the pinned `tokenizer.json`, generated
by `tool/omni_prompt_oracle.py`. The script renders the template as data with
jinja2; it does not run the checkpoint's Python code. The template leaves each
media placeholder single; the processor repeats it per encoder row. Tests use a
small synthetic byte tokenizer for ID placement and the oracle's marker IDs
only as documentation of the real vocabulary.
