# Long reasoning on small models

Small models (Qwen3-0.6B and 1.7B here) write long replies fluently, but
they go wrong long before they stop. This page covers what `ChatSession`
offers against that and what a live canary measured.

## What the session offers

- **Presence penalty** (`SamplingParams.presencePenalty`): subtracted from
  the logit of every token the reply has already sampled, which makes a
  loop less likely to start. Qwen suggests up to 1.5 for its quantized
  models.
- **Thinking budget** (`ChatSession.thinkingBudget`,
  `LocalModel.chat(thinkingBudget:)`): with `enableThinking`, reasoning
  that reaches the budget gets `thinkingCutoff` fed as if generated, and
  the model answers from there (`ChatTurn.thinkingCut`).
- **Loop detection** (`RepetitionGuard`): a reply that ends in the same
  block of tokens repeated closes its reasoning like the budget does, or
  ends, outside the reasoning (`ChatTurn.stoppedOnRepetition`).
- **Line checks** (`send(checkLine:, lineRetries:)`): each line is
  checked when it ends. A rejected line is rewound and sampled again,
  hotter than `sampling`, up to `lineRetries` times; then the reply ends at
  the last line that passed (`ChatTurn.stoppedOnCheck`). Rewinding is a
  step on Qwen3 and a line's worth of tokens on LFM3, from the checkpoint
  at the previous line.
- **Inserted text** (`send(insert:)`): after each generated token a tool
  may write the next part of the reply, for example the result of a sum
  the model has just written out. The model continues from it
  (`ChatTurn.insertedTokens`).

## The canary

`tool/reasoning_canary.dart` gives the model a running sum: add N
two-digit numbers one at a time, one line per step, `i: previous + number
= total`. Every line can be graded as it streams in, and each mistake is
one of:

| Kind | The line |
|---|---|
| `format` | is not `i: a + b = c` |
| `step` | has the wrong step number |
| `operand` | adds a number other than the step's |
| `carry` | does not start from the line before's total |
| `arithmetic` | adds wrong |

The canary reports how many steps were right before the first mistake.
Options: `--indexed=on` labels each number with its step (`31) 71`),
`--verify=on` rejects wrong lines through `checkLine`, and
`--calculator=on` writes each sum through `insert`, from the model's own
two numbers, so the model still has to count, find and carry.

```bash
dart run tool/reasoning_canary.dart <bundle> --context=8160 \
  --steps=25,50,100,200 --rounds=24 --indexed=on --calculator=on \
  --out=build/canary/run.jsonl
```

## Results

Qwen3 Q8 on an Apple silicon Mac (Metal), context 8,160, temperature 0.7,
top-k 20, top-p 0.8, seeds 1 to 6, six rounds a length. Each cell is the
median count of steps right before the first mistake, with the rounds
fully right in parentheses. Plain runs also had 300-step rounds,
whose first mistakes are counted in the last column. A 200-step answer is about 4,000 tokens.

| Model and conditions | N=25 | N=50 | N=100 | N=200 | First mistakes |
|---|---|---|---|---|---|
| 0.6B | 25 (4/6) | 29 (1/6) | 10 (0/6) | 5 (0/6) | operand 4, arithmetic 11, carry 3 |
| 0.6B, indexed | 25 (5/6) | 50 (4/6) | 66 (2/6) | 50 (0/6) | arithmetic 12, carry 1 |
| 0.6B, indexed, verified | 25 (5/6) | 50 (4/6) | 67 (2/6) | 67 (0/6) | arithmetic 2 |
| 0.6B, indexed, calculator | 25 (6/6) | 50 (6/6) | 100 (5/6) | 199 (3/6) | operand 2, carry 1, step 1 |
| 1.7B | 25 (4/6) | 36.5 (2/6) | 39.5 (1/6) | 39 (0/6) | operand 11, arithmetic 10, format 2 |
| 1.7B, indexed | 25 (4/6) | 41 (3/6) | 44 (1/6) | 61.5 (0/6) | arithmetic 16 |
| 1.7B, indexed, verified | 25 (5/6) | 50 (5/6) | 44 (1/6) | 70.5 (0/6) | arithmetic (rejected 5 times) |
| 1.7B, indexed, calculator | 25 (6/6) | 50 (6/6) | 100 (6/6) | 200 (6/6) | none |

What the runs show:

- **Plain, both models lose their place.** 0.6B also often quits early,
  writing a tenth of a long list. The bigger model holds about twice as
  long, but its first mistakes are still half finding the wrong number.
- **Numbering the inputs removes those.** With each number labeled, no
  round of either size added the wrong number first. What is left is
  adding wrong, and then 0.6B and 1.7B are about the same.
- **Checking and resampling fixes only slips.** A wrong line the model is
  unsure of comes back right on a hotter retry, for about 1.1 times the
  tokens. A wrong sum it is sure of comes back wrong on every retry, so
  verified rounds stop there.
- **A calculator removes the rest.** With the sums written for it, 1.7B
  got every round right up to 200 steps, and 0.6B 21 of 24, its last
  mistakes again finding or carrying a number. The calculator writes
  about 30% of the reply's tokens.

In short: make each step depend only on what is in front of the model,
check its lines to catch slips, and give a tool what the model gets
confidently wrong. A bigger model alone moved none of these limits much.
