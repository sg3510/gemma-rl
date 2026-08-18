# Failure taxonomy

130 rollouts: **90 passed**, 40 did not.

| category | count | % of failures | what it means |
|---|---|---|---|
| `malformed_tool_args` | 9 | 22% | Emitted tool-call arguments that were not valid JSON. |
| `tool_arg_error` | 10 | 25% | Called a real tool with arguments the env rejected. |
| `no_mutation` | 3 | 8% | Never called any write tool, though the task required one. |
| `missing_output` | 8 | 20% | Did the right mutations but never stated required info. |
| `wrong_mutation` | 10 | 25% | Wrote to the database, but the final state was wrong. |

**Mechanical** (harness/format problems): 19 (48% of failures)
**Capability** (the thing RL is for): 21 (52% of failures)

## Evidence

**`malformed_tool_args`**

- task 0 (trial_0): `Extra data: line 1 column 29 (char 28)`
- task 1 (trial_0): `Extra data: line 1 column 29 (char 28)`
- task 7 (trial_0): `Extra data: line 1 column 29 (char 28)`

**`tool_arg_error`**

- task 4 (trial_0): `Error: user not found`
- task 19 (trial_0): `Error: non-delivered order cannot be exchanged`
- task 27 (trial_0): `Error: non-delivered order cannot be exchanged`

**`no_mutation`**

- task 92 (trial_0): `task needed 2 write(s), agent made 0`
- task 96 (trial_0): `task needed 2 write(s), agent made 0`
- task 109 (trial_0): `task needed 1 write(s), agent made 0`

**`missing_output`**

- task 3 (trial_0): `never stated: ['10']`
- task 34 (trial_0): `never stated: ['1093.34']`
- task 60 (trial_0): `never stated: ['242.92']`

**`wrong_mutation`**

- task 28 (trial_0): `called ['cancel_pending_order', 'return_delivered_order_items']`
- task 49 (trial_0): `called ['exchange_delivered_order_items']`
- task 64 (trial_0): `called ['modify_pending_order_items']`

