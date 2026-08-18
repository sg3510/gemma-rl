# Failure taxonomy

460 rollouts: **302 passed**, 158 did not.

| category | count | % of failures | what it means |
|---|---|---|---|
| `infra_error` | 1 | 1% | Rollout crashed (provider/API). Excluded from metrics. |
| `malformed_tool_args` | 49 | 31% | Emitted tool-call arguments that were not valid JSON. |
| `tool_arg_error` | 35 | 22% | Called a real tool with arguments the env rejected. |
| `no_mutation` | 9 | 6% | Never called any write tool, though the task required one. |
| `missing_output` | 18 | 11% | Did the right mutations but never stated required info. |
| `wrong_mutation` | 46 | 29% | Wrote to the database, but the final state was wrong. |

**Mechanical** (harness/format problems): 84 (53% of failures)
**Capability** (the thing RL is for): 73 (46% of failures)

## Evidence

**`infra_error`**

- task 45 (trial_2): `litellm.BadRequestError: OpenrouterException - {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"Requests ending wit`

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

