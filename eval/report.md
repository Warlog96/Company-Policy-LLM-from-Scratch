# Evaluation report

Checkpoint: `checkpoints/qa.pt`. Greedy decoding on 155 held-out in-scope questions (new wording of trained facts) and 10 unseen out-of-scope questions.

## Summary

| Metric | Value |
|---|---|
| Answer accuracy (correct value appears) | 100.0% |
| Source accuracy (exact tag) | 100.0% |
| Answer + source both correct | 100.0% |
| Refusal accuracy (out-of-scope) | 80.0% |
| False refusals (in-scope refused) | 0.0% |
| Hallucination rate (in-scope, confident + wrong) | 0.0% |

## Per topic

| Topic | Questions | Answer acc. | Source acc. |
|---|---|---|---|
| attendance | 14 | 100% | 100% |
| benefits | 19 | 100% | 100% |
| code of conduct | 13 | 100% | 100% |
| it policy | 19 | 100% | 100% |
| leave | 26 | 100% | 100% |
| onboarding | 13 | 100% | 100% |
| reimbursement | 17 | 100% | 100% |
| travel | 18 | 100% | 100% |
| working hours | 16 | 100% | 100% |

## 10 good answers

| Question | Model answer | Model source | Expected |
|---|---|---|---|
| Hi, how many casual leaves do I get each year? | Employees get 12 days of casual leave per year. | HR Leave Policy §3 | Employees get 12 days of casual leave per year. [HR Leave Policy §3] |
| Regarding leave: How many sick leave days do I get per year? | Employees get 10 days of paid sick leave per year. | HR Leave Policy §4 | Employees get 10 days of paid sick leave per year. [HR Leave Policy §4] |
| Regarding leave: How many days of earned leave do I get per year? | Employees earn 18 days of earned leave per year, credited at 1.5 days per month. | HR Leave Policy §5 | Employees earn 18 days of earned leave per year, credited at 1.5 days per month. [HR Leave Policy §5] |
| am I allowed to carry forward unused casual leave | Unused casual leave lapses on 31 December and cannot be carried forward. | HR Leave Policy §3.2 | Unused casual leave lapses on 31 December and cannot be carried forward. [HR Leave Policy §3.2] |
| As an Arcvale employee, how much earned leave can I carry forward? | Up to 30 days of unused earned leave can be carried forward to the next year. | HR Leave Policy §5.3 | Up to 30 days of unused earned leave can be carried forward to the next year. [HR Leave Policy §5.3] |
| I'm a new employee. Am I allowed to encash my earned leave? | Earned leave above 30 days can be encashed at the basic salary rate at year end. | HR Leave Policy §5.4 | Earned leave above 30 days can be encashed at the basic salary rate at year end. [HR Leave Policy §5.4] |
| Could you check the policy? Do I need a medical certificate for sick leave? | A medical certificate is required for sick leave longer than 2 consecutive days. | HR Leave Policy §4.2 | A medical certificate is required for sick leave longer than 2 consecutive days. [HR Leave Policy §4.2] |
| As an Arcvale employee, what is the duration of maternity leave? | Female employees get 26 weeks of paid maternity leave. | HR Leave Policy §7 | Female employees get 26 weeks of paid maternity leave. [HR Leave Policy §7] |
| Quick question: What is the duration of paternity leave? | As per company policy, male employees get 10 working days of paid paternity leave, taken within 3 months of the birth. | HR Leave Policy §8 | Male employees get 10 working days of paid paternity leave, taken within 3 months of the birth. [HR Leave Policy §8] |
| Hi, how much adoption leave does an employee get? | Employees adopting a child under 3 years old get 12 weeks of paid adoption leave. | HR Leave Policy §9 | Employees adopting a child under 3 years old get 12 weeks of paid adoption leave. [HR Leave Policy §9] |

## 10 bad answers / hallucinations

| Question | Model answer | Model source | Expected |
|---|---|---|---|
| Can you help me with my maths homework? | As per company policy, multi-factor authentication is mandatory for email, VPN and all cloud apps. | IT Security Policy §3 | I don't know. This is not covered in the company policies. |
| When will the new office building open? | As per company policy, new employees attend a 3-day induction program in their first week. | Onboarding Policy §2 | I don't know. This is not covered in the company policies. |
