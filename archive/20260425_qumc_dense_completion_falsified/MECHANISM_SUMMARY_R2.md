# Mechanism Summary — qumc_minimal_strict_v2

Teacher PPL: 24.23
Basis: carlini_recovered (access=strict_black_box, queries=100)

| Variant | PPL m±s | KL m±s | Top-1 m±s | kd/ce early | kd/ce late | tail_w late | budget topk/probe |
|---|---|---|---|---|---|---|---|
| strict_topk_kd | 411.12 ± 7.11 | 3.2941 ± 0.0200 | 0.4057 ± 0.0025 | 1.8988 | 1.4044 | n/a | 40000/0 |
| completion_no_unc_strict | 524561.44 ± 32967.94 | 10.9754 ± 0.0708 | 0.0022 ± 0.0001 | 1.1780 | 0.6409 | n/a | 40000/80,000,000 |
| completion_uncertainty_strict | 123.90 ± 16.11 | 2.3661 ± 0.1216 | 0.3957 ± 0.0121 | 0.0009 | 0.0013 | 0.0011 | 40000/80,000,000 |
| full_logit_upper | 115.03 ± 16.88 | 1.9608 ± 0.1403 | 0.4036 ± 0.0158 | 0.6691 | 0.5078 | n/a | 0/0 |
| old_lc_simulator | 524561.44 ± 32967.94 | 10.9754 ± 0.0708 | 0.0022 ± 0.0001 | 1.1780 | 0.6409 | n/a | 0/0 |