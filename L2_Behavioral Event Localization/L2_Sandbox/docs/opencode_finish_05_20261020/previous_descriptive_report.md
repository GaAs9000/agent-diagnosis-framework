# OpenCode finite-batch terminal report

Plan: `69c0c6fe13d903a1d87e90877973de7daab7bd7a738b791422b31eaa974f5c55`. Cross-actor exploratory; D2 remains closed.

Planned 2282; valid 1792; invalid/incomplete 78; unstarted 412. Units 30; cells 163.

Reported settled actual tokens: 644288496; unknown reserved tokens: 355176472 (233 requests).

HTTP status counts: `{'200': 16858, '400': 1, '503': 2, 'no_http_status': 230}`. Unknown-usage HTTP statuses: `{'400': 1, '503': 2, 'no_http_status': 230}`.

Costs are usage equivalents, not billed cash. Unknown reservations are retained.

| Population | Arm | Valid | Invalid/incomplete | Unstarted | Eligible units | Matched units | Arm mean | Matched P mean | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure | D | 264 | 7 | 65 | 24 | 24 | 0.004167 | 0.050000 | -0.045833 |
| failure | P | 255 | 4 | 77 | 24 | 24 | 0.050000 | 0.050000 | 0.000000 |
| failure | R_explicit | 253 | 13 | 70 | 24 | 24 | 0.020833 | 0.050000 | -0.029167 |
| failure | R_implicit | 249 | 19 | 68 | 24 | 24 | 0.318687 | 0.050000 | 0.268687 |
| failure | R_single | 190 | 11 | 65 | 19 | 19 | 0.335141 | 0.063158 | 0.271983 |
| failure | S | 254 | 15 | 67 | 24 | 24 | 0.042593 | 0.050000 | -0.007407 |
| passing | D | 80 | 4 | 0 | 6 | 6 | 1.000000 | 1.000000 | 0.000000 |
| passing | P | 84 | 0 | 0 | 6 | 6 | 1.000000 | 1.000000 | 0.000000 |
| passing | R_explicit | 83 | 1 | 0 | 6 | 6 | 0.988095 | 1.000000 | -0.011905 |
| passing | S | 80 | 4 | 0 | 6 | 6 | 0.988095 | 1.000000 | -0.011905 |

All means above weight each matched unit equally. Per-cell valid n follows.

| Unit | Arm | Planned | Valid | Invalid/incomplete | Unstarted | Target mean |
|---|---|---:|---:|---:|---:|---:|
| officeval_003.r00::ITEM::0 | D | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_003.r00::ITEM::0 | P | 14 | 10 | 1 | 3 | 0.600000 |
| officeval_003.r00::ITEM::0 | R_explicit | 14 | 10 | 0 | 4 | 0.100000 |
| officeval_003.r00::ITEM::0 | R_implicit | 14 | 10 | 1 | 3 | 0.300000 |
| officeval_003.r00::ITEM::0 | R_single | 14 | 10 | 1 | 3 | 0.300000 |
| officeval_003.r00::ITEM::0 | S | 14 | 10 | 0 | 4 | 0.500000 |
| officeval_003.r01::ITEM::0 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_003.r01::ITEM::0 | P | 14 | 10 | 0 | 4 | 0.100000 |
| officeval_003.r01::ITEM::0 | R_explicit | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_003.r01::ITEM::0 | R_implicit | 14 | 10 | 1 | 3 | 0.200000 |
| officeval_003.r01::ITEM::0 | R_single | 14 | 11 | 0 | 3 | 0.090909 |
| officeval_003.r01::ITEM::0 | S | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_003.r02::ITEM::0 | D | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_003.r02::ITEM::0 | P | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_003.r02::ITEM::0 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_003.r02::ITEM::0 | R_implicit | 14 | 10 | 0 | 4 | 0.100000 |
| officeval_003.r02::ITEM::0 | R_single | 14 | 10 | 1 | 3 | 0.200000 |
| officeval_003.r02::ITEM::0 | S | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_003.r03::ITEM::0 | D | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_003.r03::ITEM::0 | P | 14 | 10 | 0 | 4 | 0.400000 |
| officeval_003.r03::ITEM::0 | R_explicit | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_003.r03::ITEM::0 | R_implicit | 14 | 10 | 1 | 3 | 0.500000 |
| officeval_003.r03::ITEM::0 | R_single | 14 | 11 | 0 | 3 | 0.454545 |
| officeval_003.r03::ITEM::0 | S | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_003.r04::ITEM::0 | D | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_003.r04::ITEM::0 | P | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_003.r04::ITEM::0 | R_explicit | 14 | 10 | 0 | 4 | 0.100000 |
| officeval_003.r04::ITEM::0 | R_implicit | 14 | 11 | 0 | 3 | 0.181818 |
| officeval_003.r04::ITEM::0 | R_single | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_003.r04::ITEM::0 | S | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_004.r00::PASSING_ITEM::0 | D | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::0 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::0 | R_explicit | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::0 | S | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::1 | D | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::1 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::1 | R_explicit | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::1 | S | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::10 | D | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::10 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::10 | R_explicit | 14 | 14 | 0 | 0 | 0.928571 |
| officeval_004.r00::PASSING_ITEM::10 | S | 14 | 14 | 0 | 0 | 0.928571 |
| officeval_004.r00::PASSING_ITEM::2 | D | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::2 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::2 | R_explicit | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::2 | S | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::4 | D | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::4 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::4 | R_explicit | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::4 | S | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::5 | D | 14 | 13 | 1 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::5 | P | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::5 | R_explicit | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_004.r00::PASSING_ITEM::5 | S | 14 | 14 | 0 | 0 | 1.000000 |
| officeval_010.r01::ITEM::0 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::0 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::0 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::0 | R_implicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::0 | R_single | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::0 | S | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::1 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::1 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::1 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::1 | R_implicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::1 | R_single | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::1 | S | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::10 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::10 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::10 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::10 | R_implicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::10 | R_single | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::10 | S | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::7 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r01::ITEM::7 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r01::ITEM::7 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::7 | R_implicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r01::ITEM::7 | R_single | 14 | 9 | 1 | 4 | 0.000000 |
| officeval_010.r01::ITEM::7 | S | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r02::ITEM::31 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_010.r02::ITEM::31 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_010.r02::ITEM::31 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r02::ITEM::31 | R_implicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_010.r02::ITEM::31 | R_single | 14 | 9 | 1 | 4 | 0.222222 |
| officeval_010.r02::ITEM::31 | S | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r00::ITEM::32 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r00::ITEM::32 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_050.r00::ITEM::32 | R_explicit | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r00::ITEM::32 | R_implicit | 14 | 9 | 2 | 3 | 0.000000 |
| officeval_050.r00::ITEM::32 | R_single | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_050.r00::ITEM::32 | S | 14 | 9 | 1 | 4 | 0.000000 |
| officeval_050.r00::ITEM::35 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r00::ITEM::35 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_050.r00::ITEM::35 | R_explicit | 14 | 9 | 2 | 3 | 0.000000 |
| officeval_050.r00::ITEM::35 | R_implicit | 14 | 9 | 2 | 3 | 0.777778 |
| officeval_050.r00::ITEM::35 | R_single | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_050.r00::ITEM::35 | S | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_050.r02::ITEM::19 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r02::ITEM::19 | P | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_050.r02::ITEM::19 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_050.r02::ITEM::19 | R_implicit | 14 | 9 | 1 | 4 | 0.777778 |
| officeval_050.r02::ITEM::19 | R_single | 14 | 9 | 1 | 4 | 0.000000 |
| officeval_050.r02::ITEM::19 | S | 14 | 10 | 1 | 3 | 0.100000 |
| officeval_050.r04::ITEM::19 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r04::ITEM::19 | P | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_050.r04::ITEM::19 | R_explicit | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_050.r04::ITEM::19 | R_implicit | 14 | 9 | 2 | 3 | 0.111111 |
| officeval_050.r04::ITEM::19 | R_single | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_050.r04::ITEM::19 | S | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_070.r03::ITEM::1 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_070.r03::ITEM::1 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_070.r03::ITEM::1 | R_explicit | 14 | 10 | 0 | 4 | 0.200000 |
| officeval_070.r03::ITEM::1 | R_implicit | 14 | 11 | 0 | 3 | 1.000000 |
| officeval_070.r03::ITEM::1 | R_single | 14 | 11 | 0 | 3 | 1.000000 |
| officeval_070.r03::ITEM::1 | S | 14 | 9 | 2 | 3 | 0.222222 |
| officeval_085.r00::GATE_NO_VALID_DELIVERY | D | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r00::GATE_NO_VALID_DELIVERY | P | 14 | 12 | 0 | 2 | 0.000000 |
| officeval_085.r00::GATE_NO_VALID_DELIVERY | R_explicit | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r00::GATE_NO_VALID_DELIVERY | R_implicit | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r00::GATE_NO_VALID_DELIVERY | S | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r01::GATE_NO_VALID_DELIVERY | D | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r01::GATE_NO_VALID_DELIVERY | P | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r01::GATE_NO_VALID_DELIVERY | R_explicit | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r01::GATE_NO_VALID_DELIVERY | R_implicit | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r01::GATE_NO_VALID_DELIVERY | S | 14 | 12 | 0 | 2 | 0.000000 |
| officeval_085.r02::GATE_NO_VALID_DELIVERY | D | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r02::GATE_NO_VALID_DELIVERY | P | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r02::GATE_NO_VALID_DELIVERY | R_explicit | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r02::GATE_NO_VALID_DELIVERY | R_implicit | 14 | 12 | 0 | 2 | 0.000000 |
| officeval_085.r02::GATE_NO_VALID_DELIVERY | S | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r03::GATE_NO_VALID_DELIVERY | D | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r03::GATE_NO_VALID_DELIVERY | P | 14 | 12 | 0 | 2 | 0.000000 |
| officeval_085.r03::GATE_NO_VALID_DELIVERY | R_explicit | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r03::GATE_NO_VALID_DELIVERY | R_implicit | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r03::GATE_NO_VALID_DELIVERY | S | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r04::GATE_NO_VALID_DELIVERY | D | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r04::GATE_NO_VALID_DELIVERY | P | 14 | 12 | 0 | 2 | 0.000000 |
| officeval_085.r04::GATE_NO_VALID_DELIVERY | R_explicit | 14 | 13 | 0 | 1 | 0.000000 |
| officeval_085.r04::GATE_NO_VALID_DELIVERY | R_implicit | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_085.r04::GATE_NO_VALID_DELIVERY | S | 14 | 12 | 1 | 1 | 0.000000 |
| officeval_090.r00::ITEM::2 | D | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_090.r00::ITEM::2 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_090.r00::ITEM::2 | R_explicit | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_090.r00::ITEM::2 | R_implicit | 14 | 10 | 1 | 3 | 1.000000 |
| officeval_090.r00::ITEM::2 | R_single | 14 | 11 | 0 | 3 | 1.000000 |
| officeval_090.r00::ITEM::2 | S | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r01::ITEM::2 | D | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r01::ITEM::2 | P | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_090.r01::ITEM::2 | R_explicit | 14 | 9 | 1 | 4 | 0.000000 |
| officeval_090.r01::ITEM::2 | R_implicit | 14 | 10 | 0 | 4 | 0.900000 |
| officeval_090.r01::ITEM::2 | R_single | 14 | 10 | 1 | 3 | 1.000000 |
| officeval_090.r01::ITEM::2 | S | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r02::ITEM::2 | D | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r02::ITEM::2 | P | 14 | 11 | 0 | 3 | 0.000000 |
| officeval_090.r02::ITEM::2 | R_explicit | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_090.r02::ITEM::2 | R_implicit | 14 | 10 | 0 | 4 | 0.900000 |
| officeval_090.r02::ITEM::2 | R_single | 14 | 9 | 2 | 3 | 1.000000 |
| officeval_090.r02::ITEM::2 | S | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_090.r04::ITEM::2 | D | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r04::ITEM::2 | P | 14 | 10 | 0 | 4 | 0.000000 |
| officeval_090.r04::ITEM::2 | R_explicit | 14 | 10 | 1 | 3 | 0.000000 |
| officeval_090.r04::ITEM::2 | R_implicit | 14 | 10 | 0 | 4 | 0.900000 |
| officeval_090.r04::ITEM::2 | R_single | 14 | 10 | 0 | 4 | 1.000000 |
| officeval_090.r04::ITEM::2 | S | 14 | 10 | 1 | 3 | 0.000000 |

- Scores are frozen target rubric-item or gate endpoints, not whole-benchmark scores.
- Planned n=14 per cell does not imply 14 valid observations; consult each cell census.
- Arm comparisons use the same arm-eligible units with valid arm and P baseline means.
- Missing and invalid observations are excluded from score means and remain separately counted.
- Cross-actor exploratory results do not open D2 or establish a formal p-value gate.
- Passing controls share parent families; pooled harm comparisons are descriptive only.
- Unknown usage reservations remain charged in accounting; usage equivalents are not cash invoices.
- Package integrity was enforced by execution; portable reporting verifies receipt seals and bindings only.
