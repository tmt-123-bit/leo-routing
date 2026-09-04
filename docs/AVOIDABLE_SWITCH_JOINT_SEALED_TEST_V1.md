# Avoidable-Switch Joint Sealed-Test Protocol v1

## Authorization boundary

This protocol is fixed before any workload in `78001..78050` is instantiated.
Execution requires a separate self-hashed authorization artifact that binds
this document, the sealed-test runner, and these completed freezes:

- MAPPO training: `153b5cd85305f3f4c9a03fd549eea3e06785a0d1a447b96ec9005098c7d25742`
- MAPPO validation: `d3c039657802e9c784c82cec31024e5098b31e5a3e217e69bbc64dad5a8469d6`
- classical training: `df7cf104d30360df898ef8239b64789c16e0baa9e87c78c52c08a676ecb62eba`
- classical gate: `36bac779a6961ce38eb6872ec0f9cfa3953898ceb00d9b5bd7f10ce80fbc3a1e`

The operator authorized this one-time test on 2026-09-04 after both validation
chains completed without sealed-test access. No method, checkpoint, model,
scenario, seed, workload, endpoint, or analysis choice may change after the
authorization artifact is written.

## Frozen grid

Scenarios are `medium_load` and `hotspot_high_load`. Workloads are exactly
`78001..78050`. The evaluated methods are:

- MAPPO `qos_only_baseline`, `qos_only_constrained`, and
  `reward_shaped_control`, each with the eight frozen policy seeds;
- newly trained Q-routing and OSPF-ECMP with the same eight identities;
- deterministic Global Dijkstra with one sentinel identity.

The complete grid contains 4,100 rows: 2,400 MAPPO, 800 Q-routing, 800 OSPF,
and 100 Global Dijkstra rows. The entire grid is evaluated regardless of
interim outcomes. No result-dependent retry, seed replacement, checkpoint
selection, scenario removal, or method removal is allowed.

## Primary estimands and decision

For each scenario, the treatment is `qos_only_constrained` and the reference
is `qos_only_baseline`. Policy seed is the independent replication unit and
the 50 workloads are paired common random numbers.

- Delivery effect: equal-weight mean of within-seed paired workload means.
- Decision-level avoidable-switch rate: within each seed, ratio of summed
  avoidable decisions to summed opportunities, then equal weight over seeds.
- Rate effect: constrained rate minus baseline rate.

The paper claim is authorized only if all four conditions pass in both
scenarios:

1. the one-sided 95% crossed-bootstrap lower bound for delivery effect is at
   least `-0.02`;
2. the one-sided 95% crossed-bootstrap upper bound for rate effect is below
   zero;
3. the one-sided 95% crossed-bootstrap upper bound for the constrained rate is
   at most `0.12`;
4. every constrained policy-seed rate is at most `0.12`.

Intervals use 5,000 crossed bootstrap draws over policy seeds and workloads.
Two-sided exact sign-flip p-values over the eight policy-seed effects are
reported as sensitivity analyses and Holm-adjusted over the four
scenario-by-endpoint treatment/reference comparisons.

## Secondary comparisons and reporting

All six method means, delivery ratios, decision-level rates, packet loss,
delay, and queue metrics are reported. Paired comparisons against Q-routing
and OSPF may be labeled secondary; Global Dijkstra is descriptive because it
has one deterministic routing identity. These comparisons cannot rescue a
failed primary decision.

The final freeze must bind every row, selected checkpoint, Q model, source
freeze, authorization artifact, statistics file, and runner source hash. A
partial grid cannot support a paper claim.
