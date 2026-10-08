# Rider vs captain: top complaint themes (by priority = frequency x severity)

| # | Rider theme | % of rider reviews | Captain theme | % of captain reviews |
|---|---|---|---|---|
| 1 | Captains overcharging passengers | 23.7% | Captains earning insufficient due to high costs | 14.4% |
| 2 | Captains demand extra cash | 13.6% | Unresponsive customer support | 13.4% |
| 3 | Captains rude and demand extra cash | 7.7% | Insufficient earnings for captains | 6.4% |
| 4 | Captains frequently cancel rides | 5.9% | Insufficient order volume | 6.0% |
| 5 | Unresponsive Customer Support | 9.1% | Captains receiving low per‑km fares | 5.7% |

## Possible links (hypotheses, not findings)

- **Captains overcharging passengers** (rider) <-> **Captains receiving low per‑km fares** (captain): Drivers overcharge to compensate for low per‑km rates set by the platform *Test:* Compare trip fare breakdowns with per‑km rates and identify instances where final fare exceeds expected amount
- **Captains frequently cancel rides** (rider) <-> **Insufficient order volume** (captain): Low order volume leads drivers to cancel rides hoping for better matches *Test:* Analyze cancellation timestamps against order queue length and driver idle time metrics
- **Captains demand extra cash** (rider) <-> **Captains earning insufficient due to high costs** (captain): High operating costs make drivers request extra cash from riders *Test:* Correlate driver cost data (fuel, maintenance) with frequency of extra‑cash requests in ride logs

Pairs suggested by openai/gpt-oss-120b; they are co-occurrence hypotheses for a PM to test, not causal findings.