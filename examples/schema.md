# The panel the experiments expect

**The panels used in the paper are OptionMetrics data and are not redistributable.**
They are not in this repository. What follows is the schema the scripts read, so
that a licensed export can be dropped in.

The experiment scripts read one CSV per underlying, in the layout an end-of-day
option quote file is normally exported in. The columns actually used
are listed below; anything else in the file is ignored.

| column | meaning |
|---|---|
| `[QUOTE_DATE]` | quote date, one row per underlying-date-expiry-strike |
| `[UNDERLYING_LAST]` | spot |
| `[STRIKE]` | strike |
| `[DTE]` | days to expiry; the code divides by 365 |
| `[C_BID]`, `[C_ASK]` | call quote |
| `[P_BID]`, `[P_ASK]` | put quote |
| `[C_IV]`, `[P_IV]` | the target |
| `[C_DELTA]`, `[C_GAMMA]`, `[C_VEGA]`, `[C_THETA]` | used only by Section III of the paper, which identifies them as functions of the target and then excludes them |
| `[P_DELTA]`, `[P_GAMMA]`, `[P_VEGA]`, `[P_THETA]` | as above |

Each row is split into a call side and a put side, so one input row becomes two
observations. The cleaning that every script applies is in
`experiments/t4_clean_sweep.py`:

    0.01 < IV < 4,  DTE/365 > 1/365,  strike > 0,  ask >= bid,
    0.5 <= spot/strike <= 2.0

The admitted feature vector is then the seven columns

    m = spot/strike,  log m,  tau,  strike,  spot,  is_call,
    spread = ask - bid

and Step 1 appends the eight interaction terms of Section S4 of the supplement.
No sensitivity and no mid quote enters, for the reason given in Section III.

## Pointing the scripts at your own file

`experiments/t4_clean_sweep.py` holds the path in `PANEL`. Set it to your own
export, or set the environment variable and edit that one line; every other
script imports `load()` from it.
