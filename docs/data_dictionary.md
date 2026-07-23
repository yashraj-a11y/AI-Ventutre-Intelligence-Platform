# Data Dictionary

Column-level reference for the **AI Venture Intelligence Platform**, based on the
**Kickstarter Projects** dataset (`ks-projects-201801.csv`) — 378,661 campaigns,
15 raw columns.

- **Grain:** one row = one campaign
- **Primary key:** `ID`
- **Prediction target:** `state`

This is a living document — we'll expand descriptions, categories, and future-module
mappings as the project grows.

| Column | Type | Description | Category | Future Module | Notes |
| ------ | ---- | ----------- | -------- | ------------- | ----- |
| ID | Numeric | Unique campaign identifier | Identifier | — | Drop before modeling; not a feature |
| name | Text | Campaign title | Text | NLP Pipeline | Requires cleaning; 4 nulls; near-unique (375,764 / 378,661) |
| category | Categorical | Project sub-category (e.g. Product Design) | Categorical | Feature Engineering | High cardinality (159 levels); nested in main_category |
| main_category | Categorical | Top-level category (e.g. Film & Video) | Categorical | Feature Engineering | 15 levels; prefer over `category` for encoding |
| currency | Categorical | Native currency of the campaign | Categorical | Feature Engineering | 14 levels; ~78% USD; near-redundant with `country` |
| deadline | Date | Funding deadline | Temporal | Feature Engineering | Stored as string; parse to datetime |
| goal | Numeric | Funding target (native currency) | Numerical | Prediction Engine | Scale later; heavy right skew — log transform |
| launched | Date | Launch timestamp | Temporal | Feature Engineering | Parse to datetime; watch 1970-01-01 sentinel |
| pledged | Numeric | Amount pledged (native currency) | Numerical | Prediction Engine | ⚠️ Post-outcome — leakage risk for pre-launch model |
| state | Categorical | Final campaign outcome | **Target** | Prediction Engine | 6 classes; filter to successful/failed for binary task |
| backers | Numeric | Number of backers | Numerical | Prediction Engine | ⚠️ Post-outcome — leakage risk; heavy skew |
| country | Categorical | Creator's country | Categorical | Feature Engineering | 23 levels; ~77% US; group rare levels |
| usd pledged | Numeric | Pledged amount in USD (Kickstarter conv.) | Numerical | Prediction Engine | 3,797 nulls; legacy — prefer `usd_pledged_real` |
| usd_pledged_real | Numeric | Pledged amount in USD (Fixer.io conv.) | Numerical | Prediction Engine | Complete; ⚠️ post-outcome leakage risk |
| usd_goal_real | Numeric | Goal in USD (Fixer.io conv.) | Numerical | Prediction Engine | Scale later; use for cross-campaign comparison |

## Column categories at a glance

- **Identifier:** `ID`
- **Target:** `state`
- **Text:** `name`
- **Temporal:** `launched`, `deadline`
- **Numerical:** `goal`, `pledged`, `backers`, `usd pledged`, `usd_pledged_real`, `usd_goal_real`
- **Categorical:** `category`, `main_category`, `currency`, `country`

## Notes to revisit as the project expands

- **Currency comparability:** `goal` / `pledged` are in native currency and are not
  comparable across campaigns — prefer the `usd_*_real` columns for any math.
- **Leakage:** `pledged`, `backers`, `usd pledged`, `usd_pledged_real`, and anything
  derived from them are known only *after* a campaign ends. Exclude from a
  forward-looking (pre-launch) predictor; keep for descriptive analysis.
- **Engineered features** (from `src/preprocessing/feature_engineering.py`) will get
  their own section here as they stabilize.
