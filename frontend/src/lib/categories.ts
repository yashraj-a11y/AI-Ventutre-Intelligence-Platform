// Day 19 — the REAL Kickstarter option sets the backend understands (15 main categories).
export const MAIN_CATEGORIES = [
  "Games", "Technology", "Design", "Film & Video", "Music", "Publishing", "Food",
  "Fashion", "Art", "Comics", "Theater", "Photography", "Crafts", "Journalism", "Dance",
] as const;

export const CURRENCIES = ["USD", "GBP", "EUR", "CAD", "AUD", "SEK", "DKK"] as const;
export const COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "NL", "SE", "DK"] as const;
