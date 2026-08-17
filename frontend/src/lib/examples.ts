// Fictional demo campaigns (not in the Kickstarter dataset), dated inside the 2009–2017 training range.
// The same three are used in docs/DEMO_GUIDE.md, so results here match the documented walkthrough.
import type { CampaignRequest } from "../api/types";

export interface Example {
  label: string;
  blurb: string;
  campaign: CampaignRequest;
}

export const EXAMPLES: Example[] = [
  {
    label: "Smart planter",
    blurb: "Technology · $25k goal",
    campaign: {
      name: "Lumen: solar smart garden planter",
      main_category: "Technology",
      category: "Gadgets",
      currency: "USD",
      country: "US",
      usd_goal_real: 25000,
      launched: "2017-03-06 10:00:00",
      deadline: "2017-04-05 10:00:00",
    },
  },
  {
    label: "Card game",
    blurb: "Games · $8k goal",
    campaign: {
      name: "Hollow Kingdom: a cooperative dungeon card game",
      main_category: "Games",
      category: "Tabletop Games",
      currency: "USD",
      country: "US",
      usd_goal_real: 8000,
      launched: "2016-09-06 10:00:00",
      deadline: "2016-10-06 10:00:00",
    },
  },
  {
    label: "Home robot",
    blurb: "Technology · $120k goal",
    campaign: {
      name: "Atlas One: modular home robot",
      main_category: "Technology",
      category: "Hardware",
      currency: "USD",
      country: "US",
      usd_goal_real: 120000,
      launched: "2017-01-10 10:00:00",
      deadline: "2017-03-11 10:00:00",
    },
  },
];

export const LINKS = {
  linkedin: "https://www.linkedin.com/in/yash-raj-a54a67323/",
};
